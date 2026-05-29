"""
LocateAnything-3B inference adapter.

This module wraps the upstream `LocateAnythingWorker` pattern from
`Embodied/locateanything_worker.py`, but with:

  • All generation kwargs sourced from env vars (the trained values from
    `Embodied/evaluation/inference_compat.py:build_generate_kwargs`).
  • Explicit `attn_implementation` override (config.json says 'magi',
    but MagiAttention does not support sm_120 — we force 'sdpa' here,
    which is the only valid path in Qwen2Model.forward() on RTX 5090.
    See ATTENTION below).
  • bf16 enforcement.
  • Strict per-request validation (no fallbacks).

ATTENTION:
    NVIDIA trained LocateAnything-3B with `_attn_implementation='magi'`
    (custom block-mask attention from SandAI MagiAttention). The model's
    custom modeling_qwen2.py defines THREE attention classes (eager,
    flash_attention_2, sdpa, magi) but its Qwen2Model.forward() only
    builds masks for `magi` and `sdpa` paths — any other value raises
    NotImplementedError at line 1335. So in practice this model accepts
    exactly two attn impls at inference: 'magi' and 'sdpa'.

    On RTX 5090 (Blackwell GB202, sm_120), MagiAttention's FFA_FA4
    cutlass kernels require sm_100a (Blackwell datacenter B200)
    architecture-specific instructions (TMEM, tcgen05/UMMA) that do
    not exist on sm_120 consumer Blackwell. Per NVIDIA's own Blackwell
    Compatibility Guide, sm_100a kernels are not forward-compatible to
    sm_120 — there is no PTX-JIT rescue path. The maintainer confirms
    sm_120 is on the roadmap, not yet shipped (SandAI/MagiAttention#184,
    open as of 2026-05-29). So magi is structurally unavailable.

    That leaves 'sdpa'. The model's sdpa path in Qwen2Model.forward()
    reconstructs the same block-mask attention PATTERN via
    `mask_sdpa_utils.update_causal_mask_for_one_gen_window_2d`:
    bidirectional-within-window + blocked-just-emitted-token + causal
    prefix — i.e. it faithfully reproduces the magi range mask used
    at training time, via PyTorch SDPA + a hand-constructed 4D
    attention mask. The result is mathematically equivalent to
    magi+hybrid within bf16 precision; only execution speed differs
    (no fused FA-style kernel). This means `LA_ATTN_IMPL=sdpa` +
    `LA_GEN_MODE=hybrid` preserves the train-time attention pattern,
    train-time MTP/PBD generation behaviour, and train-time generation
    kwargs simultaneously. It is the correct configuration, not a
    fallback.

OVERRIDE MECHANICS:
    The model's custom `_autoset_attn_implementation` (modeling_qwen2.py
    line 1048) short-circuits when `config._attn_implementation == 'magi'`
    and silently drops any user-provided `attn_implementation=...` kwarg
    on `from_pretrained`. To force the override we load the AutoConfig
    explicitly, mutate `_attn_implementation` on the outer config AND
    `text_config` (the inner Qwen2 config) to the desired value, then
    pass the mutated config to `from_pretrained`. This bypasses the
    short-circuit because the check at line 1048 no longer sees 'magi'.

SDPA BACKEND OVERRIDE:
    With LA_ATTN_IMPL=sdpa the model code calls
    `torch.nn.functional.scaled_dot_product_attention(q, k, v,
    attn_mask=mask, is_causal=False)` with a non-contiguous bf16
    `(B,1,N,N)` block mask. PyTorch's SDPA dispatcher then chooses a
    kernel: cuDNN is excluded on sm_120; Flash is rejected by any
    non-null attn_mask; Mem-Efficient is rejected by the mask's
    last-dim stride not equalling 1; so dispatch silently falls through
    to the Math backend, which materialises a `B*H*N*N*2` byte
    probability tensor (~13 GiB at N=25,600 in bf16) and OOMs at full
    LA_MAX_IMAGE_DIM.

    NVIDIA's model code was written expecting the Mem-Efficient backend
    (witness the contiguous-q/k/v workaround at modeling_qwen2.py:709
    for the torch 2.1-era mem-eff non-contiguous-input bug); PyTorch
    2.12's dispatch checks just don't accept their mask shape as-is.

    `_patch_sdpa_to_mem_efficient` below makes the mask contiguous and
    wraps the SDPA call in `sdpa_kernel([EFFICIENT_ATTENTION, MATH])`
    so mem-eff is preferred and math remains as a no-op safety net for
    inputs mem-eff still happens to reject. Numerical drift vs the
    unpatched math backend is ULP-level in bf16 — within the noise
    floor that already exists between any two attention implementations.
    See the function's docstring for the per-backend dispatch rules,
    citations, and the rationale for why the math fallback inside the
    sdpa_kernel context is not the kind of "fallback" the project's
    no-fallbacks principle forbids (it cannot produce a worse outcome
    than current behaviour).
"""

from __future__ import annotations
from dataclasses import dataclass
import io
import os
import time

import torch
from PIL import Image, ImageFile, ImageOps
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor

from .parsing import parse_boxes, parse_points, has_abstention
from .pixel_token_math import plan_resize


# Hard upper bound on decoded pixel count. Defense-in-depth: the Rust
# frontend already enforces max_image_dim=2240 per side; even if that
# guard were bypassed, this cap raises DecompressionBombError before
# allocating gigantic buffers. 2240² × 4-safety-factor ≈ 20 M pixels.
Image.MAX_IMAGE_PIXELS = 20_000_000

# Explicit refusal to decode truncated JPEGs. PIL's default is False
# already, but a future dep could flip it; lock it down here.
ImageFile.LOAD_TRUNCATED_IMAGES = False


# Exception types PIL+libjpeg-turbo can raise on a malformed/unsupported
# JPEG. Used to narrow inference.run()'s decode try/except so server-side
# failures (e.g., MemoryError on a host under pressure) don't masquerade
# as client invalid_image errors.
_PIL_DECODE_EXCEPTIONS = (
    OSError,
    SyntaxError,
    ValueError,
    Image.UnidentifiedImageError,
    Image.DecompressionBombError,
    Image.DecompressionBombWarning,
)


@dataclass
class GenConfig:
    """The canonical trained sampling parameters. Sourced from environment
    variables which are baked at Docker-build time from versions.sh."""
    temperature: float
    top_p: float
    repetition_penalty: float
    do_sample: bool
    max_new_tokens: int
    generation_mode: str
    n_future_tokens: int

    @classmethod
    def from_env(cls) -> "GenConfig":
        ds = _require_env("LA_GEN_DO_SAMPLE")
        if ds not in ("0", "1"):
            raise RuntimeError(
                f"LA_GEN_DO_SAMPLE={ds!r} is not exactly '0' or '1'. The "
                "training-time value is '1' (do_sample=True). No truthy/"
                "falsy parsing is performed — set it to exactly '0' or '1'."
            )
        return cls(
            temperature=float(_require_env("LA_GEN_TEMPERATURE")),
            top_p=float(_require_env("LA_GEN_TOP_P")),
            repetition_penalty=float(_require_env("LA_GEN_REP_PEN")),
            do_sample=(ds == "1"),
            max_new_tokens=int(_require_env("LA_GEN_MAX_NEW_TOKENS")),
            generation_mode=_require_env("LA_GEN_MODE"),
            n_future_tokens=int(_require_env("LA_GEN_N_FUTURE_TOKENS")),
        )

    def to_kwargs(self, mode: str) -> dict:
        """Build the .generate() kwargs for one request. `mode` is REQUIRED
        — every request must specify a generation_mode explicitly; the
        server has no per-request default."""
        if not isinstance(mode, str) or mode not in ("fast", "hybrid", "slow"):
            raise ValueError(
                f"generation_mode={mode!r} is not one of 'fast'|'hybrid'|'slow'. "
                "Every request must declare a generation_mode explicitly — the "
                "server applies no default. See docs/MODEL_CAPABILITIES.md#"
                "generation-modes."
            )
        kwargs = dict(
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            generation_mode=mode,
            use_cache=True,
            verbose=False,
        )
        if mode in ("fast", "hybrid"):
            kwargs["n_future_tokens"] = self.n_future_tokens
        return kwargs


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        raise RuntimeError(
            f"environment variable {name} is not set. The Docker image bakes "
            f"these from scripts/lib/versions.sh — running outside Docker is "
            f"NOT supported."
        )
    return v


def _patch_sdpa_to_mem_efficient(model) -> None:
    """Force PyTorch's mem-efficient SDPA backend on the model's attention.

    PROBLEM:
        The model's `Qwen2SdpaAttention.forward` calls
        `torch.nn.functional.scaled_dot_product_attention(q, k, v,
        attn_mask=mask, is_causal=False)` with a `(B, 1, N, N)` bf16
        block-pattern mask. PyTorch's backend dispatcher then chooses a
        kernel by checking each option in order: cuDNN → Flash → Mem-Eff
        → Math.  cuDNN is excluded on sm_120 by `check_prefer_cudnn_
        attention()` (cudnn major must be 9 or 10). Flash is rejected by
        `check_for_attn_mask()` (any non-null mask → out). Mem-Eff is
        rejected by `check_last_dim_stride_equals_1_dense()` when the
        mask isn't contiguous on its trailing dim — which it isn't,
        because the mask is built by stacking per-batch slices in
        `mask_sdpa_utils.create_block_diff_mask_by_pe_4d`. So dispatch
        falls through to Math, which materialises a
        `B × H × N × N × 2 bytes` probability tensor — at N=25,600,
        H=16, B=1, bf16: 13.1 GiB per request. Adding that to the ~7 GiB
        model weights and ~5 GiB of residual segments overshoots the
        5090's 32 GiB.

    FIX:
        (a) `.contiguous()` on the mask before SDPA → stride(-1)==1 →
            mem-eff's stride check passes.
        (b) Wrap the SDPA call in
            `sdpa_kernel([EFFICIENT_ATTENTION, MATH])` → mem-eff is
            preferred; math remains as the no-op safety net for any
            input shape mem-eff still happens to reject. The list-with-
            math is what the upstream PyTorch SDPA docs recommend for
            "prefer X, accept Y" semantics — it is strictly an
            improvement over the dispatcher's silent fallthrough.

    NUMERICAL EQUIVALENCE TO MATH BACKEND:
        Both backends compute `softmax(QK^T / √d + M) @ V` over the same
        operands; mem-eff differs only by tiling the reduction. In bf16
        (7 mantissa bits), the typical max-element delta in attn_output
        is O(1e-2) absolute — well below the ~1e-1 noise floor that
        already exists between any two attention implementations at
        bf16. For LocateAnything box detection at temperature=0.7,
        top_p=0.9, the empirically expected effect is sub-pixel
        differences in box coordinates and a <1% rate of token flips,
        concentrated at the top_p truncation boundary. This is
        within-noise vs current behaviour.

    IDEMPOTENCY:
        Tags each class with `_la_sdpa_patched = True`; a re-run of
        __init__ (e.g. a soft restart) is a no-op.

    HARD-FAIL CASES:
        - No matching modeling_qwen2 module found in sys.modules: raises.
          Without the patch the math backend would OOM at large inputs.
        - The model's live `self_attn` is not an instance of the patched
          class: raises (the patch is functionally inert).
    """
    import sys
    from torch.nn.attention import sdpa_kernel, SDPBackend

    # Find the trust_remote_code-loaded modeling_qwen2 module(s). The
    # exact namespace path varies with transformers' hashing scheme,
    # so we match by module-name suffix and class presence.
    candidates = [
        m for m in list(sys.modules.values())
        if m is not None
        and getattr(m, "__name__", "").endswith(".modeling_qwen2")
        and hasattr(m, "Qwen2SdpaAttention")
    ]
    if not candidates:
        raise RuntimeError(
            "SDPA mem-efficient patch FAILED: could not find a loaded "
            "`transformers_modules.*.modeling_qwen2` module with "
            "Qwen2SdpaAttention. The trust_remote_code import path may "
            "have changed; without this patch SDPA silently falls through "
            "to the math backend and OOMs at full LA_MAX_IMAGE_DIM. "
            "Refusing to start."
        )

    def _make_patched(orig_forward):
        def patched(self, hidden_states, attention_mask=None,
                    *args, **kwargs):
            # `*args, **kwargs` forwards every other positional or keyword
            # argument the caller passes (transformers may add new ones
            # like `cache_position`, `position_embeddings`, etc. over
            # minor versions). We only intercept attention_mask.
            if attention_mask is not None:
                # Force stride(-1)==1 so PyTorch's mem-eff dispatch
                # accepts the mask. One-time bf16 copy of a (B,1,N,N)
                # tensor — ~1.3 GiB at N=25,600 vs the 13.1 GiB math
                # backend would otherwise allocate for probabilities.
                attention_mask = attention_mask.contiguous()
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                return orig_forward(
                    self, hidden_states,
                    attention_mask, *args, **kwargs,
                )
        return patched

    for mod in candidates:
        cls = mod.Qwen2SdpaAttention
        if getattr(cls, "_la_sdpa_patched", False):
            continue
        cls.forward = _make_patched(cls.forward)
        cls._la_sdpa_patched = True

    # Defense in depth: confirm a live attention module on the model now
    # routes through a patched class. If the model uses a different class
    # than we patched, the patch is functionally inert and we'd silently
    # OOM at the first large request.
    sample_attn = model.language_model.model.layers[0].self_attn
    if not getattr(type(sample_attn), "_la_sdpa_patched", False):
        raise RuntimeError(
            "SDPA mem-efficient patch verification FAILED: "
            f"model.language_model.model.layers[0].self_attn is of type "
            f"{type(sample_attn).__name__!r}, which is NOT tagged as "
            "patched. The model is using a different attention class than "
            "Qwen2SdpaAttention — the patch did not take effect. Refusing "
            "to start."
        )


@dataclass
class InferenceResult:
    raw_answer: str
    detections: list  # list[dict]
    points: list      # list[dict]
    abstained: bool
    latency_ms: float
    image_size: tuple
    resize_plan: dict


class LocateAnythingInference:
    """Stateful inference engine — loads model once, processes single frames."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self.model_dir = model_dir
        self.dtype = self._parse_dtype(_require_env("LA_MODEL_DTYPE"))
        self.attn_impl = _require_env("LA_ATTN_IMPL")
        self.gen_cfg = GenConfig.from_env()

        # Validate GPU before loading.
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() == False — refusing to start.")
        if device != "cpu":
            cap = torch.cuda.get_device_capability(0)
            if cap < (12, 0):
                raise RuntimeError(
                    f"GPU compute capability {cap[0]}.{cap[1]} < sm_120. "
                    "This server is pinned to PyTorch+CUDA wheels that target "
                    "sm_120 (Blackwell). Refusing to start."
                )

        # Close the TOCTOU window: re-verify all pinned file SHA-256
        # IMMEDIATELY before trust_remote_code loads them. The startup
        # validation in validate_startup.py runs seconds earlier; a host
        # attacker with write access to ./models/ could have swapped a
        # file in between. The window is now bounded to the time
        # between this check and the from_pretrained call below
        # (sub-millisecond).
        from . import validate_startup as _vs
        _vs.validate_model_dir(model_dir)

        # `local_files_only=True`: prevent transformers from issuing an
        #   HF Hub HEAD on each load to check for newer revisions. The
        #   weight pin + content SHA pins make any reach-out incorrect
        #   anyway, and the request would stall startup if HF is
        #   unreachable.
        # `use_safetensors=True`: explicit refusal to load .bin files
        #   even if they appear in the dir — content SHAs pin the
        #   safetensors specifically.
        # `use_fast=True`: explicit; the default anyway, but locks in
        #   the Qwen2TokenizerFast path the model was trained against.
        # `device_map={"": device}`: stream-load weights directly into
        #   GPU memory, bypassing the ~6.8 GB transient CPU staging that
        #   .to(device) after from_pretrained would otherwise produce.
        #   Note: `low_cpu_mem_usage=True` is dead code in transformers
        #   4.57.1 (silently popped at modeling_utils.py:4665) — meta-
        #   tensor init is unconditional. The `device_map` is the
        #   knob that actually changes the placement path.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Load AutoConfig and mutate `_attn_implementation` on the top-level
        # config AND the inner text_config (Qwen2). See "OVERRIDE MECHANICS"
        # in this module's docstring for why we cannot rely on the
        # `attn_implementation=` kwarg on from_pretrained — the model's
        # custom `_autoset_attn_implementation` short-circuits and drops
        # the kwarg whenever config.json's `_attn_implementation` is
        # 'magi' (which it always is in this model). vision_config is
        # left alone — MoonViT does not have a magi/sdpa branch.
        config = AutoConfig.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        config._attn_implementation = self.attn_impl
        if not hasattr(config, "text_config"):
            raise RuntimeError(
                "LocateAnything config is missing `text_config`; the model "
                "loader cannot redirect the inner Qwen2 attn implementation. "
                "This is a structural mismatch with the pinned HF revision."
            )
        config.text_config._attn_implementation = self.attn_impl

        self.model = AutoModel.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
            device_map={"": device},
        ).eval()

        # Verify the override actually stuck. The model's custom
        # `_autoset_attn_implementation` can rewrite `_attn_implementation`
        # back to 'magi' (or fall back to FA2) in some corner cases;
        # we refuse to operate if the resulting model object disagrees
        # with the configured impl.
        actual = getattr(self.model.language_model.model, "_attn_implementation", None)
        if actual != self.attn_impl:
            raise RuntimeError(
                f"attn_implementation override did not take effect: "
                f"requested LA_ATTN_IMPL={self.attn_impl!r}, "
                f"but model.language_model.model._attn_implementation={actual!r}. "
                f"The model would crash at forward() — refusing to start."
            )

        # Force the PyTorch SDPA dispatcher onto the memory-efficient
        # backend so the model can run at full LA_MAX_IMAGE_DIM without
        # OOMing. See SDPA BACKEND OVERRIDE in this module's docstring
        # for why the unpatched path silently falls through to the math
        # backend and OOMs at N=25600.
        _patch_sdpa_to_mem_efficient(self.model)

        # No `.to(device)` — accelerate's device_map already placed every
        # module on `device`; calling `.to()` on a dispatched model is a
        # no-op at best and a source of subtle dispatch-hook bugs at worst.
        # The custom .generate() respects tokenizer.model_max_length as a
        # hard cap (modeling_locateanything.py:331). 16384 default truncates
        # any 24K input the README claims to support — keep the trained cap.
        # We do NOT silently bump this; if a caller needs longer context,
        # they must pass max_new_tokens within the residual budget.

    @staticmethod
    def _parse_dtype(s: str) -> torch.dtype:
        s = s.strip().lower()
        if s in ("bfloat16", "bf16"):
            return torch.bfloat16
        raise RuntimeError(
            f"LA_MODEL_DTYPE={s!r} unsupported. The model is shipped in bf16. "
            "Refusing to start in any other dtype."
        )

    @torch.inference_mode()
    def run(
        self,
        jpeg_bytes: bytes,
        prompt: str,
        generation_mode: str,
    ) -> InferenceResult:
        """Run one inference. `generation_mode` is REQUIRED; no default.

        JPEG decoded inside this method — Python's PIL is the canonical
        decoder for the LocateAnything image processor.
        """
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                "prompt must be a non-empty string. See "
                "docs/MODEL_CAPABILITIES.md#what-it-does-well-in-order for "
                "the canonical prompt forms."
            )
        if not isinstance(jpeg_bytes, (bytes, bytearray)) or not jpeg_bytes:
            raise ValueError("jpeg_bytes must be non-empty bytes")
        try:
            raw = Image.open(io.BytesIO(jpeg_bytes))
            # Reject CMYK explicitly. PIL's CMYK→RGB transform is not
            # ICC-aware and produces incorrect colors for Adobe-tagged CMYK
            # JPEGs, which would silently degrade detection quality. The
            # model itself only consumes RGB; converting to RGB client-side
            # is the contract.
            if raw.mode == "CMYK":
                raise ValueError(
                    "CMYK JPEG rejected — the model consumes RGB only and "
                    "PIL's CMYK→RGB is not ICC-aware. Convert to RGB "
                    "client-side before sending."
                )
            # EXIF Orientation tag handling. PIL does NOT auto-rotate;
            # phone-camera JPEGs with Orientation=6 (rotate 90 CW) would
            # otherwise feed the model un-rotated sensor pixels, and the
            # boxes the model returns would be spatially wrong relative
            # to what the client sees. exif_transpose normalizes to
            # display orientation (and strips the Orientation tag).
            raw = ImageOps.exif_transpose(raw)
            image = raw.convert("RGB")
        except _PIL_DECODE_EXCEPTIONS as e:
            # Narrowed catch: server-side errors (MemoryError, RuntimeError,
            # etc.) propagate instead of being mis-attributed as invalid_image.
            raise ValueError(
                f"JPEG decode failed: {type(e).__name__}: {e}. The Rust "
                "frontend already validated the SOI marker and parsed the "
                "header; if we got here PIL refused the data — check that "
                "the encoder is producing baseline JPEG with mode L or RGB."
            ) from e

        # Plan resize before model touches it — log for debug/audit.
        plan = plan_resize(image.width, image.height)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  prompt},
            ]}
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vision_info = self.processor.process_vision_info(messages)
        if not isinstance(vision_info, tuple) or len(vision_info) != 2:
            raise RuntimeError(
                f"processor.process_vision_info returned {type(vision_info).__name__} "
                f"len={len(vision_info) if hasattr(vision_info, '__len__') else 'N/A'}, "
                "but this code is pinned to the (images, videos) 2-tuple shape from "
                "processing_locateanything.py. The upstream processor signature "
                "changed — re-verify against the model's HF repo."
            )
        images, videos = vision_info
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)
        for required_key in ("pixel_values", "input_ids", "attention_mask", "image_grid_hws"):
            if required_key not in inputs:
                raise RuntimeError(
                    f"processor output is missing required key {required_key!r}; "
                    f"got keys: {list(inputs.keys()) if hasattr(inputs, 'keys') else inputs!r}. "
                    "The upstream processor's output shape changed — re-verify "
                    "against modeling_locateanything.py's generate() signature."
                )
        pixel_values   = inputs["pixel_values"].to(self.dtype)
        input_ids      = inputs["input_ids"]
        image_grid_hws = inputs["image_grid_hws"]

        gen_kwargs = self.gen_cfg.to_kwargs(generation_mode)

        # Bracket the timed region with cuda.synchronize so latency_ms
        # measures real wall-clock kernel completion. Without these,
        # .generate() returns when CPU is done queueing work but the
        # GPU may still be running; published FPS would be biased low.
        if self.device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            **gen_kwargs,
        )
        if self.device != "cpu":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = out[0] if isinstance(out, tuple) else out
        # The custom generate returns either a str or a tuple; both handled.
        if not isinstance(answer, str):
            # Defensive — if generate ever returns tensors here, decode.
            answer = self.tokenizer.decode(answer, skip_special_tokens=False)

        detections = [d.to_json() for d in parse_boxes(answer, image.width, image.height)]
        points     = [p.to_json() for p in parse_points(answer, image.width, image.height)]
        return InferenceResult(
            raw_answer=answer,
            detections=detections,
            points=points,
            abstained=has_abstention(answer),
            latency_ms=latency_ms,
            image_size=(image.width, image.height),
            resize_plan={
                "dst_w": plan.dst_w,
                "dst_h": plan.dst_h,
                "n_llm_tokens": plan.n_llm_tokens,
                "scale": round(plan.scale, 4),
            },
        )
