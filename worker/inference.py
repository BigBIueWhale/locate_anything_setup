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
    (no fused FA-style kernel). This means `LA_ATTN_IMPL=sdpa`
    preserves the train-time attention pattern and train-time
    generation kwargs simultaneously; combined with any per-request
    `generation_mode`, MTP/PBD generation behaviour matches training.
    It is the correct configuration, not a fallback.

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

    The MoonViT vision encoder has its own independent SDPA call in
    `modeling_vit.sdpa_attention` with THREE independent blockers
    (3D q/k/v, 3D mask, bool dtype). `_patch_vit_sdpa_to_mem_efficient`
    monkey-patches the module-level function with a replacement that
    rewrites all three to 4D float-mask form. The encoder layer looks
    up the attention function from `VL_VISION_ATTENTION_FUNCTIONS`
    on every forward call (line 463 of modeling_vit.py), so the
    module-level swap propagates without per-instance rebinding.

VISION-ENCODER FA2 OVERRIDE:
    Independent of the LLM-side magi→sdpa story above, the MoonViT
    vision encoder was *trained* with FlashAttention 2's varlen kernel
    (Moonshot's Kimi-VL paper §"variable-length sequence attention
    mechanism supported by FlashAttention"; mirrored in
    `/tmp/la_research/kimi_vl.txt:270-271`). The model code at
    `models/LocateAnything-3B/modeling_vit.py:571` declares
    `_supports_flash_attn_2 = True`, and NVIDIA's outer dispatcher at
    `models/LocateAnything-3B/modeling_locateanything.py:104` defaults
    the vision attn impl to `'flash_attention_2'` whenever
    `vision_config._attn_implementation` is unset.

    HuggingFace transformers 4.57.1's `_autoset_attn_implementation`,
    however, CASCADES the top-level `_attn_implementation` value into
    every sub-config that carries `_attn_implementation_autoset: True`
    when `from_pretrained` is called. Our LLM-side override at
    `_attn_implementation = 'sdpa'` (see OVERRIDE MECHANICS above)
    therefore silently propagates the `'sdpa'` value to
    `vision_config._attn_implementation` as a side effect, even though
    the only structural reason we set 'sdpa' was the LLM's
    sm_120/magi unbuildability — MoonViT has FA2 available and was
    trained on it.

    `_force_vit_flash_attn_2(config)` below counter-overrides this
    cascade BEFORE `from_pretrained` runs: it sets
    `config.vision_config._attn_implementation = 'flash_attention_2'`
    after strict pre-checks on FA2 availability, dispatch-dict shape,
    and vision-config structural identity. The post-load verification
    then re-reads `self.model.vision_model.config._attn_implementation`
    AND every encoder block's `attn_implementation` attribute to refuse
    the boot if FA2 did not propagate. Empirically (see
    `docs/PINNED_VERSIONS.md` §"vision-encoder FA2 override") this
    restores ~2-3× vit forward speedup at high resolution, translating
    to 1.8-2.1× end-to-end speedup at 2K-2240² — recovering the
    train-correct default that HF's auto-cascade had silently
    regressed.
"""

from __future__ import annotations
from dataclasses import dataclass
import io
import os
import time
import warnings

import torch
from PIL import Image, ImageFile, ImageOps
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor

from . import prompts
from .parsing import parse_boxes, parse_points
from .pixel_token_math import plan_resize


# Hard upper bound on decoded pixel count. Defense-in-depth: the Rust
# frontend already enforces max_image_dim=2240 per side; even if that
# guard were bypassed, this makes PIL refuse the image at the
# decompression-bomb check (before allocating the full pixel buffer).
# 2240² × 4-safety-factor ≈ 20 M pixels.
#
# PIL's default policy is asymmetric: it only *warns*
# (DecompressionBombWarning) above MAX_IMAGE_PIXELS and only *raises*
# (DecompressionBombError) above 2×MAX_IMAGE_PIXELS. A bare cap would
# therefore let the 20–40 M-pixel band through silently. We promote the
# warning to a hard error so anything above 20 M pixels fails loud, closing
# that gap. DecompressionBombWarning is also in _PIL_DECODE_EXCEPTIONS below,
# so the raised error is reported as a clean client invalid_image, not a
# server fault.
Image.MAX_IMAGE_PIXELS = 20_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

# Explicit refusal to decode truncated JPEGs. PIL's default is False
# already, but a future dep could flip it; lock it down here.
ImageFile.LOAD_TRUNCATED_IMAGES = False


# Expected flash-attn version, mirrored from
# `scripts/lib/versions.sh:LA_FLASH_ATTN_VERSION`. Kept as a module-level
# constant rather than read from an env var because the vision-encoder
# FA2 override is a baked-in train-correct default, not flag-toggleable
# (cf. the LA_ATTN_IMPL env contract for the LLM, which IS
# flag-toggleable for the magi/sdpa choice). If versions.sh is ever
# updated, update both sides — the Dockerfile already pip-installs
# from versions.sh, so a mismatch here will surface immediately at
# boot via `_force_vit_flash_attn_2`'s strict version check.
_EXPECTED_FLASH_ATTN_VERSION = "2.8.3"


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
    import inspect
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

    # ---- STRICT PRE-PATCH SHAPE CHECK -----------------------------------
    # Refuse to apply the patch unless the target class and its forward
    # method look EXACTLY like what we developed against. The model file
    # itself is already SHA-256 pinned in validate_startup.py, so this is
    # defense-in-depth — but in case anyone ever forgets to update the
    # pin alongside an upstream code change, the patch must NOT silently
    # wrap a function it doesn't understand.
    EXPECTED_QWEN2_FORWARD_PARAMS = (
        "self", "hidden_states", "attention_mask", "position_ids",
        "past_key_value", "output_attentions", "use_cache",
    )
    EXPECTED_QWEN2_BASE_NAME = "Qwen2Attention"
    for mod in candidates:
        cls = mod.Qwen2SdpaAttention
        if getattr(cls, "_la_sdpa_patched", False):
            continue
        # Class identity: must inherit from Qwen2Attention.
        base_names = tuple(b.__name__ for b in cls.__mro__)
        if EXPECTED_QWEN2_BASE_NAME not in base_names:
            raise RuntimeError(
                f"SDPA mem-efficient patch ABORTED on strict pre-check: "
                f"Qwen2SdpaAttention in module {mod.__name__!r} has MRO "
                f"{base_names!r} — Qwen2Attention is not in the chain. "
                "The class shape has drifted from what the patch was "
                "developed against. Refusing to apply."
            )
        # Forward signature: must match exactly. positional / keyword,
        # parameter names, parameter order.
        sig = inspect.signature(cls.forward)
        actual_params = tuple(sig.parameters)
        if actual_params != EXPECTED_QWEN2_FORWARD_PARAMS:
            raise RuntimeError(
                f"SDPA mem-efficient patch ABORTED on strict pre-check: "
                f"Qwen2SdpaAttention.forward signature has drifted. "
                f"Expected parameters {EXPECTED_QWEN2_FORWARD_PARAMS!r}, "
                f"observed {actual_params!r}. The wrapper would forward "
                "the wrong kwargs. Refusing to apply."
            )

    def _make_patched(orig_forward):
        def patched(self, *args, **kwargs):
            # We accept ANY positional/keyword pattern the caller uses and
            # only look up `attention_mask` by name. This is more robust
            # than binding it ourselves: a caller that ever passes it
            # positionally (i.e. as args[1]) would slot-mis-align if we
            # tried to declare it as a named parameter here. The strict
            # pre-check above already asserts the upstream forward's
            # parameter NAMES are exactly what we expect, so `kwargs.get
            # ("attention_mask")` finds the right tensor; if it was
            # passed positionally we just leave it alone (not contiguous-
            # ified), which falls through to the math backend in the
            # safety-net branch of sdpa_kernel below — strictly never
            # worse than the unpatched behaviour.
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                # Force stride(-1)==1 so PyTorch's mem-eff dispatch
                # accepts the mask. One-time bf16 copy of a (B,1,N,N)
                # tensor — ~1.3 GiB at N=25,600 vs the 13.1 GiB math
                # backend would otherwise allocate for probabilities.
                kwargs["attention_mask"] = kwargs["attention_mask"].contiguous()
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                return orig_forward(self, *args, **kwargs)
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


def _patch_vit_sdpa_to_mem_efficient() -> None:
    """Force PyTorch's mem-efficient SDPA backend on the MoonViT encoder.

    Twin of `_patch_sdpa_to_mem_efficient` but for the vision side.
    `modeling_vit.sdpa_attention` (module-level helper, not a class
    method) calls `F.scaled_dot_product_attention(q, k, v, mask, ...)`
    with THREE independent blockers for the mem-eff dispatcher in
    PyTorch v2.12.0:

      (1) q/k/v are 3D `(num_heads, N, head_dim)` — heads-as-batch.
          `check_tensor_shapes` in `sdp_utils_cpp.h:303-318` requires
          all of q.dim() == k.dim() == v.dim() == 4. `_efficient_
          attention_forward` (`attention.cu:1409-1411`) re-asserts 4D
          inputs. 3D is unconditionally rejected → math fallthrough.

      (2) the attention mask is 3D `(1, N, N)`. `check_attn_mask_shape`
          (`sdp_utils_cpp.h:269-301`) accepts only `dim()==2` or
          `dim()==4`; 3D mask → mem-eff rejected.

      (3) the mask is `torch.bool`. PyTorch top-level SDPA does convert
          bool→float via `at::where` (`attention.cpp:557-559`) before
          dispatch, but the conversion allocates a fresh full-size float
          tensor — under tight VRAM, that itself can OOM. Doing the
          conversion ourselves with `masked_fill_` is in-place over a
          tensor we already allocated.

    All three blockers must be lifted together. The replacement function
    rewrites the SDPA call to:
      - bool mask → 4D bf16 additive mask `(1, 1, N, N)` with -inf where
        the bool was False (numerically identical to PyTorch's own
        at::where conversion per `attention.cpp:558-559`).
      - q/k/v transpose to `(num_heads, N, head_dim)` THEN unsqueeze to
        `(1, num_heads, N, head_dim)`. The 4D batch dim is what mem-eff
        accepts; stride view, bit-exact.
      - sdpa_kernel([EFFICIENT_ATTENTION, MATH]) — mem-eff preferred,
        math is the no-op safety net.

    The function-level replacement is harder to verify than the class
    patch in `_patch_sdpa_to_mem_efficient` (no live attribute we can
    type-check), so we verify by re-reading `mod.sdpa_attention` and
    confirming our marker attribute is present.

    NOTE: this patch DOES NOT touch the multihead_attention
    (flash-attn-varlen) or eager_attention functions in the same
    module — only `sdpa_attention`, which is the only one routed by
    `_attn_implementation="sdpa"` (the dispatch dict at line 187-191).

    DORMANCY UNDER FA2 (current happy path):
        Since `_force_vit_flash_attn_2` runs immediately before
        from_pretrained and steers the encoder onto FA2 unconditionally
        (with strict pre-checks AND a post-load verification that
        refuses the boot otherwise), the replacement function installed
        here is dormant in the standard happy path — no encoder block
        ever looks up `VL_VISION_ATTENTION_FUNCTIONS["sdpa"]` at
        runtime. It is kept installed as defense-in-depth: if a future
        change to `_force_vit_flash_attn_2` or its post-load
        verification ever allowed a partial fall-back onto SDPA without
        re-raising, the math-backend OOM at full LA_MAX_IMAGE_DIM that
        this patch was designed to prevent would silently re-emerge.
        The post-load verification would have raised first, so this is
        a belt-and-suspenders guarantee, not the primary defence.
    """
    import sys
    import inspect
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend

    candidates = [
        m for m in list(sys.modules.values())
        if m is not None
        and getattr(m, "__name__", "").endswith(".modeling_vit")
        and hasattr(m, "sdpa_attention")
    ]
    if not candidates:
        raise RuntimeError(
            "MoonViT SDPA mem-efficient patch FAILED: could not find a "
            "loaded `transformers_modules.*.modeling_vit` module with a "
            "`sdpa_attention` function. Without this patch the vision "
            "encoder silently falls through to the math SDPA backend and "
            "OOMs at full LA_MAX_IMAGE_DIM (the ~13 GiB allocation comes "
            "from materialising num_heads × N × N × bytes of attention "
            "probabilities at N=25,600). Refusing to start."
        )

    # ---- STRICT PRE-PATCH SHAPE CHECK -----------------------------------
    # The model file is SHA-256 pinned in validate_startup.py, but this
    # patch must still refuse to apply to a `sdpa_attention` function
    # whose signature has drifted in any way from what we developed
    # against. The replacement function does NOT wrap the original —
    # it reimplements it — so an unnoticed signature change would
    # silently produce wrong outputs.
    EXPECTED_VIT_SDPA_PARAMS = ("q", "k", "v", "q_cu_seqlens", "k_cu_seqlens")
    EXPECTED_VIT_DISPATCH_KEYS = frozenset({"flash_attention_2", "sdpa", "eager"})
    for mod in candidates:
        if getattr(mod.sdpa_attention, "_la_sdpa_patched", False):
            continue
        # 1. Signature must match exactly.
        sig = inspect.signature(mod.sdpa_attention)
        actual_params = tuple(sig.parameters)
        if actual_params != EXPECTED_VIT_SDPA_PARAMS:
            raise RuntimeError(
                f"MoonViT SDPA patch ABORTED on strict pre-check: "
                f"modeling_vit.sdpa_attention signature has drifted. "
                f"Expected parameters {EXPECTED_VIT_SDPA_PARAMS!r}, "
                f"observed {actual_params!r}. The replacement function "
                "would compute the wrong thing. Refusing to apply."
            )
        # 2. Dispatch dict must exist with exactly the expected key set.
        # If a new attn impl appears (e.g. "magi") or the dict structure
        # is reorganised, our swap might miss it.
        if not hasattr(mod, "VL_VISION_ATTENTION_FUNCTIONS"):
            raise RuntimeError(
                f"MoonViT SDPA patch ABORTED: module {mod.__name__!r} has "
                "no `VL_VISION_ATTENTION_FUNCTIONS` dict. The encoder "
                "layer at modeling_vit.py:463 looks attention up from "
                "this dict on every forward call — without it our patch "
                "cannot affect the live forward path."
            )
        actual_keys = frozenset(mod.VL_VISION_ATTENTION_FUNCTIONS.keys())
        if actual_keys != EXPECTED_VIT_DISPATCH_KEYS:
            raise RuntimeError(
                f"MoonViT SDPA patch ABORTED on strict pre-check: "
                f"VL_VISION_ATTENTION_FUNCTIONS keys have drifted. "
                f"Expected {sorted(EXPECTED_VIT_DISPATCH_KEYS)!r}, "
                f"observed {sorted(actual_keys)!r}. Refusing to apply."
            )

    def patched_sdpa_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
        seq_length = q.shape[0]
        # Build the segment-block mask in bool, then convert to bf16
        # additive mask in-place. Identical semantics to PyTorch's own
        # at::where(bool_mask, 0, -inf) conversion at
        # attention.cpp:557-559, but skips one fresh-tensor allocation.
        bool_mask = torch.zeros(
            (seq_length, seq_length), device=q.device, dtype=torch.bool,
        )
        for i in range(1, len(q_cu_seqlens)):
            s, e = int(q_cu_seqlens[i - 1]), int(q_cu_seqlens[i])
            bool_mask[s:e, s:e] = True
        attn_mask = torch.zeros(
            (1, 1, seq_length, seq_length), device=q.device, dtype=q.dtype,
        )
        attn_mask.masked_fill_(~bool_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        # 3D (N, H, D) → 4D (1, H, N, D). Mem-eff requires q.dim()==4
        # (sdp_utils_cpp.h:303 / attention.cu:1409). transpose + unsqueeze
        # are stride-only views; contiguous() locks the layout for the
        # dispatcher's stride checks.
        q4 = q.transpose(0, 1).unsqueeze(0).contiguous()
        k4 = k.transpose(0, 1).unsqueeze(0).contiguous()
        v4 = v.transpose(0, 1).unsqueeze(0).contiguous()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            attn_output = F.scaled_dot_product_attention(
                q4, k4, v4, attn_mask, dropout_p=0.0,
            )
        # Reverse: (1, H, N, D) → (N, H, D) → (N, H*D)
        return attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)

    # Tag the replacement so we can detect "already patched" without
    # re-wrapping on re-imports.
    patched_sdpa_attention._la_sdpa_patched = True

    for mod in candidates:
        if getattr(mod.sdpa_attention, "_la_sdpa_patched", False):
            continue
        mod.sdpa_attention = patched_sdpa_attention
        # The encoder layer at MoonVitEncoderLayer.attention_qkvpacked
        # (modeling_vit.py:463) does a fresh
        # `VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]`
        # lookup on EVERY forward call — it does not snapshot the
        # function into a per-instance attribute at __init__. So
        # updating this dict propagates to every live encoder layer
        # automatically; no per-instance rebinding is required.
        if hasattr(mod, "VL_VISION_ATTENTION_FUNCTIONS"):
            mod.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = patched_sdpa_attention

    # Defense in depth: confirm `VL_VISION_ATTENTION_FUNCTIONS["sdpa"]`
    # now points at the patched function with our marker attribute.
    # Without this, a future upstream reshuffle of the dispatch dict
    # could leave the patch functionally inert.
    for mod in candidates:
        if hasattr(mod, "VL_VISION_ATTENTION_FUNCTIONS"):
            dict_fn = mod.VL_VISION_ATTENTION_FUNCTIONS.get("sdpa")
            if dict_fn is None or not getattr(dict_fn, "_la_sdpa_patched", False):
                raise RuntimeError(
                    f"MoonViT SDPA patch verification FAILED: module "
                    f"{mod.__name__!r}'s VL_VISION_ATTENTION_FUNCTIONS['sdpa'] "
                    "is not the patched function."
                )


def _force_vit_flash_attn_2(config, model_dir: str) -> None:
    """Force the MoonViT vision encoder onto FlashAttention 2 (varlen).

    PROBLEM:
        See "VISION-ENCODER FA2 OVERRIDE" in the module docstring.
        Briefly: NVIDIA's dispatcher at
        `models/LocateAnything-3B/modeling_locateanything.py:104`
        defaults the vision encoder to `'flash_attention_2'` when
        `vision_config._attn_implementation` is unset; and MoonViT's
        `multihead_attention` at modeling_vit.py:63-121 calls
        `flash_attn.flash_attn_varlen_func` — which is what Moonshot
        actually TRAINED with (Kimi-VL paper §"variable-length sequence
        attention mechanism supported by FlashAttention";
        `/tmp/la_research/kimi_vl.txt:270-271`). But transformers
        4.57.1's `_autoset_attn_implementation` cascades the top-level
        `_attn_implementation='sdpa'` (which we set for the LLM, due to
        magi being sm_120-unbuildable) down into `vision_config`
        because `vision_config._attn_implementation_autoset: True` in
        config.json:62. Net effect: silently degrades the vision
        encoder onto PyTorch SDPA (and then the mem-eff patch above),
        losing ~2-3× of the trained-time FA2 throughput.

    FIX:
        Counter-override the cascade IMMEDIATELY before from_pretrained
        runs by setting `config.vision_config._attn_implementation =
        "flash_attention_2"`. NVIDIA's dispatcher at
        modeling_locateanything.py:104 then keeps that value (it only
        defaults the *unset* case), and the encoder block records the
        impl into per-layer `self.attn_implementation` at
        modeling_vit.py:431 — which `attention_qkvpacked` then looks
        up from `VL_VISION_ATTENTION_FUNCTIONS` on every forward call
        (modeling_vit.py:463). The post-load verification in
        `LocateAnythingInference.__init__` re-reads every encoder
        block's attribute to confirm the override propagated.

    STRICT PRE-CHECKS (refuse to apply on any drift):
        (a) flash_attn importable AND `flash_attn.__version__` EXACTLY
            equals the pinned `_EXPECTED_FLASH_ATTN_VERSION` mirroring
            `scripts/lib/versions.sh:LA_FLASH_ATTN_VERSION`. A
            same-name-different-version flash_attn could produce
            silently wrong outputs.
        (b) flash_attn.flash_attn_varlen_func importable. That is the
            specific entry point MoonViT.multihead_attention calls; a
            future flash_attn that drops the varlen path (or renames
            it) must hard-fail here rather than silently fall back
            via modeling_vit.py:84-87's `if flash_attn_varlen_func is
            None` warn-and-sdpa.
        (c) A live transformers_modules.*.modeling_vit module is in
            sys.modules with the expected dispatch dict shape:
            keys == {"flash_attention_2", "sdpa", "eager"} and
            VL_VISION_ATTENTION_FUNCTIONS["flash_attention_2"] is the
            `multihead_attention` function from that module.
        (d) `multihead_attention`'s signature is exactly the
            (q, k, v, q_cu_seqlens, k_cu_seqlens) the encoder layer at
            modeling_vit.py:463 calls it with.
        (e) `config.vision_config` is structurally what the empirical
            FA2 vs SDPA comparison was measured on: model_type
            "moonvit", num_attention_heads 16, hidden_size 1152
            (therefore head_dim 72). The FA2 varlen kernel handles
            head_dim 72 via the in-shared-memory padding path; the
            non-varlen FA2 path would reject it at
            `flash_attn/csrc/flash_attn/flash_api.cpp:154`
            (`TORCH_CHECK(d == d_rounded)`). Drift in any of these
            fields invalidates the empirical equivalence test, so
            refuse the override.
        (f) `config.vision_config._attn_implementation` is currently
            in `{None, "sdpa"}` — i.e. either unset OR auto-cascaded
            from the LLM-side sdpa override. If it is already
            `"flash_attention_2"` (e.g. an earlier override or an HF
            release that fixes the cascade), this is a no-op. Any
            other value (`"magi"`, `"eager"`, …) means someone else
            deliberately set the vision attn impl elsewhere and we
            refuse to silently overwrite that choice.

    AFTER:
        Sets `config.vision_config._attn_implementation =
        "flash_attention_2"` and stamps a marker
        `config.vision_config._la_vit_fa2_forced = True` for the
        post-load verify line in `__init__`. Logs an OK line in the
        same style as `validate_startup.ok(...)`.

    NOTE: this function only mutates the in-memory config. It does
    NOT monkey-patch any model code (cf. the upstream-correct
    dispatch path at modeling_vit.py:463) and does NOT touch the
    `models/LocateAnything-3B/` directory.
    """
    import sys
    import inspect

    # ---- (a) flash_attn importable + exact version --------------------
    try:
        import flash_attn  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "MoonViT FA2 override FAILED at pre-check (a): `import "
            f"flash_attn` raised {type(e).__name__}: {e}. The Dockerfile "
            "source-builds flash-attn (see scripts/lib/versions.sh:89 "
            "LA_FLASH_ATTN_VERSION); if this fails inside the container, "
            "the source build did not produce an importable module. "
            "Refusing to start at degraded vision-encoder throughput."
        ) from e
    fa_version = getattr(flash_attn, "__version__", None)
    if fa_version != _EXPECTED_FLASH_ATTN_VERSION:
        raise RuntimeError(
            "MoonViT FA2 override FAILED at pre-check (a): "
            f"flash_attn.__version__={fa_version!r} != "
            f"{_EXPECTED_FLASH_ATTN_VERSION!r} "
            "(pinned in scripts/lib/versions.sh:89 LA_FLASH_ATTN_VERSION; "
            "mirrored in worker/inference.py:_EXPECTED_FLASH_ATTN_VERSION). "
            "A different flash_attn build could change numerical output "
            "or break the varlen-with-head_dim=72 path. Refusing to "
            "force the vision encoder onto FA2 with an unverified build."
        )

    # ---- (b) flash_attn_varlen_func importable ------------------------
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "MoonViT FA2 override FAILED at pre-check (b): "
            "`from flash_attn import flash_attn_varlen_func` raised "
            f"{type(e).__name__}: {e}. MoonViT's `multihead_attention` "
            "at models/LocateAnything-3B/modeling_vit.py:63-121 calls "
            "this exact symbol; without it modeling_vit.py:84-87 would "
            "warn-and-fall-back to SDPA — defeating the whole point of "
            "this override. Refusing to start."
        ) from e

    # ---- (c) live modeling_vit module with expected dispatch dict ----
    # `AutoConfig.from_pretrained` does NOT trigger the modeling-module
    # import — that only happens inside `AutoModel.from_pretrained`. So
    # at this point in __init__ (before from_pretrained runs)
    # `transformers_modules.*.modeling_vit` is NOT yet in sys.modules.
    # We need to mutate the config BEFORE from_pretrained, so we cannot
    # defer the structural checks until after the model loads. The fix
    # is to materialise the modeling_vit module explicitly via
    # transformers' dynamic_module_utils — the same mechanism
    # trust_remote_code uses internally — to bring it into sys.modules,
    # then walk sys.modules the same way `_patch_vit_sdpa_to_mem_
    # efficient` does. This is read-only: we resolve a class but never
    # instantiate it, so no weights are touched.
    EXPECTED_VIT_DISPATCH_KEYS = frozenset({"flash_attention_2", "sdpa", "eager"})
    try:
        from transformers.dynamic_module_utils import (
            get_class_from_dynamic_module,
        )
        # The class path is the one the outer modeling_locateanything
        # imports from at the top of the file
        # (`from .modeling_vit import MoonVitPretrainedModel`); resolving
        # by class triggers the module import as a side effect.
        # `pretrained_model_name_or_path` is the model dir on disk.
        get_class_from_dynamic_module(
            "modeling_vit.MoonVitPretrainedModel",
            model_dir,
        )
    except Exception as e:
        raise RuntimeError(
            "MoonViT FA2 override FAILED at pre-check (c): could not "
            "trigger `modeling_vit.MoonVitPretrainedModel` import via "
            "transformers.dynamic_module_utils.get_class_from_dynamic_module: "
            f"{type(e).__name__}: {e}. Without the module loaded, we "
            "cannot verify the dispatch-dict shape before mutating the "
            "config. Refusing to start at degraded vision-encoder throughput."
        ) from e
    candidates = [
        m for m in list(sys.modules.values())
        if m is not None
        and getattr(m, "__name__", "").endswith(".modeling_vit")
        and hasattr(m, "VL_VISION_ATTENTION_FUNCTIONS")
    ]
    if not candidates:
        raise RuntimeError(
            "MoonViT FA2 override FAILED at pre-check (c): even after "
            "explicit get_class_from_dynamic_module, no loaded "
            "`transformers_modules.*.modeling_vit` module with a "
            "`VL_VISION_ATTENTION_FUNCTIONS` dict found in sys.modules. "
            "The trust_remote_code dynamic-module namespace shape has "
            "changed in this transformers version. Refusing to start at "
            "degraded vision-encoder throughput."
        )
    for mod in candidates:
        actual_keys = frozenset(mod.VL_VISION_ATTENTION_FUNCTIONS.keys())
        if actual_keys != EXPECTED_VIT_DISPATCH_KEYS:
            raise RuntimeError(
                "MoonViT FA2 override FAILED at pre-check (c): "
                f"VL_VISION_ATTENTION_FUNCTIONS keys in module "
                f"{mod.__name__!r} are {sorted(actual_keys)!r}; expected "
                f"{sorted(EXPECTED_VIT_DISPATCH_KEYS)!r}. A new key (e.g. "
                "'magi') or a missing key would mean the dispatch contract "
                "has shifted. Refusing to apply."
            )
        fa2_fn = mod.VL_VISION_ATTENTION_FUNCTIONS["flash_attention_2"]
        expected_fn = getattr(mod, "multihead_attention", None)
        if expected_fn is None:
            raise RuntimeError(
                "MoonViT FA2 override FAILED at pre-check (c): module "
                f"{mod.__name__!r} has no `multihead_attention` "
                "attribute. modeling_vit.py:188 expects "
                "VL_VISION_ATTENTION_FUNCTIONS['flash_attention_2'] to "
                "be the module-level `multihead_attention` function; "
                "without it the FA2 path is structurally broken."
            )
        if fa2_fn is not expected_fn:
            raise RuntimeError(
                "MoonViT FA2 override FAILED at pre-check (c): "
                f"VL_VISION_ATTENTION_FUNCTIONS['flash_attention_2'] in "
                f"module {mod.__name__!r} is "
                f"{getattr(fa2_fn, '__qualname__', fa2_fn)!r}, NOT the "
                "module's own `multihead_attention`. Some other code "
                "has remapped the FA2 slot — refusing to silently route "
                "the encoder through an unknown implementation."
            )
        # ---- (d) multihead_attention signature -----------------------
        EXPECTED_FA2_PARAMS = ("q", "k", "v", "q_cu_seqlens", "k_cu_seqlens")
        sig = inspect.signature(expected_fn)
        actual_params = tuple(sig.parameters)
        if actual_params != EXPECTED_FA2_PARAMS:
            raise RuntimeError(
                "MoonViT FA2 override FAILED at pre-check (d): "
                f"modeling_vit.multihead_attention signature has drifted. "
                f"Expected parameters {EXPECTED_FA2_PARAMS!r}, observed "
                f"{actual_params!r}. The encoder layer at "
                "modeling_vit.py:463-465 calls it with keyword args "
                "q_cu_seqlens=… and k_cu_seqlens=…; a parameter rename "
                "would silently break that call. Refusing to apply."
            )

    # ---- (e) vision_config structural identity ------------------------
    if not hasattr(config, "vision_config"):
        raise RuntimeError(
            "MoonViT FA2 override FAILED at pre-check (e): the loaded "
            "LocateAnything config has no `vision_config`. The model "
            "structure has drifted from the pinned HF revision "
            "(scripts/lib/versions.sh:34 LA_MODEL_HF_REVISION). "
            "Refusing to start."
        )
    vc = config.vision_config
    expected_vc = {
        "model_type":          "moonvit",
        "num_attention_heads": 16,
        "hidden_size":         1152,
    }
    vc_drift = []
    for key, want in expected_vc.items():
        got = getattr(vc, key, None)
        if got != want:
            vc_drift.append(f"vision_config.{key}={got!r} (expected {want!r})")
    if vc_drift:
        raise RuntimeError(
            "MoonViT FA2 override FAILED at pre-check (e): "
            "vision_config structural identity has drifted from the "
            "empirically-tested baseline (head_dim = hidden_size / "
            "num_attention_heads = 1152 / 16 = 72; the FA2 varlen "
            "kernel handles 72 specifically — see "
            "/tmp/la_research/flash-attention/csrc/flash_attn/flash_api.cpp:154 "
            "for why head_dim padding rules vary by entry point). "
            "Drifted fields: " + "; ".join(vc_drift)
            + ". Refusing to apply the FA2 override against an "
              "untested vision-encoder shape."
        )

    # ---- (f) current _attn_implementation value -----------------------
    current_impl = getattr(vc, "_attn_implementation", None)
    if current_impl == "flash_attention_2":
        # Idempotent: already on FA2 (e.g. some future HF release fixes
        # the cascade, or this function ran twice in the same boot).
        # No-op, log, and mark — the post-load verify line wants the
        # marker present.
        if not getattr(vc, "_la_vit_fa2_forced", False):
            vc._la_vit_fa2_forced = True
        print(
            "[validate_startup] OK: vision_config._attn_implementation "
            "already 'flash_attention_2' — no-op (idempotent re-run); "
            "MoonViT will use flash_attn_varlen_func per its trained "
            "default (modeling_locateanything.py:104).",
            flush=True,
        )
        return
    if current_impl not in (None, "sdpa"):
        raise RuntimeError(
            "MoonViT FA2 override FAILED at pre-check (f): "
            f"vision_config._attn_implementation={current_impl!r} is "
            "neither None nor 'sdpa'. The expected pre-states are: "
            "(None) the cascade hasn't run yet, OR ('sdpa') HF's "
            "_autoset_attn_implementation cascaded the LLM-side sdpa "
            "value (see config.json:62 `_attn_implementation_autoset: "
            "True` + transformers 4.57.1 modeling_utils.py cascade "
            "behaviour). Any other value means some other code has "
            "deliberately set the vision attn impl — refusing to "
            "silently overwrite that choice. Refusing to start."
        )

    # All pre-checks passed; apply the mutation.
    vc._attn_implementation = "flash_attention_2"
    vc._la_vit_fa2_forced = True

    print(
        "[validate_startup] OK: forced "
        "vision_config._attn_implementation: "
        f"{current_impl!r} -> 'flash_attention_2' "
        f"(flash_attn {fa_version} + flash_attn_varlen_func verified). "
        "Restores MoonViT's trained-time FA2 path "
        "(Kimi-VL paper §FA-varlen / modeling_vit.py:571 "
        "_supports_flash_attn_2=True / modeling_locateanything.py:104 "
        "default 'flash_attention_2'). Counter-overrides the "
        "transformers 4.57.1 _autoset_attn_implementation cascade that "
        "would otherwise propagate the LLM-side sdpa override "
        "(LA_ATTN_IMPL=sdpa) into vision_config because of "
        "config.json:62 _attn_implementation_autoset:True.",
        flush=True,
    )


# Map of `prompt_task` wire name → expected response shape. The model
# was trained on a strict template → shape mapping (see worker/prompts.py
# module docstring): templates 1-6 emit 4-coord boxes; template 7 emits
# 2-coord points. The Rust validator at rust_server/src/prompt_validator
# .rs classifies every accepted prompt into one of these seven wire names
# and forwards it through the IPC header as `prompt_task`. `run()` uses
# this map to FILTER off-shape parse output before stamping the response:
# a 4-coord box that escapes through a Point prompt's response (or a
# 2-coord point that escapes through a detection prompt's response) is
# by-definition off-distribution and must not land in the typed field for
# the other shape.
#
# Empirically the model does not emit off-shape at the trained sampling
# parameters: zero cross-shape events were observed in 3,444 trials
# spanning all 7 templates × adversarial prompts (including a Point
# prompt literally containing the substring "bounding box"). The filter
# is therefore a forward-compat guard rail, not a frequent rejection
# path. Crucially it does NOT silently drop the deviation: the off-shape
# count is surfaced in `off_shape_count` (and the verbatim tokens stay in
# `raw_answer`), and `abstained` is computed from the PRE-filter parse — so
# a model that emitted geometry in the wrong shape is reported as a loud,
# queryable deviation, never misreported as "the model abstained".
#
# Keys MUST equal worker/prompts.py::TEMPLATE_WIRE_NAMES AND the wire
# names in rust_server/src/prompt_validator.rs::TemplateKind::wire_name;
# the boot drift check at worker/validate_startup.py::
# validate_prompt_template_drift fails the container start on any
# mismatch.
EXPECTED_SHAPE = {
    "detection":      "box",
    "phrase_single":  "box",
    "phrase_multi":   "box",
    "text_grounding": "box",
    "scene_text":     "box",
    "gui_box":        "box",
    "point":          "point",
}


@dataclass
class InferenceResult:
    raw_answer: str
    detections: list  # list[dict]
    points: list      # list[dict]
    # True iff the model produced NO parseable geometry at all — no boxes
    # and no points — evaluated BEFORE the task→shape filter. An off-shape
    # emission (geometry in the wrong shape for the task) is NOT abstention:
    # it is filtered out of `detections`/`points`, counted in
    # `off_shape_count`, and leaves this False. Stamped explicitly so naive
    # clients can branch without re-parsing. NVIDIA's eval pipeline has no
    # aggregate `abstained` concept; empty parsed output is the
    # aggregate-no-result signal there too (inference_grounding_ddp.py:282-300
    # + metrics/other_metric.py:140-156). Per-category abstention in a
    # multi-category detect prompt is NOT signalled here — it's recoverable
    # from `raw_answer` as the set difference between the prompt's category
    # list and `{d['label'] for d in detections}`.
    abstained: bool
    # Number of parsed geometries the model emitted in the WRONG shape for
    # the task (a box under a `point` prompt, or a point under a box prompt)
    # and that were therefore filtered out of `detections`/`points`. Normally
    # 0. Non-zero is a loud, queryable signal of a model task→shape deviation:
    # the output is NOT silently dropped (verbatim tokens remain in
    # `raw_answer`) and does NOT count as abstention. Empirically 0 across
    # 3,444 trials at the trained sampling params (see EXPECTED_SHAPE).
    off_shape_count: int
    # True iff `raw_answer` does NOT end with the model's <|im_end|>
    # end-of-turn marker. Per the custom .generate() loop at
    # models/LocateAnything-3B/modeling_locateanything.py:464,500-501
    # the loop exits ONLY on `<|im_end|>` emission OR budget exhaustion
    # (max_new_tokens=8192), so missing <|im_end|> ⇔ budget hit. The
    # response is necessarily incomplete in that case — any block whose
    # closing </box> did not fit is silently dropped by parse_boxes /
    # parse_points (same behaviour as NVIDIA's eval at
    # inference_grounding_ddp.py:282-300 numeric-only regex). Surfaces
    # the implicit-only signal as a typed boolean so clients no longer
    # need to substring-check raw_text themselves.
    model_output_truncated: bool
    # Wire name of the canonical template the prompt was classified as
    # by the Rust validator (one of EXPECTED_SHAPE.keys()). Echoed in
    # the client-facing Result body so the client can branch on the
    # authoritative field WITHOUT re-classifying the prompt themselves:
    # for `prompt_task == "point"` look at `points[]`; for any other
    # wire name look at `detections[]`. Per the trained task→shape
    # contract, the OTHER list is always empty (off-shape outputs are
    # filtered out of `detections`/`points` and counted in
    # `off_shape_count`; see run()).
    prompt_task: str
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
        # 'magi' (which it always is in this model). The vision_config is
        # NOT left alone — see _force_vit_flash_attn_2 below.
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

        # Counter-override the HF auto-cascade that would otherwise route
        # the MoonViT vision encoder onto SDPA. The mechanism:
        #   - models/LocateAnything-3B/config.json:62 declares
        #     `vision_config._attn_implementation_autoset: True`.
        #   - transformers 4.57.1's `_autoset_attn_implementation`
        #     therefore cascades the value of
        #     `config._attn_implementation` (which we just set to
        #     `self.attn_impl == 'sdpa'`) down into vision_config when
        #     `AutoModel.from_pretrained(...)` runs.
        #   - The cascaded 'sdpa' then routes through
        #     `models/LocateAnything-3B/modeling_locateanything.py:104`
        #     and lands at `models/LocateAnything-3B/modeling_vit.py:187`
        #     `VL_VISION_ATTENTION_FUNCTIONS["sdpa"]` instead of the
        #     train-time `multihead_attention` (flash_attn_varlen_func)
        #     at modeling_vit.py:63-121.
        # The whole reason we set `config._attn_implementation = 'sdpa'`
        # was the LLM's magi-on-sm_120 unbuildability — there is no
        # analogous structural blocker on the vision side, and the
        # train-time impl (per Moonshot's Kimi-VL paper §"variable-length
        # sequence attention mechanism supported by FlashAttention";
        # /tmp/la_research/kimi_vl.txt:270-271) IS FlashAttention 2. So
        # this restores the train-correct default. See "VISION-ENCODER
        # FA2 OVERRIDE" in this module's docstring for full citations.
        _force_vit_flash_attn_2(config, model_dir)

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

        # Post-load verification for the vision-encoder FA2 override.
        # The pre-load config mutation in _force_vit_flash_attn_2 sets
        # `vision_config._attn_implementation = "flash_attention_2"`,
        # and modeling_locateanything.py:104 records that into the live
        # `vision_model.config._attn_implementation` AND propagates it
        # into every encoder block's per-instance `attn_implementation`
        # attribute (modeling_vit.py:431, set from block_cfg via the
        # MoonVitEncoder constructor at modeling_vit.py:511). We re-read
        # both surfaces here; any divergence means the cascade
        # counter-override did NOT propagate as expected, and the
        # encoder would silently fall onto a different impl at the
        # first forward call. Refuse to serve at degraded throughput.
        vit_cfg_impl = getattr(self.model.vision_model.config,
                               "_attn_implementation", None)
        if vit_cfg_impl != "flash_attention_2":
            raise RuntimeError(
                "MoonViT FA2 override POST-LOAD VERIFICATION FAILED: "
                f"self.model.vision_model.config._attn_implementation="
                f"{vit_cfg_impl!r}, expected 'flash_attention_2'. The "
                "pre-load mutation in `_force_vit_flash_attn_2` (which "
                "set config.vision_config._attn_implementation) did not "
                "carry through to the live vision_model.config. Override "
                "mechanism (cite chain): "
                "models/LocateAnything-3B/modeling_locateanything.py:104 "
                "is the dispatcher that records the impl onto "
                "vision_model.config; "
                "models/LocateAnything-3B/modeling_vit.py:431 records it "
                "into each encoder block's `attn_implementation`; "
                "models/LocateAnything-3B/modeling_vit.py:463 looks it "
                "up from VL_VISION_ATTENTION_FUNCTIONS at every forward. "
                "Refusing to serve at degraded vision-encoder throughput."
            )
        marker = getattr(config.vision_config, "_la_vit_fa2_forced", False)
        if marker is not True:
            raise RuntimeError(
                "MoonViT FA2 override POST-LOAD VERIFICATION FAILED: "
                f"config.vision_config._la_vit_fa2_forced={marker!r}, "
                "expected True. The marker should have been stamped by "
                "_force_vit_flash_attn_2 — if it is missing, that "
                "function did not run, OR a code path replaced the "
                "config object between the call and here. Refusing to "
                "serve at degraded vision-encoder throughput."
            )
        # Per-block verification. Walk every encoder block and confirm
        # the per-instance `attn_implementation` reads 'flash_attention_2'
        # — the dispatcher at modeling_vit.py:463 reads the per-instance
        # attribute, NOT the config, on every forward call. A bug that
        # silently leaves any one block on a different impl would only
        # surface as throughput noise, not a hard error.
        if not hasattr(self.model.vision_model, "encoder"):
            raise RuntimeError(
                "MoonViT FA2 override POST-LOAD VERIFICATION FAILED: "
                "self.model.vision_model has no `encoder` attribute. "
                "The MoonVitPretrainedModel structure has changed from "
                "the pinned HF revision "
                "(scripts/lib/versions.sh:34 LA_MODEL_HF_REVISION). "
                "Refusing to start."
            )
        blocks = self.model.vision_model.encoder.blocks
        bad_blocks = [
            (i, getattr(block, "attn_implementation", None))
            for i, block in enumerate(blocks)
            if getattr(block, "attn_implementation", None) != "flash_attention_2"
        ]
        if bad_blocks:
            raise RuntimeError(
                "MoonViT FA2 override POST-LOAD VERIFICATION FAILED: "
                f"{len(bad_blocks)} of {len(blocks)} encoder block(s) "
                "did NOT receive the flash_attention_2 impl. First few "
                f"divergent (block_idx, attn_implementation): "
                f"{bad_blocks[:5]!r}. The dispatcher at "
                "models/LocateAnything-3B/modeling_vit.py:463 reads "
                "per-block `self.attn_implementation` on every forward "
                "call, so a divergent block would silently route through "
                "the wrong impl. Override mechanism (cite chain): "
                "modeling_locateanything.py:104 dispatches into "
                "MoonVitPretrainedModel(config.vision_config); "
                "modeling_vit.py:574-597 forwards block_cfg through "
                "MoonVitEncoder; modeling_vit.py:431 records "
                "attn_implementation onto each MoonVitEncoderLayer. "
                "Refusing to serve at degraded vision-encoder throughput."
            )
        print(
            "[validate_startup] OK: MoonViT FA2 post-load verified — "
            f"vision_model.config._attn_implementation='flash_attention_2' "
            f"+ all {len(blocks)} encoder blocks report "
            f"attn_implementation='flash_attention_2'.",
            flush=True,
        )

        # Force the PyTorch SDPA dispatcher onto the memory-efficient
        # backend in BOTH the Qwen2 LLM and the MoonViT vision encoder,
        # so the model can run at full LA_MAX_IMAGE_DIM without OOMing.
        # See SDPA BACKEND OVERRIDE in this module's docstring for why
        # the unpatched path silently falls through to the math backend
        # and OOMs at N=25600 in both subsystems.
        _patch_sdpa_to_mem_efficient(self.model)
        _patch_vit_sdpa_to_mem_efficient()

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
        prompt_task: str,
    ) -> InferenceResult:
        """Run one inference. `generation_mode` and `prompt_task` are
        REQUIRED; no defaults.

        `prompt_task` is the wire name of the canonical template the
        prompt was classified as (Rust validator output forwarded through
        the IPC header; see EXPECTED_SHAPE for valid values). Used to
        filter off-shape model output (counted in `off_shape_count`, not
        silently dropped) per the trained task→shape contract.

        JPEG decoded inside this method — Python's PIL is the canonical
        decoder for the LocateAnything image processor.
        """
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                "prompt must be a non-empty string. See worker/prompts.py "
                "(the single source of truth for the seven canonical "
                "LocateAnything-3B prompt templates) for the canonical forms."
            )
        if not isinstance(prompt_task, str) or prompt_task not in EXPECTED_SHAPE:
            raise ValueError(
                f"prompt_task={prompt_task!r} is not one of the canonical "
                f"wire names {sorted(EXPECTED_SHAPE.keys())!r}. The Rust "
                "frontend should classify every accepted prompt into one "
                "of these values via prompt_validator::TemplateKind::"
                "wire_name and forward it through the IPC header — "
                "receiving an unknown value here means the upstream "
                "contract was violated."
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

            # ICC profile-aware colour conversion to sRGB.
            # PIL's bare `.convert("RGB")` discards any embedded ICC
            # profile — Adobe-RGB / Display-P3 / ProPhoto-tagged JPEGs
            # would then be interpreted as sRGB and silently shift in
            # colour relative to NVIDIA's training distribution (which
            # is sRGB-assumed). For drone detection this is mostly a
            # boundary-confidence issue (sky and metal colours close
            # to the perceptual edge of confident class), but for any
            # colour-sensitive class boundary it's a real loss.
            # PIL.ImageCms.profileToProfile does a real
            # colour-managed transform when the source profile is
            # tagged; we fall back to a plain `.convert("RGB")` only
            # when there's no ICC tag at all (i.e. assume-sRGB, which
            # matches both the training assumption and `.convert`'s
            # documented behaviour). If the ICC transform itself errors
            # for any reason (corrupt profile bytes, unsupported intent)
            # we hard-fail rather than silently fall back to the colour
            # shift — strict-trained-correct contract.
            icc = raw.info.get("icc_profile")
            if icc:
                from PIL import ImageCms
                try:
                    src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                    src_desc = ImageCms.getProfileDescription(src_profile).strip()
                    # Case-insensitive match — common variants include
                    # "sRGB IEC61966-2.1", "sRGB Color Space Profile",
                    # "srgb" (some older LCMS-emitted profiles).
                    if "srgb" not in src_desc.lower():
                        # Truly non-sRGB tagged input — colour-manage to sRGB
                        # via PCS-LAB so out-of-gamut colours are perceptually
                        # mapped rather than clipped at the channel level.
                        dst_profile = ImageCms.createProfile("sRGB")
                        raw = ImageCms.profileToProfile(
                            raw, src_profile, dst_profile,
                            outputMode="RGB",
                            renderingIntent=ImageCms.Intent.PERCEPTUAL,
                        )
                    # else: profile is already sRGB — skip the no-op
                    # transform; convert() below will normalise mode.
                except (ValueError, OSError, ImageCms.PyCMSError) as e:
                    # PyCMSError = corrupt/unparseable ICC bytes,
                    # OSError   = profile object construction failed,
                    # ValueError = profileToProfile rejected the intent
                    # or mode. All three are client-fault, all three get
                    # a clean 400 with provenance — never a silent
                    # colour shift.
                    raise ValueError(
                        "ICC profile present but could not be converted to "
                        f"sRGB: {type(e).__name__}: {e}. Strip or correct "
                        "the profile client-side before sending — the model "
                        "is trained on sRGB-assumed inputs and a wrong-tagged "
                        "profile causes a silent colour shift."
                    ) from e
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

        # Strict trained-correct preprocessor gates — defense in depth
        # for the Rust front-end's checks. The model's
        # image_processing_locateanything.py:rescale() enforces:
        #   (a) `(W // 14) * (H // 14) <= in_token_limit (=25,600)`
        #       (line 52). FLOOR-div on the raw 14-px patch grid, NOT
        #       ceil-div on the merged 28-px grid (a prior revision of
        #       this gate had it wrong).
        #   (b) `W // 14 < 512` AND `H // 14 < 512` (line 68 — "Exceed
        #       pos emb"). The MoonViT base learned positional embedding
        #       is 64×64, bicubic-interpolated to the runtime grid;
        #       512 patches per side is the documented hard cap.
        # Both are dormant at the current LA_MAX_IMAGE_DIM=2240 (each
        # check needs W or H ≥ 2254 / ≥ 7168 respectively to trip) but
        # enforce the trained-correct contract regardless of cap.
        w, h = image.width, image.height
        PATCH_PX, IN_TOKEN_LIMIT, POS_EMB_CAP = 14, 25600, 512
        w_patches = w // PATCH_PX
        h_patches = h // PATCH_PX
        n_patches = w_patches * h_patches
        if n_patches > IN_TOKEN_LIMIT:
            raise ValueError(
                f"image dimensions {w}x{h} produce {n_patches} ViT patches "
                f"((W // {PATCH_PX}) × (H // {PATCH_PX})), exceeding the "
                f"trained `in_token_limit = {IN_TOKEN_LIMIT}`. The model's "
                "preprocessor would internally downscale (BICUBIC) to fit "
                "— the canonical training-correct path is for the client "
                f"to send within budget. Reduce dimensions so "
                f"(W // {PATCH_PX}) × (H // {PATCH_PX}) ≤ {IN_TOKEN_LIMIT}."
            )
        if w_patches >= POS_EMB_CAP or h_patches >= POS_EMB_CAP:
            raise ValueError(
                f"image dimensions {w}x{h} would map to a "
                f"{w_patches}×{h_patches} patch grid, exceeding MoonViT's "
                f"positional-embedding cap of {POS_EMB_CAP} patches per "
                f"side (= {POS_EMB_CAP * PATCH_PX} px). Reduce each "
                f"dimension to < {POS_EMB_CAP * PATCH_PX} px."
            )

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
        # Per-request bookend check. The canonical prefix/suffix strings
        # are in `prompts.CANONICAL_RENDERED_{PREFIX,SUFFIX}`. The Qwen-
        # default fallback the error message warns about lives in the
        # model's vendored `tokenizer_config.json:9346`.
        if not text.startswith(prompts.CANONICAL_RENDERED_PREFIX):
            raise RuntimeError(
                "Rendered chat-template text does NOT start with the "
                "NVIDIA-trained system-message bookend. The processor's "
                "py_apply_chat_template has silently swapped templates "
                "(most plausibly to the Qwen-default 'You are Qwen, "
                "created by Alibaba Cloud. ...'). Refusing to run "
                "inference off-distribution. "
                f"Expected first {len(prompts.CANONICAL_RENDERED_PREFIX)} "
                f"chars: {prompts.CANONICAL_RENDERED_PREFIX!r}. "
                f"Got first {len(prompts.CANONICAL_RENDERED_PREFIX)} "
                f"chars: {text[:len(prompts.CANONICAL_RENDERED_PREFIX)]!r}."
            )
        if not text.endswith(prompts.CANONICAL_RENDERED_SUFFIX):
            raise RuntimeError(
                "Rendered chat-template text does NOT end with the "
                "assistant-turn invitation that add_generation_prompt=True "
                "is supposed to append. The processor's behavior has "
                "diverged from what NVIDIA's SFT trainer used. Refusing "
                "to run inference off-distribution. "
                f"Expected last {len(prompts.CANONICAL_RENDERED_SUFFIX)} "
                f"chars: {prompts.CANONICAL_RENDERED_SUFFIX!r}. "
                f"Got last {len(prompts.CANONICAL_RENDERED_SUFFIX)} "
                f"chars: {text[-len(prompts.CANONICAL_RENDERED_SUFFIX):]!r}."
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
        # `abstained` must reflect the MODEL, not the task-shape filter below:
        # snapshot whether the model produced ANY parseable geometry BEFORE we
        # filter to the trained shape, so an off-shape emission is never
        # misreported as "the model abstained".
        parsed_geometry = bool(detections or points)
        # Trained task→shape enforcement: a `point` task returns points and
        # every other task returns boxes. Off-shape output (a box under a point
        # prompt, or vice-versa) is filtered out of the typed lists — but NOT
        # silently dropped: it is COUNTED in `off_shape_count` (verbatim tokens
        # remain in `raw_answer`), turning a model deviation into a loud,
        # queryable signal instead of an empty response. See EXPECTED_SHAPE.
        off_shape_count = 0
        expected = EXPECTED_SHAPE[prompt_task]
        if expected == "point":
            off_shape_count = len(detections)
            detections = []
        elif expected == "box":
            off_shape_count = len(points)
            points = []
        return InferenceResult(
            raw_answer=answer,
            detections=detections,
            points=points,
            # True only if the model produced no parseable geometry at all
            # (pre-filter). An off-shape emission leaves this False; see
            # off_shape_count. See InferenceResult.abstained docstring.
            abstained=not parsed_geometry,
            off_shape_count=off_shape_count,
            # Explicit truncation signal. See InferenceResult.model_output_truncated
            # docstring.
            model_output_truncated=not answer.endswith("<|im_end|>"),
            prompt_task=prompt_task,
            latency_ms=latency_ms,
            image_size=(image.width, image.height),
            resize_plan={
                "dst_w": plan.dst_w,
                "dst_h": plan.dst_h,
                "n_llm_tokens": plan.n_llm_tokens,
                "scale": round(plan.scale, 4),
            },
        )
