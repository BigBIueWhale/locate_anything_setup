"""
Strict startup self-validation. Runs BEFORE the model loads.

This module exists so that the Python worker either reaches a known-good
state and starts accepting requests, OR crashes loudly with a precise
diagnostic. There is no in-between.
"""

from __future__ import annotations
from pathlib import Path
import hashlib
import os
import sys


# ===== Content SHA-256 of every file shipped at the pinned HF revision =====
#
# We pin BOTH the Python files (executed via trust_remote_code) AND the
# weight + config files (consumed by the loader / processor / tokenizer).
# Any mismatch is a hard boot failure: either someone tampered with the
# bind-mounted ./models/ directory or the HF revision was bumped without
# updating versions.sh. Both demand explicit operator review.
#
# Files NOT pinned here (whitelisted as inference-irrelevant):
#   - LICENSE, README.md, .gitattributes (metadata)
#   - all_results.json, trainer_state.json, training_args.bin (training
#     artifacts not read by inference)
# Their presence/absence does not affect model behavior, so pinning them
# would only generate spurious churn on README/license updates.

_PINNED_PY_SHA256 = {
    "configuration_locateanything.py":    "d2738cc180add2b77e88b8cf2bc87ff012f23bd417a99150a033f61b0a8eb857",
    "configuration_qwen2.py":             "1fda5efb735cae465debd414afc673389fe731afd95c934a469faa23d3d7fdf1",
    "generate_utils.py":                  "863187051772549928bf103b58f6176263c9b786fb19ef83fd7b2e76169fa65e",
    "image_processing_locateanything.py": "5109868add766c7e244487ecfff6a6f5a4aa1497b38b385a33a969f12e23b4ec",
    "mask_magi_utils.py":                 "646b565e38b30d58cafe30aeecf0283aee83d198ce8e57936c3488c1dc7b9276",
    "mask_sdpa_utils.py":                 "7e9d600eb25283963cc1696da060813066b9795da467cf9fb0ca68bfc8de1e1d",
    "modeling_locateanything.py":         "ffe736fb8ded5d597201704ccd85134d18a8e4dea43d309228644737234b7244",
    "modeling_qwen2.py":                  "aadb676c0a587a16b7071977c159df16299fad22d88ee8ed9754881ab7f59575",
    "modeling_vit.py":                    "96479eb121c840f009a32830c78740154171290419108caefcf8778580700373",
    "processing_locateanything.py":       "682145ed054b1e912e66273b476e51a25b2d666d4a37b26385af9300b66d40d8",
}

# Weights + configs. Safetensors shards are the largest (~7.66 GiB total)
# so we stream-hash with 1 MiB chunks; ~3-10s additional boot time.
_PINNED_WEIGHT_SHA256 = {
    # 2 safetensors shards
    "model-00001-of-00002.safetensors": "923cfc10fed19808067da6df85a9a4220ddc1f9eb91ceee94c0fecd05d0f2d58",
    "model-00002-of-00002.safetensors": "3459ba101f40594f3f62d3312014f1f8378b4ba3da3b1d562480045938fc7d47",
    # safetensors index (shard map)
    "model.safetensors.index.json":     "2ecc63fee5f958ffc8142fa29ff7b704a58e80349e9c9ca155a9710d97700271",
    # 11 inference-relevant config / tokenizer files
    "config.json":               "59e6b5104f9d948db6a38f778e29f86d5c01e373f46d02008fc3070377917007",
    "tokenizer_config.json":     "930b057de30312f861a22780017b09eccd87e893d966bd004bf4d70eec0e2652",
    "preprocessor_config.json":  "34f0dc33b40ee26d280d7ed93614c90b1d41a68d6ccfe5a8341e274e6169f94e",
    "processor_config.json":     "1274db3b9504d37e57ea41ba7de547194d381265c5946f6db8560f851a940992",
    "chat_template.json":        "a0cb84f5108587c8a2e944ad7d4b123bb413c34baeef30e3d6d7a3bb486a835d",
    "added_tokens.json":         "1a87d2bec4c707c3946235046555d91f9df6986b8dd4a3ac53e6d0b24c36d176",
    "generation_config.json":    "f15f5de33244a61325923e99bad2c061029acb8d6dd5c57f8458b3949ddd8f97",
    "special_tokens_map.json":   "bfadce2f545458bf9d39fc9153cd2ac1077371ed5ab553fa4988061fefe67ac7",
    "vocab.json":                "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
    "merges.txt":                "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}

REQUIRED_FILES = (
    # The minimal set of files we need from the HF model directory.
    # If any are missing, the model load will fail later anyway, but
    # we surface the error early with a clean message.
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "added_tokens.json",
    "modeling_locateanything.py",
    "processing_locateanything.py",
    "image_processing_locateanything.py",
    "modeling_vit.py",
    "modeling_qwen2.py",
    "generation_config.json",
    "generate_utils.py",
)


def fail(msg: str) -> None:
    """Print to stderr and exit 1. No fallbacks."""
    print(f"[validate_startup] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[validate_startup] OK: {msg}", flush=True)


def validate_model_dir(model_dir: str) -> None:
    p = Path(model_dir)
    if not p.is_dir():
        fail(f"model directory does not exist: {model_dir}")
    missing = [f for f in REQUIRED_FILES if not (p / f).is_file()]
    if missing:
        fail(
            f"model directory {model_dir} is missing required files: {missing}. "
            "Re-run scripts/01_download_weights.sh on the host to refetch."
        )
    safetensors = list(p.glob("*.safetensors"))
    if not safetensors:
        fail(
            f"no .safetensors weights present in {model_dir}. "
            "Re-run scripts/01_download_weights.sh."
        )

    # Deny-list any *.py file present in the model directory that is NOT
    # in the pinned set. Closes the "drop a new __init__.py" attack class:
    # transformers' trust_remote_code path uses standard Python imports,
    # so any unpinned .py file in the package directory could be sourced
    # transitively (e.g., a malicious __init__.py).
    unpinned_py = [
        f.name for f in p.glob("*.py") if f.name not in _PINNED_PY_SHA256
    ]
    if unpinned_py:
        fail(
            f"unpinned .py file(s) present in {model_dir}: {unpinned_py}. "
            "trust_remote_code=True can transitively import these. Refusing "
            "to load. Either (a) remove the file(s), or (b) if they are "
            "legitimate at a new HF revision, add their SHA-256 to "
            "_PINNED_PY_SHA256 and verify the new revision was reviewed."
        )
    total_bytes = sum(f.stat().st_size for f in safetensors)
    if total_bytes < 7 * 1024 * 1024 * 1024:  # 7 GiB minimum
        fail(
            f"weight files in {model_dir} total only {total_bytes / 1e9:.2f} GB; "
            "expected ~7.66 GB for nvidia/LocateAnything-3B. Re-download."
        )
    ok(f"model directory {model_dir} OK ({total_bytes / 1e9:.2f} GB weights)")

    # CONTENT SHA-256 of every .py file plus every safetensors shard and
    # inference-relevant config/tokenizer file. Defense against
    # trust_remote_code=True executing tampered code from a writable
    # bind-mount, and against silent corruption of the weights themselves
    # (cosmic bit flip, partial-write power loss, attacker injection).
    # The manifest hash in la_worker._compute_weight_manifest_sha256 only
    # fingerprints (name, size); a same-size byte change escapes that check.
    # This pin fails the boot loudly on any byte change.
    import time as _time
    t0 = _time.perf_counter()
    mismatches = []
    missing = []
    bytes_hashed = 0
    for fname, expected_sha in {**_PINNED_PY_SHA256, **_PINNED_WEIGHT_SHA256}.items():
        fpath = p / fname
        if not fpath.is_file():
            missing.append(fname)
            continue
        actual_sha = _sha256_of_file(fpath)
        bytes_hashed += fpath.stat().st_size
        if actual_sha != expected_sha:
            mismatches.append(
                f"  {fname}: expected sha256-{expected_sha}, got sha256-{actual_sha}"
            )
    if missing:
        fail(
            f"required model file(s) missing from {p}: {missing}. "
            "Re-run scripts/01_download_weights.sh to restore."
        )
    if mismatches:
        fail(
            "model file content has drifted from the pinned SHA-256:\n"
            + "\n".join(mismatches)
            + "\nThis is either (a) an attacker swapped a file in the "
            "bind-mounted ./models/ directory, (b) the HF revision was "
            "bumped without updating _PINNED_*_SHA256 in this module, or "
            "(c) on-disk corruption (cosmic bit flip / partial write). "
            "Investigate before re-running."
        )
    elapsed = _time.perf_counter() - t0
    ok(f"all {len(_PINNED_PY_SHA256) + len(_PINNED_WEIGHT_SHA256)} pinned files "
       f"content-verified ({bytes_hashed / 1e9:.2f} GB in {elapsed:.1f}s)")


def _sha256_of_file(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed with 1 MiB chunks so memory
    stays bounded even for the 5 GiB safetensors shards."""
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_gpu() -> None:
    import torch
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() == False")
    if torch.cuda.device_count() == 0:
        fail("torch.cuda.device_count() == 0")
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 12:
        fail(
            f"GPU 0 compute capability {cap[0]}.{cap[1]} < 12.0 (sm_120). "
            "This image is pinned to PyTorch+CUDA wheels with sm_120 kernels. "
            "Run on Blackwell-class hardware (RTX 5090 / GB202)."
        )
    name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem < 24:
        fail(
            f"GPU 0 has only {total_mem:.1f} GB VRAM; the model needs ≥24 GB "
            "in bf16 for comfortable operation. Refusing to start."
        )
    arches = torch.cuda.get_arch_list()
    if not any("sm_120" in a for a in arches):
        fail(
            f"torch.cuda.get_arch_list()={arches} does not include sm_120. "
            "The installed torch wheel was not built with Blackwell support."
        )
    ok(f"GPU 0: {name}, {total_mem:.1f} GB, cap={cap[0]}.{cap[1]}, arches OK")


def validate_env() -> None:
    required = [
        "LA_MODEL_DTYPE", "LA_ATTN_IMPL",
        "LA_GEN_TEMPERATURE", "LA_GEN_TOP_P", "LA_GEN_DO_SAMPLE",
        "LA_GEN_REP_PEN", "LA_GEN_MAX_NEW_TOKENS",
        "LA_GEN_MODE", "LA_GEN_N_FUTURE_TOKENS",
        "LA_IPC_SOCKET",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        fail(f"missing env vars (these MUST be set in the Dockerfile): {missing}")
    # Hard-fail on any deviation from the canonical training-time values.
    # These are the kwargs every benchmark in the paper used
    # (Embodied/evaluation/inference_compat.py:42-68). The model was not
    # trained on any other combination; running with different values
    # = not using the model as trained = forbidden.
    #
    # LA_ATTN_IMPL exception: config.json's `_attn_implementation='magi'`
    # is the train-time value, but MagiAttention does not support sm_120
    # (RTX 5090). The model's custom Qwen2Model.forward() accepts exactly
    # two paths — 'magi' and 'sdpa' — and the sdpa path reconstructs the
    # same block-mask attention pattern (via
    # mask_sdpa_utils.update_causal_mask_for_one_gen_window_2d) that the
    # model was trained with under magi. So sdpa preserves the train-time
    # attention pattern within bf16 precision, just at lower throughput.
    # See worker/inference.py module docstring "ATTENTION" for the full
    # provenance.
    canonical = {
        "LA_GEN_TEMPERATURE":      "0.7",
        "LA_GEN_TOP_P":            "0.9",
        "LA_GEN_REP_PEN":          "1.1",
        "LA_GEN_DO_SAMPLE":        "1",
        "LA_GEN_MODE":             "hybrid",
        "LA_GEN_N_FUTURE_TOKENS":  "6",
        "LA_GEN_MAX_NEW_TOKENS":   "8192",
        "LA_MODEL_DTYPE":          "bfloat16",
        "LA_ATTN_IMPL":            "sdpa",
    }
    drift = []
    for k, expected in canonical.items():
        got = os.environ[k]
        if got != expected:
            drift.append(f"{k}={got!r} (expected {expected!r})")
    if drift:
        fail(
            "Generation-parameter / dtype / attention drift detected — the "
            "image was rebuilt with values that differ from how the model was "
            "trained. Each must equal the canonical value from "
            "Embodied/evaluation/inference_compat.py. Drifted variables: "
            + "; ".join(drift)
            + ". Either revert scripts/lib/versions.sh + Dockerfile to the "
              "canonical values, or accept that this is no longer "
              "'as-trained' inference and remove this check explicitly."
        )
    ok(f"env validated; mode={os.environ['LA_GEN_MODE']}, "
       f"attn={os.environ['LA_ATTN_IMPL']}, dtype={os.environ['LA_MODEL_DTYPE']}")


def validate_preprocessor_config(model_dir: str) -> None:
    """Hard-fail on any drift in preprocessor_config.json from the canonical
    LocateAnything-3B values.

    The model card and NVIDIA's training pipeline pin these values; the
    `image_processing_locateanything.py` default for `in_token_limit` is
    *4096* — a 32× silent downgrade vs the trained value of 25,600 — so a
    missing or wrong-typed config field would degrade detection quality
    in a way that's invisible from the worker logs. Validate explicitly
    so an upstream config drift hard-fails the boot.

    Values verified against the cloned NVlabs/EAGLE Embodied training
    pipeline (eaglevl/utils/locany/preprocessor_config.json carries the
    same in_token_limit/patch_size/merge_kernel_size; the HF release adds
    explicit image_mean/image_std/patch_size/processor_class to defend
    against the code-side default footgun)."""
    import json
    cfg_path = Path(model_dir) / "preprocessor_config.json"
    if not cfg_path.is_file():
        fail(f"preprocessor_config.json missing at {cfg_path!s} — the model "
             "directory is incomplete; re-run scripts/01_download_weights.sh.")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"preprocessor_config.json could not be parsed: {type(e).__name__}: {e}")

    EXPECTED = {
        "image_processor_type":  "LocateAnythingImageProcessor",
        "processor_class":       "LocateAnythingProcessor",
        "in_token_limit":        25600,
        "patch_size":            14,
        "merge_kernel_size":     [2, 2],
        "image_mean":            [0.5, 0.5, 0.5],
        "image_std":             [0.5, 0.5, 0.5],
    }
    drift = []
    for key, want in EXPECTED.items():
        if key not in cfg:
            drift.append(f"missing key {key!r} (expected {want!r})")
            continue
        got = cfg[key]
        if got != want:
            drift.append(f"{key}={got!r} (expected {want!r})")
    if drift:
        fail(
            "preprocessor_config.json drift detected — refusing to start because "
            "the model would be invoked with parameters different from how "
            "NVIDIA trained it, silently degrading detection quality. "
            "Drifted fields: " + "; ".join(drift)
            + ". The canonical values are mirrored from the HF model card at "
              "the pinned revision; if a real revision bump is intended, update "
              "scripts/lib/versions.sh's content-SHA pin for "
              "preprocessor_config.json AND update EXPECTED in "
              "worker/validate_startup.py:validate_preprocessor_config — in "
              "that order."
        )
    ok("preprocessor_config validated; in_token_limit=25600 patch_size=14 "
       "merge_kernel_size=[2,2] image_mean=image_std=[0.5,0.5,0.5]")


def validate_prompt_template_drift() -> None:
    """Hard-fail on any drift between the Rust binary's embedded prompt
    template constants and worker/prompts.py.

    worker/prompts.py is THE single source of truth for the seven canonical
    LocateAnything-3B templates (file-level banner declares this; the Rust
    validator at rust_server/src/prompt_validator.rs mirrors those constants
    byte-for-byte for fast per-frame WebSocket-edge validation). If the two
    sides ever drift, the WS frontend would accept or reject prompts
    inconsistently with what the Python worker considers canonical —
    silently changing the contract the client must satisfy.

    This check runs `la_server --print-canonical-templates`, parses the
    JSON output, and dict-equals it against prompts.CANONICAL_TEMPLATES.
    Any difference is a hard boot failure naming the drifted key(s)."""
    import json as _json
    import subprocess
    from . import prompts

    binary = os.environ.get("LA_SERVER_BIN", "/usr/local/bin/la_server")
    if not Path(binary).is_file():
        fail(
            f"la_server binary not found at {binary}; set LA_SERVER_BIN to "
            "the correct path or rebuild the container. The Rust↔Python "
            "prompt-template drift check cannot run without it."
        )
    try:
        result = subprocess.run(
            [binary, "--print-canonical-templates"],
            capture_output=True, timeout=5, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        fail(
            f"failed to run `{binary} --print-canonical-templates`: "
            f"{type(e).__name__}: {e}"
        )
    if result.returncode != 0:
        fail(
            f"`{binary} --print-canonical-templates` exited "
            f"{result.returncode}; stderr first 1000 chars: "
            + result.stderr.decode("utf-8", errors="replace")[:1000]
        )
    try:
        rust_constants = _json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError) as e:
        fail(
            f"could not parse `{binary} --print-canonical-templates` "
            f"output as JSON: {type(e).__name__}: {e}. Output first "
            "1000 chars: "
            + result.stdout[:1000].decode("utf-8", errors="replace")
        )
    if not isinstance(rust_constants, dict):
        fail(
            f"Rust drift-check output is not a JSON object; got "
            f"{type(rust_constants).__name__}"
        )

    py_constants = prompts.CANONICAL_TEMPLATES
    py_keys, rust_keys = set(py_constants.keys()), set(rust_constants.keys())
    if py_keys != rust_keys:
        missing_in_rust = sorted(py_keys - rust_keys)
        extra_in_rust   = sorted(rust_keys - py_keys)
        parts = ["prompt template constant key DRIFT between Rust and Python:"]
        if missing_in_rust:
            parts.append(
                f"  Keys in worker/prompts.py.CANONICAL_TEMPLATES "
                f"but missing in Rust output: {missing_in_rust}"
            )
        if extra_in_rust:
            parts.append(
                f"  Keys in Rust output but missing in "
                f"worker/prompts.py.CANONICAL_TEMPLATES: {extra_in_rust}"
            )
        parts.append(
            "Reconcile both sides. worker/prompts.py is the single source "
            "of truth per its file-level banner; update "
            "rust_server/src/prompt_validator.rs to match and rebuild."
        )
        fail("\n".join(parts))

    drift = []
    for key in sorted(py_keys):
        py_val   = py_constants[key]
        rust_val = rust_constants[key]
        if py_val != rust_val:
            drift.append(f"  {key}: Python={py_val!r}, Rust={rust_val!r}")
    if drift:
        fail(
            "prompt template constant DRIFT between Rust binary at "
            + binary
            + " and worker/prompts.py — the values disagree:\n"
            + "\n".join(drift)
            + "\nUpdate rust_server/src/prompt_validator.rs to match "
              "worker/prompts.py byte-for-byte and rebuild the container. "
              "NVIDIA's training code uses these exact strings; any drift "
              "moves inference off-distribution."
        )
    ok(
        f"prompt template constants verified in lockstep between Rust "
        f"binary ({binary}) and worker/prompts.py "
        f"({len(py_constants)} entries)"
    )


def run_all(model_dir: str) -> None:
    validate_env()
    validate_gpu()
    validate_model_dir(model_dir)
    validate_preprocessor_config(model_dir)
    validate_prompt_template_drift()
    ok("all preflight checks passed")
