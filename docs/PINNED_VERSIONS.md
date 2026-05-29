# Pinned versions — what each pin is and why

Every pin lives in [`scripts/lib/versions.sh`](../scripts/lib/versions.sh).
The Dockerfile, the host validator, and the Bash orchestration all
read from that one file. To change a version, edit one line there
and re-run `setup.sh`.

---

## Host preconditions (validated, not installed)

| Var | Value | Why |
|---|---|---|
| `LA_REQUIRE_UBUNTU_CODENAME`     | `noble`        | 24.04 LTS — what the personal_server stack pins. |
| `LA_REQUIRE_DRIVER_BRANCH`       | `595`          | Matches `nvidia-driver-595-open` from personal_server §10. |
| `LA_REQUIRE_DRIVER_MIN`          | `595.45.04`    | Minimum that supports CUDA 13.0 per NVIDIA's release notes. |
| `LA_REQUIRE_GPU_COMPUTE_CAP`     | `12.0`         | RTX 5090 / Blackwell sm_120. The torch wheels and flash-attn build below target sm_120 specifically. |
| `LA_REQUIRE_GPU_MEM_MIN_MIB`     | `24000`        | bf16 weights ~7 GiB + KV/activation headroom for 16K context. |
| `LA_REQUIRE_DOCKER_MAJOR`        | `29`           | Matches `docker-ce = 5:29.4.1-1` from personal_server §12. |
| `LA_REQUIRE_NVCTK_VERSION`       | `1.19.0`       | personal_server §13 pin. |
| `LA_REQUIRE_DISK_FREE_GIB`       | `30`           | ~12 GiB image + 8 GiB weights + headroom. |

## Model identity

| Var | Value | Source |
|---|---|---|
| `LA_MODEL_HF_REPO`     | `nvidia/LocateAnything-3B`     | Verified live on HF; published under the `nvidia` org. |
| `LA_MODEL_HF_REVISION` | `7a81d81`                       | HF main commit at 2026-05-28. Pinned so `main` can move without affecting this build. |

## Container base image

| Var | Value | Why |
|---|---|---|
| `LA_CUDA_BASE_IMAGE` | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` | Matches the host's CUDA 13.0.x driver capability. `-devel` because we source-build flash-attn. cuDNN for transformer kernels. ubuntu24.04 matches the host. |
| `LA_RUST_BUILDER_IMAGE` | `rust:1.95-bookworm` | Rust 1.95 stable on Debian 12. glibc 2.36 builds; runs fine on the runtime's glibc 2.39 (forward compat). |

## Python + PyTorch

| Var | Value | Why |
|---|---|---|
| `LA_PYTHON_PKG`          | `python3.12`        | Ubuntu 24.04 default; all pinned deps have py312 wheels. |
| `LA_TORCH_VERSION`       | `2.12.0`            | Latest stable as of 2026-05-28; cu130 wheels have native sm_120 kernels. |
| `LA_TORCHVISION_VERSION` | `0.27.0`            | Matches torch 2.12. |
| `LA_TORCH_CUDA_TAG`      | `cu130`             | Stable Blackwell-capable. cu132 is "experimental". |
| `LA_TORCH_INDEX_URL`     | `https://download.pytorch.org/whl/cu130` | PyTorch's pinned cu130 index. |

## flash-attn (source build, sm_120 only)

| Var | Value | Why |
|---|---|---|
| `LA_FLASH_ATTN_VERSION` | `2.8.3` | Latest 2.x on PyPI; source-built with sm_120 kernels. The model's modeling_qwen2.py conditionally imports flash_attn at module load (`if is_flash_attn_2_available(): from flash_attn import ...`), so we keep it installed even though the active attn path is sdpa (see below) — letting the conditional import succeed avoids module-load surprises. FA4 (next generation) does NOT run on RTX 5090: sm_120 lacks the TMEM hardware FA4 requires. |
| `LA_FLASH_ATTN_ARCHS`   | `120`   | Build only sm_120 kernels. Shortens build time ~5× vs. the default `80;90;100;110;120` sweep. |

`magi_attention` is **omitted**. The model's `config.json` declares
`_attn_implementation='magi'` (SandAI MagiAttention), but the FFA_FA4
cutlass kernels target `sm_100a` (Blackwell datacenter B200) using
architecture-specific instructions (TMEM, tcgen05/UMMA) that do **not**
exist on sm_120 consumer Blackwell. Per NVIDIA's own Blackwell
Compatibility Guide, sm_100a binaries are not forward-compatible to
sm_120 — there is no PTX-JIT rescue path. The MagiAttention maintainer
confirms sm_120 is on the roadmap, not yet shipped
(SandAI/MagiAttention#184).

We override the model's attention to **sdpa** via `LA_ATTN_IMPL=sdpa`.
This is **not** a "fallback" in the degraded-quality sense — it is the
only viable path on sm_120 that preserves the train-time attention
pattern. The model's custom `modeling_qwen2.py:Qwen2Model.forward()`
has exactly two valid branches: `magi` and `sdpa`. Any other value
(including `flash_attention_2`) raises `NotImplementedError` at
line 1335. The `sdpa` branch reconstructs the same block-mask
attention pattern (bidirectional-within-window + blocked-just-emitted-
token + causal prefix) via
`mask_sdpa_utils.update_causal_mask_for_one_gen_window_2d`, then runs
it through PyTorch SDPA. The result is mathematically equivalent to
`magi+hybrid` within bf16 precision; only execution speed differs.

Override mechanics: NVIDIA's model code defines a custom
`_autoset_attn_implementation` that silently drops user-provided
`attn_implementation=` kwargs on `from_pretrained` whenever
config.json says `'magi'`. To force the override, `worker/inference.py`
loads the AutoConfig explicitly, mutates `_attn_implementation` on
both the top-level config and the inner `text_config`, then passes
the mutated config to `from_pretrained`. A boot-time verification
then re-reads the attribute on the constructed model and refuses to
serve if the override did not stick.

## Model-mandated Python deps (pinned EXACTLY)

These match the upstream `nvlabs/Eagle/Embodied/pyproject.toml` and
the model card's quoted install recipe. The model code imports
specific behavior from these versions — don't loosen the pins.

| Var | Value |
|---|---|
| `LA_TRANSFORMERS_VERSION` | `4.57.1` |
| `LA_TOKENIZERS_VERSION`   | `0.22.0` |
| `LA_ACCELERATE_VERSION`   | `1.5.2`  |
| `LA_PEFT_VERSION`         | `0.12.0` |
| `LA_SENTENCEPIECE_VERSION`| `0.2.0`  |
| `LA_NUMPY_VERSION`        | `1.25.0` |
| `LA_PILLOW_VERSION`       | `11.1.0` |
| `LA_OPENCV_VERSION`       | `4.11.0.86` (opencv-python-headless) |
| `LA_DECORD_VERSION`       | `0.6.0`  |
| `LA_LMDB_VERSION`         | `1.7.5`  |

## Python sidecar deps (server, not model)

| Var | Value | Why |
|---|---|---|
| `LA_HFHUB_VERSION`       | `0.27.0` | `huggingface_hub.snapshot_download`. |
| `LA_HF_TRANSFER_VERSION` | `0.1.8`  | hf_transfer for fast parallel download. |
| `LA_PSUTIL_VERSION`      | `6.1.0`  | Optional, used in /v1/info. |

## Rust crate pins

Authoritative in [`rust_server/Cargo.toml`](../rust_server/Cargo.toml)
with `=X.Y.Z` exact-equality requirements. The full transitive
closure is frozen in [`rust_server/Cargo.lock`](../rust_server/Cargo.lock).
`cargo build --locked` enforces both. There is no shadow copy of
these versions anywhere else in the repo — to upgrade a crate, edit
`Cargo.toml` and run `cargo generate-lockfile`. All pins were
verified live against crates.io at 2026-05-28.

## Generation parameters (baked at build time, validated at boot)

These ENV vars are set in the Dockerfile and read by
`worker/inference.py:GenConfig.from_env()`. They EQUAL the values
used in `Embodied/evaluation/inference_compat.py:build_generate_kwargs`
for all benchmark runs in the paper.

| ENV var                    | Value             | Source |
|---|---|---|
| `LA_MODEL_DTYPE`           | `bfloat16`        | `config.json:torch_dtype`. |
| `LA_ATTN_IMPL`             | `sdpa`              | Override of `config.json:_attn_implementation='magi'` — magi unbuildable on sm_120, sdpa is the only other valid branch in `Qwen2Model.forward()` and reconstructs the same block-mask pattern. See `## magi_attention` section above. |
| `LA_GEN_TEMPERATURE`       | `0.7`             | `inference_compat.py:55`. |
| `LA_GEN_TOP_P`             | `0.9`             | `inference_compat.py:56`. |
| `LA_GEN_DO_SAMPLE`         | `1`               | `inference_compat.py:54`. |
| `LA_GEN_REP_PEN`           | `1.1`             | `inference_compat.py:57`. |
| `LA_GEN_MAX_NEW_TOKENS`    | `8192`            | README recommendation. |
| `LA_GEN_MODE`              | `hybrid`          | Default; per-request override supported. |
| `LA_GEN_N_FUTURE_TOKENS`   | `6`               | Trained block size (`config.json:text_config.block_size`). |

The Python worker **hard-fails at boot** if any of these deviates
from the canonical values — see `worker/validate_startup.py` →
`validate_env` → the `canonical` dict at the bottom of the function.
There is no fallback and no opt-in flag. If you intentionally want
to deviate (e.g., to A/B-test a sampling change), edit the env
pin in `versions.sh` AND edit the canonical baseline in
`validate_startup.py` together so the boot check still passes —
this leaves an auditable two-place diff in the repo so anyone
reading the code can see the deviation is deliberate.

## What's NOT pinned (intentionally)

* The host driver minor version (we check ≥ `LA_REQUIRE_DRIVER_MIN`
  but accept any patch).
* The host Docker version's patch level (only major must match).
* `pip` / `setuptools` / `wheel` (pinned inside the Dockerfile to
  `25.2 / 75.6.0 / 0.45.1`; intentionally not surfaced to versions.sh
  because they are build-tool, not runtime).
