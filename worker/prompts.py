"""
Canonical LocateAnything-3B prompt templates — THE SINGLE SOURCE OF TRUTH
=========================================================================

Every allowed prompt template the model was trained on lives here, verbatim.

Verbatim provenance from NVIDIA's released code (Eagle repo commit
`e21b6ac` "Release LocateAnything Code"):
  * `Embodied/locateanything_worker.py:103-137` — the reference worker
    that NVIDIA ships in their HF model card example.
  * `Embodied/evaluation/inference_detection_ddp.py:163-164,209-210`
  * `Embodied/evaluation/inference_grounding_ddp.py:245-248,303-311,
     765-789,794-801`
  * `Embodied/evaluation/inference_sspro_ddp.py:225-226`
  * `Embodied/document/DATA_PREPARATION.md:155-203` (training samples)
  * Paper Table 9 (cross-checked).

The model is sensitive to per-word prompt phrasing — the asymmetry between
`matches` (templates 1, 2, 6, 7) and `match` (template 3), the `</c>`
separator with no whitespace, and the trailing period are all part of
what the model was trained on. DO NOT paraphrase.

Anything that references the list of allowed prompts MUST resolve back
to this file. The repository contains:
  * a strict Rust validator in `rust_server/src/prompt_validator.rs` that
    mirrors these constants byte-for-byte and rejects malformed prompts at
    the WebSocket edge — its constants are compared against THIS file at
    container boot (worker/validate_startup.py:validate_prompt_template_drift),
    so any drift between Rust and Python is a hard boot failure;
  * documentation (docs/CLIENT_PROTOCOL.md, docs/MODEL_CAPABILITIES.md,
    docs/DRONE_DETECTION.md) which only POINTS at this file rather than
    duplicating its content;
  * the GET /v1/capabilities response, which carries
    CANONICAL_REFERENCE_URL so clients can render a link to it.
"""

from __future__ import annotations
from typing import Iterable, List


# ===========================================================================
# Canonical reference — what every error message and capability response
# points clients at when they need to look up the allowed templates.
# ===========================================================================

CANONICAL_REFERENCE_URL = (
    "https://github.com/BigBIueWhale/locate_anything_setup/blob/main/worker/prompts.py"
)


# ===========================================================================
# Verbatim template constants — THESE are the single source of truth.
# ===========================================================================
# Every other place in the repo (Rust validator, docs) MUST mirror these
# exactly. The boot-time drift check enforces this for the Rust side; docs
# point here rather than duplicate, by policy.

CATEGORY_SEPARATOR = "</c>"

# Templates 1, 2, 6: take a non-empty bracketed slot after the prefix and
# end with a literal period.
DETECTION_PREFIX      = "Locate all the instances that matches the following description: "
SINGLE_PHRASE_PREFIX  = "Locate a single instance that matches the following description: "
GUI_BOX_PREFIX        = "Locate the region that matches the following description: "

# Template 3: phrase grounding multi — note `match` (no trailing 's').
# The asymmetry vs templates 1/2/6 is intentional and trained, verified
# verbatim in locateanything_worker.py:114 and DATA_PREPARATION.md:171.
MULTI_PHRASE_PREFIX   = "Locate all the instances that match the following description: "

# Template 4: text grounding — `referred as`, no "the following description".
TEXT_GROUNDING_PREFIX = "Please locate the text referred as "

# Template 5: scene-text detection — fixed, takes no slot.
SCENE_TEXT_EXACT      = "Detect all the text in box format."

# Template 7: pointing / GUI-point. Single category per call only; for
# multi-category pointing the NVIDIA eval (inference_grounding_ddp.py:297-312)
# loops one category at a time and merges client-side.
POINT_PREFIX          = "Point to: "


# ---------------------------------------------------------------------------
# Canonical chat-template wrapping — the EXACT bookend strings NVIDIA's SFT
# trainer fed the model. The processor's `py_apply_chat_template` renders
# every conversation as:
#
#   {CANONICAL_RENDERED_PREFIX}<image-1>{user_prompt}{CANONICAL_RENDERED_SUFFIX}
#
# These bookends are an invariant across every Frame the model has ever
# seen. They are checked per-request in `worker/inference.py::run()` so that
# any silent swap to a different chat template (e.g. the Qwen-default
# tokenizer template that produces `You are Qwen, created by Alibaba
# Cloud. ...`) hard-fails immediately rather than via output-quality
# regression. Verified live against the running tokenizer + verified
# against the SFT trainer's render at
# /tmp/nvlabs_eagle/Embodied/eaglevl/train/locany_finetune_magi_stream.py.
# ---------------------------------------------------------------------------

CANONICAL_RENDERED_PREFIX = (
    "<|im_start|>system\n"
    "You are a helpful assistant.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
)

CANONICAL_RENDERED_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# Aggregated for the boot-time drift check (Python ↔ Rust). The Rust binary
# emits its embedded constants via `la_server --print-canonical-templates`;
# we compare the JSON output against this dict.
CANONICAL_TEMPLATES = {
    "category_separator":    CATEGORY_SEPARATOR,
    "detection_prefix":      DETECTION_PREFIX,
    "single_phrase_prefix":  SINGLE_PHRASE_PREFIX,
    "multi_phrase_prefix":   MULTI_PHRASE_PREFIX,
    "text_grounding_prefix": TEXT_GROUNDING_PREFIX,
    "scene_text_exact":      SCENE_TEXT_EXACT,
    "gui_box_prefix":        GUI_BOX_PREFIX,
    "point_prefix":          POINT_PREFIX,
}


# ===========================================================================
# Helper builders — every helper uses the constants above so the constants
# remain the single source of truth.
# ===========================================================================


def detect_categories(categories: Iterable[str]) -> str:
    """Closed-class detection (template 1).

    Categories are joined with the literal `</c>` separator (no whitespace);
    the verbatim form is `{DETECTION_PREFIX}A</c>B</c>C.`.
    """
    cats = [c.strip() for c in categories if c.strip()]
    if not cats:
        raise ValueError("detect_categories requires at least one category")
    joined = CATEGORY_SEPARATOR.join(cats)
    return f"{DETECTION_PREFIX}{joined}."


def ground_single(phrase: str) -> str:
    """Phrase grounding — single instance (template 2)."""
    if not phrase or not phrase.strip():
        raise ValueError("ground_single requires a non-empty phrase")
    return f"{SINGLE_PHRASE_PREFIX}{phrase.strip()}."


def ground_multi(phrase: str) -> str:
    """Phrase grounding — multiple instances (template 3).

    Note `match` (no trailing 's') — the asymmetry vs the other Locate-style
    templates is intentional and trained.
    """
    if not phrase or not phrase.strip():
        raise ValueError("ground_multi requires a non-empty phrase")
    return f"{MULTI_PHRASE_PREFIX}{phrase.strip()}."


def ground_text(phrase: str) -> str:
    """Text grounding (template 4)."""
    if not phrase or not phrase.strip():
        raise ValueError("ground_text requires a non-empty phrase")
    return f"{TEXT_GROUNDING_PREFIX}{phrase.strip()}."


def detect_text() -> str:
    """Scene-text detection (template 5)."""
    return SCENE_TEXT_EXACT


def ground_gui(phrase: str, output_type: str = "box") -> str:
    """GUI grounding — box (template 6) or point (template 7)."""
    if not phrase or not phrase.strip():
        raise ValueError("ground_gui requires a non-empty phrase")
    if output_type == "point":
        return f"{POINT_PREFIX}{phrase.strip()}."
    return f"{GUI_BOX_PREFIX}{phrase.strip()}."


def point_to(phrase: str) -> str:
    """Pointing (template 7) — single category/phrase per call.

    Multi-category pointing is N separate calls, merged client-side, per
    NVIDIA's inference_grounding_ddp.py:297-312.
    """
    if not phrase or not phrase.strip():
        raise ValueError("point_to requires a non-empty phrase")
    return f"{POINT_PREFIX}{phrase.strip()}."


# ===========================================================================
# Demo and preset prompt bundles — concrete, well-formed prompts the server
# advertises in /v1/capabilities.preset_prompts for clients to start from.
# The model was NOT trained on aerial / sky / drone imagery; see
# docs/DRONE_DETECTION.md for the honest caveats.
# ===========================================================================

DRONE_PROMPTS_RANKED: List[str] = [
    point_to("drone in the sky"),
    detect_categories(["drone"]),
    ground_multi("a small drone in the sky"),
    point_to("quadcopter"),
    detect_categories(["drone", "quadcopter", "uav", "aircraft"]),
    ground_multi("a flying object in the sky"),
    ground_single("the drone"),
]

HOUSEHOLD_PROMPTS = {
    "office":      detect_categories(["bottle", "cup", "laptop", "keyboard", "mouse", "monitor", "book"]),
    "living_room": detect_categories(["person", "dog", "cat", "couch", "tv", "laptop", "book", "cup"]),
    "kitchen":     detect_categories(["bottle", "cup", "bowl", "knife", "spoon", "refrigerator", "microwave"]),
    "street":      detect_categories(["person", "car", "bus", "bicycle", "traffic light", "stop sign"]),
    "demo":        detect_categories(["person", "laptop", "bottle", "cup", "book", "monitor", "keyboard"]),
}
