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
# regression. Cross-verified against the running tokenizer and against
# NVlabs/Eagle's SFT trainer at
# `Embodied/eaglevl/train/locany_finetune_magi_stream.py`.
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


# Stable enum-string per template, used for the per-request `prompt_task`
# field on the Rust→Python IPC header AND surfaced verbatim in the client-
# facing Result body. Order must match
# `rust_server/src/prompt_validator.rs::TemplateKind` declaration order
# AND the keys of `worker/inference.py::EXPECTED_SHAPE`; the boot drift
# check (worker/validate_startup.py::validate_prompt_template_drift)
# enforces both equalities so a future rename on either side fails the
# container start.
TEMPLATE_WIRE_NAMES = [
    "detection",
    "phrase_single",
    "phrase_multi",
    "text_grounding",
    "scene_text",
    "gui_box",
    "point",
]


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
    "template_wire_names":   TEMPLATE_WIRE_NAMES,
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


def classify_prompt(prompt: str) -> str:
    """Classify a canonical prompt string into its `prompt_task` wire name
    (one of TEMPLATE_WIRE_NAMES). This is the Python mirror of the Rust
    validator's classification step at
    `rust_server/src/prompt_validator.rs::validate()` — used by
    `worker/calibration.py` and other internal call sites that need the
    wire name but don't go through the WebSocket path (where the Rust
    validator already classifies).

    Strict: raises ValueError if `prompt` doesn't start with one of the
    canonical prefixes. Prefix matching is byte-exact — the
    `matches`/`match` and `description: ` distinctions are load-bearing.
    The boot drift check ensures the wire names returned here remain
    in lockstep with the Rust validator's `TemplateKind::wire_name`.
    """
    if prompt == SCENE_TEXT_EXACT:
        return "scene_text"
    if prompt.startswith(DETECTION_PREFIX):
        return "detection"
    if prompt.startswith(SINGLE_PHRASE_PREFIX):
        return "phrase_single"
    if prompt.startswith(MULTI_PHRASE_PREFIX):
        return "phrase_multi"
    if prompt.startswith(TEXT_GROUNDING_PREFIX):
        return "text_grounding"
    if prompt.startswith(GUI_BOX_PREFIX):
        return "gui_box"
    if prompt.startswith(POINT_PREFIX):
        return "point"
    raise ValueError(
        f"prompt does not match any of the seven canonical LocateAnything-3B "
        f"template prefixes (see {CANONICAL_REFERENCE_URL}): {prompt!r}"
    )


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


# ===========================================================================
# Typed PromptRequest constructors — the wire-v2 shape that
# /v1/capabilities.preset_prompts advertises (A.5). With the request now a
# typed sum (mirroring rust_server/src/protocol.rs::PromptRequest, internally
# tagged on `task`), advertising bare prompt STRINGS would be
# wire-inconsistent: a client cannot send a string, only a typed request. So
# the preset bundles below are lists of {label, request, generation_mode}
# objects whose `request` is exactly a PromptRequest dict.
#
# Each `task` value is one of TEMPLATE_WIRE_NAMES; the slot field names match
# the Rust *Req structs (DetectionReq.categories, PhraseReq.phrase,
# TextReq.text, DescReq.description, SceneTextReq{}). Slot rules are now
# server-authoritative (validated in the Rust builder), but presets are kept
# clean here (strip + non-empty) so every advertised preset is accepted as-is.
# ===========================================================================


def _clean_slot(s: str, what: str) -> str:
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"{what} requires a non-empty string")
    return s.strip()


def req_detection(categories: Iterable[str]) -> dict:
    """Typed detection request (task `detection`). Mirrors DetectionReq."""
    cats = [c.strip() for c in categories if isinstance(c, str) and c.strip()]
    if not cats:
        raise ValueError("req_detection requires at least one category")
    return {"task": "detection", "categories": cats}


def req_phrase_single(phrase: str) -> dict:
    """Typed single-instance phrase grounding (task `phrase_single`)."""
    return {"task": "phrase_single", "phrase": _clean_slot(phrase, "req_phrase_single")}


def req_phrase_multi(phrase: str) -> dict:
    """Typed multi-instance phrase grounding (task `phrase_multi`)."""
    return {"task": "phrase_multi", "phrase": _clean_slot(phrase, "req_phrase_multi")}


def req_text_grounding(text: str) -> dict:
    """Typed text grounding (task `text_grounding`). Mirrors TextReq."""
    return {"task": "text_grounding", "text": _clean_slot(text, "req_text_grounding")}


def req_scene_text() -> dict:
    """Typed scene-text detection (task `scene_text`). Mirrors SceneTextReq{}."""
    return {"task": "scene_text"}


def req_gui_box(description: str) -> dict:
    """Typed GUI-region grounding (task `gui_box`). Mirrors DescReq."""
    return {"task": "gui_box", "description": _clean_slot(description, "req_gui_box")}


def req_point(phrase: str) -> dict:
    """Typed pointing request (task `point`). Mirrors PhraseReq."""
    return {"task": "point", "phrase": _clean_slot(phrase, "req_point")}


def _preset(label: str, request: dict, generation_mode: str) -> dict:
    """One advertised preset: a label, a typed PromptRequest, and a mode."""
    if generation_mode not in ("fast", "hybrid", "slow"):
        raise ValueError(
            f"preset generation_mode={generation_mode!r} must be "
            "'fast'|'hybrid'|'slow'"
        )
    return {"label": label, "request": request, "generation_mode": generation_mode}


# Typed equivalents of DRONE_PROMPTS_RANKED, in the same ranked order. `point`
# is the structurally cleanest drone prompt (few output tokens, MTP fast path);
# slow mode is the most accurate for the off-distribution aerial use case.
DRONE_PRESETS_RANKED: List[dict] = [
    _preset("drone (point)",                 req_point("drone in the sky"),                                   "slow"),
    _preset("drone (detect)",                req_detection(["drone"]),                                        "slow"),
    _preset("small drone (phrase, multi)",   req_phrase_multi("a small drone in the sky"),                    "slow"),
    _preset("quadcopter (point)",            req_point("quadcopter"),                                         "slow"),
    _preset("drone/uav/aircraft (detect)",   req_detection(["drone", "quadcopter", "uav", "aircraft"]),       "slow"),
    _preset("flying object (phrase, multi)", req_phrase_multi("a flying object in the sky"),                  "slow"),
    _preset("the drone (phrase, single)",    req_phrase_single("the drone"),                                  "slow"),
]

# Typed equivalents of HOUSEHOLD_PROMPTS (closed-class detection bundles).
HOUSEHOLD_PRESETS: List[dict] = [
    _preset("office",      req_detection(["bottle", "cup", "laptop", "keyboard", "mouse", "monitor", "book"]),   "hybrid"),
    _preset("living_room", req_detection(["person", "dog", "cat", "couch", "tv", "laptop", "book", "cup"]),       "hybrid"),
    _preset("kitchen",     req_detection(["bottle", "cup", "bowl", "knife", "spoon", "refrigerator", "microwave"]), "hybrid"),
    _preset("street",      req_detection(["person", "car", "bus", "bicycle", "traffic light", "stop sign"]),      "hybrid"),
    _preset("demo",        req_detection(["person", "laptop", "bottle", "cup", "book", "monitor", "keyboard"]),   "hybrid"),
]
