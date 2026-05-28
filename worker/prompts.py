"""
Canonical LocateAnything-3B prompt templates.

Verbatim from `Embodied/locateanything_worker.py` and Table 9 of the paper.
The model is sensitive to prompt phrasing — these forms are what it was
trained on. DO NOT paraphrase.
"""

from __future__ import annotations
from typing import Iterable, List

# Joiner between categories in the closed-class detection prompt.
CATEGORY_SEPARATOR = "</c>"


def detect_categories(categories: Iterable[str]) -> str:
    """`Locate all the instances that matches the following description: A</c>B</c>C.`

    Matches the LocateAnything detection prompt. Categories are joined by
    `</c>` (no space). Use for closed-class detection.
    """
    cats = [c.strip() for c in categories if c.strip()]
    if not cats:
        raise ValueError("detect_categories requires at least one category")
    joined = CATEGORY_SEPARATOR.join(cats)
    return f"Locate all the instances that matches the following description: {joined}."


def ground_multi(phrase: str) -> str:
    """`Locate all the instances that match the following description: <phrase>.`

    Phrase grounding for multiple instances of a free-form referring phrase.
    """
    if not phrase or not phrase.strip():
        raise ValueError("ground_multi requires a non-empty phrase")
    return f"Locate all the instances that match the following description: {phrase.strip()}."


def ground_single(phrase: str) -> str:
    """`Locate a single instance that matches the following description: <phrase>.`"""
    if not phrase or not phrase.strip():
        raise ValueError("ground_single requires a non-empty phrase")
    return f"Locate a single instance that matches the following description: {phrase.strip()}."


def ground_text(phrase: str) -> str:
    """`Please locate the text referred as <phrase>.`"""
    if not phrase or not phrase.strip():
        raise ValueError("ground_text requires a non-empty phrase")
    return f"Please locate the text referred as {phrase.strip()}."


def detect_text() -> str:
    """`Detect all the text in box format.`"""
    return "Detect all the text in box format."


def ground_gui(phrase: str, output_type: str = "box") -> str:
    """GUI grounding (box or point form)."""
    if not phrase or not phrase.strip():
        raise ValueError("ground_gui requires a non-empty phrase")
    if output_type == "point":
        return f"Point to: {phrase.strip()}."
    return f"Locate the region that matches the following description: {phrase.strip()}."


def point_to(phrase: str) -> str:
    """`Point to: <phrase>.`"""
    if not phrase or not phrase.strip():
        raise ValueError("point_to requires a non-empty phrase")
    return f"Point to: {phrase.strip()}."


# --------- Demo and preset prompt bundles ---------------------------------
# These bundles are *useful sets* for the documented use cases. The model
# was NOT trained on aerial / sky / drone imagery — see docs/MODEL_CAPABILITIES.md.

DRONE_PROMPTS_RANKED: List[str] = [
    # Ordered best→worst expected per research subagent's analysis.
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
