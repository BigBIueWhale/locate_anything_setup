//! Strict, regex-free prompt BUILDER for the seven canonical
//! LocateAnything-3B prompt templates.
//!
//! The client no longer sends a free-form prompt string. It sends a typed
//! [`crate::protocol::PromptRequest`] (a sum over the seven trained tasks).
//! This module COMPILES that typed request into the exact trained prompt
//! string — byte-equal to `worker/prompts.py` — after validating the typed
//! slot values. Illegal slots are REJECTED with an information-generous
//! diagnostic; they are NEVER silently normalized or filtered (we do not
//! port prompts.py's `.strip()` / empty-category drop).
//!
//! Single source of truth for the template literals = `worker/prompts.py`.
//! The constants in this file mirror that Python module byte-for-byte; drift
//! is caught at container boot by
//! `worker/validate_startup.py::validate_prompt_template_drift`, which runs
//! `la_server --print-canonical-templates` and dict-equals the JSON against
//! `prompts.CANONICAL_TEMPLATES`. The `TemplateKind`, `TemplateKind::wire_name`,
//! `template_wire_names` (inside `canonical_templates_json`) and the template
//! constants below are load-bearing for that drift chain — keep them exact.
//!
//! Why no regex: slot validation must produce information-generous error
//! messages that name exactly what went wrong, not just "did not match". The
//! if/else flow lets each check attach its own diagnostic — the client always
//! learns whether they sent an empty slot, surrounding whitespace, a stray
//! `</c>`, a trailing period, a control character, a non-NFC form, etc.

use crate::protocol::{
    DescReq, DetectionReq, PhraseReq, PromptRequest, SceneTextReq, TextReq, MAX_PROMPT_CHARS,
};
use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// Single-source-of-truth constants. Mirror of worker/prompts.py.
// (KEEP byte-exact — the boot drift check and `--print-canonical-templates`
//  depend on these and on `canonical_templates_json()` below.)
// ---------------------------------------------------------------------------

/// Repository link clients are pointed at from every prompt error message
/// and from `GET /v1/capabilities.prompt_templates_reference_url`.
pub const CANONICAL_REFERENCE_URL: &str =
    "https://github.com/BigBIueWhale/locate_anything_setup/blob/main/worker/prompts.py";

pub const CATEGORY_SEPARATOR: &str = "</c>";

pub const DETECTION_PREFIX:      &str = "Locate all the instances that matches the following description: ";
pub const SINGLE_PHRASE_PREFIX:  &str = "Locate a single instance that matches the following description: ";
pub const MULTI_PHRASE_PREFIX:   &str = "Locate all the instances that match the following description: ";
pub const TEXT_GROUNDING_PREFIX: &str = "Please locate the text referred as ";
pub const SCENE_TEXT_EXACT:      &str = "Detect all the text in box format.";
pub const GUI_BOX_PREFIX:        &str = "Locate the region that matches the following description: ";
pub const POINT_PREFIX:          &str = "Point to: ";

/// Per-slot maximum length in CHARACTERS (A.1). Applies to every typed slot
/// value (each category, each phrase, the text/description). The compiled
/// prompt is separately capped at `MAX_PROMPT_CHARS`.
pub const MAX_SLOT_CHARS: usize = 200;

/// Maximum number of categories in a closed-class detection request (A.1).
pub const MAX_DETECTION_CATEGORIES: usize = 10;

/// JSON dump used by `la_server --print-canonical-templates` for the
/// boot-time drift check against worker/prompts.py. The Python side's
/// `prompts.CANONICAL_TEMPLATES` dict must match this object exactly
/// (dict-equality); a mismatch in any key causes the worker to refuse
/// to start (worker/validate_startup.py::validate_prompt_template_drift).
///
/// `template_wire_names` enumerates the seven stable enum strings used
/// for the per-request `prompt_task` field (Rust→Python IPC). Listed in
/// TemplateKind declaration order, which is also the order they appear in
/// worker/prompts.py's TEMPLATE_WIRE_NAMES.
pub fn canonical_templates_json() -> serde_json::Value {
    serde_json::json!({
        "category_separator":    CATEGORY_SEPARATOR,
        "detection_prefix":      DETECTION_PREFIX,
        "single_phrase_prefix":  SINGLE_PHRASE_PREFIX,
        "multi_phrase_prefix":   MULTI_PHRASE_PREFIX,
        "text_grounding_prefix": TEXT_GROUNDING_PREFIX,
        "scene_text_exact":      SCENE_TEXT_EXACT,
        "gui_box_prefix":        GUI_BOX_PREFIX,
        "point_prefix":          POINT_PREFIX,
        "template_wire_names": [
            TemplateKind::Detection.wire_name(),
            TemplateKind::PhraseSingle.wire_name(),
            TemplateKind::PhraseMulti.wire_name(),
            TemplateKind::TextGrounding.wire_name(),
            TemplateKind::SceneText.wire_name(),
            TemplateKind::GuiBox.wire_name(),
            TemplateKind::Point.wire_name(),
        ],
    })
}

// ---------------------------------------------------------------------------
// Template kind — the stable wire-name catalog (KEEP: drift chain).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateKind {
    Detection,
    PhraseSingle,
    PhraseMulti,
    TextGrounding,
    SceneText,
    GuiBox,
    Point,
}

impl TemplateKind {
    /// Human-readable name used inside diagnostic messages.
    pub fn display_name(self) -> &'static str {
        match self {
            TemplateKind::Detection     => "closed-class detection",
            TemplateKind::PhraseSingle  => "phrase grounding (single instance)",
            TemplateKind::PhraseMulti   => "phrase grounding (multiple instances)",
            TemplateKind::TextGrounding => "text grounding",
            TemplateKind::SceneText     => "scene-text detection",
            TemplateKind::GuiBox        => "GUI grounding (box output)",
            TemplateKind::Point         => "pointing / GUI grounding (point output)",
        }
    }

    /// Verbatim canonical form (with `[CATS]` / `[PHRASE]` slot marker)
    /// used when echoing the expected template back to the client.
    pub fn canonical_form(self) -> &'static str {
        match self {
            TemplateKind::Detection     => "Locate all the instances that matches the following description: [CATS].",
            TemplateKind::PhraseSingle  => "Locate a single instance that matches the following description: [PHRASE].",
            TemplateKind::PhraseMulti   => "Locate all the instances that match the following description: [PHRASE].",
            TemplateKind::TextGrounding => "Please locate the text referred as [PHRASE].",
            TemplateKind::SceneText     => SCENE_TEXT_EXACT,
            TemplateKind::GuiBox        => "Locate the region that matches the following description: [PHRASE].",
            TemplateKind::Point         => "Point to: [PHRASE].",
        }
    }

    /// Stable enum-string used for the per-request `prompt_task` field
    /// on the Rust→Python IPC header. Must match the keys of
    /// worker/inference.py::EXPECTED_SHAPE exactly — drift detected at
    /// boot by worker/validate_startup.py::validate_prompt_template_drift
    /// via the `template_wire_names` list in canonical_templates_json().
    pub fn wire_name(self) -> &'static str {
        match self {
            TemplateKind::Detection     => "detection",
            TemplateKind::PhraseSingle  => "phrase_single",
            TemplateKind::PhraseMulti   => "phrase_multi",
            TemplateKind::TextGrounding => "text_grounding",
            TemplateKind::SceneText     => "scene_text",
            TemplateKind::GuiBox        => "gui_box",
            TemplateKind::Point         => "point",
        }
    }
}

// ---------------------------------------------------------------------------
// Slot error.
// ---------------------------------------------------------------------------

/// A typed-slot validation failure. `message()` returns the full,
/// information-generous English diagnostic — what specifically failed, which
/// canonical template it belongs to, and the canonical-templates URL. The WS
/// edge maps this onto a per-frame `error{code:"invalid_request"}`.
#[derive(Debug, Clone)]
pub struct SlotError {
    message: String,
}

impl SlotError {
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for SlotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SlotError {}

// ---------------------------------------------------------------------------
// Builder entry point.
// ---------------------------------------------------------------------------

/// Compile a typed [`PromptRequest`] into the exact trained prompt string
/// plus the `prompt_task` wire-name (one of `TemplateKind::wire_name`).
///
/// The returned `String` is byte-equal to what `worker/prompts.py` would
/// build for the same (already-valid) slot values. Slots are validated per
/// A.1 BEFORE assembly; any violation returns a [`SlotError`] (→ the WS edge
/// emits `error{code:"invalid_request"}`) — we REJECT, never normalize/strip.
pub fn compile(request: &PromptRequest) -> Result<(String, &'static str), SlotError> {
    let (prompt, kind) = match request {
        PromptRequest::Detection(DetectionReq { categories }) => {
            (compile_detection(categories)?, TemplateKind::Detection)
        }
        PromptRequest::PhraseSingle(PhraseReq { phrase }) => {
            validate_slot(phrase, TemplateKind::PhraseSingle, "phrase", false)?;
            (format!("{SINGLE_PHRASE_PREFIX}{phrase}."), TemplateKind::PhraseSingle)
        }
        PromptRequest::PhraseMulti(PhraseReq { phrase }) => {
            validate_slot(phrase, TemplateKind::PhraseMulti, "phrase", false)?;
            (format!("{MULTI_PHRASE_PREFIX}{phrase}."), TemplateKind::PhraseMulti)
        }
        PromptRequest::TextGrounding(TextReq { text }) => {
            validate_slot(text, TemplateKind::TextGrounding, "text", false)?;
            (format!("{TEXT_GROUNDING_PREFIX}{text}."), TemplateKind::TextGrounding)
        }
        PromptRequest::SceneText(SceneTextReq {}) => {
            // No slot; the literal trained string, no period appended.
            (SCENE_TEXT_EXACT.to_string(), TemplateKind::SceneText)
        }
        PromptRequest::GuiBox(DescReq { description }) => {
            validate_slot(description, TemplateKind::GuiBox, "description", false)?;
            (format!("{GUI_BOX_PREFIX}{description}."), TemplateKind::GuiBox)
        }
        PromptRequest::Point(PhraseReq { phrase }) => {
            // Pointing forbids commas (A.1): NVIDIA's eval issues one
            // `Point to: <single>.` per category and merges client-side.
            validate_slot(phrase, TemplateKind::Point, "phrase", true)?;
            (format!("{POINT_PREFIX}{phrase}."), TemplateKind::Point)
        }
    };

    // Hard cap on the assembled prompt (A.1). The model's
    // tokenizer.model_max_length is 16384 tokens; even on ASCII this char
    // cap is generous. Per-slot ≤200 keeps us well under for normal inputs;
    // this gate also bounds the worst case of 10 maxed categories.
    let char_count = prompt.chars().count();
    if char_count > MAX_PROMPT_CHARS {
        return Err(build_error(
            &format!(
                "compiled prompt length {char_count} chars exceeds \
                 MAX_PROMPT_CHARS={MAX_PROMPT_CHARS} (the model's \
                 tokenizer.model_max_length is 16384 tokens; even pure-ASCII \
                 this character cap is generous)"
            ),
            Some(kind),
        ));
    }

    Ok((prompt, kind.wire_name()))
}

// ---------------------------------------------------------------------------
// Slot validators.
// ---------------------------------------------------------------------------

/// Compile + validate the closed-class detection categories slot. `[CATS]` is
/// 1..=10 categories joined by the literal `</c>` separator; each category is
/// validated as a slot string AND must contain no comma.
fn compile_detection(categories: &[String]) -> Result<String, SlotError> {
    if categories.is_empty() {
        return Err(build_error(
            "closed-class detection requires at least one category \
             (`request.categories` was empty)",
            Some(TemplateKind::Detection),
        ));
    }
    if categories.len() > MAX_DETECTION_CATEGORIES {
        return Err(build_error(
            &format!(
                "closed-class detection has {} categories; the maximum is \
                 {MAX_DETECTION_CATEGORIES}. Split into multiple Frames, or \
                 use multi-instance phrase grounding for a single free-form \
                 phrase.",
                categories.len()
            ),
            Some(TemplateKind::Detection),
        ));
    }
    for (i, cat) in categories.iter().enumerate() {
        // A category is a slot string with the extra no-comma rule; surface
        // its 1-based position in every diagnostic.
        validate_slot(cat, TemplateKind::Detection, &format!("category at position {}", i + 1), true)?;
    }
    let joined = categories.join(CATEGORY_SEPARATOR);
    Ok(format!("{DETECTION_PREFIX}{joined}."))
}

/// Validate a single typed slot string against the A.1 rules:
///   - non-empty,
///   - NFC (rejected if not already NFC — never normalized),
///   - no control characters,
///   - no leading/trailing whitespace,
///   - no literal `</c>`,
///   - must not end with `.`,
///   - length ≤ MAX_SLOT_CHARS chars,
///   - and (if `forbid_comma`) no `,`.
///
/// `field` is a short noun phrase ("phrase", "text", "description",
/// "category at position N") spliced into the diagnostic.
fn validate_slot(
    slot: &str,
    template: TemplateKind,
    field: &str,
    forbid_comma: bool,
) -> Result<(), SlotError> {
    if slot.is_empty() {
        return Err(build_error(
            &format!(
                "{} {field} is empty; a non-empty value is required",
                template.display_name()
            ),
            Some(template),
        ));
    }

    // Length cap (char count, not byte count, so multi-byte UTF-8 can't
    // sneak past a generous ASCII budget).
    let char_count = slot.chars().count();
    if char_count > MAX_SLOT_CHARS {
        return Err(build_error(
            &format!(
                "{} {field} length {char_count} chars exceeds the {MAX_SLOT_CHARS}-char \
                 per-slot cap",
                template.display_name()
            ),
            Some(template),
        ));
    }

    // No leading/trailing whitespace. NVIDIA's training samples have none and
    // prompts.py would `.strip()` — we REJECT instead of stripping so the
    // client knows the compiled prompt is exactly what they sent.
    if slot != slot.trim() {
        let leading = slot.len() - slot.trim_start().len();
        let trailing = slot.len() - slot.trim_end().len();
        return Err(build_error(
            &format!(
                "{} {field} has surrounding whitespace ({leading} leading byte(s), \
                 {trailing} trailing byte(s)); slots carry no padding. Trim before \
                 sending — the server will not strip it for you.",
                template.display_name()
            ),
            Some(template),
        ));
    }

    // No control characters (matches the website's §23.6 rule: any
    // `char::is_control()` is rejected). This also excludes interior
    // newlines/tabs that are not caught by the trim check.
    if let Some(c) = slot.chars().find(|c| c.is_control()) {
        return Err(build_error(
            &format!(
                "{} {field} contains a control character (U+{:04X}); control \
                 characters are not allowed in a prompt slot",
                template.display_name(),
                c as u32
            ),
            Some(template),
        ));
    }

    // NFC: reject (never normalize) anything not already in Normalization
    // Form Canonical Composition. Same predicate as the website's §23.6 rule 9
    // (`raw.chars().eq(raw.nfc())`) and the same crate pin — silently
    // normalizing would be MORE lenient than the model's tokenizer.
    if !slot.chars().eq(slot.nfc()) {
        return Err(build_error(
            &format!(
                "{} {field} is not in Unicode Normalization Form Canonical \
                 Composition (NFC); please retype the input. The server rejects \
                 non-NFC text rather than silently normalizing it (silent \
                 normalization would be more lenient than the model's tokenizer).",
                template.display_name()
            ),
            Some(template),
        ));
    }

    // No literal `</c>` — that three-character separator is structural and is
    // only assembled by the server between detection categories.
    if slot.contains(CATEGORY_SEPARATOR) {
        return Err(build_error(
            &format!(
                "{} {field} contains the literal '</c>'. That is the structural \
                 category separator; the server inserts it between detection \
                 categories. For multiple categories use a Detection request with \
                 a `categories` list — do not embed '</c>' in a slot.",
                template.display_name()
            ),
            Some(template),
        ));
    }

    // Must not end with '.' — the server appends the single trailing period
    // (except scene_text, which has no slot). A slot ending in '.' would
    // produce a double period.
    if slot.ends_with('.') {
        return Err(build_error(
            &format!(
                "{} {field} ends with '.'; the server appends the single \
                 trailing period, so a slot ending in '.' would produce '..'. \
                 Remove the trailing period from the slot value.",
                template.display_name()
            ),
            Some(template),
        ));
    }

    if forbid_comma && slot.contains(',') {
        // For detection categories and for pointing, a comma is
        // overwhelmingly a mis-join: detection categories use the `</c>`
        // separator (server-assembled from the `categories` list), and
        // NVIDIA's pointing eval calls `Point to: <single>.` once per
        // category and merges client-side.
        let advice = match template {
            TemplateKind::Detection =>
                "detection categories are joined by the server from the \
                 `categories` list (no commas, no '</c>' inside a name); if \
                 you really meant one category that contains a comma, that is \
                 off-distribution — pick a comma-free name.",
            TemplateKind::Point =>
                "NVIDIA's pointing eval calls `Point to: <single>.` once per \
                 category and merges client-side; send one Frame per category \
                 and merge on the client. The phrase must be a SINGLE category \
                 or referring expression.",
            _ => "remove the comma.",
        };
        return Err(build_error(
            &format!(
                "{} {field} contains ',': {advice}",
                template.display_name()
            ),
            Some(template),
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Error builder.
// ---------------------------------------------------------------------------

/// Build a [`SlotError`] whose `message()` is an information-generous English
/// diagnostic. Always appends the canonical reference URL so the client knows
/// where to look.
fn build_error(detail: &str, template: Option<TemplateKind>) -> SlotError {
    let mut buf = String::with_capacity(512 + detail.len());
    buf.push_str(detail);
    if let Some(k) = template {
        buf.push_str(". Template: ");
        buf.push_str(k.display_name());
        buf.push_str(" — `");
        buf.push_str(k.canonical_form());
        buf.push('`');
    }
    buf.push_str(". The seven allowed templates are defined verbatim in ");
    buf.push_str(CANONICAL_REFERENCE_URL);
    buf.push_str(" — that file is the single source of truth.");
    SlotError { message: buf }
}

// ---------------------------------------------------------------------------
// Unit tests — run with `cargo test -p la_server prompt_validator`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::*;

    fn det(cats: &[&str]) -> PromptRequest {
        PromptRequest::Detection(DetectionReq {
            categories: cats.iter().map(|s| s.to_string()).collect(),
        })
    }
    fn phrase(v: PromptRequest) -> PromptRequest { v }

    fn ok(req: &PromptRequest) -> (String, &'static str) {
        compile(req).unwrap_or_else(|e| panic!("expected OK, got: {}", e.message()))
    }
    fn err(req: &PromptRequest) -> String {
        compile(req).err().unwrap_or_else(|| panic!("expected ERR")).message().to_string()
    }

    // ---- Canonical happy path: byte-exact compiled strings -------------
    #[test]
    fn detection_single_cat() {
        let (p, t) = ok(&det(&["drone"]));
        assert_eq!(p, "Locate all the instances that matches the following description: drone.");
        assert_eq!(t, "detection");
    }
    #[test]
    fn detection_multi_cat() {
        let (p, t) = ok(&det(&["bottle", "cup", "laptop"]));
        assert_eq!(p, "Locate all the instances that matches the following description: bottle</c>cup</c>laptop.");
        assert_eq!(t, "detection");
    }
    #[test]
    fn phrase_single() {
        let (p, t) = ok(&PromptRequest::PhraseSingle(PhraseReq { phrase: "the red car".into() }));
        assert_eq!(p, "Locate a single instance that matches the following description: the red car.");
        assert_eq!(t, "phrase_single");
    }
    #[test]
    fn phrase_multi_uses_match_not_matches() {
        let (p, t) = ok(&PromptRequest::PhraseMulti(PhraseReq { phrase: "people wearing hats".into() }));
        assert_eq!(p, "Locate all the instances that match the following description: people wearing hats.");
        assert_eq!(t, "phrase_multi");
    }
    #[test]
    fn text_grounding() {
        let (p, t) = ok(&PromptRequest::TextGrounding(TextReq { text: "STOP".into() }));
        assert_eq!(p, "Please locate the text referred as STOP.");
        assert_eq!(t, "text_grounding");
    }
    #[test]
    fn scene_text_literal_no_period_appended() {
        let (p, t) = ok(&PromptRequest::SceneText(SceneTextReq {}));
        assert_eq!(p, "Detect all the text in box format.");
        assert_eq!(t, "scene_text");
    }
    #[test]
    fn gui_box() {
        let (p, t) = ok(&PromptRequest::GuiBox(DescReq { description: "the search button".into() }));
        assert_eq!(p, "Locate the region that matches the following description: the search button.");
        assert_eq!(t, "gui_box");
    }
    #[test]
    fn point() {
        let (p, t) = ok(&PromptRequest::Point(PhraseReq { phrase: "drone in the sky".into() }));
        assert_eq!(p, "Point to: drone in the sky.");
        assert_eq!(t, "point");
    }

    // ---- Slot rejects ---------------------------------------------------
    #[test]
    fn rejects_empty_categories() {
        let e = err(&det(&[]));
        assert!(e.contains("at least one category"), "got: {}", e);
    }
    #[test]
    fn rejects_too_many_categories() {
        let cats: Vec<&str> = (0..11).map(|_| "x").collect();
        let e = err(&det(&cats));
        assert!(e.contains("maximum is 10"), "got: {}", e);
    }
    #[test]
    fn allows_exactly_ten_categories() {
        let cats: Vec<String> = (0..10).map(|i| format!("c{i}")).collect();
        let req = PromptRequest::Detection(DetectionReq { categories: cats });
        assert!(compile(&req).is_ok());
    }
    #[test]
    fn rejects_empty_category_entry() {
        let e = err(&det(&["dog", "", "cat"]));
        assert!(e.contains("position 2") && e.contains("empty"), "got: {}", e);
    }
    #[test]
    fn rejects_comma_in_category() {
        let e = err(&det(&["dog, cat"]));
        assert!(e.contains("','"), "got: {}", e);
    }
    #[test]
    fn rejects_separator_in_category() {
        let e = err(&det(&["dog</c>cat"]));
        assert!(e.contains("'</c>'"), "got: {}", e);
    }
    #[test]
    fn rejects_leading_whitespace_in_phrase() {
        let e = err(&phrase(PromptRequest::Point(PhraseReq { phrase: " drone".into() })));
        assert!(e.contains("leading"), "got: {}", e);
    }
    #[test]
    fn rejects_trailing_whitespace_in_phrase() {
        let e = err(&phrase(PromptRequest::Point(PhraseReq { phrase: "drone ".into() })));
        assert!(e.contains("trailing"), "got: {}", e);
    }
    #[test]
    fn rejects_trailing_period_in_phrase() {
        let e = err(&PromptRequest::PhraseSingle(PhraseReq { phrase: "the red car.".into() }));
        assert!(e.to_lowercase().contains("ends with '.'") || e.contains("'..'"), "got: {}", e);
    }
    #[test]
    fn rejects_separator_in_phrase() {
        let e = err(&PromptRequest::PhraseSingle(PhraseReq { phrase: "dog</c>cat".into() }));
        assert!(e.contains("'</c>'"), "got: {}", e);
    }
    #[test]
    fn rejects_comma_in_point() {
        let e = err(&PromptRequest::Point(PhraseReq { phrase: "dog, cat".into() }));
        assert!(e.contains("','") && e.contains("pointing"), "got: {}", e);
    }
    #[test]
    fn allows_comma_in_phrase_multi() {
        // Free-form phrases may contain commas (only detection/point forbid).
        let req = PromptRequest::PhraseMulti(PhraseReq { phrase: "a man, woman and child".into() });
        let (p, _) = ok(&req);
        assert_eq!(p, "Locate all the instances that match the following description: a man, woman and child.");
    }
    #[test]
    fn rejects_control_char_in_slot() {
        let e = err(&PromptRequest::PhraseSingle(PhraseReq { phrase: "red\u{0007}car".into() }));
        assert!(e.contains("control character"), "got: {}", e);
    }
    #[test]
    fn rejects_interior_newline() {
        let e = err(&PromptRequest::PhraseSingle(PhraseReq { phrase: "red\ncar".into() }));
        assert!(e.contains("control character"), "got: {}", e);
    }
    #[test]
    fn rejects_non_nfc_slot() {
        // "é" as e + U+0301 (combining acute) is NFD; NFC is U+00E9.
        let e = err(&PromptRequest::PhraseSingle(PhraseReq { phrase: "cafe\u{0301}".into() }));
        assert!(e.contains("NFC"), "got: {}", e);
    }
    #[test]
    fn accepts_nfc_slot() {
        // Precomposed "é" (U+00E9) is already NFC.
        let req = PromptRequest::PhraseSingle(PhraseReq { phrase: "caf\u{00E9}".into() });
        assert!(compile(&req).is_ok());
    }
    #[test]
    fn rejects_overlong_slot() {
        let long = "x".repeat(201);
        let e = err(&PromptRequest::PhraseSingle(PhraseReq { phrase: long }));
        assert!(e.contains("exceeds the 200-char"), "got: {}", e);
    }
    #[test]
    fn allows_exactly_200_char_slot() {
        let exact = "x".repeat(200);
        let req = PromptRequest::PhraseSingle(PhraseReq { phrase: exact });
        assert!(compile(&req).is_ok());
    }

    // ---- URL is always present ------------------------------------------
    #[test]
    fn error_always_contains_url() {
        let e = err(&det(&[]));
        assert!(e.contains(CANONICAL_REFERENCE_URL), "got: {}", e);
    }

    // ---- Drift-chain invariants (KEEP byte-exact) -----------------------
    #[test]
    fn wire_names_match_canonical_order() {
        let j = canonical_templates_json();
        let names = j.get("template_wire_names").unwrap().as_array().unwrap();
        let got: Vec<&str> = names.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(got, ["detection", "phrase_single", "phrase_multi",
                         "text_grounding", "scene_text", "gui_box", "point"]);
    }
    #[test]
    fn canonical_json_has_all_template_constants() {
        let j = canonical_templates_json();
        assert_eq!(j.get("category_separator").unwrap(), "</c>");
        assert_eq!(j.get("scene_text_exact").unwrap(), SCENE_TEXT_EXACT);
        assert_eq!(j.get("point_prefix").unwrap(), POINT_PREFIX);
    }
}
