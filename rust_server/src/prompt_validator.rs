//! Strict, regex-free validator for the seven canonical LocateAnything-3B
//! prompt templates.
//!
//! Single source of truth = `worker/prompts.py`. The constants in this file
//! mirror that Python module byte-for-byte; drift is caught at container
//! boot by `worker/validate_startup.py::validate_prompt_template_drift`,
//! which runs `la_server --print-canonical-templates` and dict-equals the
//! JSON against `prompts.CANONICAL_TEMPLATES`.
//!
//! Provenance for every literal below is in `worker/prompts.py`'s module
//! docstring (Eagle commit e21b6ac, file:line citations into NVIDIA's
//! `locateanything_worker.py` / `evaluation/inference_*.py` /
//! `document/DATA_PREPARATION.md`).
//!
//! Why no regex: per-frame validation must produce information-generous
//! error messages that name exactly what went wrong, not just "did not
//! match". Regex hides the failure structure behind a boolean. If/else
//! flow lets each branch attach its own diagnostic — the client always
//! learns whether they hit the wrong prefix, missed the trailing period,
//! used a comma instead of `</c>`, sent an empty slot, etc.

use crate::protocol::MAX_PROMPT_CHARS;

// ---------------------------------------------------------------------------
// Single-source-of-truth constants. Mirror of worker/prompts.py.
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

/// JSON dump used by `la_server --print-canonical-templates` for the
/// boot-time drift check against worker/prompts.py.
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
    })
}

// ---------------------------------------------------------------------------
// Public result types.
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
}

/// Outcome of validation. The `message()` method returns the full,
/// information-generous English diagnostic — what specifically failed,
/// which canonical template was the closest match (if any), any English
/// heuristic suggestions, and the canonical-templates URL.
#[derive(Debug, Clone)]
pub struct PromptValidationError {
    message: String,
}

impl PromptValidationError {
    pub fn message(&self) -> &str {
        &self.message
    }
}

// ---------------------------------------------------------------------------
// Validator entry point.
// ---------------------------------------------------------------------------

pub fn validate(prompt: &str) -> Result<TemplateKind, PromptValidationError> {
    // -- 1. Empty ----------------------------------------------------------
    if prompt.is_empty() {
        return Err(build_error(
            "prompt is empty; a non-empty prompt is required",
            None,
            &[],
        ));
    }

    // -- 2. Length cap (char-count, not byte-count, so multi-byte UTF-8
    //       can't sneak past a generous ASCII budget) ---------------------
    let char_count = prompt.chars().count();
    if char_count > MAX_PROMPT_CHARS {
        return Err(build_error(
            &format!(
                "prompt length {} chars exceeds MAX_PROMPT_CHARS={} \
                 (the model's tokenizer.model_max_length is 16384 tokens; \
                 even pure-ASCII this character cap is generous)",
                char_count, MAX_PROMPT_CHARS
            ),
            None,
            &[],
        ));
    }

    // -- 3. Strict whitespace check: prompts must have no leading or
    //       trailing whitespace. NVIDIA's training samples have none. ------
    if prompt != prompt.trim() {
        let trimmed = prompt.trim();
        let detail = if trimmed.is_empty() {
            "prompt is whitespace-only".to_string()
        } else {
            let leading  = prompt.len() - prompt.trim_start().len();
            let trailing = prompt.len() - prompt.trim_end().len();
            format!(
                "prompt has surrounding whitespace ({} leading byte(s), \
                 {} trailing byte(s)); canonical templates have no \
                 surrounding whitespace. Trim before sending.",
                leading, trailing
            )
        };
        return Err(build_error(
            &detail,
            best_template_guess(trimmed),
            &english_heuristics(prompt),
        ));
    }

    // -- 4. Scene-text exact form (no slot, single-line template) ---------
    if prompt == SCENE_TEXT_EXACT {
        return Ok(TemplateKind::SceneText);
    }

    // -- 5. Trailing period --------------------------------------------------
    // Every template ends in '.'. Strip it before attempting prefix match
    // so the slot validator sees only the bracketed content.
    let body = match prompt.strip_suffix('.') {
        Some(s) => s,
        None => {
            let last = prompt.chars().last().unwrap_or('\0');
            return Err(build_error(
                &format!(
                    "prompt does not end with a period ('.'); every \
                     canonical template ends in a trailing '.'. Last \
                     character was U+{:04X} {:?}.",
                    last as u32, last,
                ),
                best_template_guess(prompt),
                &english_heuristics(prompt),
            ));
        }
    };

    // -- 6. Try each prefix in turn ---------------------------------------
    // The seven prefixes are byte-distinct after the position where they
    // diverge (DETECTION uses "matches " while MULTI uses "match "), so
    // any single prompt matches at most one prefix.
    if let Some(rest) = body.strip_prefix(DETECTION_PREFIX) {
        validate_detection_slot(rest)?;
        return Ok(TemplateKind::Detection);
    }
    if let Some(rest) = body.strip_prefix(SINGLE_PHRASE_PREFIX) {
        validate_phrase_slot(rest, TemplateKind::PhraseSingle)?;
        return Ok(TemplateKind::PhraseSingle);
    }
    if let Some(rest) = body.strip_prefix(MULTI_PHRASE_PREFIX) {
        validate_phrase_slot(rest, TemplateKind::PhraseMulti)?;
        return Ok(TemplateKind::PhraseMulti);
    }
    if let Some(rest) = body.strip_prefix(TEXT_GROUNDING_PREFIX) {
        validate_phrase_slot(rest, TemplateKind::TextGrounding)?;
        return Ok(TemplateKind::TextGrounding);
    }
    if let Some(rest) = body.strip_prefix(GUI_BOX_PREFIX) {
        validate_phrase_slot(rest, TemplateKind::GuiBox)?;
        return Ok(TemplateKind::GuiBox);
    }
    if let Some(rest) = body.strip_prefix(POINT_PREFIX) {
        validate_point_slot(rest)?;
        return Ok(TemplateKind::Point);
    }

    // -- 7. No prefix matched — diagnose what the client probably meant -
    Err(build_error(
        "prompt does not match any of the seven canonical LocateAnything-3B \
         prompt templates",
        best_template_guess(prompt),
        &english_heuristics(prompt),
    ))
}

// ---------------------------------------------------------------------------
// Slot validators.
// ---------------------------------------------------------------------------

/// Validate the categories slot of the closed-class detection template:
/// `[CATS]` is one-or-more category strings joined by the literal `</c>`
/// separator (no whitespace around the separator). Each category itself
/// must be non-empty and have no leading/trailing whitespace.
fn validate_detection_slot(slot: &str) -> Result<(), PromptValidationError> {
    if slot.is_empty() {
        return Err(build_error(
            "closed-class detection template requires at least one category \
             after the colon (the part before the trailing '.' is empty)",
            Some(TemplateKind::Detection),
            &[],
        ));
    }

    let parts: Vec<&str> = slot.split(CATEGORY_SEPARATOR).collect();
    for (i, cat) in parts.iter().enumerate() {
        if cat.is_empty() {
            // Empty category — either two adjacent `</c>` or leading/trailing.
            return Err(build_error(
                &format!(
                    "closed-class detection category list contains an empty \
                     category at position {} (split index {}); empty \
                     categories between '</c>' separators are not allowed",
                    i + 1, i
                ),
                Some(TemplateKind::Detection),
                &[],
            ));
        }
        if *cat != cat.trim() {
            return Err(build_error(
                &format!(
                    "closed-class detection category at position {} has \
                     surrounding whitespace: {:?}. Categories are joined \
                     with the literal '</c>' (no spaces around it).",
                    i + 1, cat
                ),
                Some(TemplateKind::Detection),
                &[],
            ));
        }
        if cat.contains(',') {
            // Heuristic: a comma inside a single category is overwhelmingly
            // the client meaning to join categories with ',' rather than
            // with the trained `</c>` separator. NVIDIA's training prompts
            // use `</c>` between categories with no whitespace; commas
            // inside a category name are vanishingly rare in the training
            // distribution and are almost always a mis-join.
            return Err(build_error(
                &format!(
                    "closed-class detection category at position {} \
                     contains ',': {:?}. The trained category separator \
                     is the literal three-character string '</c>' (no \
                     whitespace around it), not ',' — e.g. \
                     'cat</c>dog</c>bottle.'. If you really did mean a \
                     single category that contains a comma, that combination \
                     is off-distribution; pick one of the comma-free names.",
                    i + 1, cat
                ),
                Some(TemplateKind::Detection),
                &[],
            ));
        }
    }
    Ok(())
}

/// Validate a free-form phrase slot: non-empty, no leading/trailing
/// whitespace, no stray `</c>` (the category separator is only legal in
/// the closed-class detection template).
fn validate_phrase_slot(
    slot: &str,
    template: TemplateKind,
) -> Result<(), PromptValidationError> {
    if slot.is_empty() {
        return Err(build_error(
            &format!(
                "{} template requires a non-empty phrase after the colon \
                 (the part before the trailing '.' is empty)",
                template.display_name()
            ),
            Some(template),
            &[],
        ));
    }
    if slot != slot.trim() {
        return Err(build_error(
            &format!(
                "{} template phrase has surrounding whitespace: {:?}. The \
                 phrase comes immediately after the canonical prefix and \
                 ends immediately before the trailing '.', with no padding.",
                template.display_name(), slot
            ),
            Some(template),
            &[],
        ));
    }
    if slot.contains(CATEGORY_SEPARATOR) {
        return Err(build_error(
            &format!(
                "{} template phrase contains '</c>' — that separator is \
                 only legal inside the closed-class detection template's \
                 category list. For multiple categories use the detection \
                 template (`{}`); for multi-instance phrase grounding use \
                 a single free-form phrase.",
                template.display_name(),
                TemplateKind::Detection.canonical_form(),
            ),
            Some(template),
            &[],
        ));
    }
    Ok(())
}

/// Validate the pointing slot. Same rules as phrase, plus: no comma —
/// NVIDIA's evaluation (inference_grounding_ddp.py:297-312) makes one
/// `Point to:` call per category and merges client-side; the single-prompt
/// comma-joined form is off-distribution for pointing.
fn validate_point_slot(slot: &str) -> Result<(), PromptValidationError> {
    validate_phrase_slot(slot, TemplateKind::Point)?;
    if slot.contains(',') {
        return Err(build_error(
            &format!(
                "pointing template phrase contains ',' — NVIDIA's pointing \
                 evaluation calls `Point to: <single_category>.` once per \
                 category and merges results client-side (per \
                 inference_grounding_ddp.py:297-312). Use one Frame per \
                 category and merge on the client; the phrase here must \
                 be a SINGLE category or referring expression. Got phrase: \
                 {:?}.",
                slot
            ),
            Some(TemplateKind::Point),
            &[],
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Closest-template guess and English heuristics.
// ---------------------------------------------------------------------------

/// Score the prompt against each canonical template by length of common
/// prefix; return the highest-scoring template (or None if every template
/// shares fewer than 4 characters with the input — too little to suggest).
/// Cheap O(7·N) scan; runs only on the failure path.
fn best_template_guess(prompt: &str) -> Option<TemplateKind> {
    let candidates: &[(TemplateKind, &str)] = &[
        (TemplateKind::Detection,     DETECTION_PREFIX),
        (TemplateKind::PhraseSingle,  SINGLE_PHRASE_PREFIX),
        (TemplateKind::PhraseMulti,   MULTI_PHRASE_PREFIX),
        (TemplateKind::TextGrounding, TEXT_GROUNDING_PREFIX),
        (TemplateKind::SceneText,     SCENE_TEXT_EXACT),
        (TemplateKind::GuiBox,        GUI_BOX_PREFIX),
        (TemplateKind::Point,         POINT_PREFIX),
    ];
    let mut best: Option<(TemplateKind, usize)> = None;
    for &(kind, canonical) in candidates {
        // Common prefix length in bytes (ASCII templates → bytes == chars).
        let common = prompt.bytes()
            .zip(canonical.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        if common < 4 { continue; }
        if best.map_or(true, |(_, len)| common > len) {
            best = Some((kind, common));
        }
    }
    best.map(|(k, _)| k)
}

/// Heuristic English-language hints. We surface common mistakes — wrong
/// verb, wrong preposition, comma instead of `</c>`, missing capital,
/// "Point to: x</c>y." etc — as plain-English suggestions appended to the
/// diagnostic. None of these are required; they are advisory only.
fn english_heuristics(prompt: &str) -> Vec<String> {
    let mut hints = Vec::new();
    let trimmed = prompt.trim();

    if trimmed.is_empty() {
        return hints;
    }

    // ---- Capitalization ---------------------------------------------------
    if let Some(first) = trimmed.chars().next() {
        if first.is_ascii_lowercase() {
            hints.push(format!(
                "first character is lowercase ({:?}); canonical templates \
                 begin with a capital ('Locate', 'Point', 'Please', 'Detect')",
                first
            ));
        }
    }

    // ---- Wrong opening verb ----------------------------------------------
    // Heuristic order: longest match first so we don't fire "Find" for
    // "Find me the". Keep the list short — only common mistakes.
    static OPENING_HINTS: &[(&str, &str)] = &[
        ("Find ",      "did you mean 'Locate' (closed-class detection / phrase grounding) or 'Point to:' (pointing)?"),
        ("find ",      "did you mean 'Locate' (closed-class detection / phrase grounding) or 'Point to:' (pointing)?"),
        ("Where ",     "the model does not support free-form questions; use 'Locate ...' or 'Point to: ...'"),
        ("where ",     "the model does not support free-form questions; use 'Locate ...' or 'Point to: ...'"),
        ("Show ",      "use 'Locate ...' or 'Point to: ...' — there is no 'Show' template"),
        ("Identify ",  "use 'Locate all the instances that matches the following description: <cats>.' (closed-class detection)"),
        ("Detect ",    "the only 'Detect' template is exactly 'Detect all the text in box format.' — no slot, no other variant"),
        ("Point at ",  "did you mean 'Point to:' (canonical pointing prefix)?"),
        ("Point at:",  "did you mean 'Point to:' (canonical pointing prefix)?"),
        ("Pointing ",  "did you mean 'Point to: <phrase>.' (canonical pointing template)?"),
        ("Please find ",   "did you mean 'Please locate the text referred as <phrase>.' (text grounding)?"),
        ("Please locate ", "the only 'Please locate' template is text grounding: 'Please locate the text referred as <phrase>.'"),
    ];
    for &(needle, hint) in OPENING_HINTS {
        if trimmed.starts_with(needle) {
            hints.push(format!("opening '{}' is not a canonical template prefix — {}", needle.trim_end(), hint));
            break;
        }
    }

    // ---- match/matches asymmetry -----------------------------------------
    // Pattern "Locate all the instances that matches" is detection (with
    // `</c>`-joined categories). Pattern "Locate all the instances that
    // match" is multi-instance phrase grounding (with a single free-form
    // phrase). Surface confusions.
    let has_matches_prefix = trimmed.starts_with("Locate all the instances that matches the following description:");
    let has_match_prefix   = trimmed.starts_with("Locate all the instances that match the following description:")
        && !has_matches_prefix;
    if has_match_prefix && trimmed.contains(CATEGORY_SEPARATOR) {
        hints.push(
            "your prompt uses 'match' (no 's') AND contains the '</c>' \
             separator — those don't go together. 'match' is multi-instance \
             phrase grounding which takes a single free-form phrase. For a \
             '</c>'-joined category list use 'matches' (closed-class \
             detection).".into()
        );
    }
    if has_matches_prefix && !trimmed.contains(CATEGORY_SEPARATOR) && trimmed.contains(',') {
        hints.push(
            "your prompt uses 'matches' (the closed-class detection \
             template) but the category list is comma-separated; the \
             trained separator is the literal three characters '</c>' \
             with no whitespace around it (e.g. 'cat</c>dog</c>bottle').".into()
        );
    }

    // ---- Comma in non-detection templates --------------------------------
    if trimmed.starts_with("Point to:") && trimmed.contains(',') {
        hints.push(
            "'Point to:' is a SINGLE-category template — NVIDIA's eval \
             calls it once per category and merges client-side. Loop on \
             the client instead of comma-joining.".into()
        );
    }

    // ---- Missing colon after 'Point to' ----------------------------------
    if (trimmed.starts_with("Point to ") || trimmed.starts_with("point to ")) && !trimmed.starts_with("Point to: ") {
        hints.push("the pointing prefix is 'Point to:' (with a colon AND a space — exactly 'Point to: ')".into());
    }

    // ---- Wrong sentence punctuation --------------------------------------
    if trimmed.ends_with('?') {
        hints.push("prompt ends with '?'; canonical templates end with '.'".into());
    } else if trimmed.ends_with('!') {
        hints.push("prompt ends with '!'; canonical templates end with '.'".into());
    }

    // ---- Multi-sentence / chain-of-thought -------------------------------
    // Count internal periods (excluding the trailing one if any).
    let internal_periods = {
        let stripped = trimmed.strip_suffix('.').unwrap_or(trimmed);
        stripped.matches('.').count()
    };
    if internal_periods >= 1 {
        hints.push(format!(
            "prompt contains {} internal period(s); canonical templates \
             are single sentences with exactly one trailing '.'",
            internal_periods
        ));
    }

    hints
}

// ---------------------------------------------------------------------------
// Error builder.
// ---------------------------------------------------------------------------

/// Build a `PromptValidationError` whose `message()` is an
/// information-generous English diagnostic. Always appends the canonical
/// reference URL so the client knows where to look.
fn build_error(
    detail: &str,
    closest: Option<TemplateKind>,
    hints: &[String],
) -> PromptValidationError {
    let mut buf = String::with_capacity(512 + detail.len());
    buf.push_str(detail);
    if let Some(k) = closest {
        buf.push_str(". Closest canonical template: ");
        buf.push_str(k.display_name());
        buf.push_str(" — `");
        buf.push_str(k.canonical_form());
        buf.push('`');
    }
    if !hints.is_empty() {
        buf.push_str(". Hints: ");
        for (i, h) in hints.iter().enumerate() {
            if i > 0 { buf.push_str("; "); }
            buf.push_str(h);
        }
    }
    buf.push_str(". The seven allowed templates are defined verbatim in ");
    buf.push_str(CANONICAL_REFERENCE_URL);
    buf.push_str(" — that file is the single source of truth.");
    PromptValidationError { message: buf }
}

// ---------------------------------------------------------------------------
// Unit tests — run with `cargo test -p la_server prompt_validator`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(p: &str) -> TemplateKind {
        validate(p).unwrap_or_else(|e| panic!("expected OK for {:?}, got: {}", p, e.message()))
    }
    fn err(p: &str) -> String {
        validate(p).err().unwrap_or_else(|| panic!("expected ERR for {:?}", p)).message().to_string()
    }

    // ---- Canonical happy path -------------------------------------------
    #[test]
    fn detection_single_cat() {
        assert_eq!(ok("Locate all the instances that matches the following description: drone."), TemplateKind::Detection);
    }
    #[test]
    fn detection_multi_cat() {
        assert_eq!(
            ok("Locate all the instances that matches the following description: bottle</c>cup</c>laptop."),
            TemplateKind::Detection
        );
    }
    #[test]
    fn phrase_single() {
        assert_eq!(ok("Locate a single instance that matches the following description: the red car."), TemplateKind::PhraseSingle);
    }
    #[test]
    fn phrase_multi() {
        assert_eq!(ok("Locate all the instances that match the following description: people wearing hats."), TemplateKind::PhraseMulti);
    }
    #[test]
    fn text_grounding() {
        assert_eq!(ok("Please locate the text referred as STOP."), TemplateKind::TextGrounding);
    }
    #[test]
    fn scene_text() {
        assert_eq!(ok("Detect all the text in box format."), TemplateKind::SceneText);
    }
    #[test]
    fn gui_box() {
        assert_eq!(ok("Locate the region that matches the following description: the search button."), TemplateKind::GuiBox);
    }
    #[test]
    fn point() {
        assert_eq!(ok("Point to: drone in the sky."), TemplateKind::Point);
    }

    // ---- match / matches asymmetry --------------------------------------
    #[test]
    fn match_multi_with_cls_separator_is_rejected() {
        let e = err("Locate all the instances that match the following description: dog</c>cat.");
        assert!(e.contains("'</c>'"), "expected hint about separator misuse, got: {}", e);
    }
    #[test]
    fn matches_detection_with_comma_warns() {
        let e = err("Locate all the instances that matches the following description: dog, cat.");
        assert!(e.contains("'</c>'"), "expected a `</c>` hint, got: {}", e);
    }

    // ---- Structural rejects ---------------------------------------------
    #[test]
    fn rejects_empty() {
        assert!(validate("").is_err());
    }
    #[test]
    fn rejects_leading_whitespace() {
        let e = err(" Point to: drone.");
        assert!(e.contains("leading"), "got: {}", e);
    }
    #[test]
    fn rejects_trailing_whitespace() {
        let e = err("Point to: drone. ");
        assert!(e.contains("trailing"), "got: {}", e);
    }
    #[test]
    fn rejects_missing_period() {
        let e = err("Point to: drone");
        assert!(e.to_lowercase().contains("period"), "got: {}", e);
    }
    #[test]
    fn rejects_empty_category() {
        let e = err("Locate all the instances that matches the following description: dog</c></c>cat.");
        assert!(e.contains("empty category"), "got: {}", e);
    }
    #[test]
    fn rejects_whitespace_in_category() {
        let e = err("Locate all the instances that matches the following description: dog </c>cat.");
        assert!(e.contains("whitespace"), "got: {}", e);
    }
    #[test]
    fn rejects_stray_separator_in_phrase() {
        let e = err("Locate a single instance that matches the following description: dog</c>cat.");
        assert!(e.contains("'</c>'"), "got: {}", e);
    }
    #[test]
    fn rejects_comma_in_point() {
        let e = err("Point to: dog, cat.");
        assert!(e.contains("Point to:"), "got: {}", e);
    }
    #[test]
    fn rejects_question_mark_end() {
        let e = err("Point to: drone?");
        assert!(e.contains("period"), "got: {}", e);
    }

    // ---- Heuristic hints ------------------------------------------------
    #[test]
    fn heuristic_find_suggestion() {
        let e = err("Find the dog.");
        assert!(e.contains("Locate") || e.contains("Point to"), "got: {}", e);
    }
    #[test]
    fn heuristic_point_at() {
        let e = err("Point at: drone.");
        assert!(e.contains("Point to"), "got: {}", e);
    }
    #[test]
    fn heuristic_lowercase_first() {
        let e = err("locate all the instances that matches the following description: drone.");
        assert!(e.contains("lowercase") || e.contains("capital"), "got: {}", e);
    }
    #[test]
    fn heuristic_missing_colon_in_point() {
        let e = err("Point to drone.");
        assert!(e.contains("Point to:"), "got: {}", e);
    }

    // ---- URL is always present ------------------------------------------
    #[test]
    fn error_always_contains_url() {
        let e = err("");
        assert!(e.contains(CANONICAL_REFERENCE_URL), "got: {}", e);
    }
}
