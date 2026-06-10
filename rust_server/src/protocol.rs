//! Wire protocol — both the Rust↔Python IPC and the WebSocket external API.
//!
//! ## External (client ↔ Rust server) — WebSocket /v1/stream
//!
//! Stop-and-wait per connection: client opens WS, immediately sends a Frame,
//! reads exactly one Response, optionally sends the next Frame, and so on.
//! There is no handshake — the WS opens "hot." Capabilities are fetched out
//! of band via `GET /v1/capabilities`.
//!
//! Every WebSocket binary message is a Frame:
//!   `[ 4 bytes BE u32 header_len ] [ header_len bytes UTF-8 JSON ] [ JPEG bytes ]`
//!
//! The header is a typed [`InferHeader`] carrying a typed [`PromptRequest`]
//! (a sum type over the seven trained tasks) and a typed [`GenerationMode`].
//! Illegal states are unrepresentable: serde rejects any unknown field, any
//! non-snake_case task tag, and any generation mode that is not exactly one
//! of `fast`/`hybrid`/`slow` — at parse time, before any inference is run.
//!
//! The reply is exactly one typed [`Response`] — a flat tagged union on
//! `type` (`boxes` XOR `points` XOR `abstained` XOR `error`). The Rust edge
//! DESERIALIZES the worker's reply into this type, so a drifted Python field
//! (a stale `off_shape_count`, a missing `latency_ms`, a duplicate key) fails
//! loud instead of being forwarded to the client.
//!
//! ## Internal (Rust ↔ Python) — Unix domain socket /tmp/la.sock
//!
//! Length-prefixed frames using LengthDelimitedCodec (4-byte BE u32).
//! Each request is TWO consecutive frames sent by the Rust side:
//!   1) Header JSON. For frames the Rust side builds `{kind:"frame",
//!      frame_id, prompt, prompt_task, generation_mode, jpeg_len}` — note the
//!      worker receives the COMPILED prompt string + the `prompt_task`
//!      wire-name, NOT the typed `PromptRequest` (the typed request is the
//!      external client surface only). Control queries
//!      (`{"kind":"capabilities"|"info"}`) carry the kind only.
//!   2) JPEG bytes (zero-length for capability/info queries).
//!
//! The Python worker responds with ONE frame: a JSON body carrying a
//! top-level `ok:bool` discriminator (internal to the UDS hop). `ok:true`
//! → the body deserializes into a success [`Response`] variant; `ok:false`
//! → an error [`Response::Error`]. The `ok` field never reaches the client.

use serde::{Deserialize, Serialize};

// ===========================================================================
// REQUEST — JSON header of the binary Frame (A.1)
// ===========================================================================

/// The typed JSON header the client sends with every WebSocket Frame.
///
/// **Every field is required.** Missing-or-malformed → WS Close 1008.
/// `deny_unknown_fields` here rejects any stray top-level key at parse time.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct InferHeader {
    /// Client-assigned correlation id. Echoed unchanged in the response.
    /// 1..=256 chars.
    pub frame_id: String,

    /// The typed inference request — a sum over the seven trained tasks. The
    /// server compiles this to the exact trained prompt string via
    /// `prompt_validator::compile`; the template constants in that module
    /// remain the single source of truth and stay boot-drift-checked.
    pub request: PromptRequest,

    /// Exactly one of `fast`, `hybrid`, `slow`. No server default — every
    /// Frame commits to a mode. Serde rejects any other string at parse time.
    pub generation_mode: GenerationMode,

    /// Declared JPEG payload size in bytes. Must equal the actual JPEG bytes
    /// that follow the header in the binary frame.
    pub jpeg_len: usize,
}

/// The three trained generation modes. snake_case on the wire.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenerationMode {
    Fast,
    Hybrid,
    Slow,
}

/// The typed inference request: internally tagged on `task`, one variant per
/// trained prompt template.
///
/// Newtype-around-named-struct pattern: container-level `deny_unknown_fields`
/// is NOT expressible on enum variants, so each variant wraps a struct that
/// carries its own `deny_unknown_fields`. `SceneText` wraps an EMPTY struct
/// (NOT a unit variant) — under internal tagging a unit variant silently
/// ignores extra sibling fields, while the empty-struct newtype rejects them.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum PromptRequest {
    Detection(DetectionReq),
    PhraseSingle(PhraseReq),
    PhraseMulti(PhraseReq),
    TextGrounding(TextReq),
    SceneText(SceneTextReq),
    GuiBox(DescReq),
    Point(PhraseReq),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct DetectionReq {
    pub categories: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct PhraseReq {
    pub phrase: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct TextReq {
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct DescReq {
    pub description: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SceneTextReq {}

// ===========================================================================
// RESPONSE — one Text reply per Frame; flat tagged union on `type` (A.2)
// ===========================================================================

/// The client-facing reply to a Frame. Exactly one of `boxes` / `points` /
/// `abstained` / `error` — mutually exclusive, exhaustive. The Rust edge
/// DESERIALIZES the worker's reply into this type (egress schema-enforced).
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Boxes(BoxesBody),
    Points(PointsBody),
    Abstained(AbstainedBody),
    Error(ErrorBody),
}

// The `deny_unknown_fields` on each of the three success bodies below is the
// LOAD-BEARING egress refusal: it is what makes a drifted/garbled worker reply
// fail loud instead of being forwarded to the client. Verified empirically
// against the pinned serde 1.0.228 (see the `resp_rejects_*` wire_tests): when a
// struct carries BOTH `deny_unknown_fields` AND a `#[serde(flatten)]` field, the
// OUTER attribute rejects any field unknown to both the named outer fields AND
// the flattened `Meta`. The common belief that flatten disables the outer
// `deny_unknown_fields` is FALSE here; what is actually inert is a
// `deny_unknown_fields` on the flattened struct itself (`Meta`). So this
// attribute must NOT be dropped in the belief that `Meta`'s own `deny` covers
// the meta fields — it does not, in flattened position.
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct BoxesBody {
    pub frame_id: String,
    pub boxes: Vec<LabeledBox>,
    #[serde(flatten)]
    pub meta: Meta,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct PointsBody {
    pub frame_id: String,
    pub points: Vec<LabeledPoint>,
    #[serde(flatten)]
    pub meta: Meta,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct AbstainedBody {
    pub frame_id: String,
    #[serde(flatten)]
    pub meta: Meta,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ErrorBody {
    pub frame_id: String,
    pub code: ErrorCode,
    pub message: String,
}

/// A single detected box. `label` is REQUIRED (never null). `bbox_norm` is
/// the post-clamp 0..=1000 normalized box; `bbox_px` MUST stay f32 for
/// byte-exactness against the Python side.
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct LabeledBox {
    pub label: String,
    pub bbox_norm: [u16; 4],
    pub bbox_px: [f32; 4],
}

/// A single pointed location. `label` is REQUIRED. `point_px` MUST stay f32.
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct LabeledPoint {
    pub label: String,
    pub point_norm: [u16; 2],
    pub point_px: [f32; 2],
}

/// Per-reply metadata, flattened into the boxes/points/abstained variants.
/// NOTE: `Meta` is only ever deserialized in FLATTENED position, where its own
/// `deny_unknown_fields` is INERT (serde 1.0.228) — the enclosing body's
/// `deny_unknown_fields` is what rejects stray meta fields. The attribute is
/// kept here purely as a defensive backstop in case `Meta` is ever deserialized
/// standalone; it is NOT what guards the flattened replies.
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    pub raw_text: String,
    pub model_output_truncated: bool,
    /// Off-contract items dropped (off-shape geometry / unlabeled). Usually
    /// 0. Non-fatal — co-emitted valid geometry is still returned.
    pub deviations_dropped: u32,
    pub image_size: [u32; 2],
    pub resize_plan: ResizePlan,
    pub generation_mode_used: GenerationMode,
    pub latency_ms: f64,
    pub total_ms: f64,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ResizePlan {
    pub dst_w: u32,
    pub dst_h: u32,
    pub n_llm_tokens: u32,
    pub scale: f64,
}

/// The four client-facing error codes. snake_case on the wire. Replaces the
/// old numeric (400/500/504) per-frame codes.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    InvalidImage,
    ModelDeviation,
    Internal,
}

impl ErrorCode {
    /// The exact wire string for this code. Handy for constructing error
    /// `Response`s without round-tripping through serde.
    pub fn as_wire(self) -> &'static str {
        match self {
            ErrorCode::InvalidRequest => "invalid_request",
            ErrorCode::InvalidImage => "invalid_image",
            ErrorCode::ModelDeviation => "model_deviation",
            ErrorCode::Internal => "internal",
        }
    }
}

// ===========================================================================
// Shared limits.
// ===========================================================================

/// Minimum allowed image dimension (per side, in pixels). Below this
/// the model has zero useful spatial information.
pub const MIN_IMAGE_DIM: u16 = 32;

/// Maximum allowed COMPILED prompt length in CHARACTERS. The actual
/// tokenizer.model_max_length is 16384 *tokens*, which is always ≥ the
/// equivalent character count for ASCII / Western prompts. This is a
/// generous-but-finite character cap on the assembled prompt string.
pub const MAX_PROMPT_CHARS: usize = 16384;

// ===========================================================================
// Wire round-trip tests — every A.1 request variant and every A.2 response
// variant is round-tripped through serde_json against the EXACT spec bytes
// (contract_v2_spec.md §A.1 / §A.2). The Python worker emits these same
// bytes; an external test round-trips both sides.
// ===========================================================================
#[cfg(test)]
mod wire_tests {
    use super::*;

    /// Parse `wire`, assert it matches `check`, re-serialize, and assert the
    /// re-serialized bytes parse back to a Value EQUAL to the original wire
    /// (semantic byte round-trip — robust to key ordering, which serde fixes
    /// by struct declaration order anyway).
    fn round_trip<T>(wire: &str, check: impl FnOnce(&T))
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let parsed: T = serde_json::from_str(wire)
            .unwrap_or_else(|e| panic!("deserialize failed for {wire}: {e}"));
        check(&parsed);
        let reser = serde_json::to_string(&parsed).expect("serialize");
        let a: serde_json::Value = serde_json::from_str(wire).unwrap();
        let b: serde_json::Value = serde_json::from_str(&reser).unwrap();
        assert_eq!(a, b, "round-trip value mismatch.\n  wire:  {wire}\n  reser: {reser}");
    }

    fn rejects<T: serde::de::DeserializeOwned>(wire: &str) {
        assert!(
            serde_json::from_str::<T>(wire).is_err(),
            "expected REJECT but it parsed: {wire}"
        );
    }

    // ---- A.1 REQUEST variants (exact spec bytes) ------------------------
    #[test]
    fn req_detection() {
        let w = r#"{"frame_id":"f-1","request":{"task":"detection","categories":["drone","bird"]},"generation_mode":"slow","jpeg_len":300456}"#;
        round_trip::<InferHeader>(w, |h| {
            assert_eq!(h.frame_id, "f-1");
            assert_eq!(h.generation_mode, GenerationMode::Slow);
            assert_eq!(h.jpeg_len, 300456);
            match &h.request {
                PromptRequest::Detection(d) => assert_eq!(d.categories, ["drone", "bird"]),
                other => panic!("wrong variant: {other:?}"),
            }
        });
    }
    #[test]
    fn req_point() {
        let w = r#"{"frame_id":"f-2","request":{"task":"point","phrase":"drone in the sky"},"generation_mode":"slow","jpeg_len":300456}"#;
        round_trip::<InferHeader>(w, |h| match &h.request {
            PromptRequest::Point(p) => assert_eq!(p.phrase, "drone in the sky"),
            other => panic!("wrong variant: {other:?}"),
        });
    }
    #[test]
    fn req_scene_text() {
        let w = r#"{"frame_id":"f-3","request":{"task":"scene_text"},"generation_mode":"hybrid","jpeg_len":300456}"#;
        round_trip::<InferHeader>(w, |h| {
            assert_eq!(h.generation_mode, GenerationMode::Hybrid);
            assert!(matches!(h.request, PromptRequest::SceneText(_)));
        });
    }
    #[test]
    fn req_phrase_single_multi_text_gui() {
        round_trip::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"phrase_single","phrase":"the red car"},"generation_mode":"fast","jpeg_len":1}"#,
            |h| assert!(matches!(h.request, PromptRequest::PhraseSingle(_))),
        );
        round_trip::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"phrase_multi","phrase":"people"},"generation_mode":"fast","jpeg_len":1}"#,
            |h| assert!(matches!(h.request, PromptRequest::PhraseMulti(_))),
        );
        round_trip::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"text_grounding","text":"STOP"},"generation_mode":"fast","jpeg_len":1}"#,
            |h| assert!(matches!(h.request, PromptRequest::TextGrounding(_))),
        );
        round_trip::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"gui_box","description":"the search button"},"generation_mode":"fast","jpeg_len":1}"#,
            |h| assert!(matches!(h.request, PromptRequest::GuiBox(_))),
        );
    }

    // ---- A.1 REQUEST negative (illegal states unrepresentable) ----------
    #[test]
    fn req_rejects_unknown_top_level_field() {
        rejects::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"scene_text"},"generation_mode":"slow","jpeg_len":1,"extra":true}"#,
        );
    }
    #[test]
    fn req_rejects_unknown_body_field() {
        rejects::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"detection","categories":["x"],"limit":5},"generation_mode":"slow","jpeg_len":1}"#,
        );
    }
    #[test]
    fn req_scene_text_rejects_extra_sibling_field() {
        // The EMPTY-struct newtype (NOT a unit variant) must reject extras.
        rejects::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"scene_text","phrase":"oops"},"generation_mode":"slow","jpeg_len":1}"#,
        );
    }
    #[test]
    fn req_rejects_bad_generation_mode() {
        rejects::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"scene_text"},"generation_mode":"turbo","jpeg_len":1}"#,
        );
    }
    #[test]
    fn req_rejects_unknown_task_tag() {
        rejects::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"segment"},"generation_mode":"slow","jpeg_len":1}"#,
        );
    }
    #[test]
    fn req_rejects_wrong_slot_for_task() {
        // detection needs `categories`, not `phrase`.
        rejects::<InferHeader>(
            r#"{"frame_id":"a","request":{"task":"detection","phrase":"x"},"generation_mode":"slow","jpeg_len":1}"#,
        );
    }

    // ---- A.2 RESPONSE variants (exact spec bytes) -----------------------
    #[test]
    fn resp_boxes() {
        let w = r#"{"type":"boxes","frame_id":"f-1","boxes":[{"label":"drone","bbox_norm":[420,510,560,640],"bbox_px":[806.4,550.8,1075.2,691.2]}],"raw_text":"...","model_output_truncated":false,"deviations_dropped":0,"image_size":[1920,1080],"resize_plan":{"dst_w":1932,"dst_h":1092,"n_llm_tokens":2691,"scale":1.006},"generation_mode_used":"slow","latency_ms":812.4,"total_ms":821.0}"#;
        round_trip::<Response>(w, |r| match r {
            Response::Boxes(b) => {
                assert_eq!(b.frame_id, "f-1");
                assert_eq!(b.boxes.len(), 1);
                assert_eq!(b.boxes[0].label, "drone");
                assert_eq!(b.boxes[0].bbox_norm, [420, 510, 560, 640]);
                assert_eq!(b.boxes[0].bbox_px, [806.4, 550.8, 1075.2, 691.2]);
                assert_eq!(b.meta.deviations_dropped, 0);
                assert_eq!(b.meta.image_size, [1920, 1080]);
                assert_eq!(b.meta.generation_mode_used, GenerationMode::Slow);
                assert_eq!(b.meta.total_ms, 821.0);
            }
            other => panic!("wrong variant: {other:?}"),
        });
    }
    #[test]
    fn resp_points() {
        let w = r#"{"type":"points","frame_id":"f-2","points":[{"label":"drone in the sky","point_norm":[500,300],"point_px":[960.0,324.0]}],"raw_text":"...","model_output_truncated":false,"deviations_dropped":0,"image_size":[1920,1080],"resize_plan":{"dst_w":1932,"dst_h":1092,"n_llm_tokens":2691,"scale":1.006},"generation_mode_used":"slow","latency_ms":1.0,"total_ms":2.0}"#;
        round_trip::<Response>(w, |r| match r {
            Response::Points(p) => {
                assert_eq!(p.points[0].label, "drone in the sky");
                assert_eq!(p.points[0].point_norm, [500, 300]);
                assert_eq!(p.points[0].point_px, [960.0, 324.0]);
            }
            other => panic!("wrong variant: {other:?}"),
        });
    }
    #[test]
    fn resp_abstained() {
        let w = r#"{"type":"abstained","frame_id":"f-3","raw_text":"...","model_output_truncated":false,"deviations_dropped":0,"image_size":[1920,1080],"resize_plan":{"dst_w":1932,"dst_h":1092,"n_llm_tokens":2691,"scale":1.006},"generation_mode_used":"hybrid","latency_ms":1.0,"total_ms":2.0}"#;
        round_trip::<Response>(w, |r| match r {
            Response::Abstained(a) => {
                assert_eq!(a.frame_id, "f-3");
                assert_eq!(a.meta.generation_mode_used, GenerationMode::Hybrid);
            }
            other => panic!("wrong variant: {other:?}"),
        });
    }
    #[test]
    fn resp_error() {
        let w = r#"{"type":"error","frame_id":"f-4","code":"invalid_request","message":"..."}"#;
        round_trip::<Response>(w, |r| match r {
            Response::Error(e) => {
                assert_eq!(e.frame_id, "f-4");
                assert_eq!(e.code, ErrorCode::InvalidRequest);
            }
            other => panic!("wrong variant: {other:?}"),
        });
        // every ErrorCode wire string parses
        for (s, c) in [
            ("invalid_request", ErrorCode::InvalidRequest),
            ("invalid_image", ErrorCode::InvalidImage),
            ("model_deviation", ErrorCode::ModelDeviation),
            ("internal", ErrorCode::Internal),
        ] {
            let w = format!(r#"{{"type":"error","frame_id":"x","code":"{s}","message":"m"}}"#);
            round_trip::<Response>(&w, |r| match r {
                Response::Error(e) => assert_eq!(e.code, c),
                _ => panic!("not error"),
            });
        }
    }

    // ---- A.2 RESPONSE negative (egress schema-enforced) -----------------
    #[test]
    fn resp_rejects_stale_off_shape_count() {
        rejects::<Response>(
            r#"{"type":"abstained","frame_id":"f","raw_text":"x","model_output_truncated":false,"deviations_dropped":0,"off_shape_count":0,"image_size":[1,1],"resize_plan":{"dst_w":1,"dst_h":1,"n_llm_tokens":1,"scale":1.0},"generation_mode_used":"slow","latency_ms":1.0,"total_ms":2.0}"#,
        );
    }
    #[test]
    fn resp_rejects_missing_latency_ms() {
        rejects::<Response>(
            r#"{"type":"abstained","frame_id":"f","raw_text":"x","model_output_truncated":false,"deviations_dropped":0,"image_size":[1,1],"resize_plan":{"dst_w":1,"dst_h":1,"n_llm_tokens":1,"scale":1.0},"generation_mode_used":"slow","total_ms":2.0}"#,
        );
    }
    #[test]
    fn resp_rejects_unlabeled_box() {
        // label is REQUIRED.
        rejects::<Response>(
            r#"{"type":"boxes","frame_id":"f","boxes":[{"bbox_norm":[0,0,1,1],"bbox_px":[0.0,0.0,1.0,1.0]}],"raw_text":"x","model_output_truncated":false,"deviations_dropped":0,"image_size":[1,1],"resize_plan":{"dst_w":1,"dst_h":1,"n_llm_tokens":1,"scale":1.0},"generation_mode_used":"slow","latency_ms":1.0,"total_ms":2.0}"#,
        );
    }
    #[test]
    fn resp_rejects_unlabeled_point() {
        // Symmetric to resp_rejects_unlabeled_box: a point's `label` is REQUIRED.
        // It is the queried phrase the worker attributes to the model's BARE
        // pointing output (`<box><x><y></box>`, no `<ref>`); a label-less point
        // is rejected at the egress, so the worker producing one would fail loud
        // here rather than reach the client.
        rejects::<Response>(
            r#"{"type":"points","frame_id":"f","points":[{"point_norm":[500,300],"point_px":[960.0,324.0]}],"raw_text":"x","model_output_truncated":false,"deviations_dropped":0,"image_size":[1,1],"resize_plan":{"dst_w":1,"dst_h":1,"n_llm_tokens":1,"scale":1.0},"generation_mode_used":"slow","latency_ms":1.0,"total_ms":2.0}"#,
        );
    }
    #[test]
    fn resp_rejects_negative_norm() {
        // bbox_norm is u16 → a negative coordinate is unrepresentable.
        rejects::<Response>(
            r#"{"type":"boxes","frame_id":"f","boxes":[{"label":"x","bbox_norm":[-1,0,1,1],"bbox_px":[0.0,0.0,1.0,1.0]}],"raw_text":"x","model_output_truncated":false,"deviations_dropped":0,"image_size":[1,1],"resize_plan":{"dst_w":1,"dst_h":1,"n_llm_tokens":1,"scale":1.0},"generation_mode_used":"slow","latency_ms":1.0,"total_ms":2.0}"#,
        );
    }
    #[test]
    fn resp_rejects_boxes_with_points_field() {
        // boxes variant must not also carry points (deny_unknown_fields).
        rejects::<Response>(
            r#"{"type":"boxes","frame_id":"f","boxes":[],"points":[],"raw_text":"x","model_output_truncated":false,"deviations_dropped":0,"image_size":[1,1],"resize_plan":{"dst_w":1,"dst_h":1,"n_llm_tokens":1,"scale":1.0},"generation_mode_used":"slow","latency_ms":1.0,"total_ms":2.0}"#,
        );
    }
    #[test]
    fn resp_rejects_points_with_stray_meta_field() {
        // Symmetric guard for the THIRD flatten-bearing body (points): a field
        // unknown to both the body and the flattened Meta must reject. Together
        // with resp_rejects_stale_off_shape_count (abstained body) and
        // resp_rejects_boxes_with_points_field (boxes body), this pins the
        // load-bearing body-level deny_unknown_fields across all three bodies —
        // so the flatten/deny mechanism documented on BoxesBody can't silently rot.
        rejects::<Response>(
            r#"{"type":"points","frame_id":"f","points":[],"stray":true,"raw_text":"x","model_output_truncated":false,"deviations_dropped":0,"image_size":[1,1],"resize_plan":{"dst_w":1,"dst_h":1,"n_llm_tokens":1,"scale":1.0},"generation_mode_used":"slow","latency_ms":1.0,"total_ms":2.0}"#,
        );
    }
    #[test]
    fn resp_rejects_unknown_type_tag() {
        rejects::<Response>(r#"{"type":"detections","frame_id":"f"}"#);
    }
}
