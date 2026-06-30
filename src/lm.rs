//! The LLM-touching seam.
//!
//! [`LmOps`] abstracts the three LLM operations BinEval needs; [`DsrsOps`] is the production
//! implementation backed by DSRs (`dspy-rs`). Keeping DSRs behind this trait lets the deterministic
//! orchestration in [`crate::generate`] and [`crate::evaluate`] be unit-tested with fakes.
//!
//! ## Design notes
//!
//! All DSRs signatures declare their structured outputs as `String` and we parse the JSON
//! ourselves. DSRs' `ChatAdapter` parses any non-`String` output field with
//! `serde_json::from_str(..).unwrap()`, which **panics** on malformed model output — unacceptable
//! for the crate's resilient-failure contract. Parsing here turns malformed output into a typed
//! [`BinEvalError::Parse`] that flows into retry / non-fatal failure handling.
//!
//! Reasoning signatures use `#[Signature(cot)]`; the auto-added `reasoning` output field is used as
//! the per-question explanation `e_i`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dspy_rs::{Example, LM, Predict, Prediction, Predictor, Signature};
use serde::Deserialize;
use serde_json::Value;

use crate::error::BinEvalError;
use crate::types::{Question, Requirement};

// ----------------------------------------------------------------------------
// Signatures (the only place the "task" is described — declaratively, via DSRs)
// ----------------------------------------------------------------------------

/// List the distinct requirements an acceptable output must satisfy, each tagged with one of the
/// given evaluation dimensions.
#[Signature(cot)]
struct SummarizeSig {
    #[input(desc = "The task specification describing what a good output must do.")]
    pub task_prompt: String,
    #[input(desc = "Comma-separated evaluation dimensions to organize requirements under.")]
    pub dimensions: String,
    #[output(
        desc = "A JSON array of objects, each {\"dimension\": <one of the given dimensions>, \"text\": <a distinct requirement>}. Output JSON only."
    )]
    pub requirements: String,
}

/// Decompose each requirement into minimal binary yes/no questions where answering "yes" means the
/// requirement is satisfied. Prefer few, non-redundant questions; do not over-decompose.
#[Signature(cot)]
struct DecomposeSig {
    #[input(desc = "The task specification.")]
    pub task_prompt: String,
    #[input(desc = "Comma-separated evaluation dimensions.")]
    pub dimensions: String,
    #[input(desc = "A JSON array of requirements, each {\"id\", \"dimension\", \"text\"}.")]
    pub requirements: String,
    #[output(
        desc = "A JSON array of objects, each {\"requirement_id\": <an id from the requirements input>, \"dimension\": <its dimension>, \"text\": <a yes/no question where 'yes' means satisfied>, \"violation_example\": <a short example of a 'no', or null>}. Output JSON only."
    )]
    pub questions: String,
}

/// Decide whether the output satisfies the given binary question about the source.
#[Signature(cot)]
struct BinaryEvalSig {
    #[input(desc = "The source/input the output is about (may be empty).")]
    pub source: String,
    #[input(desc = "The output being evaluated.")]
    pub output: String,
    #[input(desc = "A yes/no question; 'yes' means the requirement is satisfied.")]
    pub question: String,
    #[input(desc = "An example of what a 'no' looks like; may be empty.")]
    pub violation_example: String,
    #[output(desc = "Answer exactly 'yes' if the output satisfies the question, otherwise 'no'.")]
    pub verdict: String,
}

// ----------------------------------------------------------------------------
// Trait seam + transport-agnostic data
// ----------------------------------------------------------------------------

/// A requirement as returned by the model, before the crate assigns a stable id.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RequirementDraft {
    pub(crate) dimension: String,
    pub(crate) text: String,
}

/// A question as returned by the model, before the crate assigns a stable id and validates provenance.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct QuestionDraft {
    pub(crate) requirement_id: String,
    pub(crate) dimension: String,
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) violation_example: Option<String>,
}

/// A single binary verdict plus its explanation.
pub(crate) struct EvalAnswer {
    pub(crate) satisfied: bool,
    pub(crate) explanation: String,
}

/// The three LLM operations BinEval depends on. Implemented by [`DsrsOps`] (production) and by
/// fakes in tests.
#[allow(async_fn_in_trait)]
pub(crate) trait LmOps {
    async fn summarize(
        &self,
        task: &str,
        dims: &[String],
    ) -> Result<Vec<RequirementDraft>, BinEvalError>;

    async fn decompose(
        &self,
        task: &str,
        dims: &[String],
        reqs: &[Requirement],
    ) -> Result<Vec<QuestionDraft>, BinEvalError>;

    async fn eval_question(
        &self,
        source: &str,
        output: &str,
        question: &Question,
    ) -> Result<EvalAnswer, BinEvalError>;
}

// ----------------------------------------------------------------------------
// Production implementation (DSRs)
// ----------------------------------------------------------------------------

/// Production [`LmOps`] backed by DSRs. Holds per-role LMs and the `Predict` modules; every call
/// uses `forward_with_config` so the crate never touches the global `configure` singleton.
pub(crate) struct DsrsOps {
    generator_lm: Arc<LM>,
    evaluator_lm: Arc<LM>,
    summarize: Predict,
    decompose: Predict,
    binary_eval: Predict,
}

impl DsrsOps {
    pub(crate) fn new(generator_lm: Arc<LM>, evaluator_lm: Arc<LM>) -> Self {
        Self {
            generator_lm,
            evaluator_lm,
            summarize: Predict::new(SummarizeSig::new()),
            decompose: Predict::new(DecomposeSig::new()),
            binary_eval: Predict::new(BinaryEvalSig::new()),
        }
    }
}

impl LmOps for DsrsOps {
    async fn summarize(
        &self,
        task: &str,
        dims: &[String],
    ) -> Result<Vec<RequirementDraft>, BinEvalError> {
        let ex = input_example(&[("task_prompt", task), ("dimensions", &dims.join(", "))]);
        let pred = self
            .summarize
            .forward_with_config(ex, Arc::clone(&self.generator_lm))
            .await
            .map_err(|e| BinEvalError::Lm(e.to_string()))?;
        parse_json("requirements", &get_field(&pred, "requirements")?)
    }

    async fn decompose(
        &self,
        task: &str,
        dims: &[String],
        reqs: &[Requirement],
    ) -> Result<Vec<QuestionDraft>, BinEvalError> {
        let reqs_json = serde_json::to_string(reqs)?;
        let ex = input_example(&[
            ("task_prompt", task),
            ("dimensions", &dims.join(", ")),
            ("requirements", &reqs_json),
        ]);
        let pred = self
            .decompose
            .forward_with_config(ex, Arc::clone(&self.generator_lm))
            .await
            .map_err(|e| BinEvalError::Lm(e.to_string()))?;
        parse_json("questions", &get_field(&pred, "questions")?)
    }

    async fn eval_question(
        &self,
        source: &str,
        output: &str,
        question: &Question,
    ) -> Result<EvalAnswer, BinEvalError> {
        let violation = question.violation_example.as_deref().unwrap_or("");
        let ex = input_example(&[
            ("source", source),
            ("output", output),
            ("question", &question.text),
            ("violation_example", violation),
        ]);
        let pred = self
            .binary_eval
            .forward_with_config(ex, Arc::clone(&self.evaluator_lm))
            .await
            .map_err(|e| BinEvalError::Lm(e.to_string()))?;
        let satisfied = parse_verdict(&get_field(&pred, "verdict")?)?;
        // `cot` adds a "reasoning" field; use it as the explanation e_i.
        let explanation = get_field(&pred, "reasoning").unwrap_or_default();
        Ok(EvalAnswer {
            satisfied,
            explanation,
        })
    }
}

// ----------------------------------------------------------------------------
// Helpers (parsing the dynamic Prediction / building Examples)
// ----------------------------------------------------------------------------

fn input_example(pairs: &[(&str, &str)]) -> Example {
    let mut data = HashMap::new();
    let mut input_keys = Vec::with_capacity(pairs.len());
    for (k, v) in pairs {
        data.insert((*k).to_string(), Value::String((*v).to_string()));
        input_keys.push((*k).to_string());
    }
    Example::new(data, input_keys, vec![])
}

fn get_field(pred: &Prediction, field: &str) -> Result<String, BinEvalError> {
    match pred.data.get(field) {
        None => Err(BinEvalError::MissingField {
            field: field.to_string(),
        }),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Ok(other.to_string()),
    }
}

/// Strip a leading/trailing Markdown code fence (```json ... ```), if present.
fn strip_fences(s: &str) -> &str {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        let rest = rest.trim_start_matches(['\n', '\r', ' ']);
        let rest = rest.strip_suffix("```").unwrap_or(rest);
        return rest.trim().trim_end_matches('`').trim();
    }
    t
}

fn parse_json<T: for<'de> Deserialize<'de>>(field: &str, raw: &str) -> Result<T, BinEvalError> {
    serde_json::from_str::<T>(strip_fences(raw)).map_err(|e| BinEvalError::Parse {
        field: field.to_string(),
        message: e.to_string(),
    })
}

fn parse_verdict(raw: &str) -> Result<bool, BinEvalError> {
    let t = raw.trim().trim_matches('"').trim().to_ascii_lowercase();
    if t.starts_with("yes") || t.starts_with("true") || t == "y" || t == "1" {
        Ok(true)
    } else if t.starts_with("no") || t.starts_with("false") || t == "n" || t == "0" {
        Ok(false)
    } else {
        Err(BinEvalError::Parse {
            field: "verdict".to_string(),
            message: format!("unrecognized verdict: {raw:?}"),
        })
    }
}

/// Exponential backoff for the `attempt`-th retry (1-based), capped at 5s.
pub(crate) fn backoff(attempt: u32) -> Duration {
    let shift = attempt.clamp(1, 6) - 1;
    let ms = 200u64.saturating_mul(1u64 << shift);
    Duration::from_millis(ms.min(5000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_parsing_is_lenient() {
        assert!(parse_verdict("yes").unwrap());
        assert!(parse_verdict(" Yes.").unwrap());
        assert!(parse_verdict("\"true\"").unwrap());
        assert!(!parse_verdict("no").unwrap());
        assert!(!parse_verdict("No, because ...").unwrap());
        assert!(parse_verdict("maybe").is_err());
    }

    #[test]
    fn strips_code_fences_before_parsing() {
        let raw = "```json\n[{\"dimension\":\"coherence\",\"text\":\"reads well\"}]\n```";
        let drafts: Vec<RequirementDraft> = parse_json("requirements", raw).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].dimension, "coherence");
    }

    #[test]
    fn parse_error_is_retryable_not_a_panic() {
        let err = parse_json::<Vec<RequirementDraft>>("requirements", "not json").unwrap_err();
        assert!(err.is_retryable());
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff(1), Duration::from_millis(200));
        assert_eq!(backoff(2), Duration::from_millis(400));
        assert_eq!(backoff(3), Duration::from_millis(800));
        assert_eq!(backoff(100), Duration::from_millis(5000));
    }
}
