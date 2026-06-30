//! Core domain types for BinEval.
//!
//! All types are `serde`-serializable so a [`QuestionSet`] can be generated once and persisted,
//! and an [`EvalReport`] can be emitted as JSON. See `docs/prd.md` for the full model.

use serde::{Deserialize, Serialize};

/// The paper's default evaluation dimensions (coherence / consistency / fluency / relevance).
///
/// Callers may override these; see [`default_dimensions`] for an owned `Vec<String>`.
pub const DEFAULT_DIMENSIONS: [&str; 4] = ["coherence", "consistency", "fluency", "relevance"];

/// The [`DEFAULT_DIMENSIONS`] as an owned `Vec<String>`.
pub fn default_dimensions() -> Vec<String> {
    DEFAULT_DIMENSIONS.iter().map(|s| s.to_string()).collect()
}

/// An explicit requirement extracted from the task prompt during generation (Step 1, "summarize").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirement {
    /// Stable identifier, referenced by [`Question::requirement_id`].
    pub id: String,
    /// The evaluation dimension this requirement belongs to.
    pub dimension: String,
    /// The requirement, in natural language.
    pub text: String,
}

/// A single binary (yes/no) question, phrased so that **"yes" means the requirement is satisfied**.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Question {
    /// Stable identifier for this question.
    pub id: String,
    /// The evaluation dimension this question belongs to.
    pub dimension: String,
    /// The question text (a "yes" answer indicates the requirement is met).
    pub text: String,
    /// Optional concise example of what a violation ("no") looks like.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub violation_example: Option<String>,
    /// Provenance: the [`Requirement::id`] this question was decomposed from.
    pub requirement_id: String,
}

/// The reusable artifact produced by question generation: `𝒬 = ℱ(T; M)`.
///
/// Depends only on the task prompt — generate once, then evaluate many `(source, output)` pairs
/// against it. Can also be hand-authored or loaded from JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionSet {
    /// The task prompt `T` this set was derived from.
    pub task_prompt: String,
    /// The dimensions these questions are organized under.
    pub dimensions: Vec<String>,
    /// The intermediate requirements `R` (retained for inspectability and provenance).
    pub requirements: Vec<Requirement>,
    /// The binary questions.
    pub questions: Vec<Question>,
}

/// The outcome of evaluating one question against an `(x, y)` pair.
///
/// Failures are non-fatal by default (see the crate's resilient evaluation policy) and are recorded
/// here so a partial score is never mistaken for a complete one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum QuestionOutcome {
    /// The evaluator answered: `satisfied` is the binary verdict, `explanation` is `e_i`.
    Answered {
        /// `true` ⇒ "yes" ⇒ the requirement is satisfied (`f_E = 1`).
        satisfied: bool,
        /// The evaluator's natural-language explanation for the verdict.
        explanation: String,
    },
    /// The evaluator call failed (after retries); excluded from scoring.
    Failed {
        /// A short classification of the error (e.g. the `PredictError` class).
        class: String,
        /// Human-readable error detail.
        message: String,
    },
}

/// A per-question verdict in an [`EvalReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionVerdict {
    /// The [`Question::id`] this verdict is for.
    pub question_id: String,
    /// The question's dimension (denormalized for convenient aggregation/inspection).
    pub dimension: String,
    /// The outcome.
    pub outcome: QuestionOutcome,
}

impl QuestionVerdict {
    /// Construct an `Answered` verdict.
    pub fn answered(
        question_id: impl Into<String>,
        dimension: impl Into<String>,
        satisfied: bool,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            question_id: question_id.into(),
            dimension: dimension.into(),
            outcome: QuestionOutcome::Answered {
                satisfied,
                explanation: explanation.into(),
            },
        }
    }

    /// Construct a `Failed` verdict.
    pub fn failed(
        question_id: impl Into<String>,
        dimension: impl Into<String>,
        class: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            question_id: question_id.into(),
            dimension: dimension.into(),
            outcome: QuestionOutcome::Failed {
                class: class.into(),
                message: message.into(),
            },
        }
    }
}

/// Aggregated score for one dimension, with coverage so partial scores are visible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionScore {
    /// The dimension name.
    pub dimension: String,
    /// `S_d` — mean of satisfied verdicts over the **answered** questions, in `[0, 1]`
    /// (or rescaled via [`EvalReport::rescaled`]).
    pub score: f32,
    /// Number of questions in this dimension that were successfully answered.
    pub answered: usize,
    /// Total number of questions in this dimension (answered + failed).
    pub intended: usize,
}

/// The result of evaluating an `(x, y)` pair against a [`QuestionSet`].
///
/// Rich and eager: every score is backed by the per-question verdicts and explanations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// One entry per question (including failures), preserving question order.
    pub per_question: Vec<QuestionVerdict>,
    /// Per-dimension scores. Dimensions with zero answered questions are omitted.
    pub per_dimension: Vec<DimensionScore>,
    /// `S` — mean of satisfied verdicts over all answered questions, in `[0, 1]`
    /// (or rescaled). `0.0` if no question was answered.
    pub overall: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_set_round_trips_through_json() {
        let qs = QuestionSet {
            task_prompt: "Write a faithful, concise summary.".into(),
            dimensions: default_dimensions(),
            requirements: vec![Requirement {
                id: "r1".into(),
                dimension: "consistency".into(),
                text: "Every claim must be supported by the source.".into(),
            }],
            questions: vec![Question {
                id: "q1".into(),
                dimension: "consistency".into(),
                text: "Are all claims supported by the source?".into(),
                violation_example: Some("States a fact not present in the source.".into()),
                requirement_id: "r1".into(),
            }],
        };
        let json = serde_json::to_string(&qs).unwrap();
        let back: QuestionSet = serde_json::from_str(&json).unwrap();
        assert_eq!(qs, back);
    }

    #[test]
    fn eval_report_round_trips_through_json() {
        let report = EvalReport {
            per_question: vec![
                QuestionVerdict::answered("q1", "coherence", true, "reads cleanly"),
                QuestionVerdict::failed("q2", "consistency", "Lm", "timeout"),
            ],
            per_dimension: vec![DimensionScore {
                dimension: "coherence".into(),
                score: 1.0,
                answered: 1,
                intended: 1,
            }],
            overall: 1.0,
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: EvalReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, back);
    }

    #[test]
    fn outcome_is_status_tagged_and_violation_is_optional() {
        let answered = QuestionOutcome::Answered {
            satisfied: true,
            explanation: "ok".into(),
        };
        let json = serde_json::to_string(&answered).unwrap();
        assert!(json.contains("\"status\":\"answered\""));

        let no_violation = Question {
            id: "q".into(),
            dimension: "d".into(),
            text: "t".into(),
            violation_example: None,
            requirement_id: "r".into(),
        };
        let json = serde_json::to_string(&no_violation).unwrap();
        assert!(!json.contains("violation_example"));
    }
}
