//! Core domain types for BinEval.
//!
//! All types are `serde`-serializable: a [`QuestionSet`] is generated once and can be persisted to
//! JSON, then reused to produce a [`Report`] for each `(source, output)` pair.

use serde::{Deserialize, Serialize};

/// A single requirement extracted from the task prompt, with the binary questions it decomposes into.
///
/// Each question is phrased so that **"yes" (`true`) means the requirement is satisfied**.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirement {
    /// The requirement, in natural language.
    pub text: String,
    /// The binary yes/no questions this requirement decomposes into.
    pub questions: Vec<String>,
}

/// The reusable artifact produced by generation: the requirements and binary questions for a task.
///
/// Depends only on the task prompt — generate once, then evaluate many `(source, output)` pairs
/// against it. Can also be hand-authored or loaded from JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionSet {
    /// The task prompt this set was derived from.
    pub task: String,
    /// The requirements, each with its binary questions.
    pub requirements: Vec<Requirement>,
}

impl QuestionSet {
    /// Iterate over every `(requirement, question)` pair across all requirements.
    pub fn questions(&self) -> impl Iterator<Item = (&str, &str)> {
        self.requirements.iter().flat_map(|r| {
            r.questions
                .iter()
                .map(move |q| (r.text.as_str(), q.as_str()))
        })
    }

    /// Total number of questions across all requirements.
    pub fn question_count(&self) -> usize {
        self.requirements.iter().map(|r| r.questions.len()).sum()
    }
}

/// The evaluator's answer to one binary question about an `(source, output)` pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Answer {
    /// The requirement the question came from.
    pub requirement: String,
    /// The binary question that was answered.
    pub question: String,
    /// The verdict: `true` ⇒ "yes" ⇒ the requirement is satisfied.
    pub verdict: bool,
    /// The evaluator's natural-language explanation for the verdict (from chain-of-thought).
    pub reasoning: String,
}

/// The result of evaluating one `(source, output)` pair against a [`QuestionSet`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// The task prompt the evaluation was for.
    pub task: String,
    /// One answer per question, in question order.
    pub answers: Vec<Answer>,
    /// Overall score: the fraction of questions answered "yes", in `[0, 1]`.
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> QuestionSet {
        QuestionSet {
            task: "Write a faithful, concise summary.".into(),
            requirements: vec![
                Requirement {
                    text: "Every claim is supported by the source.".into(),
                    questions: vec![
                        "Are all claims supported by the source?".into(),
                        "Are there no fabricated facts?".into(),
                    ],
                },
                Requirement {
                    text: "The summary is concise.".into(),
                    questions: vec!["Is the summary free of redundancy?".into()],
                },
            ],
        }
    }

    #[test]
    fn question_set_round_trips_through_json() {
        let qs = sample();
        let json = serde_json::to_string(&qs).unwrap();
        let back: QuestionSet = serde_json::from_str(&json).unwrap();
        assert_eq!(qs, back);
    }

    #[test]
    fn questions_iterates_all_pairs() {
        let qs = sample();
        assert_eq!(qs.question_count(), 3);
        let pairs: Vec<_> = qs.questions().collect();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, "Every claim is supported by the source.");
        assert_eq!(pairs[2].0, "The summary is concise.");
    }
}
