//! Question generation: the two-step pipeline (summarize → global decompose) plus deterministic
//! id-assignment and provenance validation. The LLM calls go through [`crate::lm::LmOps`]; this
//! module's logic is unit-tested with a fake.

use std::collections::HashSet;

use crate::Config;
use crate::error::BinEvalError;
use crate::lm::{LmOps, backoff};
use crate::types::{Question, QuestionSet, Requirement};

/// Generate a [`QuestionSet`] from a task prompt.
///
/// Summarize → assign requirement ids → decompose (one global call) → validate `requirement_id`s
/// against the requirements (dropping strays), assign question ids → assemble.
pub(crate) async fn generate_questions<O: LmOps>(
    ops: &O,
    cfg: &Config,
    task: &str,
) -> Result<QuestionSet, BinEvalError> {
    let dims = &cfg.dimensions;

    // Step 1: summarize into requirements (with bounded retry).
    let req_drafts = {
        let mut attempt = 0u32;
        loop {
            match ops.summarize(task, dims).await {
                Ok(v) => break v,
                Err(e) if e.is_retryable() && attempt < cfg.max_retries => {
                    attempt += 1;
                    tokio::time::sleep(backoff(attempt)).await;
                }
                Err(e) => return Err(e),
            }
        }
    };
    let requirements: Vec<Requirement> = req_drafts
        .into_iter()
        .enumerate()
        .map(|(i, d)| Requirement {
            id: format!("r{}", i + 1),
            dimension: d.dimension,
            text: d.text,
        })
        .collect();

    // Step 2: decompose into questions (one global call, with bounded retry).
    let q_drafts = {
        let mut attempt = 0u32;
        loop {
            match ops.decompose(task, dims, &requirements).await {
                Ok(v) => break v,
                Err(e) if e.is_retryable() && attempt < cfg.max_retries => {
                    attempt += 1;
                    tokio::time::sleep(backoff(attempt)).await;
                }
                Err(e) => return Err(e),
            }
        }
    };

    // Validate provenance and assign stable question ids.
    let known: HashSet<&str> = requirements.iter().map(|r| r.id.as_str()).collect();
    let mut questions = Vec::new();
    for d in q_drafts {
        if !known.contains(d.requirement_id.as_str()) {
            continue; // drop strays referencing unknown requirements
        }
        let id = format!("q{}", questions.len() + 1);
        questions.push(Question {
            id,
            dimension: d.dimension,
            text: d.text,
            violation_example: d.violation_example.filter(|s| !s.trim().is_empty()),
            requirement_id: d.requirement_id,
        });
    }

    Ok(QuestionSet {
        task_prompt: task.to_string(),
        dimensions: dims.clone(),
        requirements,
        questions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{EvalAnswer, LmOps, QuestionDraft, RequirementDraft};

    struct GenFake {
        reqs: Vec<RequirementDraft>,
        questions: Vec<QuestionDraft>,
    }

    impl LmOps for GenFake {
        async fn summarize(
            &self,
            _t: &str,
            _d: &[String],
        ) -> Result<Vec<RequirementDraft>, BinEvalError> {
            Ok(self.reqs.clone())
        }
        async fn decompose(
            &self,
            _t: &str,
            _d: &[String],
            _r: &[Requirement],
        ) -> Result<Vec<QuestionDraft>, BinEvalError> {
            Ok(self.questions.clone())
        }
        async fn eval_question(
            &self,
            _s: &str,
            _o: &str,
            _q: &Question,
        ) -> Result<EvalAnswer, BinEvalError> {
            unreachable!("generation tests never evaluate")
        }
    }

    fn cfg() -> Config {
        Config {
            dimensions: vec!["coherence".into(), "consistency".into()],
            concurrency: 4,
            max_retries: 0,
            strict: false,
        }
    }

    fn req(dim: &str, text: &str) -> RequirementDraft {
        RequirementDraft {
            dimension: dim.into(),
            text: text.into(),
        }
    }

    fn qd(req_id: &str, dim: &str, text: &str, ve: Option<&str>) -> QuestionDraft {
        QuestionDraft {
            requirement_id: req_id.into(),
            dimension: dim.into(),
            text: text.into(),
            violation_example: ve.map(str::to_string),
        }
    }

    #[tokio::test]
    async fn assigns_ids_and_keeps_provenance() {
        let fake = GenFake {
            reqs: vec![
                req("coherence", "reads well"),
                req("consistency", "is factual"),
            ],
            questions: vec![
                qd("r1", "coherence", "Reads well?", None),
                qd("r2", "consistency", "Factual?", Some("makes up facts")),
            ],
        };
        let qs = generate_questions(&fake, &cfg(), "task").await.unwrap();

        assert_eq!(qs.requirements.len(), 2);
        assert_eq!(qs.requirements[0].id, "r1");
        assert_eq!(qs.requirements[1].id, "r2");
        assert_eq!(qs.questions.len(), 2);
        assert_eq!(qs.questions[0].id, "q1");
        assert_eq!(qs.questions[0].requirement_id, "r1");
        assert_eq!(
            qs.questions[1].violation_example.as_deref(),
            Some("makes up facts")
        );
        assert_eq!(qs.task_prompt, "task");
    }

    #[tokio::test]
    async fn drops_questions_with_unknown_requirement_id() {
        let fake = GenFake {
            reqs: vec![req("coherence", "reads well")],
            questions: vec![
                qd("r1", "coherence", "ok?", None),
                qd("r99", "coherence", "stray?", None),
            ],
        };
        let qs = generate_questions(&fake, &cfg(), "task").await.unwrap();
        assert_eq!(qs.questions.len(), 1);
        assert_eq!(qs.questions[0].requirement_id, "r1");
    }

    #[tokio::test]
    async fn blank_violation_example_becomes_none() {
        let fake = GenFake {
            reqs: vec![req("coherence", "reads well")],
            questions: vec![qd("r1", "coherence", "ok?", Some("   "))],
        };
        let qs = generate_questions(&fake, &cfg(), "task").await.unwrap();
        assert_eq!(qs.questions[0].violation_example, None);
    }
}
