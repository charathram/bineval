//! Evaluation orchestration: answer every question independently and concurrently against an
//! `(source, output)` pair, with bounded retry and non-fatal per-question failures, then aggregate.
//! LLM calls go through [`crate::lm::LmOps`]; this logic is unit-tested with fakes.

use futures::stream::{self, StreamExt};

use crate::Config;
use crate::error::BinEvalError;
use crate::lm::{LmOps, backoff};
use crate::score::aggregate;
use crate::types::{EvalReport, Question, QuestionOutcome, QuestionSet, QuestionVerdict};

/// Evaluate one `(source, output)` pair against a [`QuestionSet`].
pub(crate) async fn evaluate_pair<O: LmOps>(
    ops: &O,
    cfg: &Config,
    questions: &QuestionSet,
    source: &str,
    output: &str,
) -> Result<EvalReport, BinEvalError> {
    let concurrency = cfg.concurrency.max(1);

    let verdicts: Vec<QuestionVerdict> = stream::iter(questions.questions.iter())
        .map(|q| eval_one(ops, cfg, source, output, q))
        .buffered(concurrency) // bounded concurrency, results in question order
        .collect()
        .await;

    if cfg.strict
        && let Some(failed) = verdicts
            .iter()
            .find(|v| matches!(v.outcome, QuestionOutcome::Failed { .. }))
    {
        return Err(BinEvalError::StrictFailure(failed.question_id.clone()));
    }

    Ok(aggregate(&questions.dimensions, verdicts))
}

/// Evaluate a single question with bounded retry. Never returns `Err`: an exhausted/non-retryable
/// failure becomes a [`QuestionOutcome::Failed`] verdict (strict-mode handling is in the caller).
async fn eval_one<O: LmOps>(
    ops: &O,
    cfg: &Config,
    source: &str,
    output: &str,
    q: &Question,
) -> QuestionVerdict {
    let mut attempt = 0u32;
    loop {
        match ops.eval_question(source, output, q).await {
            Ok(answer) => {
                return QuestionVerdict::answered(
                    q.id.as_str(),
                    q.dimension.as_str(),
                    answer.satisfied,
                    answer.explanation,
                );
            }
            Err(e) => {
                if e.is_retryable() && attempt < cfg.max_retries {
                    attempt += 1;
                    tokio::time::sleep(backoff(attempt)).await;
                    continue;
                }
                return QuestionVerdict::failed(
                    q.id.as_str(),
                    q.dimension.as_str(),
                    e.class(),
                    e.to_string(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{EvalAnswer, LmOps, QuestionDraft, RequirementDraft};
    use crate::types::Requirement;
    use std::sync::Mutex;

    fn cfg(strict: bool, max_retries: u32) -> Config {
        Config {
            dimensions: vec!["coherence".into(), "consistency".into()],
            concurrency: 2,
            max_retries,
            strict,
        }
    }

    fn question(id: &str, dim: &str) -> Question {
        Question {
            id: id.into(),
            dimension: dim.into(),
            text: format!("{id}?"),
            violation_example: None,
            requirement_id: "r1".into(),
        }
    }

    fn two_question_set() -> QuestionSet {
        QuestionSet {
            task_prompt: "t".into(),
            dimensions: vec!["coherence".into(), "consistency".into()],
            requirements: vec![],
            questions: vec![question("q1", "coherence"), question("q2", "consistency")],
        }
    }

    // Answers coherence questions "yes" and others "no".
    struct Mixed;
    impl LmOps for Mixed {
        async fn summarize(
            &self,
            _t: &str,
            _d: &[String],
        ) -> Result<Vec<RequirementDraft>, BinEvalError> {
            unreachable!()
        }
        async fn decompose(
            &self,
            _t: &str,
            _d: &[String],
            _r: &[Requirement],
        ) -> Result<Vec<QuestionDraft>, BinEvalError> {
            unreachable!()
        }
        async fn eval_question(
            &self,
            _s: &str,
            _o: &str,
            q: &Question,
        ) -> Result<EvalAnswer, BinEvalError> {
            Ok(EvalAnswer {
                satisfied: q.dimension == "coherence",
                explanation: "x".into(),
            })
        }
    }

    // Always fails q2 (retryable Lm error); answers others "yes".
    struct FailQ2;
    impl LmOps for FailQ2 {
        async fn summarize(
            &self,
            _t: &str,
            _d: &[String],
        ) -> Result<Vec<RequirementDraft>, BinEvalError> {
            unreachable!()
        }
        async fn decompose(
            &self,
            _t: &str,
            _d: &[String],
            _r: &[Requirement],
        ) -> Result<Vec<QuestionDraft>, BinEvalError> {
            unreachable!()
        }
        async fn eval_question(
            &self,
            _s: &str,
            _o: &str,
            q: &Question,
        ) -> Result<EvalAnswer, BinEvalError> {
            if q.id == "q2" {
                Err(BinEvalError::Lm("boom".into()))
            } else {
                Ok(EvalAnswer {
                    satisfied: true,
                    explanation: "ok".into(),
                })
            }
        }
    }

    // Fails the first attempt, then succeeds.
    struct FlakyOnce {
        attempts: Mutex<u32>,
    }
    impl LmOps for FlakyOnce {
        async fn summarize(
            &self,
            _t: &str,
            _d: &[String],
        ) -> Result<Vec<RequirementDraft>, BinEvalError> {
            unreachable!()
        }
        async fn decompose(
            &self,
            _t: &str,
            _d: &[String],
            _r: &[Requirement],
        ) -> Result<Vec<QuestionDraft>, BinEvalError> {
            unreachable!()
        }
        async fn eval_question(
            &self,
            _s: &str,
            _o: &str,
            _q: &Question,
        ) -> Result<EvalAnswer, BinEvalError> {
            let mut a = self.attempts.lock().unwrap();
            *a += 1;
            if *a <= 1 {
                Err(BinEvalError::Lm("transient".into()))
            } else {
                Ok(EvalAnswer {
                    satisfied: true,
                    explanation: "ok".into(),
                })
            }
        }
    }

    #[tokio::test]
    async fn evaluates_and_scores() {
        let report = evaluate_pair(&Mixed, &cfg(false, 0), &two_question_set(), "x", "y")
            .await
            .unwrap();
        assert_eq!(report.per_question.len(), 2);
        assert!((report.overall - 0.5).abs() < 1e-6); // 1 of 2 satisfied
    }

    #[tokio::test]
    async fn failed_question_is_recorded_non_fatally() {
        let report = evaluate_pair(&FailQ2, &cfg(false, 1), &two_question_set(), "x", "y")
            .await
            .unwrap();
        let q2 = report
            .per_question
            .iter()
            .find(|v| v.question_id == "q2")
            .unwrap();
        assert!(matches!(q2.outcome, QuestionOutcome::Failed { .. }));
        // q1 answered + satisfied → overall over answered questions = 1.0
        assert!((report.overall - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn strict_mode_aborts_on_failure() {
        let result = evaluate_pair(&FailQ2, &cfg(true, 0), &two_question_set(), "x", "y").await;
        assert!(matches!(result, Err(BinEvalError::StrictFailure(_))));
    }

    #[tokio::test]
    async fn retries_then_succeeds() {
        let qs = QuestionSet {
            task_prompt: "t".into(),
            dimensions: vec!["coherence".into()],
            requirements: vec![],
            questions: vec![question("q1", "coherence")],
        };
        let report = evaluate_pair(
            &FlakyOnce {
                attempts: Mutex::new(0),
            },
            &cfg(false, 3),
            &qs,
            "x",
            "y",
        )
        .await
        .unwrap();
        assert!(matches!(
            report.per_question[0].outcome,
            QuestionOutcome::Answered {
                satisfied: true,
                ..
            }
        ));
    }
}
