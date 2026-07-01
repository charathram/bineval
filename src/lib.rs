//! # bineval
//!
//! An interpretable LLM-evaluation library. Instead of a single opaque score, BinEval decomposes an
//! eval task into atomic yes/no questions, answers each one, and aggregates the verdicts into a score
//! backed by per-question explanations.
//!
//! ## Two phases
//!
//! 1. [`BinEval::generate`] — extract the requirements of a task and decompose each into binary
//!    questions, producing a reusable [`QuestionSet`].
//! 2. [`BinEval::evaluate`] — answer every question for a `(source, output)` pair and score the
//!    result, producing a [`Report`].
//!
//! ```no_run
//! use bineval::BinEval;
//!
//! # async fn run() -> anyhow::Result<()> {
//! let be = BinEval::from_env().await?;
//! let questions = be.generate("Write a faithful, concise summary of a news article.").await?;
//! let report = be.evaluate(&questions, "the source article", "the candidate summary").await?;
//! println!("score: {:.2}", report.score);
//! # Ok(())
//! # }
//! ```
//!
//! BinEval uses two LMs — a *generator* (question generation) and an *evaluator* (answering) — and
//! gets typed structured output from each via [`rig`](https://crates.io/crates/rig-core)'s
//! `Extractor` (tool-calling). See [`Lm`] for provider configuration.

mod llm;
mod score;
pub mod types;

use anyhow::Result;
use futures::stream::{self, StreamExt};

pub use llm::Lm;
pub use score::fraction_yes;
pub use types::{Answer, QuestionSet, Report, Requirement};

use llm::{
    ANSWER_PREAMBLE, DECOMPOSE_PREAMBLE, EXTRACT_PREAMBLE, QuestionList, RequirementList, VerdictOut,
};

/// Maximum number of concurrent LLM extractions in flight (per generate/evaluate call).
const CONCURRENCY: usize = 8;

/// The BinEval evaluator. Build with [`BinEval::new`] (explicit LMs) or [`BinEval::from_env`].
pub struct BinEval {
    generator: Lm,
    evaluator: Lm,
}

impl BinEval {
    /// Construct from explicit generator and evaluator LMs.
    pub fn new(generator: Lm, evaluator: Lm) -> Self {
        Self {
            generator,
            evaluator,
        }
    }

    /// Build from the environment, loading a `.env` file first.
    ///
    /// Builds two LMs — a generator and an evaluator. Each role reads `BINEVAL_<ROLE>_<KEY>` and
    /// falls back to the shared `BINEVAL_<KEY>` (roles: `GENERATOR`, `EVALUATOR`; keys: `MODEL`,
    /// `TEMPERATURE`, `MAX_TOKENS`, `BASE_URL`, `API_KEY`). Provider keys (e.g. `OPENAI_API_KEY`)
    /// are used automatically when `API_KEY` is unset.
    pub async fn from_env() -> Result<BinEval> {
        dotenvy::dotenv().ok();
        let generator = env::build_lm("GENERATOR")?;
        let evaluator = env::build_lm("EVALUATOR")?;
        Ok(BinEval::new(generator, evaluator))
    }

    /// Phase 1 — extract requirements from `task` and decompose each into binary questions.
    #[tracing::instrument(skip(self), fields(task_len = task.len()))]
    pub async fn generate(&self, task: &str) -> Result<QuestionSet> {
        tracing::info!("extracting requirements");
        let extracted: RequirementList = self.generator.extract(EXTRACT_PREAMBLE, task).await?;
        tracing::info!(count = extracted.requirements.len(), "extracted requirements");

        // Decompose each requirement into questions, concurrently, with the generator LM.
        let results: Vec<Result<Requirement>> = stream::iter(extracted.requirements.into_iter())
            .map(|text| async move {
                let input = format!("Task:\n{task}\n\nRequirement:\n{text}");
                let ql: QuestionList = self.generator.extract(DECOMPOSE_PREAMBLE, &input).await?;
                tracing::debug!(requirement = %text, questions = ql.questions.len(), "decomposed");
                Ok(Requirement {
                    text,
                    questions: ql.questions,
                })
            })
            .buffered(CONCURRENCY)
            .collect()
            .await;
        let requirements = results.into_iter().collect::<Result<Vec<_>>>()?;

        let qs = QuestionSet {
            task: task.to_string(),
            requirements,
        };
        tracing::info!(
            requirements = qs.requirements.len(),
            questions = qs.question_count(),
            "generated question set"
        );
        Ok(qs)
    }

    /// Phase 2 — answer every question in `questions` for the `(source, output)` pair and score it.
    #[tracing::instrument(skip(self, questions, source, output), fields(questions = questions.question_count()))]
    pub async fn evaluate(
        &self,
        questions: &QuestionSet,
        source: &str,
        output: &str,
    ) -> Result<Report> {
        let pairs: Vec<(&str, &str)> = questions.questions().collect();
        tracing::info!(count = pairs.len(), "answering questions");

        let results: Vec<Result<Answer>> = stream::iter(pairs.into_iter())
            .map(|(requirement, question)| async move {
                let input = format!(
                    "SOURCE:\n{source}\n\nOUTPUT:\n{output}\n\nQUESTION:\n{question}"
                );
                let v: VerdictOut = self.evaluator.extract(ANSWER_PREAMBLE, &input).await?;
                tracing::debug!(question = %question, verdict = v.verdict, "answered");
                Ok(Answer {
                    requirement: requirement.to_string(),
                    question: question.to_string(),
                    verdict: v.verdict,
                    reasoning: v.reasoning,
                })
            })
            .buffered(CONCURRENCY)
            .collect()
            .await;
        let answers = results.into_iter().collect::<Result<Vec<_>>>()?;

        let score = fraction_yes(&answers);
        tracing::info!(score, answers = answers.len(), "evaluation complete");
        Ok(Report {
            task: questions.task.clone(),
            answers,
            score,
        })
    }
}

/// Environment-variable helpers for [`BinEval::from_env`].
mod env {
    use crate::llm::Lm;
    use anyhow::Result;

    /// `BINEVAL_<ROLE>_<KEY>`, falling back to the shared `BINEVAL_<KEY>`.
    fn role_var(role: &str, key: &str) -> Option<String> {
        std::env::var(format!("BINEVAL_{role}_{key}"))
            .ok()
            .or_else(|| std::env::var(format!("BINEVAL_{key}")).ok())
    }

    pub(super) fn build_lm(role: &str) -> Result<Lm> {
        let model = role_var(role, "MODEL").unwrap_or_else(|| "openai:gpt-4o-mini".to_string());
        let temperature = role_var(role, "TEMPERATURE")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0_f64);
        let max_tokens = role_var(role, "MAX_TOKENS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096_u64);
        Lm::new(
            &model,
            temperature,
            max_tokens,
            role_var(role, "API_KEY"),
            role_var(role, "BASE_URL"),
        )
    }
}
