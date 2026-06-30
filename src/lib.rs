//! # bineval
//!
//! An interpretable LLM-evaluation library — a Rust implementation of **BinEval**
//! (*"Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement"*,
//! Cho et al., arXiv 2606.27226), built on [DSRs](https://crates.io/crates/dspy-rs).
//!
//! Instead of a single opaque scalar judgment, BinEval decomposes an evaluation task into atomic
//! yes/no questions, answers each independently, and aggregates the verdicts into per-dimension and
//! overall scores — each grounded in a natural-language explanation.
//!
//! ## Two phases
//!
//! 1. [`BinEval::generate`] — build a reusable [`QuestionSet`] from a task prompt `T` (once per task).
//! 2. [`BinEval::evaluate`] — score any `(source, output)` pair against that set, producing an [`EvalReport`].
//!
//! ```no_run
//! use bineval::{BinEval, LM};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let lm = LM::builder().model("openai:gpt-4o-mini".to_string()).temperature(0.0).build().await?;
//! let be = BinEval::builder().generator_lm(lm).build()?;
//!
//! let questions = be.generate("Write a faithful, concise summary of a news article.").await?;
//! let report = be.evaluate(&questions, "the source article", "the candidate summary").await?;
//! println!("overall: {:.2}", report.overall);
//! # Ok(())
//! # }
//! ```
//!
//! All LLM logic is expressed through DSRs Signatures/Modules; there are no hand-authored prompt
//! strings. See `docs/prd.md` for the full design.

mod error;
mod evaluate;
mod generate;
mod lm;
pub mod score;
pub mod types;

use std::sync::Arc;

pub use error::BinEvalError;
pub use score::aggregate;
pub use types::{
    DEFAULT_DIMENSIONS, DimensionScore, EvalReport, Question, QuestionOutcome, QuestionSet,
    QuestionVerdict, Requirement, default_dimensions,
};

/// Re-exported from `dspy-rs` so callers can build models without depending on DSRs directly.
pub use dspy_rs::LM;

/// Runtime configuration shared by generation and evaluation.
#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub dimensions: Vec<String>,
    pub concurrency: usize,
    pub max_retries: u32,
    pub strict: bool,
}

/// The BinEval evaluator. Construct with [`BinEval::builder`].
pub struct BinEval {
    ops: lm::DsrsOps,
    config: Config,
}

impl BinEval {
    /// Start building a [`BinEval`].
    pub fn builder() -> BinEvalBuilder {
        BinEvalBuilder::default()
    }

    /// Build a fully-configured [`BinEval`] from the environment, loading a `.env` file first.
    ///
    /// Builds **two** LMs — a generator and an evaluator — from per-role variables (see
    /// [`lms_from_env`]) and reads optional `BINEVAL_DIMENSIONS` (comma-separated),
    /// `BINEVAL_CONCURRENCY`, `BINEVAL_MAX_RETRIES`, and `BINEVAL_STRICT`.
    pub async fn from_env() -> Result<BinEval, BinEvalError> {
        let (generator, evaluator) = lms_from_env().await?;
        let mut builder = BinEval::builder()
            .generator_lm(generator)
            .evaluator_lm(evaluator);
        if let Some(dimensions) = env::dimensions() {
            builder = builder.dimensions(dimensions);
        }
        if let Some(concurrency) = env::parsed("BINEVAL_CONCURRENCY") {
            builder = builder.concurrency(concurrency);
        }
        if let Some(max_retries) = env::parsed("BINEVAL_MAX_RETRIES") {
            builder = builder.max_retries(max_retries);
        }
        if let Some(strict) = env::boolean("BINEVAL_STRICT") {
            builder = builder.strict(strict);
        }
        builder.build()
    }

    /// Phase 1 — generate a reusable [`QuestionSet`] for a task. Runs once per task.
    pub async fn generate(&self, task_prompt: &str) -> Result<QuestionSet, BinEvalError> {
        generate::generate_questions(&self.ops, &self.config, task_prompt).await
    }

    /// Phase 2 — evaluate one `(source, output)` pair against a [`QuestionSet`].
    pub async fn evaluate(
        &self,
        questions: &QuestionSet,
        source: &str,
        output: &str,
    ) -> Result<EvalReport, BinEvalError> {
        evaluate::evaluate_pair(&self.ops, &self.config, questions, source, output).await
    }

    /// Evaluate many `(source, output)` pairs against one [`QuestionSet`]. Pairs are evaluated
    /// sequentially (each pair already fans its questions out concurrently); one pair's failure does
    /// not affect the others.
    pub async fn evaluate_many(
        &self,
        questions: &QuestionSet,
        pairs: &[(String, String)],
    ) -> Vec<Result<EvalReport, BinEvalError>> {
        let mut reports = Vec::with_capacity(pairs.len());
        for (source, output) in pairs {
            reports.push(self.evaluate(questions, source, output).await);
        }
        reports
    }
}

/// Builder for [`BinEval`].
#[derive(Default)]
pub struct BinEvalBuilder {
    generator_lm: Option<LM>,
    evaluator_lm: Option<LM>,
    dimensions: Option<Vec<String>>,
    concurrency: Option<usize>,
    max_retries: Option<u32>,
    strict: bool,
}

impl BinEvalBuilder {
    /// The LM used for question generation. **Required.**
    pub fn generator_lm(mut self, lm: LM) -> Self {
        self.generator_lm = Some(lm);
        self
    }

    /// The LM used for evaluating questions. Defaults to the generator LM if unset.
    pub fn evaluator_lm(mut self, lm: LM) -> Self {
        self.evaluator_lm = Some(lm);
        self
    }

    /// Evaluation dimensions. Defaults to [`DEFAULT_DIMENSIONS`].
    pub fn dimensions(mut self, dimensions: Vec<String>) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Maximum number of question evaluations in flight at once (default 8).
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = Some(concurrency);
        self
    }

    /// Maximum retries per LM call on retryable errors (default 3).
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// If true, the first unrecovered question failure aborts `evaluate` with an error
    /// (default false — failures are recorded non-fatally).
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Build the [`BinEval`]. Errors if no generator LM was supplied.
    pub fn build(self) -> Result<BinEval, BinEvalError> {
        let generator = Arc::new(
            self.generator_lm
                .ok_or_else(|| BinEvalError::Config("generator_lm is required".into()))?,
        );
        let evaluator = match self.evaluator_lm {
            Some(lm) => Arc::new(lm),
            None => Arc::clone(&generator),
        };
        let config = Config {
            dimensions: self.dimensions.unwrap_or_else(default_dimensions),
            concurrency: self.concurrency.unwrap_or(8),
            max_retries: self.max_retries.unwrap_or(3),
            strict: self.strict,
        };
        Ok(BinEval {
            ops: lm::DsrsOps::new(generator, evaluator),
            config,
        })
    }
}

/// Load environment variables from a `.env` file in the current directory (or a parent), if present.
///
/// Existing process environment variables take precedence. Returns `true` if a file was loaded.
/// Useful for supplying provider API keys (e.g. `OPENAI_API_KEY`) and the `BINEVAL_*` settings.
pub fn load_dotenv() -> bool {
    dotenvy::dotenv().is_ok()
}

/// Build the generator and evaluator [`LM`]s from environment variables, loading `.env` first.
///
/// Each role reads `BINEVAL_<ROLE>_<KEY>` and falls back to the shared `BINEVAL_<KEY>`, for roles
/// `GENERATOR` and `EVALUATOR` and keys:
/// - `MODEL` — `provider:model` (default `openai:gpt-4o-mini`)
/// - `TEMPERATURE` (default `0.0`)
/// - `MAX_TOKENS` (default `4096`)
/// - `BASE_URL` (optional) — for OpenAI-compatible/local servers
/// - `API_KEY` (optional) — explicit key; otherwise the provider's standard variable
///   (e.g. `OPENAI_API_KEY`) is used.
///
/// With only a shared `BINEVAL_MODEL` set, both LMs are configured identically.
pub async fn lms_from_env() -> Result<(LM, LM), BinEvalError> {
    load_dotenv();
    let generator = env::build_lm("GENERATOR").await?;
    let evaluator = env::build_lm("EVALUATOR").await?;
    Ok((generator, evaluator))
}

/// Environment-variable helpers for [`lms_from_env`] and [`BinEval::from_env`].
mod env {
    use super::{BinEvalError, LM};

    /// `BINEVAL_<ROLE>_<KEY>`, falling back to the shared `BINEVAL_<KEY>`.
    fn role_var(role: &str, key: &str) -> Option<String> {
        std::env::var(format!("BINEVAL_{role}_{key}"))
            .ok()
            .or_else(|| std::env::var(format!("BINEVAL_{key}")).ok())
    }

    pub(super) fn parsed<T: std::str::FromStr>(key: &str) -> Option<T> {
        std::env::var(key).ok().and_then(|v| v.parse().ok())
    }

    pub(super) fn boolean(key: &str) -> Option<bool> {
        std::env::var(key).ok().map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    }

    pub(super) fn dimensions() -> Option<Vec<String>> {
        let dims: Vec<String> = std::env::var("BINEVAL_DIMENSIONS")
            .ok()?
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        (!dims.is_empty()).then_some(dims)
    }

    pub(super) async fn build_lm(role: &str) -> Result<LM, BinEvalError> {
        let model = role_var(role, "MODEL").unwrap_or_else(|| "openai:gpt-4o-mini".to_string());
        let temperature = role_var(role, "TEMPERATURE")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0_f32);
        let max_tokens = role_var(role, "MAX_TOKENS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096_u32);
        LM::builder()
            .model(model)
            .temperature(temperature)
            .max_tokens(max_tokens)
            .maybe_base_url(role_var(role, "BASE_URL"))
            .maybe_api_key(role_var(role, "API_KEY"))
            .build()
            .await
            .map_err(|e| {
                BinEvalError::Config(format!("failed to build {role} LM from environment: {e}"))
            })
    }
}
