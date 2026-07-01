//! The LLM seam: a thin, provider-agnostic wrapper over [`rig`](https://crates.io/crates/rig-core)
//! that pulls **typed** structured output out of a model via Rig's `Extractor` (tool-calling).
//!
//! BinEval uses two [`Lm`]s — a generator and an evaluator. All prompt intent lives in the three
//! preamble constants below plus the `schemars` doc-comments on the output DTOs (which become the
//! tool's JSON-schema field descriptions). Parsing is Rig's job — there is no hand-rolled JSON
//! parsing, and malformed model output surfaces as an error rather than a panic.
//!
//! Structured extraction relies on provider **function/tool-calling**, so OpenAI-compatible local
//! servers that lack tool support are not supported.

use anyhow::{Context, Result, anyhow};
use rig_core::client::CompletionClient;
use rig_core::providers::{anthropic, openai};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A configured LLM behind one of the supported providers.
pub struct Lm {
    provider: Provider,
    model: String,
    temperature: f64,
    max_tokens: u64,
}

enum Provider {
    OpenAI(openai::Client),
    Anthropic(anthropic::Client),
}

impl Lm {
    /// Build from a `"provider:model"` spec (e.g. `openai:gpt-4o-mini`,
    /// `anthropic:claude-sonnet-4-5`), with optional explicit API key / base URL. When `api_key` is
    /// `None`, the provider's standard env var (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) is used.
    pub fn new(
        spec: &str,
        temperature: f64,
        max_tokens: u64,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Result<Lm> {
        let (provider_name, model) = spec
            .split_once(':')
            .ok_or_else(|| anyhow!("model spec must be `provider:model`, got {spec:?}"))?;

        let provider = match provider_name {
            "openai" => {
                let key = api_key
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .context("no OpenAI API key (set OPENAI_API_KEY or BINEVAL_*_API_KEY)")?;
                let mut builder = openai::Client::builder().api_key(&key);
                if let Some(url) = &base_url {
                    builder = builder.base_url(url);
                }
                Provider::OpenAI(builder.build().context("building OpenAI client")?)
            }
            "anthropic" => {
                let key = api_key
                    .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                    .context("no Anthropic API key (set ANTHROPIC_API_KEY or BINEVAL_*_API_KEY)")?;
                let mut builder = anthropic::Client::builder().api_key(&key);
                if let Some(url) = &base_url {
                    builder = builder.base_url(url);
                }
                Provider::Anthropic(builder.build().context("building Anthropic client")?)
            }
            other => {
                return Err(anyhow!(
                    "unsupported provider {other:?} (supported: openai, anthropic)"
                ));
            }
        };

        Ok(Lm {
            provider,
            model: model.to_string(),
            temperature,
            max_tokens,
        })
    }

    /// Extract typed structured output `T` from `input`, guided by `preamble`, via Rig's `Extractor`
    /// (tool-calling). Rig coerces the model's response into `T`; failures return an error.
    pub async fn extract<T>(&self, preamble: &str, input: &str) -> Result<T>
    where
        T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
    {
        // The extractor builder has no `.temperature()`; pass it through `additional_params`.
        let params = serde_json::json!({ "temperature": self.temperature });
        let out = match &self.provider {
            Provider::OpenAI(client) => client
                .extractor::<T>(&self.model)
                .preamble(preamble)
                .max_tokens(self.max_tokens)
                .additional_params(params)
                .build()
                .extract(input)
                .await
                .context("Rig extraction failed (OpenAI)")?,
            Provider::Anthropic(client) => client
                .extractor::<T>(&self.model)
                .preamble(preamble)
                .max_tokens(self.max_tokens)
                .additional_params(params)
                .build()
                .extract(input)
                .await
                .context("Rig extraction failed (Anthropic)")?,
        };
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// Structured-output DTOs. schemars doc-comments become the tool schema's field
// descriptions, so they carry the per-field intent to the model.
// ----------------------------------------------------------------------------

/// Output of the requirement-extraction step.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct RequirementList {
    /// The distinct requirements an acceptable output must satisfy — each a short, self-contained
    /// statement. Prefer few, non-overlapping requirements.
    pub requirements: Vec<String>,
}

/// Output of the per-requirement question-generation step.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct QuestionList {
    /// Minimal, non-redundant binary yes/no questions for the requirement, each phrased so that
    /// "yes" means the requirement is satisfied. Do not over-decompose.
    pub questions: Vec<String>,
}

/// Output of the per-question answering step.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct VerdictOut {
    /// `true` if the output satisfies the question, otherwise `false`.
    pub verdict: bool,
    /// A brief justification for the verdict.
    pub reasoning: String,
}

/// Instruction for extracting requirements from a task prompt.
pub(crate) const EXTRACT_PREAMBLE: &str = "You are an expert evaluation designer. Given a task \
    specification, identify the distinct requirements that an acceptable output must satisfy. Prefer \
    a small set of non-overlapping requirements, each a short, self-contained statement.";

/// Instruction for decomposing one requirement into binary questions.
pub(crate) const DECOMPOSE_PREAMBLE: &str = "Given a task and a single requirement, produce minimal, \
    non-redundant binary yes/no questions that check whether the requirement is satisfied, where \
    'yes' means satisfied. Prefer few questions; do not over-decompose.";

/// Instruction for answering one binary question about an (source, output) pair.
pub(crate) const ANSWER_PREAMBLE: &str = "You judge whether an OUTPUT satisfies a yes/no QUESTION \
    about it, given the SOURCE it is based on. Set verdict=true only if the output clearly satisfies \
    the question, and give a brief reasoning.";

/// Live tests that exercise **one extractor at a time** against a real LLM.
///
/// Ignored by default. Run a single one (and see logs) with, e.g.:
/// ```sh
/// OPENAI_API_KEY=... cargo test --lib -- --ignored --nocapture extract_requirements
/// ```
#[cfg(test)]
mod live_tests {
    use super::*;

    /// Build an [`Lm`] from the environment, or `None` to skip.
    fn lm_from_env() -> Option<Lm> {
        dotenvy::dotenv().ok();
        let model = std::env::var("BINEVAL_MODEL").ok().or_else(|| {
            std::env::var("OPENAI_API_KEY")
                .ok()
                .map(|_| "openai:gpt-4o-mini".to_string())
        })?;
        Lm::new(&model, 0.0, 4096, None, None).ok()
    }

    /// Install a `tracing` subscriber that prints through libtest capture (shown with `--nocapture`).
    fn init_tracing() {
        use tracing_subscriber::EnvFilter;
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("bineval=debug"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .with_target(false)
            .try_init();
    }

    #[tokio::test]
    #[ignore = "requires a provider API key; run with `--ignored --nocapture`"]
    async fn extract_requirements_live() {
        init_tracing();
        let Some(lm) = lm_from_env() else {
            eprintln!("skipping: set OPENAI_API_KEY or BINEVAL_MODEL");
            return;
        };
        tracing::info!("calling ExtractRequirements …");
        let out: RequirementList = lm
            .extract(
                EXTRACT_PREAMBLE,
                "Extract a complete definition of a source code file written in COBOL. The definition should include the file's purpose, its inputs and outputs, and any dependencies or constraints.",
            )
            .await
            .expect("extract");
        tracing::info!(count = out.requirements.len(), "got requirements");
        for r in &out.requirements {
            println!("- {r}");
        }
        assert!(!out.requirements.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires a provider API key; run with `--ignored --nocapture`"]
    async fn generate_questions_live() {
        init_tracing();
        let Some(lm) = lm_from_env() else {
            eprintln!("skipping: set OPENAI_API_KEY or BINEVAL_MODEL");
            return;
        };
        let input = "Task:\nWrite a faithful, concise summary of a news article.\n\n\
                     Requirement:\nEvery claim in the summary is supported by the source.";
        tracing::info!("calling GenerateQuestions …");
        let out: QuestionList = lm
            .extract(DECOMPOSE_PREAMBLE, input)
            .await
            .expect("extract");
        tracing::info!(count = out.questions.len(), "got questions");
        for q in &out.questions {
            println!("- {q}");
        }
        assert!(!out.questions.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires a provider API key; run with `--ignored --nocapture`"]
    async fn answer_question_live() {
        init_tracing();
        let Some(lm) = lm_from_env() else {
            eprintln!("skipping: set OPENAI_API_KEY or BINEVAL_MODEL");
            return;
        };
        let input = "SOURCE:\nThe Pentagon called the intercept unsafe and unprofessional.\n\n\
                     OUTPUT:\nThe Russian Defense Ministry called the intercept unsafe.\n\n\
                     QUESTION:\nIs every claim in the output supported by the source?";
        tracing::info!("calling AnswerQuestion …");
        let out: VerdictOut = lm.extract(ANSWER_PREAMBLE, input).await.expect("extract");
        tracing::info!(verdict = out.verdict, "got verdict");
        println!("verdict={} reasoning={}", out.verdict, out.reasoning);
    }
}
