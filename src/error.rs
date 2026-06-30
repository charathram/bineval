//! Error type for BinEval.

use thiserror::Error;

/// Errors produced by generation and evaluation.
///
/// Per-question evaluation failures are recorded non-fatally in the [`crate::EvalReport`] (unless
/// strict mode is enabled); this type surfaces from `generate` and from `evaluate` in strict mode.
#[derive(Debug, Error)]
pub enum BinEvalError {
    /// An underlying LM call failed (network, provider, timeout, etc.).
    #[error("LM call failed: {0}")]
    Lm(String),

    /// The model's output for a field could not be parsed.
    #[error("failed to parse model output for `{field}`: {message}")]
    Parse {
        /// The output field that failed to parse.
        field: String,
        /// Parser detail.
        message: String,
    },

    /// An expected output field was missing from the model's response.
    #[error("model output field `{field}` was missing")]
    MissingField {
        /// The missing field.
        field: String,
    },

    /// The crate was misconfigured (e.g. no generator LM supplied).
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Strict-mode evaluation aborted because a question could not be answered.
    #[error("strict mode: question `{0}` could not be evaluated")]
    StrictFailure(String),

    /// Filesystem I/O error (CLI).
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// (De)serialization error.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl BinEvalError {
    /// A short, stable label for this error, recorded on a failed [`crate::QuestionVerdict`].
    pub fn class(&self) -> String {
        match self {
            BinEvalError::Lm(_) => "lm",
            BinEvalError::Parse { .. } => "parse",
            BinEvalError::MissingField { .. } => "missing_field",
            BinEvalError::Config(_) => "config",
            BinEvalError::StrictFailure(_) => "strict",
            BinEvalError::Io(_) => "io",
            BinEvalError::Serde(_) => "serde",
        }
        .to_string()
    }

    /// Whether retrying the failed operation might succeed.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            BinEvalError::Lm(_) | BinEvalError::Parse { .. } | BinEvalError::MissingField { .. }
        )
    }
}
