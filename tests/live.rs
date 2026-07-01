//! Live end-to-end test against a real LLM.
//!
//! Ignored by default so the offline suite stays green. Run explicitly with a provider API key:
//! ```sh
//! OPENAI_API_KEY=... cargo test --test live -- --ignored --nocapture
//! BINEVAL_MODEL=anthropic:claude-sonnet-4-5 ANTHROPIC_API_KEY=... cargo test --test live -- --ignored --nocapture
//! ```

use bineval::BinEval;

/// True if enough env is present to build an LM (else the test skips).
fn have_credentials() -> bool {
    std::env::var("BINEVAL_MODEL").is_ok()
        || std::env::var("OPENAI_API_KEY").is_ok()
        || std::env::var("ANTHROPIC_API_KEY").is_ok()
}

/// Install a `tracing` subscriber that prints through libtest capture (shown with `--nocapture`).
/// Honors `RUST_LOG`, else `bineval=debug` so per-requirement / per-question events are visible.
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
#[ignore = "requires a provider API key; run with `--ignored`"]
async fn live_generate_and_evaluate() {
    init_tracing();
    if !have_credentials() {
        eprintln!("skipping: set BINEVAL_MODEL or OPENAI_API_KEY / ANTHROPIC_API_KEY to run");
        return;
    }

    let be = BinEval::from_env().await.expect("build BinEval from env");

    // Phase 1: generation.
    let questions = be
        .generate("Write a faithful, concise summary of a news article.")
        .await
        .expect("generate");
    assert!(
        !questions.requirements.is_empty(),
        "expected some requirements"
    );
    assert!(questions.question_count() > 0, "expected some questions");

    // Phase 2: evaluation of a deliberately-flawed summary.
    let source = "A Russian SU-27 intercepted a U.S. RC-135 reconnaissance aircraft over the Baltic \
                  Sea. The Pentagon called the intercept unsafe and unprofessional.";
    let flawed =
        "The RC-135 was intercepted over the Baltic Sea and escorted to a Russian airbase.";
    let report = be
        .evaluate(&questions, source, flawed)
        .await
        .expect("evaluate");

    assert_eq!(report.answers.len(), questions.question_count());
    assert!(
        (0.0..=1.0).contains(&report.score),
        "score {} out of range",
        report.score
    );
}
