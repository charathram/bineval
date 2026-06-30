//! Live end-to-end tests against a real LLM.
//!
//! Ignored by default so the offline suite stays green. Run explicitly with a provider API key:
//! ```sh
//! OPENAI_API_KEY=... cargo test --test live -- --ignored
//! BINEVAL_MODEL=anthropic:claude-sonnet-4-5-20250929 ANTHROPIC_API_KEY=... cargo test --test live -- --ignored
//! ```

use bineval::{BinEval, LM, QuestionOutcome};

/// Resolve a model string from the environment, or `None` to skip.
fn model_from_env() -> Option<String> {
    if let Ok(m) = std::env::var("BINEVAL_MODEL") {
        return Some(m);
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return Some("openai:gpt-4o-mini".to_string());
    }
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        return Some("anthropic:claude-sonnet-4-5-20250929".to_string());
    }
    None
}

#[tokio::test]
#[ignore = "requires a provider API key; run with `--ignored`"]
async fn live_generate_and_evaluate() {
    let Some(model) = model_from_env() else {
        eprintln!("skipping: set BINEVAL_MODEL or OPENAI_API_KEY / ANTHROPIC_API_KEY to run");
        return;
    };

    let lm = LM::builder()
        .model(model)
        .temperature(0.0)
        .max_tokens(4096)
        .build()
        .await
        .expect("build LM");
    let be = BinEval::builder()
        .generator_lm(lm)
        .build()
        .expect("build BinEval");

    // Phase 1: generation.
    let questions = be
        .generate("Write a faithful, concise summary of a news article.")
        .await
        .expect("generate");
    assert!(!questions.questions.is_empty(), "expected some questions");
    assert!(
        !questions.requirements.is_empty(),
        "expected some requirements"
    );
    for q in &questions.questions {
        assert!(
            !q.text.trim().is_empty(),
            "question text should be non-empty"
        );
    }

    // Phase 2: evaluation of a deliberately-flawed summary.
    let source = "A Russian SU-27 intercepted a U.S. RC-135 reconnaissance aircraft over the Baltic \
                  Sea. The Pentagon called the intercept unsafe and unprofessional.";
    let flawed =
        "The RC-135 was intercepted over the Baltic Sea and escorted to a Russian airbase.";
    let report = be
        .evaluate(&questions, source, flawed)
        .await
        .expect("evaluate");

    assert!(
        (0.0..=1.0).contains(&report.overall),
        "overall score {} out of range",
        report.overall
    );
    let answered = report
        .per_question
        .iter()
        .filter(|v| matches!(v.outcome, QuestionOutcome::Answered { .. }))
        .count();
    assert!(answered > 0, "expected at least one answered question");

    // Rescaling stays in range.
    let rescaled = report.rescaled(1.0, 5.0);
    assert!((1.0..=5.0).contains(&rescaled.overall));
}
