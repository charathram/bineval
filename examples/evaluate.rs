//! Generate a QuestionSet, then evaluate a deliberately-flawed output against it and print the
//! per-dimension scores and per-question verdicts with explanations.
//!
//! Model config and API keys are read from the environment / a `.env` file (see the README).
//! Run with, e.g.:
//! ```sh
//! OPENAI_API_KEY=... cargo run --example evaluate
//! ```

use bineval::{BinEval, QuestionOutcome};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Loads `.env` and builds the generator + evaluator LMs from `BINEVAL_*` variables.
    let be = BinEval::from_env().await?;

    let task = "Write a faithful, concise summary of a news article.";
    let source = "A Russian SU-27 fighter intercepted a U.S. RC-135 reconnaissance aircraft over the \
                  Baltic Sea. The Pentagon called the intercept unsafe and unprofessional.";
    // A summary with a fabricated detail and a misattribution.
    let output = "The U.S. RC-135 was intercepted over the Baltic Sea. The Russian Defense Ministry \
                  called the intercept unsafe, and the plane was escorted to a nearby airbase.";

    let questions = be.generate(task).await?;
    let report = be.evaluate(&questions, source, output).await?;

    println!("Overall: {:.2}\n", report.overall);
    println!("Per-dimension:");
    for d in &report.per_dimension {
        println!(
            "  {:<12} {:.2}  ({}/{} answered)",
            d.dimension, d.score, d.answered, d.intended
        );
    }

    println!("\nPer-question:");
    for v in &report.per_question {
        match &v.outcome {
            QuestionOutcome::Answered {
                satisfied,
                explanation,
            } => println!(
                "  [{}] ({}) {} — {}",
                v.question_id,
                v.dimension,
                if *satisfied { "YES" } else { "NO " },
                explanation
            ),
            QuestionOutcome::Failed { class, message } => println!(
                "  [{}] ({}) FAILED [{class}]: {message}",
                v.question_id, v.dimension
            ),
        }
    }

    println!(
        "\nRescaled to 1–5 overall: {:.2}",
        report.rescaled(1.0, 5.0).overall
    );
    Ok(())
}
