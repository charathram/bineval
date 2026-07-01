//! Generate a QuestionSet for a task, then evaluate a deliberately-flawed output against it and
//! print the score and the per-question verdicts with explanations.
//!
//! Model config and API keys are read from the environment / a `.env` file (see the README):
//! ```sh
//! OPENAI_API_KEY=... cargo run --example end_to_end
//! ```

use bineval::BinEval;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Loads `.env` and builds the generator + evaluator LMs from `BINEVAL_*` variables.
    let be = BinEval::from_env().await?;

    let task = "Write a faithful, concise summary of a news article.";
    let source = "A Russian SU-27 fighter intercepted a U.S. RC-135 reconnaissance aircraft over the \
                  Baltic Sea. The Pentagon called the intercept unsafe and unprofessional.";
    // A summary with a fabricated detail and a misattribution.
    let output = "The U.S. RC-135 was intercepted over the Baltic Sea. The Russian Defense Ministry \
                  called the intercept unsafe, and the plane was escorted to a nearby airbase.";

    let questions = be.generate(task).await?;
    println!(
        "Generated {} questions across {} requirements.\n",
        questions.question_count(),
        questions.requirements.len()
    );

    let report = be.evaluate(&questions, source, output).await?;
    println!("Score: {:.2}\n", report.score);

    for answer in &report.answers {
        println!(
            "[{}] {} — {}",
            if answer.verdict { "YES" } else { "NO " },
            answer.question,
            answer.reasoning
        );
    }
    Ok(())
}
