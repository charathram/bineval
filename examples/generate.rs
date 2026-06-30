//! Generate a BinEval QuestionSet for a task and print it.
//!
//! Model config and API keys are read from the environment / a `.env` file (see the README).
//! Run with, e.g.:
//! ```sh
//! OPENAI_API_KEY=... cargo run --example generate
//! # or put BINEVAL_MODEL / OPENAI_API_KEY in a .env file in the project root
//! ```

use bineval::BinEval;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Loads `.env` and builds the generator + evaluator LMs from `BINEVAL_*` variables.
    let be = BinEval::from_env().await?;

    let task = "Write a faithful, concise summary of a news article.";
    let questions = be.generate(task).await?;

    println!(
        "Generated {} questions for task:\n  {task}\n",
        questions.questions.len()
    );
    for q in &questions.questions {
        println!("[{}] ({}) {}", q.id, q.dimension, q.text);
        if let Some(example) = &q.violation_example {
            println!("      violation example: {example}");
        }
    }

    println!("\nFull QuestionSet (JSON):");
    println!("{}", serde_json::to_string_pretty(&questions)?);
    Ok(())
}
