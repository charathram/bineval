//! `bineval` CLI — generate a QuestionSet and evaluate outputs from the shell.
//!
//! Model configuration for both roles (generator + evaluator) comes from the environment / a `.env`
//! file via `BINEVAL_*` variables (see the README), as do provider API keys (e.g. `OPENAI_API_KEY`).

use std::io::Read;
use std::path::PathBuf;

use bineval::{BinEval, QuestionSet};
use clap::{Parser, Subcommand};

type DynError = Box<dyn std::error::Error>;

#[derive(Parser)]
#[command(
    name = "bineval",
    about = "Interpretable LLM evaluation via binary questions (BinEval)."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a reusable QuestionSet from a task prompt (printed as JSON).
    Generate {
        /// Task prompt: a file path, or `-` for stdin.
        #[arg(long)]
        task: String,
        /// Comma-separated evaluation dimensions (default: the paper's four).
        #[arg(long, value_delimiter = ',')]
        dims: Option<Vec<String>>,
    },
    /// Evaluate a (source, output) pair against a QuestionSet (report printed as JSON).
    Evaluate {
        /// Path to a QuestionSet JSON file (from `generate`).
        #[arg(long)]
        questions: PathBuf,
        /// Source/input the output is about: a file path, or `-` for stdin.
        #[arg(long)]
        source: String,
        /// The output being evaluated: a file path, or `-` for stdin.
        #[arg(long)]
        output: String,
        /// Max concurrent question evaluations.
        #[arg(long)]
        concurrency: Option<usize>,
        /// Rescale scores to a custom range, e.g. `1,5`.
        #[arg(long, value_delimiter = ',')]
        rescale: Option<Vec<f32>>,
    },
}

/// Read `arg` as a file path, or stdin when it is `-`.
fn read_input(arg: &str) -> Result<String, DynError> {
    if arg == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        Ok(buf)
    } else {
        Ok(std::fs::read_to_string(arg)?)
    }
}

#[tokio::main]
async fn main() -> Result<(), DynError> {
    let cli = Cli::parse();
    // Both LMs (generator + evaluator) are configured from the environment / `.env`.
    let (generator, evaluator) = bineval::lms_from_env().await?;

    match cli.command {
        Command::Generate { task, dims } => {
            let task_text = read_input(&task)?;
            let mut builder = BinEval::builder()
                .generator_lm(generator)
                .evaluator_lm(evaluator);
            if let Some(dims) = dims {
                builder = builder.dimensions(dims);
            }
            let be = builder.build()?;
            let questions = be.generate(task_text.trim()).await?;
            println!("{}", serde_json::to_string_pretty(&questions)?);
        }
        Command::Evaluate {
            questions,
            source,
            output,
            concurrency,
            rescale,
        } => {
            let question_set: QuestionSet =
                serde_json::from_str(&std::fs::read_to_string(&questions)?)?;
            let source_text = read_input(&source)?;
            let output_text = read_input(&output)?;

            let mut builder = BinEval::builder()
                .generator_lm(generator)
                .evaluator_lm(evaluator);
            if let Some(concurrency) = concurrency {
                builder = builder.concurrency(concurrency);
            }
            let be = builder.build()?;

            let mut report = be
                .evaluate(&question_set, source_text.trim(), output_text.trim())
                .await?;
            if let Some(range) = rescale {
                match range.as_slice() {
                    [a, b] => report = report.rescaled(*a, *b),
                    _ => return Err("--rescale expects two numbers, e.g. 1,5".into()),
                }
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}
