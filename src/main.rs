//! `bineval` CLI — generate a QuestionSet and evaluate outputs from the shell.
//!
//! Model configuration for both roles (generator + evaluator) comes from the environment / a `.env`
//! file via `BINEVAL_*` variables (see the README), as do provider API keys (e.g. `OPENAI_API_KEY`).
//!
//! Logs go to stderr (so JSON on stdout stays clean). Use `-v` for debug, `-vv` for trace, or set
//! `RUST_LOG` for full control.

use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result};
use bineval::{BinEval, QuestionSet};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "bineval",
    about = "Interpretable LLM evaluation via binary questions (BinEval)."
)]
struct Cli {
    /// Increase log verbosity: `-v` = debug, `-vv` = trace (overridden by `RUST_LOG`).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,
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
        /// Write the QuestionSet here instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
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
    },
    /// Generate questions for a task and immediately evaluate a (source, output) pair against them.
    Run {
        /// Task prompt: a file path, or `-` for stdin.
        #[arg(long)]
        task: String,
        /// Source/input the output is about: a file path, or `-` for stdin.
        #[arg(long)]
        source: String,
        /// The output being evaluated: a file path, or `-` for stdin.
        #[arg(long)]
        output: String,
    },
}

/// Read `arg` as a file path, or stdin when it is `-`.
fn read_input(arg: &str) -> Result<String> {
    if arg == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("reading stdin")?;
        Ok(buf)
    } else {
        std::fs::read_to_string(arg).with_context(|| format!("reading {arg}"))
    }
}

fn init_tracing(verbose: u8) {
    let default = match verbose {
        0 => "bineval=info",
        1 => "bineval=debug",
        _ => "bineval=trace",
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    let be = BinEval::from_env().await?;

    match cli.command {
        Command::Generate { task, out } => {
            let task = read_input(&task)?;
            let questions = be.generate(task.trim()).await?;
            let json = serde_json::to_string_pretty(&questions)?;
            match out {
                Some(path) => {
                    std::fs::write(&path, json).with_context(|| format!("writing {path:?}"))?;
                    tracing::info!(?path, "wrote question set");
                }
                None => println!("{json}"),
            }
        }
        Command::Evaluate {
            questions,
            source,
            output,
        } => {
            let qs: QuestionSet = serde_json::from_str(
                &std::fs::read_to_string(&questions)
                    .with_context(|| format!("reading {questions:?}"))?,
            )
            .context("parsing QuestionSet JSON")?;
            let source = read_input(&source)?;
            let output = read_input(&output)?;
            let report = be.evaluate(&qs, source.trim(), output.trim()).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Run {
            task,
            source,
            output,
        } => {
            let task = read_input(&task)?;
            let source = read_input(&source)?;
            let output = read_input(&output)?;
            let qs = be.generate(task.trim()).await?;
            let report = be.evaluate(&qs, source.trim(), output.trim()).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}
