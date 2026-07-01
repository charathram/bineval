# bineval

An interpretable LLM-evaluation library for Rust — an implementation of **BinEval**, built on
**[Rig](https://crates.io/crates/rig-core)** (`rig-core`).

> **Ask, don't judge.** Instead of squeezing an LLM's quality into one opaque scalar, BinEval
> decomposes an evaluation task into atomic **yes/no questions**, answers each independently, and
> aggregates the verdicts into an overall score — each answer backed by a natural-language
> explanation. The result is evaluation you can **inspect, debug, and act on**.

Based on *Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement*
(Cho et al., [arXiv 2606.27226](https://arxiv.org/abs/2606.27226)).

📄 **Full design & requirements: [docs/prd.md](docs/prd.md).**

---

## Status

🚧 **v1 in development.** Scope: the minimal core — **extract requirements → binary questions →
answer them → score**. Per-dimension scoring, prompt-optimization loops (cross-model and self
update), and benchmark reproduction are **out of scope for v1** (see the [PRD](docs/prd.md)).

## How it works

1. **Generate** a reusable `QuestionSet` from a task prompt — extract the requirements an acceptable
   output must satisfy, then decompose each requirement into minimal binary yes/no questions.
2. **Evaluate** any `(source, output)` pair against that `QuestionSet` — each question is answered
   (`yes` = requirement satisfied) with a natural-language explanation, and the verdicts aggregate
   into an overall score in `[0, 1]`.

A `QuestionSet` is generated once and reused across many outputs. Structured output is obtained via
Rig's `Extractor` (tool-calling) into typed Rust structs — no hand-rolled JSON parsing — so malformed
model output surfaces as an error, not a panic. BinEval uses two LMs: a *generator* (question
generation) and an *evaluator* (answering).

> **Note:** structured extraction relies on provider **function/tool-calling** (OpenAI and Anthropic
> support it). OpenAI-compatible local servers without tool support are not supported.

## Install

```toml
[dependencies]
bineval = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Requires Rust ≥ 1.85 (edition 2024). Configure models and API keys via the environment or a `.env`
file (see [Configuration](#configuration)); copy [`.env.example`](.env.example) to `.env` to start.

## Quickstart (library)

```rust
use bineval::BinEval;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Loads `.env` and builds the generator + evaluator LMs from BINEVAL_* variables.
    let be = BinEval::from_env().await?;

    // Phase 1 — once per task: produce a reusable, serializable QuestionSet.
    let questions = be.generate("Write a faithful, concise summary of a news article.").await?;

    // Phase 2 — per output: score a candidate against the source.
    let report = be.evaluate(&questions, source_article, candidate_summary).await?;

    println!("score: {:.2}", report.score);
    for answer in &report.answers {
        println!("[{}] {} — {}", if answer.verdict { "yes" } else { "no" }, answer.question, answer.reasoning);
    }
    Ok(())
}
```

Prefer to wire models yourself? Build the two `Lm`s explicitly from a `provider:model` spec:

```rust
use bineval::{BinEval, Lm};

// Lm::new(spec, temperature, max_tokens, api_key, base_url)
let generator = Lm::new("openai:gpt-4o-mini", 0.0, 4096, None, None)?;
let evaluator = Lm::new("anthropic:claude-sonnet-4-5", 0.0, 4096, None, None)?;
let be = BinEval::new(generator, evaluator);
```

Every score is backed by per-question verdicts and explanations in `report.answers`, so you can see
exactly *why* an output scored the way it did.

## CLI

Models and keys come from the environment / `.env` (see [Configuration](#configuration)). Logs go to
stderr (so JSON on stdout stays clean); pass `-v`/`-vv` to raise verbosity.

```sh
# Generate a question set for a task (reuse it across many outputs).
bineval generate --task task.txt --out qs.json

# Evaluate an output against it.
bineval evaluate --questions qs.json --source article.txt --output summary.txt > report.json

# Or do both in one go, with verbose logging.
bineval -v run --task task.txt --source article.txt --output summary.txt > report.json
```

## Configuration

BinEval uses **two** LMs — a *generator* (question generation) and an *evaluator* (answering) — both
configured from the environment. `BinEval::from_env()` loads a `.env` file (real environment
variables take precedence); each role reads `BINEVAL_<ROLE>_<KEY>` and falls back to the shared
`BINEVAL_<KEY>` (roles: `GENERATOR`, `EVALUATOR`):

| Variable | Default | Notes |
| --- | --- | --- |
| `BINEVAL_MODEL` | `openai:gpt-4o-mini` | `provider:model`; per-role: `BINEVAL_GENERATOR_MODEL`, `BINEVAL_EVALUATOR_MODEL` |
| `BINEVAL_TEMPERATURE` | `0.0` | per-role overridable |
| `BINEVAL_MAX_TOKENS` | `4096` | per-role overridable |
| `BINEVAL_BASE_URL` | — | OpenAI-compatible endpoints (server must support tool-calling) |
| `BINEVAL_API_KEY` | — | explicit key; else the provider's standard key (`OPENAI_API_KEY`, …) |

Provider API keys (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are read automatically once `.env` is
loaded. See [`.env.example`](.env.example). To use a different model per role, e.g. a stronger
generator and a cheaper evaluator, set `BINEVAL_GENERATOR_MODEL` and `BINEVAL_EVALUATOR_MODEL`.

Logging uses [`tracing`](https://crates.io/crates/tracing); set `RUST_LOG=bineval=debug` for full
control over what the CLI emits.

## Roadmap

- **v1:** requirement extraction → binary questions → answering → scoring (library, CLI, example).
- **Later:** per-dimension scoring, cross-model prompt update, self prompt update, benchmark harness.

See [docs/prd.md](docs/prd.md) for the original specification, design decisions, and rationale.

## References

- Paper: [arXiv 2606.27226](https://arxiv.org/abs/2606.27226)
- Rig: [crate](https://crates.io/crates/rig-core) · [repo](https://github.com/0xPlaygrounds/rig)

## License

Licensed under the [MIT License](LICENSE) © 2026 Charath Ranganathan ([charath@pragmatik.tech](mailto:charath@pragmatik.tech)).
