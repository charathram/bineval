# bineval

An interpretable LLM-evaluation library for Rust — an implementation of **BinEval** built on
**DSRs** ([`dspy-rs`](https://crates.io/crates/dspy-rs), the Rust rewrite of DSPy).

> **Ask, don't judge.** Instead of squeezing an LLM's quality into one opaque scalar, BinEval
> decomposes an evaluation task into atomic **yes/no questions**, answers each independently, and
> aggregates the verdicts into per-dimension and overall scores — each backed by a natural-language
> explanation. The result is evaluation you can **inspect, debug, and act on**.

Based on *Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement*
(Cho et al., [arXiv 2606.27226](https://arxiv.org/abs/2606.27226)).

📄 **Full design & requirements: [docs/prd.md](docs/prd.md).**

---

## Status

🚧 **v1 in development.** Scope: the core of the method — **binary question generation** and
**binary evaluation & scoring**. The paper's prompt-optimization loops (cross-model and self
update) and benchmark reproduction are **out of scope for v1** (see the
[PRD](docs/prd.md#6-scope-v1)).

## How it works

1. **Generate** a reusable `QuestionSet` from a task prompt `T` — summarize `T` into requirements,
   then decompose them into binary questions organized by dimension (default:
   *coherence, consistency, fluency, relevance*; fully configurable).
2. **Evaluate** any `(source, output)` pair against that `QuestionSet` — each question is answered
   independently (`yes` = requirement satisfied), and the verdicts aggregate into per-dimension and
   overall scores in `[0, 1]` (optionally rescaled, e.g. to `1–5`).

A `QuestionSet` is generated once and reused across many outputs. All LLM logic is expressed through
DSRs Signatures and Modules — there are no hand-authored prompt strings.

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
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Loads `.env` and builds the generator + evaluator LMs from BINEVAL_* variables.
    let be = BinEval::from_env().await?;

    // Phase 1 — once per task: produce a reusable, serializable QuestionSet.
    let questions = be.generate("Write a faithful, concise summary of a news article.").await?;

    // Phase 2 — per output: score a candidate against the source.
    let report = be.evaluate(&questions, source_article, candidate_summary).await?;

    println!("overall: {:.2}", report.overall);
    for d in &report.per_dimension {
        println!("{}: {:.2} ({}/{} answered)", d.dimension, d.score, d.answered, d.intended);
    }
    Ok(())
}
```

Prefer to wire models yourself? Build them explicitly (`LM` is re-exported from `dspy-rs`):

```rust
use bineval::{BinEval, LM};

let lm = LM::builder().model("openai:gpt-4o-mini".to_string()).temperature(0.0).build().await?;
let be = BinEval::builder().generator_lm(lm).build()?;   // evaluator defaults to the generator
```

Every score is backed by per-question verdicts and explanations in `report.per_question`, so you can
see exactly *why* an output scored the way it did.

## CLI

Models and keys come from the environment / `.env` (see [Configuration](#configuration)).

```sh
# Generate a question set for a task (reuse it across many outputs).
bineval generate --task task.txt --dims coherence,consistency,fluency,relevance > qs.json

# Evaluate an output against it (optionally rescale scores to 1–5).
bineval evaluate --questions qs.json --source article.txt --output summary.txt --rescale 1,5 > report.json
```

## Configuration

BinEval uses **two** LMs — a *generator* (question generation) and an *evaluator* (`f_E`) — both
configured from the environment. `BinEval::from_env()` and `bineval::lms_from_env()` load a `.env`
file (real environment variables take precedence) and read these variables; each role reads
`BINEVAL_<ROLE>_<KEY>` and falls back to the shared `BINEVAL_<KEY>` (roles: `GENERATOR`, `EVALUATOR`):

| Variable | Default | Notes |
| --- | --- | --- |
| `BINEVAL_MODEL` | `openai:gpt-4o-mini` | `provider:model`; per-role: `BINEVAL_GENERATOR_MODEL`, `BINEVAL_EVALUATOR_MODEL` |
| `BINEVAL_TEMPERATURE` | `0.0` | per-role overridable |
| `BINEVAL_MAX_TOKENS` | `4096` | per-role overridable |
| `BINEVAL_BASE_URL` | — | OpenAI-compatible / local servers (vLLM, Ollama) |
| `BINEVAL_API_KEY` | — | explicit key; else the provider's standard key (`OPENAI_API_KEY`, …) |
| `BINEVAL_DIMENSIONS` | the four | comma-separated (read by `from_env`) |
| `BINEVAL_CONCURRENCY` / `BINEVAL_MAX_RETRIES` / `BINEVAL_STRICT` | `8` / `3` / `false` | read by `from_env` |

Provider API keys (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are read automatically once `.env` is
loaded. See [`.env.example`](.env.example). To use a different model per role, e.g. a stronger
generator and a cheaper evaluator, set `BINEVAL_GENERATOR_MODEL` and `BINEVAL_EVALUATOR_MODEL`.

## Roadmap

- **v1:** question generation + evaluation/scoring (library, CLI, examples).
- **Later:** cross-model prompt update, self prompt update, optional benchmark-reproduction harness.

See [docs/prd.md](docs/prd.md) for the complete specification, design decisions, and rationale.

## References

- Paper: [arXiv 2606.27226](https://arxiv.org/abs/2606.27226)
- DSRs: [docs](https://dsrs.herumbshandilya.com/) · [crate](https://crates.io/crates/dspy-rs) · [repo](https://github.com/krypticmouse/DSRs)

## License

Licensed under the [MIT License](LICENSE) © 2026 Charath Ranganathan ([charath@pragmatik.tech](mailto:charath@pragmatik.tech)).
