# BinEval — Product Requirements Document

**Project:** `bineval` — a Rust implementation of the BinEval algorithm, built on Rig (`rig-core`)
**Status:** Draft / v1 in design
**Last updated:** 2026-07-01

---

> ## ⚠️ v1 was simplified and re-based onto Rig (2026-07-01)
>
> The shipped v1 is a **minimal core** that diverges from the design below (which was written against
> DSRs/`dspy-rs`). This document is retained for its background, algorithm summary, and rationale, but
> the implementation now uses **`rig-core` directly**. Concretely:
>
> - **No DSRs.** We dropped `dspy-rs`: on 0.7.3 its `ChatAdapter` parses typed outputs with
>   `serde_json::from_str(..).unwrap()` and **panicked** when gpt-4o-mini echoed the injected JSON
>   schema before the array; the BAML fix exists only on unreleased `main` (vendored BAML + forked
>   `facet`/`minijinja` git deps) — too unstable. BAML's `jsonish` parser isn't usable standalone.
> - **Rig `Extractor` for structured output.** Each step defines an output struct deriving
>   `serde` + `schemars::JsonSchema`, and Rig's `Extractor` (tool-calling) coerces the model response
>   into it — robust, no hand-rolled JSON parsing, and malformed output errors rather than panics.
>   Requires provider **function/tool-calling** (OpenAI/Anthropic ✓; tool-less local servers are not
>   supported). schemars field doc-comments carry per-field intent to the model.
> - **Hand-authored preambles.** With DSRs gone, the three operations use short preamble strings
>   (this reverses the original "no hand-authored prompts via Signatures" goal — a deliberate trade
>   for stability and control).
> - **No dimensions.** Requirements and questions are a flat list; the report has a single overall
>   `score` (fraction of "yes"), no per-dimension breakdown and no affine rescale.
> - **Per-requirement decomposition**, run concurrently (`futures::buffered`).
> - **`anyhow` everywhere** instead of a custom `thiserror` enum.
> - **Two LMs retained** (generator + evaluator) behind a small provider enum (`openai`/`anthropic`),
>   built via `BinEval::from_env` / `BinEval::new` / `Lm::new`.
> - **CLI** has a `run` subcommand and `tracing`-based logging (`-v`/`-vv`, `RUST_LOG`).
>
> Current source layout: `src/llm.rs` (Rig `Lm` wrapper + `Extractor` calls + output DTOs +
> preambles), `src/lib.rs` (`BinEval` orchestration + env config), `src/types.rs`, `src/score.rs`,
> `src/bin/bineval.rs`, `examples/end_to_end.rs`, `tests/live.rs`. Per-dimension scoring and the
> prompt-optimization loops remain future work.

---

## 1. Overview

`bineval` is a reusable Rust crate that implements **BinEval**, the interpretable LLM-evaluation
method from *"Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and
Self-Improvement"* (Cho et al., arXiv 2606.27226). It is built on **DSRs** (the
[`dspy-rs`](https://crates.io/crates/dspy-rs) framework, a Rust rewrite of DSPy).

Instead of asking a model for one opaque scalar judgment ("rate this summary 1–5"), BinEval
**decomposes** an evaluation task into atomic yes/no questions, answers each one **independently**,
and **aggregates** the verdicts into per-dimension and overall scores. Every score is grounded in
the individual binary verdicts and their natural-language explanations, making evaluation
**inspectable, debuggable, and actionable**.

This document specifies **v1**, which delivers the irreducible core of the method: binary question
generation and binary evaluation/scoring.

---

## 2. Background & motivation

Evaluating LLM outputs is a bottleneck: human evaluation is slow and expensive; lexical metrics
(ROUGE/BLEU/BERTScore) correlate poorly with human judgment on open-ended generation; and holistic
"LLM-as-judge" approaches return opaque scores that are hard to debug. A single scalar is often
insufficient — if a summary gets a mediocre rating, it is unclear *why* (factual error? weak
relevance? poor fluency?).

BinEval's premise: **ask, don't judge.** Decompose each evaluation criterion into atomic,
checkable yes/no questions and aggregate the verdicts. This turns evaluation from a black-box
verdict into a structured diagnostic signal. The paper reports that BinEval matches or outperforms
strong baselines (UniEval, G-Eval, QAGS) on SummEval, Topical-Chat, and QAGS — with especially
strong results on factual consistency — while producing scores a practitioner can actually inspect.

The paper has four components:

1. **Binary question generation** — a meta-prompt turns a task prompt `T` into binary questions.
2. **Binary evaluation & scoring** — an evaluator answers each question and the verdicts are aggregated.
3. **Cross-model prompt update** — align a target evaluator's prompt to a stronger source evaluator.
4. **Self prompt update** — improve a *generator* using its own evaluation failures as feedback.

`bineval` v1 implements **(1) and (2)**. See §6 (Scope) and §13 (Roadmap).

---

## 3. Goals & non-goals

### Goals
- A **reusable, provider-agnostic Rust crate** exposing a clean public API for interpretable LLM evaluation.
- **Task-agnostic** evaluation with **caller-configurable dimensions** (default: the paper's
  coherence / consistency / fluency / relevance).
- **Inspectable, debuggable output**: every score is backed by per-question verdicts and explanations.
- A **two-phase** workflow with a reusable, serializable `QuestionSet` artifact (generate once, evaluate many).
- **All LLM logic expressed through DSRs** Signatures and Modules — no hand-authored prompt strings.
- **Resilient** at scale (bounded retry, non-fatal per-question failures, visible coverage).
- **Testable** without network access (LLM operations sit behind an internal trait seam).
- Shipped as a **library + CLI + runnable examples**.

### Non-goals (v1)
- Components (3) cross-model prompt update and (4) self prompt update — deferred; architecture leaves seams.
- **Reproducing the paper's benchmark numbers** (SummEval / Topical-Chat / QAGS / IFBench, and the
  Spearman / Kendall / Pearson correlation harness) — a separate effort, out of scope.
- Hosting/serving, model fine-tuning, or a GUI.
- Bundling datasets.

---

## 4. Users & use cases

- **Rust developers building LLM pipelines** who need an interpretable, programmatic evaluator for
  generated text (summaries, dialogue responses, instruction-following outputs).
- **Researchers / practitioners** who want per-question, per-dimension diagnostics rather than a
  single opaque score, and who want to inspect or hand-edit the question set.
- **CLI users** who want to generate a question set and score outputs from the shell, emitting JSON.

Representative flow: generate a `QuestionSet` once for a task ("summarize a news article well"),
then evaluate thousands of candidate `(source, output)` pairs against it, getting per-dimension
scores plus a per-question breakdown explaining each score.

---

## 5. Algorithm summary (as implemented in v1)

Notation follows the paper.

**Generation** — `𝒬 = ℱ(T; M)` depends only on the task prompt `T` (not on any individual input/output):
- **Step 1 — Summarize:** `T → R = {r_1, …, r_K}`, an explicit set of requirements, each tagged by dimension.
- **Step 2 — Decompose:** each `r_k →` one or more binary questions, phrased so that **"yes" = the
  requirement is satisfied**, each optionally paired with a concise violation example. Questions are
  organized by dimension: `𝒬 = ⋃_d 𝒬_d`.
- The decompose step sees **all** requirements at once (global view) to avoid redundancy and
  over-decomposition (the paper documents over-decomposition as a real failure mode that degraded
  relevance correlation).

**Evaluation & scoring** — for an evaluator `E`, source `x`, output `y`, and question `q_i`:
- `f_E(x, y, q_i) ∈ {0, 1}` (1 = "yes" = satisfied), with a natural-language explanation `e_i`.
- Questions are answered **independently** (this independence underpins the paper's variance-reduction
  argument: aggregating N weakly-correlated binary classifiers reduces error ∝ 1/N).
- Per-dimension score: `S_d = (1/|𝒬_d|) Σ_{q_i ∈ 𝒬_d} f_E(x, y, q_i)`.
- Overall score: `S = (1/N) Σ_i f_E(x, y, q_i)`. Both in `[0, 1]`.
- Optional affine rescale to `[a, b]`: `S'(x, y) = S(x, y)·(b − a) + a` (§3.2 of the paper).

---

## 6. Scope (v1)

| In scope | Out of scope (deferred) |
|---|---|
| Binary question generation (summarize → global decompose) | Cross-model prompt update (component 3) |
| Binary evaluation (independent per-question) | Self prompt update (component 4) |
| Per-dimension + overall scoring, affine rescale | Benchmark reproduction & correlation metrics |
| Serializable `QuestionSet` and `EvalReport` | Optimizers (COPRO / MIPROv2 / GEPA) |
| Library + CLI + examples | Datasets, serving, GUI |

---

## 7. Design decisions & rationale

| Decision | Choice | Rationale |
|---|---|---|
| Goal | Reusable crate | Not a paper-reproduction harness; usable by others. |
| Scope | Components 1 + 2 | Smallest complete, useful unit; de-risks DSRs integration before the harder loops. |
| Generality | Task-agnostic; dimensions a configurable `Vec<String>`, default = paper's four | Matches the paper's "task-agnostic, training-free" claim; the only option consistent with "reusable". |
| Phasing | Two phases with a first-class, serializable `QuestionSet` | `𝒬` depends only on `T`; generate once, reuse across many outputs; enables inspection, caching, and caller-authored questions; sets up loops 3/4. |
| Generation shape | Two-step: Summarize, then a **single global Decompose** call | Preserves the summarize step the paper says matters; global view guards against over-decomposition; provenance via validated `requirement_id`. |
| `QuestionSet` richness | Rich + traceable; keeps requirements `R` and question→requirement links; `violation_example` optional | Embodies "debuggable"; required by loops 3/4 later; some requirements need no negative exemplar. |
| Evaluation execution | Per-question **independent** calls, run **concurrently** with a configurable cap | Faithful to the independence assumption behind the 1/N variance-reduction result; clean per-question explanations; natural DSRs pattern. |
| Model wiring | Crate-owned, **per-role** LMs via builder (`generator_lm`, optional `evaluator_lm`) | Avoids hidden global state; supports different models per role; extends to loops 3/4 (source/target/note-taker/updater). |
| Report | Rich + eager: per-question verdict + explanation, per-dimension score + coverage, overall; `rescaled(a,b)` method | Makes interpretability concrete; canonical `[0,1]` scores with on-demand rescale. |
| Failure handling | Resilient + bounded retry; unrecovered failures recorded **non-fatally**; coverage surfaced; optional `strict` mode | N×M LLM calls make transient failures common; DSRs ships no retry; partial scores must never masquerade as complete. |
| LLM logic | **All via DSRs Signatures/Modules — no hand-authored prompt strings.** CoT for reasoning tasks, Predict otherwise, custom `Module` for composition | Idiomatic DSRs/DSPy; intent carried by typed signatures; prompts rendered by the adapter. |
| Testing | Internal `LmOps` trait seam (`DsrsOps` prod, `FakeOps` tests); pure logic unit-tested; live tests env-gated | Fast, deterministic, offline tests; keeps DSRs at the edges. |
| Packaging | Library + `bineval` CLI + `examples/` | Library is the deliverable; CLI serves non-Rust users; examples double as docs and smoke tests. |

---

## 8. Domain model

All types are `serde`-serializable.

- **`Requirement`** `{ id, dimension, text }` — an explicit requirement extracted from `T`.
- **`Question`** `{ id, dimension, text, violation_example: Option<String>, requirement_id }` —
  a binary question; `text` is phrased so "yes" = satisfied; `requirement_id` links to provenance.
- **`QuestionSet`** `{ task_prompt, dimensions, requirements, questions }` — the reusable artifact
  produced by generation; can also be hand-authored or loaded from JSON.
- **`QuestionOutcome`** — `Answered { satisfied: bool, explanation: String }` or
  `Failed { class: String, message: String }` (failures are non-fatal).
- **`QuestionVerdict`** `{ question_id, dimension, outcome }`.
- **`DimensionScore`** `{ dimension, score, answered, intended }` — `answered`/`intended` expose coverage.
- **`EvalReport`** `{ per_question, per_dimension, overall }` with method `rescaled(a, b)`.

`DEFAULT_DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]` is exposed as a constant.

---

## 9. Public API

```rust
let be = BinEval::builder()
    .generator_lm(gen_lm)   // required — LM used for question generation
    .evaluator_lm(eval_lm)  // optional — LM used for f_E; defaults to generator_lm
    .dimensions(dims)       // optional — defaults to DEFAULT_DIMENSIONS
    .concurrency(8)         // optional — max in-flight question evaluations
    .max_retries(3)         // optional — bounded retry w/ exp backoff on retryable errors
    .strict(false)          // optional — true => first unrecovered question failure aborts
    .build();

// Phase 1 — runs once per task; reusable, serializable artifact.
let qs: QuestionSet = be.generate(task_prompt).await?;

// Phase 2 — runs per (source, output) pair.
let report: EvalReport = be.evaluate(&qs, source_x, output_y).await?;

// Convenience — evaluate many outputs against one question set.
let reports = be.evaluate_many(&qs, &pairs).await; // Vec<Result<EvalReport, BinEvalError>>
```

Callers may construct a `QuestionSet` by hand or load it from JSON and skip `generate` entirely.

---

## 10. DSRs integration approach

**Principle: no hand-authored prompt strings.** All LLM logic is expressed as DSRs Signatures
(`#[Signature(...)]` structs whose docstrings and `#[input/output(desc=...)]` carry the intent);
DSRs' `ChatAdapter` renders the actual prompts. Reasoning tasks use `#[Signature(cot)]` (DSRs' CoT
mode), which auto-adds a `reasoning` output field; non-reasoning tasks would use a plain
`#[Signature]`. Modules are `Predict::new(Sig::new())`, invoked per role.

Signatures (in `src/lm.rs`):

- `SummarizeSig` — `(task_prompt, dimensions) → requirements` — `#[Signature(cot)]`.
- `DecomposeSig` — `(task_prompt, dimensions, requirements) → questions` — `#[Signature(cot)]`,
  one global call over all requirements.
- `BinaryEvalSig` — `(source, output, question, violation_example) → verdict` — `#[Signature(cot)]`;
  the auto-generated **`reasoning` field is the explanation `e_i`** we store.

Orchestration:

- **Generation** (`src/generate.rs`) is a plain async function composing `SummarizeSig` then
  `DecomposeSig`, assigning stable ids, and validating LLM-assigned `requirement_id`s against `R`
  (dropping strays) before assembling the `QuestionSet`.
- **Evaluation** (`src/evaluate.rs`) drives `BinaryEvalSig` once per question, concurrently.

> **Reality-driven deviations from the initial plan** (the real `dspy-rs` 0.7.3 API differs from
> early assumptions; verified against the crate source):
> - **`Predictor::forward_with_config(Example, Arc<LM>)`** is used for every call, supplying the
>   per-role LM explicitly and bypassing the global `configure` singleton. (A DSRs `Module` was *not*
>   used for composition: `Module::forward` only reads the global LM, and `Module::batch` fails fast
>   on the first error — neither fits per-role models or resilient evaluation.)
> - **All signature outputs are `String`, parsed by us.** DSRs' adapter parses any non-`String`
>   output with `serde_json::from_str(..).unwrap()`, which *panics* on malformed model output. We
>   instead emit/parse JSON ourselves, turning bad output into a retryable `BinEvalError::Parse` —
>   required to honor the resilient-failure contract. Code fences are stripped; verdicts parsed leniently.
> - **`Example`/`Prediction` are dynamic `HashMap<String, Value>`**, accessed by key.

Per-role LMs are built by the caller via `LM::builder().model("provider:model").temperature(0.0)
.max_tokens(..).build().await?` and handed to the builder; `evaluator_lm` defaults to `generator_lm`.
DSRs supports OpenAI, Anthropic, Gemini, Groq, OpenRouter, and Ollama; API keys come from the environment.

**Testability seam.** An internal `LmOps` trait (`src/lm.rs`) abstracts the three LLM operations
(`summarize`, `decompose`, `eval_question`). `DsrsOps` is the production implementation; tests supply
deterministic fakes. The generation/evaluation logic is generic over `LmOps`, so all assembly,
validation, scoring, retry, and concurrency logic is unit-tested with zero network.

---

## 11. Flows

- **Generation** — `Summarize` → validate/assemble requirements → `Decompose` (single global call)
  → validate `requirement_id`s and assign question ids → `QuestionSet`. Temperature 0. Runs once per task.
- **Evaluation** — build one `BinaryEval` input per question; drive `LmOps::eval_question`
  concurrently (`futures::stream::buffer_unordered(concurrency)`), each wrapped in bounded retry
  (retry when `PredictError::is_retryable()`, exponential backoff). Outcomes preserve question order.
  With `strict = true`, the first unrecovered failure returns `Err`; otherwise failures are recorded
  as `QuestionOutcome::Failed`.
- **Scoring** (pure) — `satisfied → 1.0/0.0`; `S_d` = mean over **answered** questions in the
  dimension (a dimension with zero answered questions is omitted); `overall` = mean over all
  answered questions. Per-dimension `answered` vs `intended` make partial coverage visible.
  `rescaled(a, b)` applies the §3.2 affine map.

---

## 12. Cross-cutting concerns

- **Error model** — `BinEvalError` (via `thiserror`): `Lm` (the underlying DSRs call returns
  `anyhow::Error`, whose message we wrap), `Parse` / `MissingField` (our own JSON/field parsing),
  `Config`, `StrictFailure`, and `Io` / `Serde` (CLI). `is_retryable()` is true for `Lm` / `Parse` /
  `MissingField`; `class()` gives the short label recorded on a failed verdict.
- **Concurrency** — bounded by the configurable `concurrency` cap; generation is a small fixed
  number of calls, evaluation is the N-per-output fan-out.
- **Determinism** — temperature 0 by default to reduce run-to-run variance.
- **Async** — the public API is async; callers bring their own `tokio` runtime.
- **Serialization** — `QuestionSet` and `EvalReport` round-trip through JSON (`serde_json`).
- **Configuration from `.env`** — `BinEval::from_env()` and `lms_from_env()` load a `.env` file
  (`dotenvy`; real environment variables take precedence) and build **both** LMs — generator and
  evaluator — from `BINEVAL_*` variables. Each role reads `BINEVAL_<GENERATOR|EVALUATOR>_<KEY>` with
  a shared `BINEVAL_<KEY>` fallback (`MODEL` / `TEMPERATURE` / `MAX_TOKENS` / `BASE_URL` / `API_KEY`);
  provider-standard keys (e.g. `OPENAI_API_KEY`) are read automatically. See the README's
  Configuration table and `.env.example`.

### CLI

- `bineval generate --task <file|-> [--dims a,b,c] [--model provider:model] > qs.json`
- `bineval evaluate --questions qs.json --source <file|-> --output <file|-> [--model ..] [--rescale a,b] > report.json`
- Reads/writes JSON (`QuestionSet`, `EvalReport`) and plain text; model and API key from flags/env.

### Testing strategy

- **Unit (no network, via `FakeOps`):** scoring math; empty / partial-coverage dimensions;
  `rescaled`; `requirement_id` validation/repair; retry (transient-then-success, exhausted);
  concurrency aggregation and ordering; `strict` vs resilient behavior.
- **Serde round-trip:** `QuestionSet`, `EvalReport`.
- **Live (env-gated; `#[ignore]` or a `live` feature; requires an API key):** real `generate` +
  `evaluate` end-to-end sanity — "yes = satisfied" semantics, scores in range, explanations present.

---

## 13. Dependencies & tooling

- `dspy-rs = "0.7.3"` — DSRs (Signatures, Modules, LM backends, adapters).
- `tokio` — async runtime (`macros`, `rt-multi-thread`, `time`).
- `serde`, `serde_json` — serialization.
- `futures` — bounded-concurrency `buffered`.
- `thiserror` — error enum; `clap` (derive) for the CLI.
- `dotenvy` — load `.env` for `from_env` / `lms_from_env`.
- `anyhow` and `schemars` — **required by the `#[Signature]` macro's generated code** (it references
  `anyhow::Result` and `schemars::schema_for!`), not used directly by our logic.
- Edition 2024 (already set; implies Rust ≥ 1.85).

### File layout (as built)

```
src/lib.rs            public API: BinEval, builder, Config, re-exports (incl. LM)
src/types.rs          domain types (serde)
src/generate.rs       two-step generation (summarize → global decompose) + validation
src/evaluate.rs       per-question concurrency + retry + aggregation
src/score.rs          pure scoring/aggregation/rescale
src/lm.rs             LmOps trait + DSRs signatures + DsrsOps + parsing helpers
src/error.rs          BinEvalError
src/bin/bineval.rs    CLI
examples/{generate,evaluate}.rs
tests/live.rs         env-gated live end-to-end (#[ignore])
docs/prd.md           this document
```

---

## 14. Milestones / roadmap

- **M0 — Docs:** this PRD + `README.md`.
- **M1 — Core types + scoring:** domain types, pure scoring/aggregation/rescale, unit tests.
- **M2 — DSRs integration:** `LmOps` trait, `DsrsOps`, generation and evaluation flows.
- **M3 — CLI + examples.**
- **M4 — Tests + verification:** unit suite green offline; env-gated live end-to-end.
- **Future (post-v1):** component 3 (cross-model prompt update), component 4 (self prompt update),
  optional benchmark-reproduction harness and correlation metrics.

---

## 15. Open questions / risks

- **DSRs structured-output stability** — reliability of parsing typed `Vec<Question>` /
  `Verdict` outputs across providers; if shaky, add minimal format-only few-shot to the signatures.
- **`requirement_id` provenance** — depends on the model tagging questions correctly in the global
  decompose call; mitigated by validation/repair, but worth monitoring vs. a per-requirement fan-out.
- **Cost** — evaluation is N calls per output; the concurrency cap and temperature 0 help, but
  large question sets × many outputs can be expensive.
- **Over-decomposition** — the paper's documented relevance-degradation failure mode; the generation
  signatures must steer toward minimal, non-redundant questions.
- **DSRs API churn** — `dspy-rs` is young (0.7.x); pin the version and watch for breaking changes.

---

## 16. References

- Paper: *Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement*
  — Cho et al., arXiv 2606.27226.
- DSRs (`dspy-rs`): https://dsrs.herumbshandilya.com/ · https://crates.io/crates/dspy-rs ·
  https://github.com/krypticmouse/DSRs
