//! Pure scoring and aggregation — no I/O, no LLM, fully deterministic.
//!
//! Implements the paper's scoring (§3.2): per-dimension and overall means of binary verdicts,
//! plus the optional affine rescale `S' = S·(b − a) + a`.

use crate::types::{DimensionScore, EvalReport, QuestionOutcome, QuestionVerdict};

/// Aggregate per-question verdicts into per-dimension and overall scores.
///
/// - `dimensions` fixes the output order of [`EvalReport::per_dimension`]; any dimension that
///   appears in `verdicts` but not in `dimensions` is appended afterward (nothing is silently dropped).
/// - A dimension's score is the mean of satisfied verdicts over its **answered** questions; the
///   `intended` count includes failed questions so partial coverage is visible.
/// - Dimensions with zero answered questions are omitted from `per_dimension`.
/// - `overall` is the mean of satisfied verdicts over all answered questions, or `0.0` if none.
pub fn aggregate(dimensions: &[String], verdicts: Vec<QuestionVerdict>) -> EvalReport {
    // Output order: configured dimensions first, then any extras encountered in the verdicts.
    let mut order: Vec<String> = dimensions.to_vec();
    for verdict in &verdicts {
        if !order.iter().any(|d| d == &verdict.dimension) {
            order.push(verdict.dimension.clone());
        }
    }

    let mut per_dimension = Vec::new();
    for dim in &order {
        let mut intended = 0usize;
        let mut answered = 0usize;
        let mut satisfied = 0usize;
        for verdict in &verdicts {
            if &verdict.dimension != dim {
                continue;
            }
            intended += 1;
            if let QuestionOutcome::Answered { satisfied: s, .. } = &verdict.outcome {
                answered += 1;
                if *s {
                    satisfied += 1;
                }
            }
        }
        if answered > 0 {
            per_dimension.push(DimensionScore {
                dimension: dim.clone(),
                score: satisfied as f32 / answered as f32,
                answered,
                intended,
            });
        }
    }

    let mut total_answered = 0usize;
    let mut total_satisfied = 0usize;
    for verdict in &verdicts {
        if let QuestionOutcome::Answered { satisfied: s, .. } = &verdict.outcome {
            total_answered += 1;
            if *s {
                total_satisfied += 1;
            }
        }
    }
    let overall = if total_answered > 0 {
        total_satisfied as f32 / total_answered as f32
    } else {
        0.0
    };

    EvalReport {
        per_question: verdicts,
        per_dimension,
        overall,
    }
}

impl EvalReport {
    /// Affine-rescale all scores from `[0, 1]` to `[a, b]`: `S' = S·(b − a) + a` (paper §3.2).
    ///
    /// Per-question verdicts and coverage counts are unchanged.
    pub fn rescaled(&self, a: f32, b: f32) -> EvalReport {
        let map = |s: f32| s * (b - a) + a;
        EvalReport {
            per_question: self.per_question.clone(),
            per_dimension: self
                .per_dimension
                .iter()
                .map(|d| DimensionScore {
                    dimension: d.dimension.clone(),
                    score: map(d.score),
                    answered: d.answered,
                    intended: d.intended,
                })
                .collect(),
            overall: map(self.overall),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> Vec<String> {
        vec!["coherence".into(), "consistency".into()]
    }

    fn approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn basic_aggregation() {
        let verdicts = vec![
            QuestionVerdict::answered("q1", "coherence", true, ""),
            QuestionVerdict::answered("q2", "coherence", false, ""),
            QuestionVerdict::answered("q3", "consistency", true, ""),
            QuestionVerdict::answered("q4", "consistency", true, ""),
        ];
        let report = aggregate(&dims(), verdicts);

        approx(report.overall, 0.75);
        assert_eq!(report.per_dimension.len(), 2);

        let coherence = &report.per_dimension[0];
        assert_eq!(coherence.dimension, "coherence");
        approx(coherence.score, 0.5);
        assert_eq!((coherence.answered, coherence.intended), (2, 2));

        let consistency = &report.per_dimension[1];
        assert_eq!(consistency.dimension, "consistency");
        approx(consistency.score, 1.0);
        assert_eq!((consistency.answered, consistency.intended), (2, 2));
    }

    #[test]
    fn failed_questions_are_excluded_but_counted_as_intended() {
        let verdicts = vec![
            QuestionVerdict::answered("q1", "coherence", true, ""),
            QuestionVerdict::failed("q2", "coherence", "Lm", "timeout"),
        ];
        let report = aggregate(&dims(), verdicts);

        let coherence = &report.per_dimension[0];
        assert_eq!((coherence.answered, coherence.intended), (1, 2));
        approx(coherence.score, 1.0); // only the answered question counts toward the score
        approx(report.overall, 1.0);
    }

    #[test]
    fn dimension_with_no_answered_questions_is_omitted() {
        let verdicts = vec![QuestionVerdict::failed(
            "q1",
            "consistency",
            "Lm",
            "timeout",
        )];
        let report = aggregate(&dims(), verdicts);

        assert!(report.per_dimension.is_empty());
        approx(report.overall, 0.0);
    }

    #[test]
    fn no_answered_questions_yields_zero_overall() {
        let report = aggregate(&dims(), vec![]);
        approx(report.overall, 0.0);
        assert!(report.per_dimension.is_empty());
        assert!(report.per_question.is_empty());
    }

    #[test]
    fn ordering_is_configured_dimensions_then_extras() {
        let verdicts = vec![
            QuestionVerdict::answered("q1", "relevance", true, ""), // not in configured dims
            QuestionVerdict::answered("q2", "coherence", true, ""),
        ];
        let report = aggregate(&dims(), verdicts);

        let names: Vec<&str> = report
            .per_dimension
            .iter()
            .map(|d| d.dimension.as_str())
            .collect();
        // "consistency" has no verdicts (omitted); configured "coherence" precedes extra "relevance".
        assert_eq!(names, vec!["coherence", "relevance"]);
    }

    #[test]
    fn rescale_maps_zero_one_to_one_five() {
        let verdicts = vec![
            QuestionVerdict::answered("q1", "coherence", true, ""),
            QuestionVerdict::answered("q2", "coherence", false, ""),
        ]; // overall = 0.5
        let report = aggregate(&dims(), verdicts).rescaled(1.0, 5.0);

        approx(report.overall, 3.0); // 0.5 * (5 - 1) + 1
        approx(report.per_dimension[0].score, 3.0);
        // Coverage is preserved through rescaling.
        assert_eq!(report.per_dimension[0].answered, 2);
    }
}
