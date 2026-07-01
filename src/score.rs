//! Pure scoring — no I/O, no LLM, fully deterministic.

use crate::types::Answer;

/// The overall score: the fraction of answers whose verdict is "yes" (`true`), in `[0, 1]`.
///
/// Returns `0.0` when there are no answers.
pub fn fraction_yes(answers: &[Answer]) -> f32 {
    if answers.is_empty() {
        return 0.0;
    }
    let yes = answers.iter().filter(|a| a.verdict).count();
    yes as f32 / answers.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(verdict: bool) -> Answer {
        Answer {
            requirement: "r".into(),
            question: "q".into(),
            verdict,
            reasoning: String::new(),
        }
    }

    fn approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn empty_is_zero() {
        approx(fraction_yes(&[]), 0.0);
    }

    #[test]
    fn all_yes_is_one() {
        approx(fraction_yes(&[answer(true), answer(true)]), 1.0);
    }

    #[test]
    fn half_yes() {
        approx(
            fraction_yes(&[answer(true), answer(false), answer(true), answer(false)]),
            0.5,
        );
    }
}
