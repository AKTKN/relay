use ndarray::Array1;

use crate::decoder::Bit;

pub fn residual_weight(decoded_detectors: &Array1<Bit>, detectors: ndarray::ArrayView1<'_, Bit>) -> usize {
    decoded_detectors
        .iter()
        .zip(detectors.iter())
        .filter(|(a, b)| **a != **b)
        .count()
}

pub fn fitness_from_ms_cumsum(
    residual_w: usize,
    llr_cumsum_abs_sum: f64,
    final_marginal: &Array1<f64>,
    alpha: f64,
    beta: f64,
    low_llr_mu: f64,
    low_llr_threshold: f64,
) -> f64 {
    let low_llr_count = low_llr_count(final_marginal, low_llr_threshold);
    -(alpha * residual_w as f64) + beta * llr_cumsum_abs_sum + low_llr_mu * low_llr_count as f64
}

pub fn fitness_from_final_marginal(
    residual_w: usize,
    final_marginal: &Array1<f64>,
    alpha: f64,
    beta: f64,
    low_llr_mu: f64,
    low_llr_threshold: f64,
) -> f64 {
    let confidence = final_marginal.iter().map(|v| v.abs()).sum::<f64>();
    let low_llr_count = low_llr_count(final_marginal, low_llr_threshold);
    -(alpha * residual_w as f64) + beta * confidence + low_llr_mu * low_llr_count as f64
}

pub fn low_llr_count(final_marginal: &Array1<f64>, threshold: f64) -> usize {
    let threshold = threshold.abs();
    final_marginal
        .iter()
        .filter(|&&value| value.abs() <= threshold)
        .count()
}

#[cfg(test)]
mod tests {
    use super::{fitness_from_final_marginal, fitness_from_ms_cumsum, low_llr_count};
    use ndarray::array;

    #[test]
    fn counts_low_absolute_llrs_inclusively() {
        let posterior = array![-0.5, 0.0, 0.5000001, 1.2];
        assert_eq!(low_llr_count(&posterior, 0.5), 2);
    }

    #[test]
    fn final_marginal_fitness_adds_low_llr_term() {
        let posterior = array![0.1, -0.4, 2.0];
        let score = fitness_from_final_marginal(1, &posterior, 2.0, 1.0, 3.0, 0.5);
        assert!((score - 6.5).abs() < 1e-9);
    }

    #[test]
    fn ms_cumsum_fitness_adds_low_llr_term() {
        let posterior = array![0.1, -0.4, 2.0];
        let score = fitness_from_ms_cumsum(1, 5.0, &posterior, 2.0, 1.0, -3.0, 0.5);
        assert!((score - -3.0).abs() < 1e-9);
    }
}
