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
    alpha: f64,
    beta: f64,
) -> f64 {
    -(alpha * residual_w as f64) + beta * llr_cumsum_abs_sum
}

pub fn fitness_from_final_marginal(
    residual_w: usize,
    final_marginal: &Array1<f64>,
    alpha: f64,
    beta: f64,
) -> f64 {
    let confidence = final_marginal.iter().map(|v| v.abs()).sum::<f64>();
    -(alpha * residual_w as f64) + beta * confidence
}
