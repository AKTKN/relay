use crate::decoder::Bit;
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct DisorderedBPLegTrace {
    pub success: bool,
    pub iterations: usize,
    pub negative_llr_count: usize,
    pub decoding: Array1<Bit>,
    pub posterior: Array1<f64>,
    pub alpha_mean: f64,
    pub gamma_mean: f64,
    pub bias_mean: f64,
    pub bias_applied_count: usize,
}
