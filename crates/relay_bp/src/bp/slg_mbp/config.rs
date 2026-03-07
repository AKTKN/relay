use crate::bp::min_sum::MinSumDecoderConfig;
use ndarray::Array1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitPerturbationMode {
    AdditiveGaussian,
    MultiplicativeUniform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode {
    Weighted,
    Tournament,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightedSelectionMode {
    Softmax,
    Rank,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GammaMode {
    Fixed,
    IntervalRandomPerGeneration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitStrategy {
    MinSum,
    MemBp,
}

#[derive(Clone, Debug)]
pub struct SLGMBPDecoderConfig {
    pub ensemble_size: usize,
    pub t_ms: usize,
    pub t_mem: usize,
    pub g_max: usize,
    pub sigma2: f64,
    pub delta: f64,
    pub fitness_alpha: f64,
    pub fitness_beta: f64,
    pub eta: f64,
    pub mutation_rate: f64,
    pub mutation_llr_abs_threshold: f64,
    pub elite_count: usize,
    pub tournament_size: usize,
    pub selection_mode: SelectionMode,
    pub weighted_selection_mode: WeightedSelectionMode,
    pub init_perturbation_mode: InitPerturbationMode,
    pub gamma_mode: GammaMode,
    pub gamma_fixed: f64,
    pub gamma_interval: (f64, f64),
    pub init_strategy: InitStrategy,
    pub init_gamma: f64,
    pub seed: u64,
}

impl Default for SLGMBPDecoderConfig {
    fn default() -> Self {
        Self {
            ensemble_size: 64,
            t_ms: 16,
            t_mem: 16,
            g_max: 30,
            sigma2: 0.15,
            delta: 0.2,
            fitness_alpha: 1000.0,
            fitness_beta: 1.0,
            eta: 1.0,
            mutation_rate: 0.02,
            mutation_llr_abs_threshold: 0.25,
            elite_count: 2,
            tournament_size: 3,
            selection_mode: SelectionMode::Weighted,
            weighted_selection_mode: WeightedSelectionMode::Softmax,
            init_perturbation_mode: InitPerturbationMode::AdditiveGaussian,
            gamma_mode: GammaMode::Fixed,
            gamma_fixed: 0.125,
            gamma_interval: (0.0, 0.25),
            init_strategy: InitStrategy::MinSum,
            init_gamma: 0.125,
            seed: 0,
        }
    }
}

impl SLGMBPDecoderConfig {
    pub fn min_sum_template_from_priors(&self, priors: Array1<f64>, max_iter: usize) -> MinSumDecoderConfig {
        MinSumDecoderConfig {
            error_priors: priors,
            max_iter,
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            gamma0: None,
            data_scale_value: None,
            max_data_value: None,
            int_bits: None,
            frac_bits: None,
        }
    }
}
