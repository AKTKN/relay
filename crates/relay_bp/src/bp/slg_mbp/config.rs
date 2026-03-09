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
    IntervalRandomPerVariable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveMemoryMode {
    ProbabilisticFlip,
    DirectInterval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitStrategy {
    MinSum,
    MemBp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerturbationMethod {
    Fixed,
    Resample,
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
    pub fitness_low_llr_mu: f64,
    pub fitness_low_llr_threshold: f64,
    pub eta: f64,
    pub mutation_rate: f64,
    pub mutation_llr_abs_threshold: f64,
    pub elite_count: usize,
    pub sequential_mc: bool,
    pub tournament_size: usize,
    pub selection_mode: SelectionMode,
    pub weighted_selection_mode: WeightedSelectionMode,
    pub init_perturbation_mode: InitPerturbationMode,
    pub gamma_mode: GammaMode,
    pub gamma_fixed: f64,
    pub gamma_interval: (f64, f64),
    pub adaptive_memory: bool,
    pub adaptive_memory_zeta: f64,
    pub adaptive_memory_adjacent_gamma_interval: (f64, f64),
    pub adaptive_memory_mode: AdaptiveMemoryMode,
    pub init_strategy: InitStrategy,
    pub init_gamma: f64,
    pub adaptive_perturbation: bool,
    pub adaptive_perturbation_llr_threshold: f64,
    pub adaptive_perturbation_factor: f64,
    pub continue_perturbation: bool,
    pub perturbation_method: PerturbationMethod,
    pub reset_marginal: bool,
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
            fitness_low_llr_mu: 0.0,
            fitness_low_llr_threshold: 0.0,
            eta: 1.0,
            mutation_rate: 0.02,
            mutation_llr_abs_threshold: 0.25,
            elite_count: 2,
            sequential_mc: false,
            tournament_size: 3,
            selection_mode: SelectionMode::Weighted,
            weighted_selection_mode: WeightedSelectionMode::Softmax,
            init_perturbation_mode: InitPerturbationMode::AdditiveGaussian,
            gamma_mode: GammaMode::Fixed,
            gamma_fixed: 0.125,
            gamma_interval: (0.0, 0.25),
            adaptive_memory: false,
            adaptive_memory_zeta: 1.0,
            adaptive_memory_adjacent_gamma_interval: (0.0, 0.25),
            adaptive_memory_mode: AdaptiveMemoryMode::ProbabilisticFlip,
            init_strategy: InitStrategy::MinSum,
            init_gamma: 0.125,
            adaptive_perturbation: false,
            adaptive_perturbation_llr_threshold: 0.0,
            adaptive_perturbation_factor: 1.0,
            continue_perturbation: false,
            perturbation_method: PerturbationMethod::Fixed,
            reset_marginal: false,
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
