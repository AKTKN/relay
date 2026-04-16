use crate::bp::min_sum::MinSumDecoderConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingMode {
    Fixed,
    IntervalRandom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasApplyMode {
    All,
    Filter,
}

#[derive(Clone, Debug)]
pub struct DisorderedBPDecoderConfig {
    pub t_0: usize,
    pub maximum_leg: usize,
    pub iteration_per_leg: usize,
    pub initial_alpha: f64,
    pub alpha_mode: SamplingMode,
    pub alpha_fixed: f64,
    pub alpha_interval: (f64, f64),
    pub initial_gamma: f64,
    pub gamma_mode: SamplingMode,
    pub gamma_fixed: f64,
    pub gamma_interval: (f64, f64),
    pub bias_mode: SamplingMode,
    pub bias_fixed: f64,
    pub bias_interval: (f64, f64),
    pub bias_apply_mode: BiasApplyMode,
    pub bias_filter_threshold: f64,
    pub negative_sign_prob: f64,
    pub carry_marginal_factor: f64,
    pub seed: u64,
}

impl Default for DisorderedBPDecoderConfig {
    fn default() -> Self {
        Self {
            t_0: 80,
            maximum_leg: 100,
            iteration_per_leg: 60,
            initial_alpha: 0.625,
            alpha_mode: SamplingMode::IntervalRandom,
            alpha_fixed: 0.625,
            alpha_interval: (0.6, 0.7),
            initial_gamma: 0.125,
            gamma_mode: SamplingMode::IntervalRandom,
            gamma_fixed: 0.125,
            gamma_interval: (-0.24, 0.66),
            bias_mode: SamplingMode::Fixed,
            bias_fixed: 0.0,
            bias_interval: (0.0, 0.0),
            bias_apply_mode: BiasApplyMode::All,
            bias_filter_threshold: 1.0,
            negative_sign_prob: 0.0,
            carry_marginal_factor: 1.0,
            seed: 0,
        }
    }
}

impl DisorderedBPDecoderConfig {
    pub fn min_sum_template_from_priors(&self, priors: ndarray::Array1<f64>) -> MinSumDecoderConfig {
        MinSumDecoderConfig {
            error_priors: priors,
            max_iter: self.t_0 + self.maximum_leg.saturating_sub(1) * self.iteration_per_leg,
            alpha: Some(self.initial_alpha),
            alpha_iteration_scaling_factor: 1.0,
            gamma0: Some(self.initial_gamma),
            data_scale_value: None,
            max_data_value: None,
            int_bits: None,
            frac_bits: None,
            enable_variable_message_drop: false,
            drop_probability: 0.0,
            drop_llr_threshold: 0.0,
            rng_seed: Some(self.seed),
        }
    }
}
