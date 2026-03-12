use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::slg_mbp::config::{
    AdaptiveMemoryMode, AdaptivePerturbationPriorBaseMode, AdaptivePerturbationSignMode,
    AdaptivePerturbationTarget, AdaptivePerturbationThresholdMode,
    AdaptivePerturbationVariableBaseMode, GammaMode, InitPerturbationMode, InitStrategy,
    PerturbationMethod, SLGMBPDecoderConfig,
    SelectionMode, WeightedSelectionMode,
};
use relay_bp::bp::slg_mbp::SLGMBPDecoder;
use relay_bp::decoder::Bit;

#[pyclass(extends=DynDecoder, subclass, module = "bp")]
#[allow(dead_code)]
pub struct SLGMBPDecoderF64 {}

#[pymethods]
impl SLGMBPDecoderF64 {
    #[new]
    #[pyo3(signature = (
        check_matrix,
        error_priors,
        alpha=None,
        alpha_iteration_scaling_factor=1.0,
        data_scale_value=None,
        max_data_value=None,
        ensemble_size=64,
        t_ms=16,
        t_mem=16,
        g_max=30,
        sigma2=0.15,
        delta=0.2,
        fitness_alpha=1000.0,
        fitness_beta=1.0,
        fitness_low_llr_mu=0.0,
        fitness_low_llr_threshold=0.0,
        eta=1.0,
        mutation_rate=0.02,
        mutation_llr_abs_threshold=0.25,
        elite_count=2,
        sequential_mc=false,
        tournament_size=3,
        init_perturbation_mode="gaussian".to_string(),
        selection_mode="weighted".to_string(),
        weighted_selection_mode="softmax".to_string(),
        init_strategy="min-sum".to_string(),
        init_gamma=0.125,
        adaptive_perturbation=false,
        adaptive_perturbation_llr_threshold_mode="constant".to_string(),
        adaptive_perturbation_llr_threshold=0.0,
        adaptive_perturbation_llr_threshold_min=0.0,
        adaptive_perturbation_llr_threshold_max=0.0,
        adaptive_perturbation_llr_threshold_factor=0.0,
        adaptive_perturbation_factor=1.0,
        adaptive_perturbation_sign_mode="random".to_string(),
        adaptive_perturbation_positive_sign_prob=0.5,
        adaptive_perturbation_target="prior".to_string(),
        adaptive_perturbation_prior_base_mode="initial".to_string(),
        adaptive_perturbation_variable_base_mode="previous_leg_posterior".to_string(),
        adaptive_perturbation_reset_on_threshold_exit=false,
        continue_perturbation=false,
        perturbation_method="fixed".to_string(),
        reset_marginal=false,
        marginal_carry_damping_factor=1.0,
        marginal_carry_llr_abs_threshold=-1.0,
        drop_p=0.0,
        drop_llr_threshold=0.0,
        gamma_mode="fixed".to_string(),
        gamma_fixed=0.125,
        gamma_interval=(0.0, 0.25),
        adaptive_memory=false,
        adaptive_memory_zeta=1.0,
        adaptive_memory_adjacent_gamma_interval=(0.0, 0.25),
        adaptive_memory_mode="probabilistic_flip".to_string(),
        biased_relay_mode=false,
        biased_relay_r_relay=10,
        biased_relay_maximum_round=10,
        biased_relay_t_0=80,
        biased_relay_r_relay_iter=60,
        seed=0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        check_matrix: &Bound<'_, PyAny>,
        error_priors: &Bound<'_, PyArray1<f64>>,
        alpha: Option<f64>,
        alpha_iteration_scaling_factor: f64,
        data_scale_value: Option<f64>,
        max_data_value: Option<f64>,
        ensemble_size: usize,
        t_ms: usize,
        t_mem: usize,
        g_max: usize,
        sigma2: f64,
        delta: f64,
        fitness_alpha: f64,
        fitness_beta: f64,
        fitness_low_llr_mu: f64,
        fitness_low_llr_threshold: f64,
        eta: f64,
        mutation_rate: f64,
        mutation_llr_abs_threshold: f64,
        elite_count: usize,
        sequential_mc: bool,
        tournament_size: usize,
        init_perturbation_mode: String,
        selection_mode: String,
        weighted_selection_mode: String,
        init_strategy: String,
        init_gamma: f64,
        adaptive_perturbation: bool,
        adaptive_perturbation_llr_threshold_mode: String,
        adaptive_perturbation_llr_threshold: f64,
        adaptive_perturbation_llr_threshold_min: f64,
        adaptive_perturbation_llr_threshold_max: f64,
        adaptive_perturbation_llr_threshold_factor: f64,
        adaptive_perturbation_factor: f64,
        adaptive_perturbation_sign_mode: String,
        adaptive_perturbation_positive_sign_prob: f64,
        adaptive_perturbation_target: String,
        adaptive_perturbation_prior_base_mode: String,
        adaptive_perturbation_variable_base_mode: String,
        adaptive_perturbation_reset_on_threshold_exit: bool,
        continue_perturbation: bool,
        perturbation_method: String,
        reset_marginal: bool,
        marginal_carry_damping_factor: f64,
        marginal_carry_llr_abs_threshold: f64,
        drop_p: f64,
        drop_llr_threshold: f64,
        gamma_mode: String,
        gamma_fixed: f64,
        gamma_interval: (f64, f64),
        adaptive_memory: bool,
        adaptive_memory_zeta: f64,
        adaptive_memory_adjacent_gamma_interval: (f64, f64),
        adaptive_memory_mode: String,
        biased_relay_mode: bool,
        biased_relay_r_relay: usize,
        biased_relay_maximum_round: usize,
        biased_relay_t_0: usize,
        biased_relay_r_relay_iter: usize,
        seed: u64,
    ) -> PyResult<(Self, DynDecoder)> {
        let decoder = Self {};

        let min_sum_config = MinSumDecoderConfig {
            error_priors: unsafe { error_priors.as_array() }.to_owned(),
            max_iter: t_ms.max(t_mem),
            alpha,
            alpha_iteration_scaling_factor,
            gamma0: None,
            data_scale_value,
            max_data_value,
            int_bits: None,
            frac_bits: None,
            enable_variable_message_drop: false,
            drop_probability: 0.0,
            drop_llr_threshold: 0.0,
            rng_seed: None,
        };

        let init_mode = match init_perturbation_mode.to_ascii_lowercase().as_str() {
            "multiplicative_uniform" | "uniform" | "option_b" => {
                InitPerturbationMode::MultiplicativeUniform
            }
            _ => InitPerturbationMode::AdditiveGaussian,
        };

        let selection_mode = match selection_mode.to_ascii_lowercase().as_str() {
            "tournament" | "option_b" => SelectionMode::Tournament,
            _ => SelectionMode::Weighted,
        };

        let weighted_mode = match weighted_selection_mode.to_ascii_lowercase().as_str() {
            "rank" => WeightedSelectionMode::Rank,
            _ => WeightedSelectionMode::Softmax,
        };

        let init_strategy = match init_strategy.to_ascii_lowercase().as_str() {
            "mem-bp" | "mem_bp" | "membp" => InitStrategy::MemBp,
            _ => InitStrategy::MinSum,
        };

        let perturbation_method = match perturbation_method.to_ascii_lowercase().as_str() {
            "resample" => PerturbationMethod::Resample,
            _ => PerturbationMethod::Fixed,
        };

        let gamma_mode = match gamma_mode.to_ascii_lowercase().as_str() {
            "interval_random"
            | "interval_random_per_generation"
            | "interval_random_per_variable"
            | "random_interval" => {
                GammaMode::IntervalRandomPerVariable
            }
            _ => GammaMode::Fixed,
        };

        let adaptive_memory_mode = match adaptive_memory_mode.to_ascii_lowercase().as_str() {
            "direct_interval" | "uniform_interval" | "raw_interval" => {
                AdaptiveMemoryMode::DirectInterval
            }
            _ => AdaptiveMemoryMode::ProbabilisticFlip,
        };

        let adaptive_perturbation_sign_mode =
            match adaptive_perturbation_sign_mode.to_ascii_lowercase().as_str() {
                "always_negative" | "negative" | "all_negative" => {
                    AdaptivePerturbationSignMode::AlwaysNegative
                }
                _ => AdaptivePerturbationSignMode::Random,
            };

        let adaptive_perturbation_target =
            match adaptive_perturbation_target.to_ascii_lowercase().as_str() {
                "posterior" | "marginal" | "posterior_marginal" => {
                    AdaptivePerturbationTarget::Posterior
                }
                "both" => AdaptivePerturbationTarget::Both,
                _ => AdaptivePerturbationTarget::Prior,
            };

        let adaptive_perturbation_prior_base_mode =
            match adaptive_perturbation_prior_base_mode
                .to_ascii_lowercase()
                .as_str()
            {
                "previous_biased" | "previous" | "carry" | "cumulative" => {
                    AdaptivePerturbationPriorBaseMode::PreviousBiased
                }
                _ => AdaptivePerturbationPriorBaseMode::Initial,
            };

        let adaptive_perturbation_variable_base_mode =
            match adaptive_perturbation_variable_base_mode
                .to_ascii_lowercase()
                .as_str()
            {
                "first_leg_posterior_fixed"
                | "first_leg_fixed"
                | "first_leg"
                | "fixed_first_leg" => {
                    AdaptivePerturbationVariableBaseMode::FirstLegPosteriorFixed
                }
                _ => AdaptivePerturbationVariableBaseMode::PreviousLegPosterior,
            };

        let adaptive_perturbation_llr_threshold_mode =
            match adaptive_perturbation_llr_threshold_mode
                .to_ascii_lowercase()
                .as_str()
            {
                "generation_log10_iter" | "generation_log10" | "log10_iter" | "dynamic" => {
                    AdaptivePerturbationThresholdMode::GenerationLog10Iter
                }
                _ => AdaptivePerturbationThresholdMode::Constant,
            };

        if !(0.0..=1.0).contains(&adaptive_perturbation_positive_sign_prob) {
            return Err(PyValueError::new_err(
                "adaptive_perturbation_positive_sign_prob must be between 0.0 and 1.0",
            ));
        }

        if !(0.0..=1.0).contains(&marginal_carry_damping_factor) {
            return Err(PyValueError::new_err(
                "marginal_carry_damping_factor must be between 0.0 and 1.0",
            ));
        }

        if marginal_carry_llr_abs_threshold < 0.0
            && (marginal_carry_llr_abs_threshold + 1.0).abs() > f64::EPSILON
        {
            return Err(PyValueError::new_err(
                "marginal_carry_llr_abs_threshold must be -1.0 (all variables) or >= 0.0",
            ));
        }

        let cfg = SLGMBPDecoderConfig {
            ensemble_size,
            t_ms,
            t_mem,
            g_max,
            sigma2,
            delta,
            fitness_alpha,
            fitness_beta,
            fitness_low_llr_mu,
            fitness_low_llr_threshold,
            eta,
            mutation_rate,
            mutation_llr_abs_threshold,
            elite_count,
            sequential_mc,
            tournament_size,
            selection_mode,
            weighted_selection_mode: weighted_mode,
            init_perturbation_mode: init_mode,
            init_strategy,
            init_gamma,
            adaptive_perturbation,
            adaptive_perturbation_llr_threshold_mode,
            adaptive_perturbation_llr_threshold,
            adaptive_perturbation_llr_threshold_min,
            adaptive_perturbation_llr_threshold_max,
            adaptive_perturbation_llr_threshold_factor,
            adaptive_perturbation_factor,
            adaptive_perturbation_sign_mode,
            adaptive_perturbation_positive_sign_prob,
            adaptive_perturbation_target,
            adaptive_perturbation_prior_base_mode,
            adaptive_perturbation_variable_base_mode,
            adaptive_perturbation_reset_on_threshold_exit,
            continue_perturbation,
            perturbation_method,
            reset_marginal,
            marginal_carry_damping_factor,
            marginal_carry_llr_abs_threshold,
            drop_p,
            drop_llr_threshold,
            gamma_mode,
            gamma_fixed,
            gamma_interval,
            adaptive_memory,
            adaptive_memory_zeta,
            adaptive_memory_adjacent_gamma_interval,
            adaptive_memory_mode,
            biased_relay_mode,
            biased_relay_r_relay,
            biased_relay_maximum_round,
            biased_relay_t_0,
            biased_relay_r_relay_iter,
            seed,
        };

        let inner = SLGMBPDecoder::new(
            Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
            Arc::new(min_sum_config),
            Arc::new(cfg),
        );

        Ok((decoder, DynDecoder(Box::new(inner))))
    }

    pub fn decode<'py>(
        mut self_: PyRefMut<'_, Self>,
        py: Python<'py>,
        detectors: PyReadonlyArray1<'_, Bit>,
    ) -> Bound<'py, PyArray1<Bit>> {
        self_
            .as_super()
            .inner()
            .decode(detectors.as_array())
            .into_pyarray(py)
    }

    pub fn decode_detailed(
        mut self_: PyRefMut<'_, Self>,
        detectors: PyReadonlyArray1<'_, Bit>,
    ) -> DecodeResult {
        DecodeResult::new(self_.as_super().inner().decode_detailed(detectors.as_array()))
    }

    pub fn decode_detailed_dynamics(
        mut self_: PyRefMut<'_, Self>,
        detectors: PyReadonlyArray1<'_, Bit>,
    ) -> DecodeResult {
        DecodeResult::new(
            self_
                .as_super()
                .inner()
                .decode_detailed_dynamics(detectors.as_array()),
        )
    }

    pub fn decode_batch<'py>(
        mut self_: PyRefMut<'_, Self>,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'_, Bit>,
    ) -> Bound<'py, numpy::PyArray2<Bit>> {
        self_
            .as_super()
            .inner()
            .decode_batch(detectors.as_array())
            .into_pyarray(py)
    }

    pub fn decode_detailed_batch(
        mut self_: PyRefMut<'_, Self>,
        detectors: PyReadonlyArray2<'_, Bit>,
    ) -> Vec<DecodeResult> {
        self_
            .as_super()
            .inner()
            .decode_detailed_batch(detectors.as_array())
            .into_iter()
            .map(DecodeResult::new)
            .collect()
    }
}
