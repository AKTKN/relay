use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use relay_bp::bp::adaptive_relay::{
    AdaptiveRelayDecoder, AdaptiveRelayDecoderConfig, AdaptiveRelayPerturbationMode,
    AdaptiveRelayUpdateMode, PosteriorMarginalClampMode,
};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::decoder::Bit;

#[pyclass(extends=DynDecoder, subclass, module = "bp")]
#[allow(dead_code)]
pub struct AdaptiveRelayDecoderF64 {}

#[pymethods]
impl AdaptiveRelayDecoderF64 {
    #[new]
    #[pyo3(signature = (
        check_matrix,
        error_priors,
        alpha=None,
        alpha_iteration_scaling_factor=1.0,
        data_scale_value=None,
        max_data_value=None,
        initial_gamma=0.125,
        gamma_min=-0.24,
        gamma_max=0.66,
        tau=1.0,
        beta=0.9,
        pre_decoding=false,
        pre_iteration=80,
        maximum_iteration=600,
        iter_per_leg=60,
        update_mode="per-iteration".to_string(),
        perturbation_mode="uniform".to_string(),
        perturbation_interval=(0.0, 0.0),
        perturbation_sigma=0.0,
        ensemble_size=1,
        carry_marginal_between_legs=true,
        posterior_marginal_clamp_mode="no_clamp".to_string(),
        posterior_marginal_abs_threshold=1e10,
        seed=0,
        collect_iteration_metric=false
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
        initial_gamma: f64,
        gamma_min: f64,
        gamma_max: f64,
        tau: f64,
        beta: f64,
        pre_decoding: bool,
        pre_iteration: usize,
        maximum_iteration: usize,
        iter_per_leg: usize,
        update_mode: String,
        perturbation_mode: String,
        perturbation_interval: (f64, f64),
        perturbation_sigma: f64,
        ensemble_size: usize,
        carry_marginal_between_legs: bool,
        posterior_marginal_clamp_mode: String,
        posterior_marginal_abs_threshold: f64,
        seed: u64,
        collect_iteration_metric: bool,
    ) -> PyResult<(Self, DynDecoder)> {
        let decoder = Self {};

        let min_sum_config = MinSumDecoderConfig {
            error_priors: unsafe { error_priors.as_array() }.to_owned(),
            max_iter: maximum_iteration.max(1),
            alpha,
            alpha_iteration_scaling_factor,
            gamma0: Some(initial_gamma),
            data_scale_value,
            max_data_value,
            int_bits: None,
            frac_bits: None,
            enable_variable_message_drop: false,
            drop_probability: 0.0,
            drop_llr_threshold: 0.0,
            rng_seed: None,
        };

        let update_mode = match update_mode.to_ascii_lowercase().as_str() {
            "per-leg" | "per_leg" | "leg" => AdaptiveRelayUpdateMode::PerLeg,
            _ => AdaptiveRelayUpdateMode::PerIteration,
        };

        let perturbation_mode = match perturbation_mode.to_ascii_lowercase().as_str() {
            "gaussian" | "normal" => AdaptiveRelayPerturbationMode::Gaussian,
            _ => AdaptiveRelayPerturbationMode::Uniform,
        };

        let posterior_marginal_clamp_mode =
            match posterior_marginal_clamp_mode.to_ascii_lowercase().as_str() {
                "abs_threshold_clamp" | "threshold" | "clamp" => {
                    PosteriorMarginalClampMode::AbsThresholdClamp
                }
                _ => PosteriorMarginalClampMode::NoClamp,
            };

        let adaptive_config = AdaptiveRelayDecoderConfig {
            initial_gamma,
            gamma_min,
            gamma_max,
            tau,
            beta,
            pre_decoding,
            pre_iteration,
            maximum_iteration,
            iter_per_leg,
            update_mode,
            perturbation_mode,
            perturbation_interval,
            perturbation_sigma,
            ensemble_size,
            carry_marginal_between_legs,
            posterior_marginal_clamp_mode,
            posterior_marginal_abs_threshold,
            seed,
            collect_iteration_metric,
        };

        let inner_decoder = AdaptiveRelayDecoder::<f64>::new(
            Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
            Arc::new(min_sum_config),
            Arc::new(adaptive_config),
        );

        let dyn_decoder = DynDecoder(Box::new(inner_decoder));
        Ok((decoder, dyn_decoder))
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
        DecodeResult::new(
            self_
                .as_super()
                .inner()
                .decode_detailed(detectors.as_array()),
        )
    }

    pub fn decode_batch<'py>(
        mut self_: PyRefMut<'_, Self>,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'_, Bit>,
    ) -> Bound<'py, PyArray2<Bit>> {
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
