use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use relay_bp::bp::disordered_bp::config::{BiasApplyMode, DisorderedBPDecoderConfig, SamplingMode};
use relay_bp::bp::disordered_bp::DisorderedBPDecoder;
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::decoder::Bit;

#[pyclass(extends=DynDecoder, subclass, module = "bp")]
#[allow(dead_code)]
pub struct DisorderedBPDecoderF64 {}

#[pymethods]
impl DisorderedBPDecoderF64 {
    #[new]
    #[pyo3(signature = (
        check_matrix,
        error_priors,
        alpha=None,
        alpha_iteration_scaling_factor=1.0,
        data_scale_value=None,
        max_data_value=None,
        t_0=80,
        maximum_leg=100,
        iteration_per_leg=60,
        initial_alpha=0.625,
        alpha_mode="interval_random".to_string(),
        alpha_fixed=0.625,
        alpha_interval=(0.6, 0.7),
        initial_gamma=0.125,
        gamma_mode="interval_random".to_string(),
        gamma_fixed=0.125,
        gamma_interval=(-0.24, 0.66),
        bias_mode="fixed".to_string(),
        bias_fixed=0.0,
        bias_interval=(0.0, 0.0),
        bias_apply_mode="all".to_string(),
        bias_filter_threshold=1.0,
        negative_sign_prob=0.0,
        carry_marginal_factor=1.0,
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
        t_0: usize,
        maximum_leg: usize,
        iteration_per_leg: usize,
        initial_alpha: f64,
        alpha_mode: String,
        alpha_fixed: f64,
        alpha_interval: (f64, f64),
        initial_gamma: f64,
        gamma_mode: String,
        gamma_fixed: f64,
        gamma_interval: (f64, f64),
        bias_mode: String,
        bias_fixed: f64,
        bias_interval: (f64, f64),
        bias_apply_mode: String,
        bias_filter_threshold: f64,
        negative_sign_prob: f64,
        carry_marginal_factor: f64,
        seed: u64,
    ) -> PyResult<(Self, DynDecoder)> {
        let decoder = Self {};

        let min_sum_config = MinSumDecoderConfig {
            error_priors: unsafe { error_priors.as_array() }.to_owned(),
            max_iter: t_0 + maximum_leg.saturating_sub(1) * iteration_per_leg,
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
            rng_seed: Some(seed),
        };

        let alpha_mode = match alpha_mode.to_ascii_lowercase().as_str() {
            "fixed" => SamplingMode::Fixed,
            _ => SamplingMode::IntervalRandom,
        };

        let gamma_mode = match gamma_mode.to_ascii_lowercase().as_str() {
            "fixed" => SamplingMode::Fixed,
            _ => SamplingMode::IntervalRandom,
        };

        let bias_mode = match bias_mode.to_ascii_lowercase().as_str() {
            "interval_random" | "random" => SamplingMode::IntervalRandom,
            _ => SamplingMode::Fixed,
        };

        let bias_apply_mode = match bias_apply_mode.to_ascii_lowercase().as_str() {
            "filter" => BiasApplyMode::Filter,
            _ => BiasApplyMode::All,
        };

        let dbp_config = DisorderedBPDecoderConfig {
            t_0,
            maximum_leg,
            iteration_per_leg,
            initial_alpha,
            alpha_mode,
            alpha_fixed,
            alpha_interval,
            initial_gamma,
            gamma_mode,
            gamma_fixed,
            gamma_interval,
            bias_mode,
            bias_fixed,
            bias_interval,
            bias_apply_mode,
            bias_filter_threshold,
            negative_sign_prob,
            carry_marginal_factor,
            seed,
        };

        let inner_decoder = DisorderedBPDecoder::<f64>::new(
            Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
            Arc::new(min_sum_config),
            Arc::new(dbp_config),
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
