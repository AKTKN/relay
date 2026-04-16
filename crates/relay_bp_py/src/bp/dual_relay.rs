use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use relay_bp::bp::dual_relay::{DualRelayDecoder, DualRelayDecoderConfig, DualRelayMixMode};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::relay::StoppingCriterion;
use relay_bp::decoder::Bit;

#[pyclass(extends=DynDecoder, subclass, module = "bp")]
#[allow(dead_code)]
pub struct DualRelayDecoderF64 {}

#[pymethods]
impl DualRelayDecoderF64 {
    #[new]
    #[pyo3(signature = (
        check_matrix,
        error_priors,
        alpha=None,
        alpha_iteration_scaling_factor=1.0,
        gamma0=0.125,
        data_scale_value=None,
        max_data_value=None,
        pre_iter=80,
        maximum_leg=100,
        iteration_per_leg=60,
        initial_gamma_slow=0.125,
        initial_gamma_fast=0.125,
        gamma_interval_slow=(0.1, 0.66),
        gamma_interval_fast=(0.1, 0.66),
        mix_mode="naive_average".to_string(),
        eta=0.5,
        delta=1.0,
        use_previous_message=false,
        beta=0.0,
        ensemble_mode=false,
        ensemble_size=2,
        ensemble_gamma_interval=(0.1, 0.66),
        num_pre_iteration_instance=1,
        initial_gamma=vec![0.125],
        n_solutions=1,
        stop_nconv=1,
        stopping_criterion="nconv".to_string(),
        seed=0,
        collect_iteration_metric=true
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        check_matrix: &Bound<'_, PyAny>,
        error_priors: &Bound<'_, PyArray1<f64>>,
        alpha: Option<f64>,
        alpha_iteration_scaling_factor: f64,
        gamma0: Option<f64>,
        data_scale_value: Option<f64>,
        max_data_value: Option<f64>,
        pre_iter: usize,
        maximum_leg: usize,
        iteration_per_leg: usize,
        initial_gamma_slow: f64,
        initial_gamma_fast: f64,
        gamma_interval_slow: (f64, f64),
        gamma_interval_fast: (f64, f64),
        mix_mode: String,
        eta: f64,
        delta: f64,
        use_previous_message: bool,
        beta: f64,
        ensemble_mode: bool,
        ensemble_size: usize,
        ensemble_gamma_interval: (f64, f64),
        num_pre_iteration_instance: usize,
        initial_gamma: Vec<f64>,
        n_solutions: usize,
        stop_nconv: usize,
        stopping_criterion: String,
        seed: u64,
        collect_iteration_metric: bool,
    ) -> PyResult<(Self, DynDecoder)> {
        let decoder = Self {};

        let min_sum_config = MinSumDecoderConfig {
            error_priors: unsafe { error_priors.as_array() }.to_owned(),
            max_iter: pre_iter + maximum_leg.saturating_sub(1) * iteration_per_leg,
            alpha,
            alpha_iteration_scaling_factor,
            gamma0,
            data_scale_value,
            max_data_value,
            int_bits: None,
            frac_bits: None,
            enable_variable_message_drop: false,
            drop_probability: 0.0,
            drop_llr_threshold: 0.0,
            rng_seed: Some(seed),
        };

        let stopping_criterion = match stopping_criterion.to_ascii_lowercase().as_str() {
            "pre_iter" => StoppingCriterion::PreIter,
            "all" => StoppingCriterion::All,
            _ => StoppingCriterion::NConv {
                stop_after: stop_nconv,
            },
        };

        let mix_mode = match mix_mode.to_ascii_lowercase().as_str() {
            "weighted_fast" | "weighted-fast" => DualRelayMixMode::WeightedFast,
            _ => DualRelayMixMode::NaiveAverage,
        };

        let dual_config = DualRelayDecoderConfig {
            pre_iter,
            maximum_leg,
            iteration_per_leg,
            initial_gamma_slow,
            initial_gamma_fast,
            gamma_interval_slow,
            gamma_interval_fast,
            mix_mode,
            eta,
            delta,
            use_previous_message,
            beta,
            ensemble_mode,
            ensemble_size,
            ensemble_gamma_interval,
            num_pre_iteration_instance,
            initial_gamma,
            n_solutions,
            stopping_criterion,
            seed,
            collect_iteration_metric,
        };

        let inner_decoder = DualRelayDecoder::<f64>::new(
            Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
            Arc::new(min_sum_config),
            Arc::new(dual_config),
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
