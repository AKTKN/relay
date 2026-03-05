use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::slg_mbp::config::{
    GammaMode, InitPerturbationMode, SLGMBPDecoderConfig, SelectionMode, WeightedSelectionMode,
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
        eta=1.0,
        mutation_rate=0.02,
        mutation_llr_abs_threshold=0.25,
        elite_count=2,
        tournament_size=3,
        init_perturbation_mode="gaussian".to_string(),
        selection_mode="weighted".to_string(),
        weighted_selection_mode="softmax".to_string(),
        gamma_mode="fixed".to_string(),
        gamma_fixed=0.125,
        gamma_interval=(0.0, 0.25),
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
        eta: f64,
        mutation_rate: f64,
        mutation_llr_abs_threshold: f64,
        elite_count: usize,
        tournament_size: usize,
        init_perturbation_mode: String,
        selection_mode: String,
        weighted_selection_mode: String,
        gamma_mode: String,
        gamma_fixed: f64,
        gamma_interval: (f64, f64),
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

        let gamma_mode = match gamma_mode.to_ascii_lowercase().as_str() {
            "interval_random" | "interval_random_per_generation" | "random_interval" => {
                GammaMode::IntervalRandomPerGeneration
            }
            _ => GammaMode::Fixed,
        };

        let cfg = SLGMBPDecoderConfig {
            ensemble_size,
            t_ms,
            t_mem,
            g_max,
            sigma2,
            delta,
            fitness_alpha,
            fitness_beta,
            eta,
            mutation_rate,
            mutation_llr_abs_threshold,
            elite_count,
            tournament_size,
            selection_mode,
            weighted_selection_mode: weighted_mode,
            init_perturbation_mode: init_mode,
            gamma_mode,
            gamma_fixed,
            gamma_interval,
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
