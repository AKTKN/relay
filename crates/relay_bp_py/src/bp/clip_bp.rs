// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use relay_bp::bp::clip_bp::{ClipBpConfig, ClipBpDecoder, SignUpdateMode};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::decoder::Bit;

#[pyclass(extends=DynDecoder, subclass, module = "bp")]
#[allow(dead_code)]
pub struct ClipBPDecoderF64 {}

#[pymethods]
impl ClipBPDecoderF64 {
    #[new]
    #[pyo3(signature = (
        check_matrix,
        error_priors,
        min_llr,
        sign_mode = "hysteresis".to_string(),
        gamma_first = 0.125,
        gamma_center = 0.21,
        gamma_width = 0.9,
        max_legs = 100,
        max_iter_first = 80,
        max_iter = 60,
        max_solutions = 1,
        alpha = None,
        alpha_iteration_scaling_factor = 1.0,
        gamma0 = None,
        seed = 0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        check_matrix: &Bound<'_, PyAny>,
        error_priors: &Bound<'_, PyArray1<f64>>,
        min_llr: f64,
        sign_mode: String,
        gamma_first: f64,
        gamma_center: f64,
        gamma_width: f64,
        max_legs: usize,
        max_iter_first: usize,
        max_iter: usize,
        max_solutions: usize,
        alpha: Option<f64>,
        alpha_iteration_scaling_factor: f64,
        gamma0: Option<f64>,
        seed: u64,
    ) -> PyResult<(Self, DynDecoder)> {
        let sign_mode = match sign_mode.to_ascii_lowercase().as_str() {
            "naive" => SignUpdateMode::Naive,
            "hysteresis" => SignUpdateMode::Hysteresis,
            "neighbor_only" | "neighbor-only" | "neighboronly" => SignUpdateMode::NeighborOnly,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sign_mode: {other}"
                )))
            }
        };

        let _ = gamma0;
        let min_sum_config = Arc::new(MinSumDecoderConfig {
            error_priors: unsafe { error_priors.as_array() }.to_owned(),
            max_iter: max_iter_first.max(max_iter),
            alpha,
            alpha_iteration_scaling_factor,
            // ClipBP requires priors to be present in posterior_ratios at init.
            gamma0: Some(gamma_first),
            data_scale_value: None,
            max_data_value: None,
            int_bits: None,
            frac_bits: None,
            enable_variable_message_drop: false,
            drop_probability: 0.0,
            drop_llr_threshold: 0.0,
            rng_seed: Some(seed),
        });

        let clip_config = Arc::new(ClipBpConfig {
            min_llr,
            sign_mode,
            gamma_first,
            gamma_center,
            gamma_width,
            max_legs,
            max_iter_first,
            max_iter,
            max_solutions,
            seed,
        });

        let inner = ClipBpDecoder::new(
            Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
            min_sum_config,
            clip_config,
        );

        Ok((ClipBPDecoderF64 {}, DynDecoder(Box::new(inner))))
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
