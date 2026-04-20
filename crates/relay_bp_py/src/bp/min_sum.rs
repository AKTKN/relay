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

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use relay_bp::bp::min_sum::{MinSumBPDecoder, MinSumDecoderConfig, Step1Metrics};
use relay_bp::decoder::Bit;

macro_rules! create_bp_interface {
    ($name: ident, $type: ident) => {
        #[pyclass(extends=DynDecoder, subclass, module = "bp")]
        #[allow(dead_code)]
        pub struct $name {}

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (check_matrix, error_priors, max_iter=200, alpha=None, alpha_iteration_scaling_factor=1.0, gamma0=None, data_scale_value=None, max_data_value=None, int_bits=None, frac_bits=None))]
            #[allow(clippy::missing_transmute_annotations, clippy::too_many_arguments)]
            pub fn new(
                py: Python<'_>,
                check_matrix: &Bound<'_, PyAny>,
                error_priors: &Bound<'_, PyArray1<f64>>,
                max_iter: usize,
                alpha: Option<f64>,
                alpha_iteration_scaling_factor: f64,
                gamma0: Option<f64>,
                data_scale_value: Option<f64>,
                max_data_value: Option<f64>,
                int_bits: Option<isize>,
                frac_bits: Option<isize>,
            ) -> PyResult<(Self, DynDecoder)> {
                let min_sum_decoder = Self {};

                let config = MinSumDecoderConfig {
                    error_priors: unsafe { error_priors.as_array() }.to_owned(),
                    max_iter,
                    alpha,
                    alpha_iteration_scaling_factor,
                    gamma0,
                    data_scale_value,
                    max_data_value,
                    int_bits,
                    frac_bits,
                    enable_variable_message_drop: false,
                    drop_probability: 0.0,
                    drop_llr_threshold: 0.0,
                    rng_seed: None,
                };


                let inner_decoder = MinSumBPDecoder::<$type>::new(
                    Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
                    Arc::new(config),
                );

                let dyn_decoder = DynDecoder(Box::new(inner_decoder));
                Ok((min_sum_decoder, dyn_decoder))
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

            #[pyo3(
                signature = (
                    detectors,
                    k_default=20,
                    t_warmup=0,
                    theta_r_default=0.4,
                    n_min_default=3,
                    low_threshold_0=0.1,
                    low_threshold_1=0.5,
                    endpoint_k=20,
                    collect_sign_trajectory=false,
                    snapshot_times=None
                )
            )]
            #[allow(clippy::too_many_arguments)]
            pub fn decode_detailed_step1_metrics<'py>(
                mut self_: PyRefMut<'_, Self>,
                py: Python<'py>,
                detectors: PyReadonlyArray1<'_, Bit>,
                k_default: usize,
                t_warmup: usize,
                theta_r_default: f64,
                n_min_default: usize,
                low_threshold_0: f64,
                low_threshold_1: f64,
                endpoint_k: usize,
                collect_sign_trajectory: bool,
                snapshot_times: Option<Vec<usize>>,
            ) -> PyResult<(DecodeResult, Bound<'py, PyAny>)> {
                let inner = self_.as_super().inner();
                let Some(min_sum) = inner
                    .as_any_mut()
                    .downcast_mut::<MinSumBPDecoder<$type>>()
                else {
                    return Err(PyRuntimeError::new_err(
                        "Internal decoder type mismatch; expected MinSumBPDecoder",
                    ));
                };

                let (result, metrics) = min_sum.decode_detailed_step1_metrics(
                    detectors.as_array(),
                    k_default,
                    t_warmup,
                    theta_r_default,
                    n_min_default,
                    low_threshold_0,
                    low_threshold_1,
                    endpoint_k,
                    collect_sign_trajectory,
                    snapshot_times.as_deref(),
                );

                let Step1Metrics {
                    variable_count,
                    w_sigma,
                    delta_hd,
                    delta_m_l2,
                    m_norm_l2,
                    m_bar,
                    f_low_0,
                    f_low_1,
                    f_endpoint,
                    v_osc,
                    t_stag,
                    sign_trajectory_packed,
                    sign_bytes_per_iter,
                    abs_m_snapshots,
                    snapshot_times,
                } = metrics;

                let iters = w_sigma.len();

                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("variable_count", variable_count)?;
                dict.set_item("t_stag", t_stag)?;
                dict.set_item("w_sigma", w_sigma.into_pyarray(py))?;
                dict.set_item("delta_hd", delta_hd.into_pyarray(py))?;
                dict.set_item("delta_m_l2", delta_m_l2.into_pyarray(py))?;
                dict.set_item("m_norm_l2", m_norm_l2.into_pyarray(py))?;
                dict.set_item("m_bar", m_bar.into_pyarray(py))?;
                dict.set_item("f_low_0", f_low_0.into_pyarray(py))?;
                dict.set_item("f_low_1", f_low_1.into_pyarray(py))?;
                dict.set_item("f_endpoint", f_endpoint.into_pyarray(py))?;
                dict.set_item("v_osc", v_osc.into_pyarray(py))?;
                dict.set_item("sign_bytes_per_iter", sign_bytes_per_iter)?;

                if let Some(packed) = sign_trajectory_packed {
                    dict.set_item("sign_trajectory_shape", (iters, sign_bytes_per_iter))?;
                    dict.set_item("sign_trajectory_packed", packed.into_pyarray(py))?;
                } else {
                    dict.set_item("sign_trajectory_shape", py.None())?;
                    dict.set_item("sign_trajectory_packed", py.None())?;
                }

                if let Some(snaps) = abs_m_snapshots {
                    let n_snap = snapshot_times.len();
                    dict.set_item("abs_m_snapshots_shape", (n_snap, variable_count))?;
                    dict.set_item("abs_m_snapshots", snaps.into_pyarray(py))?;
                    dict.set_item("snapshot_times", snapshot_times)?;
                } else {
                    dict.set_item("abs_m_snapshots_shape", py.None())?;
                    dict.set_item("abs_m_snapshots", py.None())?;
                    dict.set_item("snapshot_times", snapshot_times)?;
                }

                Ok((DecodeResult::new(result), dict.into_any()))
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
                    .map(|result| DecodeResult::new(result))
                    .collect()
            }
        }
    };
}

create_bp_interface!(MinSumBPDecoderF32, f32);
create_bp_interface!(MinSumBPDecoderF64, f64);

create_bp_interface!(MinSumBPDecoderI8, i8);
create_bp_interface!(MinSumBPDecoderI16, i16);
create_bp_interface!(MinSumBPDecoderI32, i32);
create_bp_interface!(MinSumBPDecoderI64, i64);
