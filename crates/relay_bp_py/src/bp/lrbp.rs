use std::sync::Arc;

use pyo3::prelude::*;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use relay_bp::bp::lrbp::{LRBPDecoder, LRBPDecoderConfig};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::relay::StoppingCriterion;
use relay_bp::decoder::Bit;

macro_rules! create_bp_interface {
    ($name: ident, $type: ident) => {
        #[pyclass(extends=DynDecoder, subclass, module = "bp")]
        #[allow(dead_code)]
        pub struct $name {}

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (check_matrix, error_priors, alpha=None, alpha_iteration_scaling_factor=1.0, gamma0=0.1, data_scale_value=None, max_data_value=None, pre_iter=80, num_sets=300,
                set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), explicit_gammas=None, stop_nconv=1,
                stopping_criterion="nconv".to_string(), logging=false, seed=0, osc_window=5, friction_slope=2.0, friction_shift=0.0, tau=0.2))]
            #[allow(clippy::missing_transmute_annotations, clippy::too_many_arguments)]
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
                num_sets: usize,
                set_max_iter: usize,
                gamma_dist_interval: (f64, f64),
                explicit_gammas: Option<&Bound<'_, PyArray2<f64>>>,
                stop_nconv: usize,
                stopping_criterion: String,
                logging: bool,
                seed: u64,
                osc_window: usize,
                friction_slope: f64,
                friction_shift: f64,
                tau: f64,
            ) -> PyResult<(Self, DynDecoder)> {
                let min_sum_decoder = Self {};

                let min_sum_config = MinSumDecoderConfig {
                    error_priors: unsafe { error_priors.as_array() }.to_owned(),
                    max_iter: pre_iter,
                    alpha,
                    alpha_iteration_scaling_factor,
                    gamma0,
                    data_scale_value,
                    max_data_value,
                    int_bits: None,
                    frac_bits: None,
                };

                let stopping_criterion = match stopping_criterion.as_str() {
                    "pre_iter" => StoppingCriterion::PreIter,
                    "nconv" => StoppingCriterion::NConv {
                        stop_after: stop_nconv,
                    },
                    "all" => StoppingCriterion::All,
                    _ => StoppingCriterion::default(),
                };

                let lrbp_config = LRBPDecoderConfig {
                    pre_iter,
                    num_sets,
                    set_max_iter,
                    gamma_dist_interval,
                    explicit_gammas: explicit_gammas
                        .map(|explicit_gammas| unsafe { explicit_gammas.as_array() }.to_owned()),
                    stopping_criterion,
                    logging,
                    seed,
                    osc_window,
                    friction_slope,
                    friction_shift,
                    tau,
                };

                let inner_decoder = LRBPDecoder::<$type>::new(
                    Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
                    Arc::new(min_sum_config),
                    Arc::new(lrbp_config),
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

create_bp_interface!(LRBPDecoderF32, f32);
create_bp_interface!(LRBPDecoderF64, f64);
create_bp_interface!(LRBPDecoderI32, i32);
create_bp_interface!(LRBPDecoderI64, i64);
