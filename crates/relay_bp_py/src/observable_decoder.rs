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

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use relay_bp::decoder::{Bit, Decoder as DecoderInner, DecoderRunner};
use relay_bp::observable_decoder::{
    ObservableDecodeResult as ObservableDecodeResultInner, ObservableDecoder,
    ObservableDecoderRunner as ObservableDecoderRunnerInner,
};
use relay_bp::ensemble_decoder::{EnsembleDecoder, SelectionStrategy, EnsembleMode, RepulsiveConfig};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::relay::{RelayDecoder, RelayDecoderConfig, StoppingCriterion};
use relay_bp::decoder::{Decoder, AutomorphismWrapperDecoder, SparseBitMatrix};

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array1;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult};
use pyo3::types::{PyList, PyDict};
use std::mem;
use rand::Rng;
use sprs::CsMat;
use rand::random;

#[pyclass(module = "observable_decoder")]
pub struct ObservableDecodeResult {
    inner: ObservableDecodeResultInner,
}

impl ObservableDecodeResult {
    pub fn new(inner: ObservableDecodeResultInner) -> Self {
        ObservableDecodeResult { inner }
    }
}
#[pymethods]
impl ObservableDecodeResult {
    #[getter]
    pub fn observables<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Bit>> {
        PyArray1::from_array(py, &self.inner.observables)
    }

    #[getter]
    pub fn error_detected(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.error_detected)
    }

    #[getter]
    pub fn error_mismatch_detected(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.error_mismatch_detected)
    }

    #[getter]
    pub fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    pub fn logical_gap(&self) -> Option<f64> {
        self.inner.logical_gap
    }

    #[getter]
    pub fn unconverged_no_error(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.unconverged_no_error)
    }

    #[getter]
    pub fn better_decoding_quality_error(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.better_decoding_quality_error)
    }

    #[getter]
    pub fn worse_decoding_quality_error(&self) -> Option<bool> {
        let true_decoding = self.inner.true_decoding.as_ref()?;
        Some(true_decoding.worse_decoding_quality_error)
    }

    #[getter]
    pub fn physical_decode_result(&self) -> Option<DecodeResult> {
        if let Some(result) = &self.inner.physical_decode_result {
            return Some(DecodeResult::new(result.clone()));
        }
        None
    }

    #[getter]
    pub fn extra<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        if let Some(phys_result) = &self.inner.physical_decode_result {
            match &phys_result.extra {
                relay_bp::decoder::BPExtraResult::None => Ok(py.None()),
                relay_bp::decoder::BPExtraResult::Ensemble(ensemble_extra) => {
                    let dict = PyDict::new(py);

                    // eprintln!("[Rust Debug] selected_coset_avg_iter: {:?}", ensemble_extra.selected_coset_avg_iter);
                    // eprintln!("[Rust Debug] runner_up_coset_avg_iter: {:?}", ensemble_extra.runner_up_coset_avg_iter);
                    // eprintln!("[Rust Debug] selected_coset_votes: {:?}", ensemble_extra.selected_coset_votes);
                    // eprintln!("[Rust Debug] runner_up_coset_votes: {:?}", ensemble_extra.runner_up_coset_votes);
                    
                    // all_corrections as list of numpy arrays
                    let corrections_list: Vec<_> = ensemble_extra
                        .all_corrections
                        .iter()
                        .map(|arr| PyArray1::from_array(py, arr).into_py(py))
                        .collect();
                    dict.set_item("all_corrections", corrections_list)?;  //empty now
                    
                    // llr_sums as list of floats
                    dict.set_item("llr_sums", ensemble_extra.llr_sums.clone())?;  //empty now
                    
                    // cosets as list of numpy arrays
                    let cosets_list: Vec<_> = ensemble_extra
                        .cosets
                        .iter()
                        .map(|arr| PyArray1::from_array(py, arr).into_py(py))
                        .collect();
                    dict.set_item("cosets", cosets_list)?;  //empty now
                    
                    dict.set_item("selected_index", ensemble_extra.selected_index)?;
                    dict.set_item("child_iterations", ensemble_extra.child_iterations.clone())?;
                    dict.set_item("child_success", ensemble_extra.child_success.clone())?;
                    dict.set_item("effective_iterations", ensemble_extra.effective_iterations)?;
                    dict.set_item("selected_coset_avg_iter", ensemble_extra.selected_coset_avg_iter)?;
                    dict.set_item("runner_up_coset_avg_iter", ensemble_extra.runner_up_coset_avg_iter)?;
                    dict.set_item("selected_coset_votes", ensemble_extra.selected_coset_votes)?;
                    dict.set_item("runner_up_coset_votes", ensemble_extra.runner_up_coset_votes)?;
                    
                    // New fields
                    dict.set_item("converged_count", ensemble_extra.converged_count)?;
                    
                    // ensemble_posterior_ratios as list of numpy arrays
                    let ensemble_posterior_list: Vec<_> = ensemble_extra
                        .ensemble_posterior_ratios
                        .iter()
                        .map(|arr| PyArray1::from_array(py, arr).into_py(py))
                        .collect();
                    dict.set_item("ensemble_posterior_ratios", ensemble_posterior_list)?;
                    
                    // ensemble_mean_posterior_ratios as numpy array or None
                    if let Some(ref mean_arr) = ensemble_extra.ensemble_mean_posterior_ratios {
                        dict.set_item("ensemble_mean_posterior_ratios", 
                            PyArray1::from_array(py, mean_arr).into_py(py))?;
                    } else {
                        dict.set_item("ensemble_mean_posterior_ratios", py.None())?;
                    }
                    
                    // ensemble_std_posterior_ratios as numpy array or None
                    if let Some(ref std_arr) = ensemble_extra.ensemble_std_posterior_ratios {
                        dict.set_item("ensemble_std_posterior_ratios", 
                            PyArray1::from_array(py, std_arr).into_py(py))?;
                    } else {
                        dict.set_item("ensemble_std_posterior_ratios", py.None())?;
                    }
                    
                    dict.set_item("ensemble_iteration_dist", ensemble_extra.ensemble_iteration_dist.clone())?;
                    dict.set_item("ensemble_mean_iteration", ensemble_extra.ensemble_mean_iteration)?;
                    dict.set_item("ensemble_std_iteration", ensemble_extra.ensemble_std_iteration)?;
                    
                    // residual_result as list of numpy arrays or None
                    if let Some(ref residual) = ensemble_extra.residual_result {
                        let residual_list: Vec<_> = residual
                            .iter()
                            .map(|arr| PyArray1::from_array(py, arr).into_py(py))
                            .collect();
                        dict.set_item("residual_result", residual_list)?;
                    } else {
                        dict.set_item("residual_result", py.None())?;
                    }
                    
                    Ok(dict.into())
                }
            }
        } else {
            Ok(py.None())
        }
    }
}

#[pyclass(module = "observable_decoder")]
#[allow(dead_code)]
pub struct ObservableDecoderRunner {
    // Static lifetime to workaround lifetime issue referenced above.
    inner: ObservableDecoderRunnerInner<'static>,
}

#[pymethods]
impl ObservableDecoderRunner {
    #[new]
    #[pyo3(signature = (decoder, observable_error_matrix, include_decode_result=false))]
    pub fn new(
        py: Python<'_>,
        decoder: DynDecoder,
        observable_error_matrix: &Bound<'_, PyAny>,
        include_decode_result: bool,
    ) -> PyResult<Self> {
        let inner: relay_bp::observable_decoder::ObservableDecoderRunner<'_> = unsafe {
            mem::transmute(ObservableDecoderRunnerInner::new(
                decoder.0,
                Arc::new(get_sprs_bit_matrix_from_python(
                    py,
                    observable_error_matrix,
                )?),
                include_decode_result,
            ))
        };
        Ok(Self { inner })
    }


    // Factory method to create an ObservableDecoderRunner with an EnsembleDecoder inside
    #[staticmethod]
    #[pyo3(signature = (ensemble_size, check_matrix, observable_matrix, error_priors, alpha=None, alpha_iteration_scaling_factor=1.0, gamma0=0.1, data_scale_value=None, max_data_value=None, pre_iter=80, num_sets=300,
        set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), explicit_gammas=None, stop_nconv=1,
        stopping_criterion="nconv".to_string(), logging=false, selection_strategy="MostLikely".to_string(), 
        perturbation_min=0.0, perturbation_max=0.0, col_permutations=None, row_permutations=None, seed=None,
        ensemble_mode="normal".to_string(), repulsive_size=0, repulsive_gamma_dist=None, abs_llr_threshold=None, pulse_per_leg=None, start_leg=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn with_ensemble_decoder(
        py: Python<'_>,
        ensemble_size: usize,
        check_matrix: &Bound<'_, PyAny>,
        observable_matrix: &Bound<'_, PyAny>,
        error_priors: PyReadonlyArray1<f64>,
        alpha: Option<f64>,
        alpha_iteration_scaling_factor: f64,
        gamma0: Option<f64>,
        data_scale_value: Option<f64>,
        max_data_value: Option<f64>,
        pre_iter: usize,
        num_sets: usize,
        set_max_iter: usize,
        gamma_dist_interval: (f64, f64),
        explicit_gammas: Option<PyReadonlyArray2<f64>>,
        stop_nconv: usize,
        stopping_criterion: String,
        logging: bool,
        selection_strategy: String,
        perturbation_min: f64,
        perturbation_max: f64,
        col_permutations: Option<&Bound<'_, PyAny>>,
        row_permutations: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
        ensemble_mode: String,
        repulsive_size: usize,
        repulsive_gamma_dist: Option<(f64, f64)>,
        abs_llr_threshold: Option<f64>,
        pulse_per_leg: Option<usize>,
        start_leg: Option<usize>,
    ) -> PyResult<Self> {
        // 1. Setyp parameters for child decoders
        let mut child_decoders: Vec<Box<dyn Decoder + Send>> = Vec::new();
        let error_priors_owned = error_priors.as_array().to_owned();

        let check_matrix_arc = Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?);
        let obs_matrix_arc = Arc::new(get_sprs_bit_matrix_from_python(py, observable_matrix)?);

        // Convert permutation matrix from Python to Rust
        let col_perms_arc: Option<Vec<Arc<SparseBitMatrix>>> = col_permutations
            .map(|any| {
                any.downcast::<PyList>()?
                    .iter()
                    .map(|p| get_sprs_bit_matrix_from_python(py, &p).map(Arc::new))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;

        let row_perms_arc: Option<Vec<Arc<SparseBitMatrix>>> = row_permutations
            .map(|any| {
                any.downcast::<PyList>()?
                    .iter()
                    .map(|p| get_sprs_bit_matrix_from_python(py, &p).map(Arc::new))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;

        let stopping_criterion_template = match stopping_criterion.as_str() {
            "pre_iter" => StoppingCriterion::PreIter,
            "nconv" => StoppingCriterion::NConv { stop_after: stop_nconv },
            "all" => StoppingCriterion::All,
            _ => StoppingCriterion::default(),
        };

        let seed = seed.unwrap_or_else(rand::random::<u64>);

        let relay_config_templete = RelayDecoderConfig {
            pre_iter, num_sets, set_max_iter, gamma_dist_interval,
            explicit_gammas: explicit_gammas.map(|arr| arr.as_array().to_owned()),
            stopping_criterion: stopping_criterion_template, logging, seed,
            repulsive_gamma_dist: None,  // Will be set per-decoder based on mode
            abs_llr_threshold: None,
            pulse_per_leg: None,
            start_leg: None,
        };   

        // Perturb error priors 
        // Initnalize random number generator
        let mut rng = rand::thread_rng();

        // Create child decoders 
        for i in 0..ensemble_size {
            // Determine if this decoder should use repulsive mode
            let use_repulsive = i < repulsive_size;
            // Generate a random perturbation strongness within the specified range
            // let alpha_perturb = if perturbation_min < perturbation_max {
            //     rng.gen_range(perturbation_min..perturbation_max)
            // } else {
            //     0.0
            // };

            
            // Apply the perturbation to the error priors
            // but include no-perturbation case for the first decoder (i == 0)
            let perturbed_priors = if (perturbation_min < perturbation_max) && (i != 0){
                let mut new_priors = error_priors_owned.clone();
                for p in new_priors.iter_mut(){
                    let factor = rng.gen_range((1.0 - perturbation_min)..=(1.0 + perturbation_max));
                    *p = (*p * factor).clamp(1e-15, 1.0 - 1e-15);
                }
                new_priors
            } else {
                error_priors_owned.clone()
            };

            // Apply permutations
            let (final_check_matrix, final_priors, col_perm, row_perm) =
                if let (Some(cp), Some(rp)) = (&col_perms_arc, &row_perms_arc) {
                    let current_col_perm = cp.get(i).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Not enough column permutations provided"))?.clone();
                    let current_row_perm = rp.get(i).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Not enough row permutations provided"))?.clone();

                    // H' = H * A
                    let transformed_h = Arc::new(&*check_matrix_arc * &*current_col_perm);

                    // p' = p * A (手動で置換)
                    let mut transformed_priors = Array1::zeros(perturbed_priors.len());
                    for (j, &val) in perturbed_priors.iter().enumerate() {
                        if let Some(k) = current_col_perm.outer_view(j).and_then(|col| col.indices().get(0).copied()) {
                            transformed_priors[k] = val;
                        }
                    }
                    (transformed_h, transformed_priors, current_col_perm, current_row_perm)
                } else {
                    // 自己同型を使わない場合は、恒等置換を生成
                    let num_vars = check_matrix_arc.cols();
                    let num_checks = check_matrix_arc.rows();
                    (check_matrix_arc.clone(), perturbed_priors, Arc::new(CsMat::eye(num_vars)), Arc::new(CsMat::eye(num_checks)))
                };

            // Create MinSumDecoderConfig with perturbed priors
            let min_sum_config = MinSumDecoderConfig {
                error_priors: final_priors, 
                max_iter: pre_iter,
                alpha,
                alpha_iteration_scaling_factor,
                gamma0,
                data_scale_value,
                max_data_value,
                int_bits: None,
                frac_bits: None,
            };

            // Create RelayDecoderConfig
            let mut relay_config = relay_config_templete.clone();
            relay_config.seed = seed + i as u64;
            
            // Apply repulsive mode settings if this is a repulsive decoder
            if use_repulsive {
                relay_config.repulsive_gamma_dist = repulsive_gamma_dist;
                relay_config.abs_llr_threshold = abs_llr_threshold;
                relay_config.pulse_per_leg = pulse_per_leg;
                relay_config.start_leg = start_leg;
            }

            // Create child decoder
            let relay_decoder = Box::new(RelayDecoder::<f64>::new(
                final_check_matrix, 
                Arc::new(min_sum_config),
                Arc::new(relay_config),
            ));

            let wrapped_decoder = Box::new(AutomorphismWrapperDecoder::new(
                relay_decoder,
                col_perm,
                row_perm,
            ));
            child_decoders.push(wrapped_decoder);
        }

        // 3. EnsembleDecoderを生成
        let strategy = match selection_strategy.to_lowercase().as_str() {
            "most-likely" | "mostlikely" => SelectionStrategy::MostLikely,
            "majority-vote" | "majorityvote" => SelectionStrategy::MajorityVote,
            _ => SelectionStrategy::MostLikely,
        };

        let original_log_priors = error_priors_owned.mapv(|p| ((1.0 - p)/p).ln());
        let original_log_priors_arc = Arc::new(original_log_priors);

        // Parse ensemble mode
        let mode = match ensemble_mode.to_lowercase().as_str() {
            "repulsive" => EnsembleMode::Repulsive,
            "normal" | _ => EnsembleMode::Normal,
        };

        // Create RepulsiveConfig if in repulsive mode
        let repulsive_config_arc = if mode == EnsembleMode::Repulsive {
            Some(Arc::new(RepulsiveConfig {
                repulsive_size,
                repulsive_gamma_dist: repulsive_gamma_dist.unwrap_or((-0.5, -0.1)),
                abs_llr_threshold: abs_llr_threshold.unwrap_or(2.0),
                pulse_per_leg: pulse_per_leg.unwrap_or(1),
                start_leg: start_leg.unwrap_or(0),
            }))
        } else {
            None
        };

        let ensemble_decoder = EnsembleDecoder::new_with_mode(
            child_decoders,
            strategy,
            Some(original_log_priors_arc),
            Some(obs_matrix_arc.clone()),
            mode,
            repulsive_config_arc,
        );

        // 4. 生成したEnsembleDecoderを使い、自分自身(ObservableDecoderRunner)のインスタンスを生成して返す
        let inner: relay_bp::observable_decoder::ObservableDecoderRunner<'_> = unsafe {
            mem::transmute(ObservableDecoderRunnerInner::new(
                Box::new(ensemble_decoder),
                obs_matrix_arc,
                true, // include_decode_result
            ))
        };
        Ok(Self { inner })
    }


    pub fn decode<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyReadonlyArray1<'_, Bit>,
    ) -> Bound<'py, PyArray1<Bit>> {
        self.inner.decode(detectors.as_array()).into_pyarray(py)
    }

    pub fn decode_detailed(&mut self, detectors: PyReadonlyArray1<'_, Bit>) -> DecodeResult {
        DecodeResult::new(self.inner.decode_detailed(detectors.as_array()))
    }

    pub fn compute_observables<'py>(
        &mut self,
        py: Python<'py>,
        errors: PyReadonlyArray1<'_, Bit>,
    ) -> Bound<'py, PyArray1<Bit>> {
        PyArray1::from_array(py, &self.inner.compute_observables(errors.as_array()))
    }

    pub fn decode_observables<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyReadonlyArray1<'_, Bit>,
    ) -> Bound<'py, PyArray1<Bit>> {
        PyArray1::from_array(py, &self.inner.decode_observables(detectors.as_array()))
    }

    pub fn from_errors_decode_observables_detailed(
        &mut self,
        errors: PyReadonlyArray1<'_, Bit>,
    ) -> ObservableDecodeResult {
        ObservableDecodeResult::new(
            self.inner
                .from_errors_decode_observables_detailed(errors.as_array()),
        )
    }

    #[pyo3(signature = (detectors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn decode_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Bound<'py, PyArray2<Bit>> {
        let results = match (parallel, progress_bar) {
            (false, false) => self.inner.decode_batch(detectors.as_array()),
            (true, false) => self.inner.par_decode_batch(detectors.as_array()),
            (false, true) => self
                .inner
                .decode_batch_progress_bar(detectors.as_array(), leave_progress_bar_on_finish),
            (true, true) => self
                .inner
                .par_decode_batch_progress_bar(detectors.as_array(), leave_progress_bar_on_finish),
        };
        results.into_pyarray(py)
    }

    #[pyo3(signature = (detectors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn decode_detailed_batch(
        &mut self,
        detectors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<DecodeResult> {
        let results = match (parallel, progress_bar) {
            (false, false) => self.inner.decode_detailed_batch(detectors.as_array()),
            (true, false) => self.inner.par_decode_detailed_batch(detectors.as_array()),
            (false, true) => self.inner.decode_detailed_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
            (true, true) => self.inner.par_decode_detailed_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
        };
        results.into_iter().map(DecodeResult::new).collect()
    }

    #[pyo3(signature = (detectors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn decode_observables_batch<'py>(
        &mut self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Bound<'py, PyArray2<Bit>> {
        let results = match (parallel, progress_bar) {
            (false, false) => self.inner.decode_observables_batch(detectors.as_array()),
            (true, false) => self
                .inner
                .par_decode_observables_batch(detectors.as_array()),
            (false, true) => self.inner.decode_observables_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
            (true, true) => self.inner.par_decode_observables_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
        };
        results.into_pyarray(py)
    }

    #[pyo3(signature = (errors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn from_errors_decode_observables_batch<'py>(
        &mut self,
        py: Python<'py>,
        errors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Bound<'py, PyArray2<Bit>> {
        let results = match (parallel, progress_bar) {
            (false, false) => self
                .inner
                .from_errors_decode_observables_batch(errors.as_array()),
            (true, false) => self
                .inner
                .par_from_errors_decode_observables_batch(errors.as_array()),
            (false, true) => self
                .inner
                .from_errors_decode_observables_batch_progress_bar(
                    errors.as_array(),
                    leave_progress_bar_on_finish,
                ),
            (true, true) => self
                .inner
                .par_from_errors_decode_observables_batch_progress_bar(
                    errors.as_array(),
                    leave_progress_bar_on_finish,
                ),
        };

        results.into_pyarray(py)
    }

    #[pyo3(signature = (detectors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn decode_observables_detailed_batch(
        &mut self,
        detectors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<ObservableDecodeResult> {
        let results = match (parallel, progress_bar) {
            (false, false) => self
                .inner
                .decode_observables_detailed_batch(detectors.as_array()),
            (true, false) => self
                .inner
                .par_decode_observables_detailed_batch(detectors.as_array()),
            (false, true) => self.inner.decode_observables_detailed_batch_progress_bar(
                detectors.as_array(),
                leave_progress_bar_on_finish,
            ),
            (true, true) => self
                .inner
                .par_decode_observables_detailed_batch_progress_bar(
                    detectors.as_array(),
                    leave_progress_bar_on_finish,
                ),
        };
        results
            .into_iter()
            .map(ObservableDecodeResult::new)
            .collect()
    }

    #[pyo3(signature = (errors, parallel=false, progress_bar=true, leave_progress_bar_on_finish=false))]
    pub fn from_errors_decode_observables_detailed_batch(
        &mut self,
        errors: PyReadonlyArray2<'_, Bit>,
        parallel: bool,
        progress_bar: bool,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<ObservableDecodeResult> {
        let results = match (parallel, progress_bar) {
            (false, false) => self
                .inner
                .from_errors_decode_observables_detailed_batch(errors.as_array()),
            (true, false) => self
                .inner
                .par_from_errors_decode_observables_detailed_batch(errors.as_array()),
            (false, true) => self
                .inner
                .from_errors_decode_observables_detailed_batch_progress_bar(
                    errors.as_array(),
                    leave_progress_bar_on_finish,
                ),
            (true, true) => self
                .inner
                .par_from_errors_decode_observables_detailed_batch_progress_bar(
                    errors.as_array(),
                    leave_progress_bar_on_finish,
                ),
        };

        results
            .into_iter()
            .map(ObservableDecodeResult::new)
            .collect()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _observable_decoder<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<ObservableDecoderRunner>()?;
    m.add_class::<ObservableDecodeResult>()?;
    Ok(())
}

pub fn init_observable_decoder<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // Workaround for https://github.com/PyO3/pyo3/issues/759
    let decoder_module = PyModule::new(_py, "_relay_bp._observable_decoder")?;

    _observable_decoder(_py, &decoder_module)?;

    m.add("_observable_decoder", &decoder_module)?;
    decoder_module.setattr("__name__", "_observable_decoder")?;
    _py.import("sys")?
        .getattr("modules")?
        .set_item("_relay_bp._observable_decoder", &decoder_module)?;
    Ok(())
}
