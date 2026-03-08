// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, PyResult};

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use relay_bp::decoder::BPExtraResult;
use relay_bp::bipartite_graph::BipartiteGraph;
use relay_bp::decoder::{
    Bit, DecodeResult as DecodeResultInner, Decoder as DecoderInner, SparseBitMatrix,
};

fn slg_mbp_dynamics_entries_to_py<'py>(
    py: Python<'py>,
    entries: &[relay_bp::bp::slg_mbp::trace::SLGMBPDynamicsEntry],
) -> PyResult<Bound<'py, PyAny>> {
    let py_entries = pyo3::types::PyList::empty(py);
    for entry in entries {
        let entry_dict = pyo3::types::PyDict::new(py);
        entry_dict.set_item("stage", entry.stage.as_str())?;
        entry_dict.set_item("generation_index", entry.generation_index)?;
        entry_dict.set_item("member_index", entry.member_index)?;
        entry_dict.set_item("residual_syndrome_count", entry.residual_syndrome_count)?;
        entry_dict.set_item(
            "residual_adjacent_variable_count",
            entry.residual_adjacent_variable_count,
        )?;
        entry_dict.set_item("score", entry.score)?;
        entry_dict.set_item(
            "residual_adjacent_variable_indices",
            entry.residual_adjacent_variable_indices.clone(),
        )?;
        entry_dict.set_item(
            "residual_adjacent_variable_llrs",
            entry.residual_adjacent_variable_llrs.clone(),
        )?;
        entry_dict.set_item("converged", entry.converged)?;
        entry_dict.set_item("iteration_count", entry.iteration_count)?;
        entry_dict.set_item("estimated_error_weight", entry.estimated_error_weight)?;
        py_entries.append(entry_dict)?;
    }
    Ok(py_entries.into_any())
}

fn slg_mbp_score_spike_trace_to_py<'py>(
    py: Python<'py>,
    trace: &relay_bp::bp::slg_mbp::trace::SLGMBPScoreSpikeTrace,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("spike_generation_index", trace.spike_generation_index)?;
    dict.set_item("previous_generation_index", trace.previous_generation_index)?;
    dict.set_item("spike_prev_score", trace.spike_prev_score)?;
    dict.set_item("spike_score", trace.spike_score)?;
    dict.set_item("spike_delta_score", trace.spike_delta_score)?;
    dict.set_item("spike_delta_order_log10", trace.spike_delta_order_log10)?;
    dict.set_item("window_start_generation", trace.window_start_generation)?;
    dict.set_item("window_end_generation", trace.window_end_generation)?;
    dict.set_item(
        "spike_residual_adjacent_variable_indices",
        trace.spike_residual_adjacent_variable_indices.clone(),
    )?;

    let snapshots = pyo3::types::PyList::empty(py);
    for snap in &trace.snapshots {
        let sd = pyo3::types::PyDict::new(py);
        sd.set_item("generation_index", snap.generation_index)?;
        sd.set_item(
            "posterior_llr_all_variables",
            snap.posterior_llr_all_variables.clone(),
        )?;
        sd.set_item(
            "memory_strength_all_variables",
            snap.memory_strength_all_variables.clone(),
        )?;
        snapshots.append(sd)?;
    }
    dict.set_item("snapshots", snapshots)?;

    Ok(dict.into_any())
}

pub fn get_sprs_bit_matrix_from_python(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
) -> PyResult<SparseBitMatrix> {
    let matrix = matrix.call_method1("astype", ("uint8",))?;

    if let Ok(dense) = matrix.downcast::<PyArray2<Bit>>() {
        let dense = dense.to_owned_array();
        Ok(SparseBitMatrix::from_dense(dense))
    } else {
        let scipy_sparse = PyModule::import(py, "scipy.sparse")?;
        if !scipy_sparse
            .getattr("issparse")?
            .call1((&matrix,))?
            .is_truthy()?
        {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Input must be a numpy array or scipy sparse matrix (csr_matrix or csc_matrix)",
            ));
        }

        let typ = matrix.get_type().name()?;
        let is_csr = typ == "csr_matrix";
        let is_csc = typ == "csc_matrix";

        if !(is_csr || is_csc) {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported sparse format '{typ}'. Only CSR and CSC are supported."
            )));
        }

        let shape = matrix.getattr("shape")?;
        let rows: usize = shape.get_item(0)?.extract()?;
        let cols: usize = shape.get_item(1)?.extract()?;

        let data: Vec<Bit> = matrix.getattr("data")?.extract()?;
        let indices: Vec<usize> = matrix.getattr("indices")?.extract()?;
        let indptr: Vec<usize> = matrix.getattr("indptr")?.extract()?;

        if is_csr {
            Ok(SparseBitMatrix::new((rows, cols), indptr, indices, data).to_csc())
        } else {
            Ok(SparseBitMatrix::new_csc(
                (rows, cols),
                indptr,
                indices,
                data,
            ))
        }
    }
}

#[pyclass(subclass, module = "decoder")]
#[derive(Clone)]
pub struct DynDecoder(pub Box<dyn DecoderInner + Send + 'static>);

impl DynDecoder {
    pub fn inner(&mut self) -> &mut dyn DecoderInner {
        self.0.as_mut()
    }
}

#[pyclass(module = "decoder")]
pub struct DecodeResult {
    inner: DecodeResultInner,
}

impl DecodeResult {
    pub fn new(inner: DecodeResultInner) -> Self {
        DecodeResult { inner }
    }
}
#[pymethods]
impl DecodeResult {
    #[getter]
    pub fn decoding<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Bit>> {
        PyArray1::from_array(py, &self.inner.decoding)
    }

    #[getter]
    pub fn decoded_detectors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Bit>> {
        PyArray1::from_array(py, &self.inner.decoded_detectors)
    }

    #[getter]
    pub fn posterior_ratios<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_array(py, &self.inner.posterior_ratios)
    }

    #[getter]
    pub fn success(&self) -> bool {
        self.inner.success
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    pub fn max_iter(&self) -> usize {
        self.inner.max_iter
    }

    #[getter]
    pub fn relay_trace<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.extra {
            BPExtraResult::RelayTrace {
                leg_success,
                leg_iterations,
                leg_negative_llr_counts,
                leg_decodings,
                leg_posteriors,
                relay_parallel_avg_iter_seconds,
                dyn_phase_avg_iter_seconds,
            } => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("leg_success", leg_success.clone())?;
                dict.set_item("leg_iterations", leg_iterations.clone())?;
                dict.set_item("leg_negative_llr_counts", leg_negative_llr_counts.clone())?;
                dict.set_item(
                    "relay_parallel_avg_iter_seconds",
                    relay_parallel_avg_iter_seconds,
                )?;
                dict.set_item("dyn_phase_avg_iter_seconds", dyn_phase_avg_iter_seconds)?;

                let py_decodings = pyo3::types::PyList::empty(py);
                for arr in leg_decodings {
                    py_decodings.append(PyArray1::from_array(py, arr))?;
                }
                dict.set_item("leg_decodings", py_decodings)?;

                let py_posteriors = pyo3::types::PyList::empty(py);
                for arr in leg_posteriors {
                    py_posteriors.append(PyArray1::from_array(py, arr))?;
                }
                dict.set_item("leg_posteriors", py_posteriors)?;
                Ok(dict.into_any())
            }
            _ => Ok(py.None().into_bound(py).into_any()),
        }
    }

    #[getter]
    pub fn slg_mbp_trace<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.extra {
            BPExtraResult::SLGMBPTrace {
                phase1_converged,
                phase1_iterations,
                total_iterations,
                generation_count,
                generation_best_fitness,
                selected_solution_posterior,
                residual_weight_history,
                gamma_history,
                detailed_dynamics: _,
                score_spike_trace,
            } => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("phase1_converged", phase1_converged)?;
                dict.set_item("phase1_iterations", phase1_iterations)?;
                dict.set_item("total_iterations", total_iterations)?;
                dict.set_item("generation_count", generation_count)?;
                dict.set_item("generation_best_fitness", generation_best_fitness)?;
                dict.set_item("residual_weight_history", residual_weight_history)?;
                dict.set_item("gamma_history", gamma_history)?;

                if let Some(arr) = selected_solution_posterior {
                    dict.set_item(
                        "selected_solution_posterior",
                        PyArray1::from_array(py, arr),
                    )?;
                } else {
                    dict.set_item("selected_solution_posterior", py.None())?;
                }

                if let Some(trace) = score_spike_trace {
                    dict.set_item("score_spike_trace", slg_mbp_score_spike_trace_to_py(py, trace)?)?;
                } else {
                    dict.set_item("score_spike_trace", py.None())?;
                }

                Ok(dict.into_any())
            }
            _ => Ok(py.None().into_bound(py).into_any()),
        }
    }

    #[getter]
    pub fn slg_mbp_detailed_dynamics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.extra {
            BPExtraResult::SLGMBPTrace {
                detailed_dynamics,
                ..
            } => {
                if let Some(trace) = detailed_dynamics {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("observed_syndrome_weight", trace.observed_syndrome_weight)?;
                    dict.set_item("entries", slg_mbp_dynamics_entries_to_py(py, &trace.entries)?)?;
                    Ok(dict.into_any())
                } else {
                    Ok(py.None().into_bound(py).into_any())
                }
            }
            _ => Ok(py.None().into_bound(py).into_any()),
        }
    }

    #[getter]
    pub fn slg_mbp_score_spike_trace<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.extra {
            BPExtraResult::SLGMBPTrace {
                score_spike_trace,
                ..
            } => {
                if let Some(trace) = score_spike_trace {
                    slg_mbp_score_spike_trace_to_py(py, trace)
                } else {
                    Ok(py.None().into_bound(py).into_any())
                }
            }
            _ => Ok(py.None().into_bound(py).into_any()),
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _decoder<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<DecodeResult>()?;
    m.add_class::<DynDecoder>()?;
    Ok(())
}

pub fn init_decoder<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // Workaround for https://github.com/PyO3/pyo3/issues/759
    let decoder_module = PyModule::new(_py, "_relay_bp._decoder")?;

    _decoder(_py, &decoder_module)?;

    m.add("_decoder", &decoder_module)?;
    decoder_module.setattr("__name__", "_decoder")?;
    _py.import("sys")?
        .getattr("modules")?
        .set_item("_relay_bp._decoder", &decoder_module)?;
    Ok(())
}
