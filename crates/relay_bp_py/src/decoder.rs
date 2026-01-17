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
use relay_bp::bipartite_graph::BipartiteGraph;
use relay_bp::decoder::{
    Bit, DecodeResult as DecodeResultInner, Decoder as DecoderInner, SparseBitMatrix,
    BPExtraResult, EnsembleExtraResult, LsdResult as LsdResultInner,
};

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

#[pyclass(module = "decoder", name = "LsdResult")]
#[derive(Clone)]
pub struct PyLsdResult {
    inner: LsdResultInner,
}

#[pymethods]
impl PyLsdResult {
    #[getter]
    pub fn cluster_sizes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_vec(py, self.inner.cluster_sizes.clone())
    }

    #[getter]
    pub fn cluster_llrs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.cluster_llrs.clone())
    }

    #[getter]
    pub fn cluster_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_vec(py, self.inner.cluster_ids.clone())
    }

    #[getter]
    pub fn elapsed_time_micros(&self) -> u64 {
        self.inner.elapsed_time_micros
    }

    #[getter]
    pub fn lsd_correction<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<Bit>>> {
        self.inner.lsd_correction.as_ref().map(|v| PyArray1::from_array(py, v))
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
    pub fn lsd(&self) -> Option<PyLsdResult> {
        self.inner.lsd.clone().map(|lsd| PyLsdResult { inner: lsd })
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
    pub fn run_time_micros(&self) -> Option<u64> {
        self.inner.run_time_micros
    }

    #[getter]
    pub fn bad_syndrome_neighbour_indices<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<usize>>> {
        self.inner.bad_syndrome_neighbour_indices.as_ref().map(|v| PyArray1::from_slice(py, v))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _decoder<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<DecodeResult>()?;
    m.add_class::<DynDecoder>()?;
    m.add_class::<PyLsdResult>()?;
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
