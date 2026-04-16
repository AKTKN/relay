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
use pyo3::prelude::*;
use relay_bp::bp::lbf::{LbfDecoder, LbfDecoderConfig};
use relay_bp::decoder::Bit;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};

#[pyclass(extends=DynDecoder, subclass, module = "bp")]
#[allow(dead_code)]
pub struct LBFDecoder {}

#[pymethods]
impl LBFDecoder {
    #[new]
    #[pyo3(signature = (check_matrix, error_priors, max_iter=200, weight=1000, k_step=2))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        check_matrix: &Bound<'_, PyAny>,
        error_priors: &Bound<'_, PyArray1<f64>>,
        max_iter: usize,
        weight: usize,
        k_step: usize,
    ) -> PyResult<(Self, DynDecoder)> {
        if k_step < 2 || k_step % 2 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k_step must be an even integer >= 2",
            ));
        }

        let check_matrix = get_sprs_bit_matrix_from_python(py, check_matrix)?;

        // Keep compatibility with older Python callers while the Rust LBF config no
        // longer uses this knob.
        if weight != 1000 {
            let warnings = PyModule::import(py, "warnings")?;
            let msg = format!(
                "LBFDecoder: 'weight' is deprecated/ignored by the current backend. Got weight={weight}."
            );
            warnings.call_method1("warn", (msg,))?;
        }

        let config = LbfDecoderConfig {
            error_priors: unsafe { error_priors.as_array() }.to_owned(),
            max_iter,
            k_step,
        };

        let inner_decoder = LbfDecoder::new(Arc::new(check_matrix), Arc::new(config));
        let dyn_decoder = DynDecoder(Box::new(inner_decoder));

        Ok((LBFDecoder {}, dyn_decoder))
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
