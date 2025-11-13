use pyo3::prelude::*;
use crate::decoder::DynDecoder;
// use relay_bp::ensemble_decoder::EnsembleDecoder;


#[pyclass(extends=DynDecoder, subclass, module = "bp")]
#[derive(Clone)]
pub struct PyEnsembleDecoder {
    // この構造体は、Python側での型ヒントのためだけに存在し、
    // 実際のデコーダーインスタンスは親クラスのDynDecoderに格納されます。
}

#[pymethods]
impl PyEnsembleDecoder {

}

pub fn init_ensemble_decoder<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnsembleDecoder>()?;
    Ok(())
}