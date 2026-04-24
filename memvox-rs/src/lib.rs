mod sentence_acc;

use pyo3::prelude::*;

/// memvox._rust — PyO3 extension for synchronous CPU hot paths.
///
/// Build:  maturin develop --manifest-path memvox-rs/Cargo.toml
/// Import: from memvox._rust import SentenceAccumulator
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sentence_acc::SentenceAccumulator>()?;
    Ok(())
}
