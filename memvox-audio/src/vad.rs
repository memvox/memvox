//! Silero VAD v4 ONNX inference.
//!
//! Replaces the energy-RMS stand-in in `ingress.rs`. Same call site: feed one
//! 1536-sample f32 frame at 16 kHz, get p(speech) in [0, 1].
//!
//! The model is *stateful* — it carries an LSTM hidden + cell state across
//! frames within a session. Construct one `SileroVad` per process and call
//! `reset()` when starting a new logical session.
//!
//! Graph signature (v4):
//!   inputs:  input  f32 [1, 1536]
//!            sr     i64 [1]
//!            h      f32 [2, 1, 64]
//!            c      f32 [2, 1, 64]
//!   outputs: output f32 [1, 1]
//!            hn     f32 [2, 1, 64]
//!            cn     f32 [2, 1, 64]

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

// `ort::Error<_>` wraps raw FFI handles and is `!Send + !Sync`, so it can't be
// auto-converted into `anyhow::Error`. Use this on every ort call site.
fn oe<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("ort: {e}")
}

pub const SAMPLE_RATE: i64 = 16_000;
pub const SPEECH_THRESHOLD: f32 = 0.5;

const STATE_LEN: usize = 2 * 1 * 64; // 128 f32s
const STATE_SHAPE: [i64; 3] = [2, 1, 64];

pub struct SileroVad {
    session: Session,
    h: Vec<f32>,
    c: Vec<f32>,
}

impl SileroVad {
    pub fn new(model_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(oe)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(oe)?
            .with_intra_threads(1)
            .map_err(oe)?
            .commit_from_file(model_path)
            .map_err(oe)
            .with_context(|| format!("loading Silero VAD model: {}", model_path.display()))?;
        Ok(Self {
            session,
            h: vec![0.0; STATE_LEN],
            c: vec![0.0; STATE_LEN],
        })
    }

    /// Score one frame. `frame.len()` must match the model's window (1536 for v4).
    pub fn score(&mut self, frame: &[f32]) -> Result<f32> {
        let input_shape = [1_i64, frame.len() as i64];

        // ort's `Tensor::from_array` accepts a `(shape, Vec<T>)` tuple — avoids
        // pulling in ndarray, which would have to be version-matched to ort's
        // internal copy.
        let outputs = self
            .session
            .run(ort::inputs![
                "input" => Tensor::from_array((input_shape, frame.to_vec())).map_err(oe)?,
                "sr"    => Tensor::from_array(([1_i64], vec![SAMPLE_RATE])).map_err(oe)?,
                "h"     => Tensor::from_array((STATE_SHAPE, self.h.clone())).map_err(oe)?,
                "c"     => Tensor::from_array((STATE_SHAPE, self.c.clone())).map_err(oe)?,
            ])
            .map_err(oe)?;

        // try_extract_tensor returns (&Shape, &[T])
        let (_, prob_data) = outputs["output"].try_extract_tensor::<f32>().map_err(oe)?;
        let prob = *prob_data.first().context("empty 'output' tensor")?;

        // Persist new LSTM state for the next frame
        let (_, hn_data) = outputs["hn"].try_extract_tensor::<f32>().map_err(oe)?;
        self.h.clear();
        self.h.extend_from_slice(hn_data);
        let (_, cn_data) = outputs["cn"].try_extract_tensor::<f32>().map_err(oe)?;
        self.c.clear();
        self.c.extend_from_slice(cn_data);

        Ok(prob)
    }

    #[allow(dead_code)] // Phase 2: called at session boundaries
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }
}

/// Resolve the path to `silero_vad.onnx`.
///
/// Order of resolution:
///   1. `MEMVOX_SILERO_VAD` env var (explicit override)
///   2. `<CARGO_MANIFEST_DIR>/models/silero_vad.onnx` (dev / `cargo run`)
///   3. `<exe_dir>/models/silero_vad.onnx` (installed binary)
pub fn silero_model_path() -> PathBuf {
    if let Ok(p) = std::env::var("MEMVOX_SILERO_VAD") {
        return PathBuf::from(p);
    }
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx");
    if dev.exists() {
        return dev;
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.join("models/silero_vad.onnx")))
        .unwrap_or(dev)
}
