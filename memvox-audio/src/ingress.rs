//! Mic capture + Silero VAD + state machine → SpeechStarted / SpeechSegment IPC.
//!
//! cpal drives mic capture on a real-time OS callback thread. Each 96ms frame
//! (1 536 samples @ 16 kHz mono) is fed to the Silero VAD ONNX model (via ort).
//! The VAD state machine (SILENT → SPEECH → TRAILING → EMIT) decides when to
//! emit a SpeechStarted signal (at onset) and a complete SpeechSegment (after
//! trailing silence).

use crate::barge_in::BargeInSignal;
use crate::ipc::{OutboundMsg, SpeechSegment, SpeechStarted};

pub struct AudioIngress {
    barge_in: BargeInSignal,
    // Phase 4: cpal stream, ort Session (Silero VAD), state machine fields
}

impl AudioIngress {
    pub fn new(barge_in: BargeInSignal) -> Self {
        Self { barge_in }
    }

    /// Start mic capture and VAD. Sends `OutboundMsg` to `tx` for each event.
    pub async fn run(
        self,
        tx: tokio::sync::mpsc::Sender<OutboundMsg>,
    ) -> anyhow::Result<()> {
        todo!("Phase 4: cpal input stream + Silero VAD loop");
    }
}

// ── VAD state machine ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadState {
    Silent,
    Speech,
    Trailing,
}
