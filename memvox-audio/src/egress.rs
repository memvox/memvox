//! Speaker playback + rubato resampling + barge-in cancel.
//!
//! Receives AudioChunk PCM (24 kHz float32 from Kokoro via IPC) and
//! CancelPlayback control messages. rubato resamples from Kokoro's 24 kHz
//! to the device-native rate (typically 48 kHz). cpal drives the output
//! stream on a real-time OS callback thread.
//!
//! Barge-in fast path: the cpal output callback polls BargeInSignal between
//! frames. When fired (by the ingress VAD thread), playback stops without
//! any Python or asyncio involvement.

use crate::barge_in::BargeInSignal;
use crate::ipc::{AudioChunk, InboundMsg};

pub struct AudioEgress {
    barge_in: BargeInSignal,
    // Phase 4: cpal output stream, rubato resampler, playback queue
}

impl AudioEgress {
    pub fn new(barge_in: BargeInSignal) -> Self {
        Self { barge_in }
    }

    /// Start playback loop. Reads `InboundMsg` from `rx`.
    pub async fn run(
        self,
        mut rx: tokio::sync::mpsc::Receiver<InboundMsg>,
    ) -> anyhow::Result<()> {
        todo!("Phase 4: cpal output stream + rubato resampler + barge-in poll");
    }
}
