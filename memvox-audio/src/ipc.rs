//! Unix socket framing for all Python ↔ Rust audio IPC.
//!
//! Wire format: 4-byte little-endian length prefix followed by a bincode payload.
//!
//! Outbound (Rust → Python):
//!   - `SpeechStarted`  — VAD onset, before full segment is ready
//!   - `SpeechSegment`  — complete utterance after trailing silence
//!
//! Inbound (Python → Rust):
//!   - `AudioChunk`     — PCM sentence burst from TTSEngine for playback
//!   - `CancelPlayback` — barge-in cancel from Python (fast path already handled internally)

use serde::{Deserialize, Serialize};

// ── Outbound (Rust → Python) ──────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub enum OutboundMsg {
    SpeechStarted(SpeechStarted),
    SpeechSegment(SpeechSegment),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpeechStarted {
    pub timestamp_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpeechSegment {
    /// PCM 16 kHz mono int16, little-endian
    pub audio: Vec<i16>,
    pub speech_prob: f32,
    pub duration_ms: f32,
    pub timestamp_start_ms: u64,
}

// ── Inbound (Python → Rust) ───────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub enum InboundMsg {
    AudioChunk(AudioChunk),
    CancelPlayback,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AudioChunk {
    /// PCM 24 kHz float32, little-endian (Kokoro output rate, resampled by egress)
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    pub is_final: bool,
}

// ── Framing helpers ───────────────────────────────────────────────────────────

use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub async fn write_msg<W, T>(writer: &mut W, msg: &T) -> anyhow::Result<()>
where
    W: AsyncWriteExt + Unpin,
    T: Serialize,
{
    let payload = bincode::serialize(msg)?;
    let len = payload.len() as u32;
    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(&payload).await?;
    Ok(())
}

pub async fn read_msg<R, T>(reader: &mut R) -> anyhow::Result<T>
where
    R: AsyncReadExt + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    Ok(bincode::deserialize(&buf)?)
}
