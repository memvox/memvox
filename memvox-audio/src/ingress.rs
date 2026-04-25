//! Mic capture + VAD + state machine → SpeechStarted / SpeechSegment IPC.
//!
//! cpal drives mic capture on a real-time OS callback thread. Each 96 ms frame
//! (1 536 samples @ 16 kHz mono) is fed to the VAD. The state machine
//! (SILENT → SPEECH → TRAILING → emit) decides when to emit a `SpeechStarted`
//! signal at onset and a complete `SpeechSegment` after trailing silence.
//!
//! Currently uses energy (RMS) VAD as a stand-in for Silero. The `ort`
//! dependency is in Cargo.toml so a `SileroVad` implementation can be dropped
//! in later behind the same call site.

use std::time::Instant;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, StreamConfig};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::barge_in::BargeInSignal;
use crate::ipc::{OutboundMsg, SpeechSegment, SpeechStarted};

// ── VAD / framing constants ───────────────────────────────────────────────────

const SAMPLE_RATE: u32       = 16_000;
const FRAME_SAMPLES: usize   = 1_536;   // 96 ms @ 16 kHz — Silero's native window
const ONSET_FRAMES: usize    = 2;       // ~192 ms of speech to confirm utterance start
const TRAILING_FRAMES: usize = 5;       // ~480 ms of silence to end utterance
const MIN_SPEECH_FRAMES: usize = 3;     // discard bursts shorter than 288 ms (likely noise)
const ENERGY_THRESHOLD: f32  = 0.01;    // RMS threshold; tune per environment

// ── State machine ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadState {
    Silent,
    Speech,
    Trailing,
}

pub struct AudioIngress {
    barge_in: BargeInSignal,
}

impl AudioIngress {
    pub fn new(barge_in: BargeInSignal) -> Self {
        Self { barge_in }
    }

    pub async fn run(
        self,
        out_tx: mpsc::Sender<OutboundMsg>,
    ) -> Result<()> {
        // Stream creation lives in a non-async helper because cpal::Stream is
        // `!Send` (it holds raw ALSA pointers). Building the stream here would
        // poison this future's auto-`Send` even after `std::mem::forget`,
        // because rustc's drop-tracking analysis is conservative across
        // `await` points. Splitting the sync setup from the async loop sides-
        // tep the issue cleanly.
        let mut sample_rx = open_input_stream()?;

        // ── State machine state ─────────────────────────────────────────────
        let mut frame_buf: Vec<f32> = Vec::with_capacity(FRAME_SAMPLES);
        let mut state = VadState::Silent;
        let mut speech_frames: Vec<Vec<f32>> = Vec::new();
        let mut onset_count: usize = 0;
        let mut trailing_count: usize = 0;
        let mut onset_ts_ms: u64 = 0;
        let session_start = Instant::now();

        // ── Main loop: pull audio chunks, assemble frames, run VAD ──────────
        while let Some(chunk) = sample_rx.recv().await {
            for &sample in &chunk {
                frame_buf.push(sample);
                if frame_buf.len() < FRAME_SAMPLES {
                    continue;
                }

                let frame: Vec<f32> = std::mem::take(&mut frame_buf);
                frame_buf.reserve(FRAME_SAMPLES);
                let is_speech = energy_vad(&frame);
                let now_ms = session_start.elapsed().as_millis() as u64;

                match state {
                    VadState::Silent => {
                        if is_speech {
                            onset_count += 1;
                            speech_frames.push(frame);
                            if onset_count >= ONSET_FRAMES {
                                state = VadState::Speech;
                                onset_ts_ms = now_ms;
                                self.barge_in.fire();   // tell egress to stop playback
                                let msg = OutboundMsg::SpeechStarted(SpeechStarted {
                                    timestamp_ms: onset_ts_ms,
                                });
                                let _ = out_tx.send(msg).await;
                                debug!("VAD ▶ SpeechStarted ts={} ms", onset_ts_ms);
                            }
                        } else {
                            onset_count = 0;
                            speech_frames.clear();
                        }
                    }
                    VadState::Speech => {
                        speech_frames.push(frame);
                        if !is_speech {
                            state = VadState::Trailing;
                            trailing_count = 1;
                        }
                    }
                    VadState::Trailing => {
                        speech_frames.push(frame);
                        if is_speech {
                            state = VadState::Speech;
                            trailing_count = 0;
                        } else {
                            trailing_count += 1;
                            if trailing_count >= TRAILING_FRAMES {
                                let n_frames = speech_frames.len();
                                if n_frames >= MIN_SPEECH_FRAMES {
                                    let audio = pcm_f32_to_i16(&speech_frames);
                                    let n_samples = audio.len();
                                    let duration_ms =
                                        n_samples as f32 / SAMPLE_RATE as f32 * 1000.0;
                                    let msg = OutboundMsg::SpeechSegment(SpeechSegment {
                                        audio,
                                        speech_prob: 0.9,
                                        duration_ms,
                                        timestamp_start_ms: onset_ts_ms,
                                    });
                                    let _ = out_tx.send(msg).await;
                                    debug!(
                                        "VAD ▶ SpeechSegment {} samples ({:.0} ms)",
                                        n_samples, duration_ms
                                    );
                                }
                                state = VadState::Silent;
                                speech_frames.clear();
                                onset_count = 0;
                                trailing_count = 0;
                            }
                        }
                    }
                }
            }
        }

        info!("ingress: task exiting");
        Ok(())
    }
}

// ── cpal stream setup (sync — keeps `!Send` Stream out of the async future) ─

fn open_input_stream() -> Result<mpsc::UnboundedReceiver<Vec<f32>>> {
    let (sample_tx, sample_rx) = mpsc::unbounded_channel::<Vec<f32>>();

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default input device")?;
    info!("ingress: input device = {}", device.name()?);

    let config = StreamConfig {
        channels: 1,
        sample_rate: SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Real-time thread — keep this fast. We allocate a Vec here for
            // simplicity; a lock-free SPSC ring buffer would avoid that.
            let _ = sample_tx.send(data.to_vec());
        },
        move |err| warn!("ingress stream error: {}", err),
        None,
    )?;
    stream.play()?;
    // Keep the device open for the program's lifetime; the OS reclaims it on
    // process exit. `forget` is necessary because Stream is !Send and we can't
    // hold it as a value in an async fn.
    std::mem::forget(stream);
    info!("ingress: mic streaming at {} Hz mono f32", SAMPLE_RATE);

    Ok(sample_rx)
}

// ── Energy VAD ────────────────────────────────────────────────────────────────

fn energy_vad(frame: &[f32]) -> bool {
    let sum_sq: f64 = frame.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (sum_sq / frame.len() as f64).sqrt() as f32;
    rms > ENERGY_THRESHOLD
}

fn pcm_f32_to_i16(frames: &[Vec<f32>]) -> Vec<i16> {
    let total: usize = frames.iter().map(|f| f.len()).sum();
    let mut out = Vec::with_capacity(total);
    for frame in frames {
        for &s in frame {
            let clamped = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32);
            out.push(clamped as i16);
        }
    }
    out
}
