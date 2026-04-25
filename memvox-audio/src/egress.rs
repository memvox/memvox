//! Speaker playback + rubato resampling + barge-in cancel.
//!
//! Receives `AudioChunk` PCM from Python (typically Kokoro at 24 kHz f32) and
//! `CancelPlayback` control messages.  Resamples to the device's native rate
//! using rubato's FFT resampler, then hands samples to a cpal output stream.
//!
//! Barge-in fast path: the cpal output callback polls `BargeInSignal` on every
//! invocation. When fired (by the ingress VAD thread) it clears the playback
//! buffer and emits silence — no Python or asyncio involved.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use rubato::{FftFixedIn, Resampler};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::barge_in::BargeInSignal;
use crate::ipc::{AudioChunk, InboundMsg};

const RESAMPLE_CHUNK: usize = 1024;   // input frames per rubato `process()` call

pub struct AudioEgress {
    barge_in: BargeInSignal,
}

impl AudioEgress {
    pub fn new(barge_in: BargeInSignal) -> Self {
        Self { barge_in }
    }

    pub async fn run(self, mut rx: mpsc::Receiver<InboundMsg>) -> Result<()> {
        // Stream creation lives in a non-async helper for the same reason as
        // ingress: cpal::Stream is `!Send` and rustc's drop-tracking poisons
        // the future even after `std::mem::forget`. The helper returns the
        // device rate and the shared playback buffer.
        let (device_rate, buffer) = open_output_stream(self.barge_in.clone())?;

        // ── Resampler state (built lazily, rebuilt if input rate changes) ───
        let mut resampler: Option<FftFixedIn<f32>> = None;
        let mut current_in_rate: u32 = 0;
        let mut input_buffer: VecDeque<f32> = VecDeque::new();

        // ── Main loop ───────────────────────────────────────────────────────
        while let Some(msg) = rx.recv().await {
            match msg {
                InboundMsg::AudioChunk(chunk) => {
                    handle_audio_chunk(
                        chunk,
                        device_rate,
                        &mut resampler,
                        &mut current_in_rate,
                        &mut input_buffer,
                        &buffer,
                        &self.barge_in,
                    )?;
                }
                InboundMsg::CancelPlayback => {
                    debug!("CancelPlayback ◼");
                    if let Ok(mut buf) = buffer.lock() {
                        buf.clear();
                    }
                    input_buffer.clear();
                }
            }
        }

        info!("egress: task exiting");
        Ok(())
    }
}

// ── cpal stream setup (sync — keeps `!Send` Stream out of the async future) ─

fn open_output_stream(
    barge_in: BargeInSignal,
) -> Result<(u32, Arc<Mutex<VecDeque<f32>>>)> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("no default output device")?;
    let supported = device.default_output_config()?;
    let device_rate = supported.sample_rate().0;
    let device_channels = supported.channels();
    info!(
        "egress: output device = {} ({} Hz, {} ch)",
        device.name()?, device_rate, device_channels
    );

    let config = StreamConfig {
        channels: device_channels,
        sample_rate: cpal::SampleRate(device_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    // Shared playback buffer (mono f32 at device_rate). Read by the cpal
    // callback (real-time thread), written by the async run loop.
    let buffer: Arc<Mutex<VecDeque<f32>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(device_rate as usize * 2)));
    let buffer_cb = buffer.clone();
    let barge_in_cb = barge_in;

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // Barge-in: drop everything queued and emit silence.
            if barge_in_cb.is_fired() {
                if let Ok(mut buf) = buffer_cb.lock() {
                    buf.clear();
                }
                for s in data.iter_mut() { *s = 0.0; }
                return;
            }

            let mut buf = match buffer_cb.lock() {
                Ok(b) => b,
                Err(_) => {
                    for s in data.iter_mut() { *s = 0.0; }
                    return;
                }
            };

            // Mono → device_channels by duplicating each sample across channels.
            let frames = data.len() / device_channels as usize;
            for frame_i in 0..frames {
                let sample = buf.pop_front().unwrap_or(0.0);
                for ch in 0..device_channels as usize {
                    data[frame_i * device_channels as usize + ch] = sample;
                }
            }
        },
        move |err| warn!("egress stream error: {}", err),
        None,
    )?;
    stream.play()?;
    // Keep the device open for the program's lifetime; OS reclaims it on exit.
    std::mem::forget(stream);

    Ok((device_rate, buffer))
}

// ── Helper: process one AudioChunk ────────────────────────────────────────────

fn handle_audio_chunk(
    chunk: AudioChunk,
    device_rate: u32,
    resampler: &mut Option<FftFixedIn<f32>>,
    current_in_rate: &mut u32,
    input_buffer: &mut VecDeque<f32>,
    playback_buffer: &Arc<Mutex<VecDeque<f32>>>,
    barge_in: &BargeInSignal,
) -> Result<()> {
    // Reset barge-in BEFORE pushing to playback buffer so the cpal callback
    // doesn't clear our just-pushed audio.
    barge_in.reset();

    if chunk.is_final || chunk.pcm.is_empty() {
        return Ok(());
    }

    if chunk.sample_rate == device_rate {
        // Fast path: no resampling needed.
        let mut buf = playback_buffer.lock().unwrap();
        for s in chunk.pcm {
            buf.push_back(s);
        }
        return Ok(());
    }

    // (Re)build the resampler if the input rate changed (e.g. TTS model swap).
    if resampler.is_none() || *current_in_rate != chunk.sample_rate {
        let r = FftFixedIn::<f32>::new(
            chunk.sample_rate as usize,
            device_rate as usize,
            RESAMPLE_CHUNK,
            2,   // sub_chunks — anti-aliasing parameter; 2 is a common default
            1,   // n_channels (mono)
        )?;
        *resampler = Some(r);
        *current_in_rate = chunk.sample_rate;
        debug!(
            "egress: resampler built {} → {} Hz ({}-sample input chunks)",
            chunk.sample_rate, device_rate, RESAMPLE_CHUNK
        );
    }
    let r = resampler.as_mut().unwrap();

    // Buffer input across calls so we always feed the resampler exactly
    // RESAMPLE_CHUNK samples at a time (its required input size).
    input_buffer.extend(chunk.pcm.iter());
    while input_buffer.len() >= RESAMPLE_CHUNK {
        let in_chunk: Vec<f32> = input_buffer.drain(..RESAMPLE_CHUNK).collect();
        let waves_out = r.process(&[in_chunk], None)?;
        let mut buf = playback_buffer.lock().unwrap();
        for &s in &waves_out[0] {
            buf.push_back(s);
        }
    }

    Ok(())
}
