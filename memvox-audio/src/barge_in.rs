//! Zero-latency internal barge-in cancel signal.
//!
//! When VAD fires a speech onset (ingress thread), it signals the playback
//! thread to stop *internally* — no Python, no asyncio queue, no IPC round trip.
//! This is the most latency-sensitive cancel path in the system.
//!
//! Implementation: a shared `AtomicBool` written by the ingress callback and
//! polled by the egress callback between audio frames.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone)]
pub struct BargeInSignal(Arc<AtomicBool>);

impl BargeInSignal {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Called by the ingress VAD callback on speech onset.
    pub fn fire(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Called by the egress playback callback to check for cancel.
    pub fn is_fired(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Reset after egress has acknowledged the cancel.
    pub fn reset(&self) {
        self.0.store(false, Ordering::Release);
    }
}
