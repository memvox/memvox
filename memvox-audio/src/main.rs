//! memvox-audio — unified Rust audio I/O binary.
//!
//! Owns hardware mic/speaker access via cpal. Runs Silero (or energy) VAD on
//! input and rubato resampling on output. Communicates with the Python
//! orchestrator over two Unix sockets using length-prefixed bincode.
//!
//! Architecture:
//!     ┌──────────┐  out_chan   ┌───────────────┐  Unix sock  ┌──────────────┐
//!     │ ingress  ├────────────▶│ outbound_task ├────────────▶│ Python read  │
//!     └──────────┘             └───────────────┘             └──────────────┘
//!     ┌──────────┐   in_chan   ┌───────────────┐  Unix sock  ┌──────────────┐
//!     │  egress  │◀────────────┤  inbound_task │◀────────────┤ Python write │
//!     └──────────┘             └───────────────┘             └──────────────┘
//!
//! The ingress and egress tasks share a `BargeInSignal` (a wrapped AtomicBool)
//! so mic onset cancels playback within a single audio frame — no IPC round trip.

mod barge_in;
mod egress;
mod ingress;
mod ipc;

use anyhow::{Context, Result};
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::barge_in::BargeInSignal;
use crate::egress::AudioEgress;
use crate::ingress::AudioIngress;
use crate::ipc::{InboundMsg, OutboundMsg};

const DEFAULT_OUT_SOCK: &str = "/tmp/memvox-audio-out.sock";
const DEFAULT_IN_SOCK:  &str = "/tmp/memvox-audio-in.sock";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "memvox_audio=info".into()),
        )
        .init();

    let (out_sock, in_sock) = parse_args();
    info!("memvox-audio starting");
    info!("  outbound socket: {}", out_sock);
    info!("  inbound socket : {}", in_sock);

    // Remove stale socket files from a previous run (bind() fails if the path exists).
    let _ = std::fs::remove_file(&out_sock);
    let _ = std::fs::remove_file(&in_sock);

    let out_listener = UnixListener::bind(&out_sock)
        .with_context(|| format!("bind {}", out_sock))?;
    let in_listener = UnixListener::bind(&in_sock)
        .with_context(|| format!("bind {}", in_sock))?;

    // Tokio mpsc channels couple the audio tasks to the socket-forwarding tasks.
    // 64 slots is plenty: ingress emits ≤ ~10 events/sec, egress receives bursts.
    let (out_tx, out_rx) = mpsc::channel::<OutboundMsg>(64);
    let (in_tx,  in_rx)  = mpsc::channel::<InboundMsg>(64);

    let barge_in = BargeInSignal::new();

    // Spawn each long-lived task. They run concurrently on the tokio runtime.
    tokio::spawn(run_ingress(barge_in.clone(), out_tx));
    tokio::spawn(run_egress(barge_in.clone(), in_rx));
    tokio::spawn(outbound_task(out_listener, out_rx));
    tokio::spawn(inbound_task(in_listener, in_tx));

    info!("running — Ctrl-C to stop");
    tokio::signal::ctrl_c().await?;
    info!("shutting down");
    Ok(())
}

// ── Audio task wrappers ───────────────────────────────────────────────────────

async fn run_ingress(bi: BargeInSignal, tx: mpsc::Sender<OutboundMsg>) {
    let ingress = AudioIngress::new(bi);
    if let Err(e) = ingress.run(tx).await {
        warn!("ingress task ended with error: {:#}", e);
    }
}

async fn run_egress(bi: BargeInSignal, rx: mpsc::Receiver<InboundMsg>) {
    let egress = AudioEgress::new(bi);
    if let Err(e) = egress.run(rx).await {
        warn!("egress task ended with error: {:#}", e);
    }
}

// ── Socket forwarders ─────────────────────────────────────────────────────────

/// Accepts AudioIngressClient connections and forwards every OutboundMsg
/// produced by the ingress task to the connected client.
async fn outbound_task(
    listener: UnixListener,
    mut rx: mpsc::Receiver<OutboundMsg>,
) -> Result<()> {
    loop {
        let (mut stream, _) = listener.accept().await?;
        info!("outbound: client connected");
        loop {
            let msg = match rx.recv().await {
                Some(m) => m,
                None => return Ok(()),  // ingress channel closed → shutdown
            };
            if let Err(e) = ipc::write_msg(&mut stream, &msg).await {
                warn!("outbound: write failed ({}); waiting for next client", e);
                break;
            }
        }
    }
}

/// Accepts AudioEgressClient connections and forwards every InboundMsg
/// the client writes into the egress task's queue.
async fn inbound_task(
    listener: UnixListener,
    tx: mpsc::Sender<InboundMsg>,
) -> Result<()> {
    loop {
        let (mut stream, _) = listener.accept().await?;
        info!("inbound: client connected");
        loop {
            let msg: InboundMsg = match ipc::read_msg(&mut stream).await {
                Ok(m) => m,
                Err(e) => {
                    warn!("inbound: read ended ({}); waiting for next client", e);
                    break;
                }
            };
            if tx.send(msg).await.is_err() {
                return Ok(());  // egress task gone
            }
        }
    }
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn parse_args() -> (String, String) {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    (
        get("--out-sock", DEFAULT_OUT_SOCK),
        get("--in-sock",  DEFAULT_IN_SOCK),
    )
}
