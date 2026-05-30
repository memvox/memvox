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
mod vad;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::barge_in::BargeInSignal;
use crate::egress::AudioEgress;
use crate::ingress::AudioIngress;
use crate::ipc::{InboundMsg, OutboundMsg};

const DEFAULT_OUT_SOCK: &str = "/tmp/memvox-audio-out.sock";
const DEFAULT_IN_SOCK:  &str = "/tmp/memvox-audio-in.sock";

#[derive(Clone, Default)]
struct Args {
    out_sock: String,
    in_sock:  String,
    input_device:  Option<String>,
    output_device: Option<String>,
    list_devices:  bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "memvox_audio=info".into()),
        )
        .init();

    let args = parse_args();

    // Always print the device inventory at startup so the user can pick names.
    list_devices();
    if args.list_devices {
        return Ok(());
    }

    info!("memvox-audio starting");
    info!("  outbound socket: {}", args.out_sock);
    info!("  inbound socket : {}", args.in_sock);
    if let Some(d) = &args.input_device  { info!("  input device  : substring '{}'", d); }
    if let Some(d) = &args.output_device { info!("  output device : substring '{}'", d); }

    // Remove stale socket files from a previous run (bind() fails if the path exists).
    let _ = std::fs::remove_file(&args.out_sock);
    let _ = std::fs::remove_file(&args.in_sock);

    let out_listener = UnixListener::bind(&args.out_sock)
        .with_context(|| format!("bind {}", args.out_sock))?;
    let in_listener = UnixListener::bind(&args.in_sock)
        .with_context(|| format!("bind {}", args.in_sock))?;

    // Tokio mpsc channels couple the audio tasks to the socket-forwarding tasks.
    // 64 slots is plenty: ingress emits ≤ ~10 events/sec, egress receives bursts.
    let (out_tx, out_rx) = mpsc::channel::<OutboundMsg>(64);
    let (in_tx,  in_rx)  = mpsc::channel::<InboundMsg>(64);

    let barge_in = BargeInSignal::new();

    // Spawn each long-lived task. They run concurrently on the tokio runtime.
    tokio::spawn(run_ingress(barge_in.clone(), out_tx, args.input_device.clone()));
    tokio::spawn(run_egress(barge_in.clone(), in_rx, args.output_device.clone()));
    tokio::spawn(outbound_task(out_listener, out_rx));
    tokio::spawn(inbound_task(in_listener, in_tx));

    info!("running — Ctrl-C to stop");
    tokio::signal::ctrl_c().await?;
    info!("shutting down");
    Ok(())
}

// ── Audio task wrappers ───────────────────────────────────────────────────────

async fn run_ingress(
    bi: BargeInSignal,
    tx: mpsc::Sender<OutboundMsg>,
    device_match: Option<String>,
) {
    let ingress = AudioIngress::new(bi);
    if let Err(e) = ingress.run(tx, device_match).await {
        warn!("ingress task ended with error: {:#}", e);
    }
}

async fn run_egress(
    bi: BargeInSignal,
    rx: mpsc::Receiver<InboundMsg>,
    device_match: Option<String>,
) {
    let egress = AudioEgress::new(bi);
    if let Err(e) = egress.run(rx, device_match).await {
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

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();

    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };

    Args {
        out_sock:      get("--out-sock").unwrap_or_else(|| DEFAULT_OUT_SOCK.to_string()),
        in_sock:       get("--in-sock").unwrap_or_else(|| DEFAULT_IN_SOCK.to_string()),
        input_device:  get("--input-device"),
        output_device: get("--output-device"),
        list_devices:  argv.iter().any(|a| a == "--list-devices"),
    }
}

/// Print every cpal-visible audio device. Used at startup so the user can
/// see what `--input-device` / `--output-device` substring matches against.
fn list_devices() {
    let host = cpal::default_host();
    println!("──── audio devices (cpal host: {}) ────", host.id().name());

    match host.input_devices() {
        Ok(devs) => {
            println!("  inputs:");
            for d in devs {
                let name = d.name().unwrap_or_else(|_| "<unnamed>".into());
                let cfg = d.default_input_config()
                    .map(|c| format!("{} Hz, {} ch, {:?}",
                        c.sample_rate().0, c.channels(), c.sample_format()))
                    .unwrap_or_else(|_| "no default config".into());
                println!("    • {}   ({})", name, cfg);
            }
        }
        Err(e) => println!("  inputs: error: {}", e),
    }

    match host.output_devices() {
        Ok(devs) => {
            println!("  outputs:");
            for d in devs {
                let name = d.name().unwrap_or_else(|_| "<unnamed>".into());
                let cfg = d.default_output_config()
                    .map(|c| format!("{} Hz, {} ch, {:?}",
                        c.sample_rate().0, c.channels(), c.sample_format()))
                    .unwrap_or_else(|_| "no default config".into());
                println!("    • {}   ({})", name, cfg);
            }
        }
        Err(e) => println!("  outputs: error: {}", e),
    }
    println!("───────────────────────────────────────");
}
