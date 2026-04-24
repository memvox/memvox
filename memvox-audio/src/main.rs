mod barge_in;
mod egress;
mod ingress;
mod ipc;

use anyhow::Result;
use tokio::net::UnixListener;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("memvox_audio=debug")
        .init();

    let socket_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/memvox-audio.sock".to_string());

    info!("memvox-audio starting on {}", socket_path);

    // Phase 4: bind Unix socket, spawn ingress + egress tasks
    todo!("Phase 4: wire ingress, egress, and IPC listener");
}
