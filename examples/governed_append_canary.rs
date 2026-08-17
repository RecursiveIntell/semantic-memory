//! Disposable canary helper: open a replication-enabled store and append one
//! governed fact through the production authority surface.
//!
//! Used by the governed-authority journaling canary: a fact appended here must
//! land in `mutation_journal` (via `append_verified_in_tx`) so that
//! `mnemes-sync-client` can export it. The idempotency suffix makes every
//! invocation distinct; rerunning with the same suffix replays the same key.
//!
//! Usage:
//! ```text
//! cargo run --example governed_append_canary -- \
//!   <base_dir> <home_device_id> <store_id> <stream_epoch> <idempotency_suffix> <content>
//! ```

use semantic_memory::{
    AuthorityIssuer, AuthorityPermit, MemoryConfig, MemoryStore, MockEmbedder, ReplicationMode,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        eprintln!(
            "usage: governed_append_canary <base_dir> <home_device_id> <store_id> \
             <stream_epoch> <idempotency_suffix> <content>"
        );
        std::process::exit(2);
    }
    let base_dir = PathBuf::from(&args[0]);
    let device = args[1].clone();
    let store_id = args[2].clone();
    let epoch: u64 = args[3].parse()?;
    let suffix = args[4].clone();
    let content = args[5].clone();

    let store = MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir,
            journal_device_id: Some(device.clone()),
            journal_store_id: Some(store_id.clone()),
            replication_mode: ReplicationMode::FactCreateRequired,
            replication_stream_epoch: epoch,
            ..Default::default()
        },
        Box::new(MockEmbedder::new(768)),
    )?;

    // Production governed path: an operator-token issuer mints the permit.
    let issuer = AuthorityIssuer::from_operator_token("canary-operator-token")
        .expect("operator token must be valid");
    let permit = issuer.mint_operator_system(
        "canary:principal",
        "canary:caller",
        AuthorityPermit::APPEND_CAPABILITY,
    );

    let receipt = store
        .authority()
        .append_with_metadata(
            permit,
            format!("canary-{suffix}"),
            "canary".into(),
            content,
            Some("canary-source".into()),
            Some(serde_json::json!({"canary": true, "suffix": suffix})),
        )
        .await?;

    println!(
        "fact_id={} after_epoch={} operation_id={}",
        receipt.affected_ids[0],
        receipt.after_epoch.0,
        receipt.operation_id
    );
    Ok(())
}
