use clap::{Parser, Subcommand};
use gpu_checkpoint::{
    checkpoint::{CheckpointConfig, CheckpointEngine, CheckpointStrategy},
    detector::CompositeDetector,
    utils,
};
use std::time::Duration;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

#[derive(Parser)]
#[command(name = "gpu-checkpoint")]
#[command(about = "GPU-aware checkpoint/restore system", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect GPU allocations in a process
    Detect {
        /// Process ID to analyze
        #[arg(short, long)]
        pid: u32,

        /// Output format (json, human)
        #[arg(short, long, default_value = "human")]
        format: String,
    },

    /// Checkpoint a process
    Checkpoint {
        /// Process ID to checkpoint
        #[arg(short, long)]
        pid: u32,

        /// Storage path for checkpoint data
        #[arg(short, long, default_value = "/tmp/gpu-checkpoint")]
        storage: String,

        /// Force specific strategy (auto, cuda, bar-sliding, hybrid)
        #[arg(long, default_value = "auto")]
        strategy: String,

        /// Storage bandwidth in MB/s
        #[arg(long, default_value = "1000")]
        bandwidth: u64,
    },

    /// Restore a process from checkpoint
    Restore {
        /// Checkpoint metadata file
        #[arg(short, long)]
        metadata: String,

        /// Storage path for checkpoint data
        #[arg(short, long, default_value = "/tmp/gpu-checkpoint")]
        storage: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    match cli.command {
        Commands::Detect { pid, format } => {
            info!("Detecting GPU allocations for PID {}", pid);

            let detector = CompositeDetector::new();
            let results = detector.detect_all(pid)?;

            if results.is_empty() {
                warn!("No GPU allocations detected for PID {}", pid);
                return Ok(());
            }

            match format.as_str() {
                "json" => {
                    println!("{}", serde_json::to_string_pretty(&results)?);
                }
                "human" => {
                    for result in &results {
                        println!("\n=== {} GPU Detection Results ===", result.vendor);
                        println!("Process ID: {}", result.pid);
                        println!(
                            "Total GPU Memory: {}",
                            utils::format_memory(result.total_gpu_memory)
                        );
                        println!("Allocations: {}", result.allocations.len());

                        if result.has_problematic_allocations() {
                            println!("\n⚠️  Problematic allocations detected!");
                        }

                        println!("\nAllocation Summary:");
                        println!("  Standard: {}", result.stats.standard_allocations);
                        println!("  UVM: {}", result.stats.uvm_allocations);
                        println!("  Managed: {}", result.stats.managed_allocations);
                        println!("  IPC: {}", result.stats.ipc_allocations);
                        println!("  Distributed: {}", result.stats.distributed_allocations);

                        if cli.verbose {
                            println!("\nDetailed Allocations:");
                            for (i, alloc) in result.allocations.iter().enumerate() {
                                println!("\n  [{}] {} allocation", i, alloc.alloc_type);
                                println!(
                                    "      Address: 0x{:016x} - 0x{:016x}",
                                    alloc.vaddr_start, alloc.vaddr_end
                                );
                                println!("      Size: {}", utils::format_memory(alloc.size));
                                if let Some(ref file) = alloc.metadata.backing_file {
                                    println!("      Backing: {file}");
                                }
                            }
                        }

                        // Recommend strategy
                        let strategy = CheckpointEngine::select_strategy(result);
                        println!("\nRecommended checkpoint strategy: {strategy:?}");
                    }
                }
                _ => {
                    error!("Unknown format: {}", format);
                    std::process::exit(1);
                }
            }
        }

        Commands::Checkpoint {
            pid,
            storage,
            strategy,
            bandwidth,
        } => {
            info!("Checkpointing PID {} to {}", pid, storage);

            // First detect to determine strategy
            let detector = CompositeDetector::new();
            let results = detector.detect_all(pid)?;

            if results.is_empty() {
                warn!("No GPU state to checkpoint for PID {}", pid);
                return Ok(());
            }

            let checkpoint_strategy = match strategy.as_str() {
                "auto" => CheckpointEngine::select_strategy(&results[0]),
                "cuda" => CheckpointStrategy::CudaCheckpoint,
                "bar-sliding" => CheckpointStrategy::BarSliding,
                "hybrid" => CheckpointStrategy::Hybrid,
                _ => {
                    error!("Unknown strategy: {}", strategy);
                    std::process::exit(1);
                }
            };

            // Create output directory if it doesn't exist
            std::fs::create_dir_all(&storage)?;

            let config = CheckpointConfig {
                strategy: checkpoint_strategy,
                storage_path: storage,
                bandwidth_mbps: bandwidth,
                timeout: Duration::from_secs(300),
                compression: false,
            };

            let engine = CheckpointEngine::new(config);

            println!("Using checkpoint strategy: {checkpoint_strategy:?}");

            let metadata = engine.checkpoint(pid, &results[0]).await?;
            println!(
                "Checkpoint completed in {}",
                utils::format_duration(metadata.duration_ms)
            );
            println!(
                "Checkpoint size: {}",
                utils::format_memory(metadata.size_bytes)
            );
            println!("Strategy used: {:?}", metadata.strategy_used);
        }

        Commands::Restore { metadata, storage } => {
            info!("Restoring from {} using storage {}", metadata, storage);
            
            // Parse the metadata path to get the checkpoint file
            let checkpoint_path = std::path::Path::new(&metadata);
            
            // Create restore engine
            let restore = gpu_checkpoint::restore::BarRestore::new();
            
            // Perform restore
            match restore.restore_from_checkpoint(checkpoint_path, None) {
                Ok(restore_metadata) => {
                    println!("Restore completed successfully!");
                    println!("Process ID: {}", restore_metadata.pid);
                    println!("Allocations restored: {}", restore_metadata.num_allocations);
                    println!(
                        "Total size: {}",
                        utils::format_memory(restore_metadata.total_size)
                    );
                    println!(
                        "Duration: {}",
                        utils::format_duration(restore_metadata.duration_ms)
                    );
                }
                Err(e) => {
                    error!("Restore failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}
