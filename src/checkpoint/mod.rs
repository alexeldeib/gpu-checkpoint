pub mod bar_sliding;

pub use bar_sliding::{BarSlidingCheckpoint, CheckpointMetadata as BarCheckpointMetadata};

use crate::detector::DetectionResult;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointStrategy {
    /// Use CUDA checkpoint API (fastest, but limited)
    CudaCheckpoint,

    /// Use BAR sliding approach (universal, but slower)
    BarSliding,

    /// Hybrid approach based on allocation types
    Hybrid,

    /// Skip GPU state (data loss)
    SkipGpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub strategy: CheckpointStrategy,
    pub storage_path: String,
    pub bandwidth_mbps: u64,
    pub timeout: Duration,
    pub compression: bool,
}

pub struct CheckpointEngine {
    _config: CheckpointConfig,
}

impl CheckpointEngine {
    pub fn new(config: CheckpointConfig) -> Self {
        Self { _config: config }
    }

    pub fn select_strategy(detection: &DetectionResult) -> CheckpointStrategy {
        // If no allocations, we can skip GPU
        if detection.allocations.is_empty() {
            return CheckpointStrategy::SkipGpu;
        }

        // If we have problematic allocations, must use BAR sliding
        if detection.has_problematic_allocations() {
            return CheckpointStrategy::BarSliding;
        }

        // Otherwise, CUDA checkpoint should work
        CheckpointStrategy::CudaCheckpoint
    }

    pub async fn checkpoint(
        &self,
        pid: u32,
        detection: &DetectionResult,
    ) -> Result<CheckpointMetadata> {
        use std::time::Instant;
        let start = Instant::now();

        match self._config.strategy {
            CheckpointStrategy::BarSliding => {
                // Use BAR sliding for problematic allocations
                let bar_checkpoint = BarSlidingCheckpoint::new();
                let output_path =
                    PathBuf::from(&self._config.storage_path).join(format!("checkpoint_{pid}.bin"));

                let bar_metadata =
                    bar_checkpoint.checkpoint_process(pid, detection, &output_path)?;

                Ok(CheckpointMetadata {
                    pid,
                    strategy_used: CheckpointStrategy::BarSliding,
                    timestamp: SystemTime::now(),
                    size_bytes: bar_metadata.size_bytes,
                    duration_ms: bar_metadata.duration_ms,
                })
            }
            CheckpointStrategy::CudaCheckpoint => {
                // TODO: Implement CUDA checkpoint
                todo!("CUDA checkpoint not yet implemented")
            }
            CheckpointStrategy::Hybrid => {
                // TODO: Implement hybrid approach
                todo!("Hybrid checkpoint not yet implemented")
            }
            CheckpointStrategy::SkipGpu => {
                // No GPU state to checkpoint
                Ok(CheckpointMetadata {
                    pid,
                    strategy_used: CheckpointStrategy::SkipGpu,
                    timestamp: SystemTime::now(),
                    size_bytes: 0,
                    duration_ms: start.elapsed().as_millis() as u64,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub pid: u32,
    pub strategy_used: CheckpointStrategy,
    pub timestamp: SystemTime,
    pub size_bytes: u64,
    pub duration_ms: u64,
}
