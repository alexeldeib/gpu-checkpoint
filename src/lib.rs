pub mod checkpoint;
pub mod detector;
pub mod restore;
pub mod utils;

pub use checkpoint::{CheckpointEngine, CheckpointStrategy};
pub use detector::{AllocationType, GpuAllocation, GpuDetector};
pub use restore::RestoreEngine;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuCheckpointError {
    #[error("Detection failed: {0}")]
    DetectionError(String),

    #[error("Checkpoint failed: {0}")]
    CheckpointError(String),

    #[error("Restore failed: {0}")]
    RestoreError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Process not found: {0}")]
    ProcessNotFound(u32),

    #[error("Permission denied")]
    PermissionDenied,

    #[error("GPU device error: {0}")]
    GpuDeviceError(String),

    #[error("Strategy selection failed: {0}")]
    StrategyError(String),
}

pub type Result<T> = std::result::Result<T, GpuCheckpointError>;
