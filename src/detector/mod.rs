mod memory;
mod nvidia;
mod process;
mod types;

pub use nvidia::NvidiaDetector;
pub use process::ProcessScanner;
pub use types::{AllocationType, DetectionResult, GpuAllocation, GpuVendor};

use crate::Result;
use std::path::Path;
use tracing::{debug, info, warn};

pub trait GpuDetector: Send + Sync {
    fn detect_allocations(&self, pid: u32) -> Result<DetectionResult>;

    fn is_gpu_process(&self, pid: u32) -> Result<bool>;

    fn get_vendor(&self) -> GpuVendor;
}

pub struct CompositeDetector {
    detectors: Vec<Box<dyn GpuDetector>>,
}

impl CompositeDetector {
    pub fn new() -> Self {
        let mut detectors: Vec<Box<dyn GpuDetector>> = Vec::new();

        // Add NVIDIA detector if available
        if Path::new("/dev/nvidia0").exists() || Path::new("/dev/nvidiactl").exists() {
            info!("NVIDIA GPU detected, adding NVIDIA detector");
            detectors.push(Box::new(NvidiaDetector::new()));
        }

        // Future: Add AMD, Intel detectors here

        if detectors.is_empty() {
            warn!("No GPU detectors available on this system");
        }

        Self { detectors }
    }

    pub fn detect_all(&self, pid: u32) -> Result<Vec<DetectionResult>> {
        let mut results = Vec::new();

        for detector in &self.detectors {
            match detector.detect_allocations(pid) {
                Ok(result) => {
                    debug!(
                        "Detector {:?} found {} allocations for PID {}",
                        detector.get_vendor(),
                        result.allocations.len(),
                        pid
                    );
                    results.push(result);
                }
                Err(e) => {
                    warn!(
                        "Detector {:?} failed for PID {}: {}",
                        detector.get_vendor(),
                        pid,
                        e
                    );
                }
            }
        }

        Ok(results)
    }
}
