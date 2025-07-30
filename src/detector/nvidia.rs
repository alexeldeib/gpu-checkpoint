use crate::Result;
use crate::detector::{GpuDetector, GpuVendor, DetectionResult, GpuAllocation, AllocationType};
use crate::detector::memory::MemoryMapParser;
use crate::detector::process::{ProcessScanner, GpuDeviceType};
use std::path::Path;
#[allow(unused_imports)]
use std::fs;
use tracing::{debug, info};

pub struct NvidiaDetector;

impl NvidiaDetector {
    pub fn new() -> Self {
        Self
    }
    
    fn detect_uvm_allocations(&self, regions: &[crate::detector::memory::MemoryRegion]) -> Vec<GpuAllocation> {
        let mut allocations = Vec::new();
        
        for region in regions {
            if let Some(pathname) = &region.pathname {
                // Direct UVM device mapping
                if pathname.contains("/dev/nvidia-uvm") {
                    let mut alloc = GpuAllocation::new(
                        region.start,
                        region.end,
                        AllocationType::Uvm
                    );
                    alloc.metadata.backing_file = Some(pathname.clone());
                    alloc.metadata.protection = region.perms.clone();
                    alloc.metadata.is_shared = region.perms.contains('s');
                    
                    debug!("Found UVM allocation: {:x}-{:x} ({} bytes)", 
                           region.start, region.end, alloc.size);
                    allocations.push(alloc);
                }
                
                // CUDA managed memory patterns
                if pathname.starts_with("[anon:") && pathname.contains("cuda") {
                    let mut alloc = GpuAllocation::new(
                        region.start,
                        region.end,
                        AllocationType::Managed
                    );
                    alloc.metadata.protection = region.perms.clone();
                    
                    debug!("Found managed memory allocation: {:x}-{:x} ({} bytes)",
                           region.start, region.end, alloc.size);
                    allocations.push(alloc);
                }
            }
        }
        
        allocations
    }
    
    fn detect_ipc_allocations(&self, regions: &[crate::detector::memory::MemoryRegion]) -> Vec<GpuAllocation> {
        let mut allocations = Vec::new();
        
        for region in regions {
            if let Some(pathname) = &region.pathname {
                // CUDA IPC shared memory patterns
                if pathname.starts_with("/dev/shm/") && 
                   (pathname.contains("cuda") || pathname.contains("nccl")) {
                    let mut alloc = GpuAllocation::new(
                        region.start,
                        region.end,
                        AllocationType::Ipc
                    );
                    alloc.metadata.backing_file = Some(pathname.clone());
                    alloc.metadata.protection = region.perms.clone();
                    alloc.metadata.is_shared = true;
                    
                    // Check if this is a distributed training allocation
                    if pathname.contains("nccl") || pathname.contains("horovod") {
                        alloc.alloc_type = AllocationType::Distributed;
                        alloc.metadata.is_distributed = true;
                    }
                    
                    debug!("Found IPC/distributed allocation: {:x}-{:x} ({} bytes)",
                           region.start, region.end, alloc.size);
                    allocations.push(alloc);
                }
            }
        }
        
        allocations
    }
    
    fn detect_bar_mappings(&self, regions: &[crate::detector::memory::MemoryRegion]) -> Vec<GpuAllocation> {
        let mut allocations = Vec::new();
        
        for region in regions {
            if let Some(pathname) = &region.pathname {
                // PCIe BAR resource mappings
                if pathname.contains("/sys/bus/pci/devices/") && 
                   pathname.contains(":00.0/resource") {
                    // Extract device info from path
                    let mut alloc = GpuAllocation::new(
                        region.start,
                        region.end,
                        AllocationType::BarMapped
                    );
                    alloc.metadata.backing_file = Some(pathname.clone());
                    alloc.metadata.protection = region.perms.clone();
                    
                    debug!("Found BAR mapping: {:x}-{:x} ({} bytes)",
                           region.start, region.end, alloc.size);
                    allocations.push(alloc);
                }
            }
        }
        
        allocations
    }
    
    fn check_nvidia_ml(&self, pid: u32) -> Result<Option<NvmlInfo>> {
        // In a real implementation, we would use nvidia-ml bindings
        // For now, we'll check for nvidia-smi output or /proc/driver/nvidia
        
        // Check if process has NVIDIA GPU context via /proc/driver/nvidia/gpus
        let nvidia_dir = "/proc/driver/nvidia/gpus";
        if Path::new(nvidia_dir).exists() {
            // This would parse actual NVML data
            debug!("NVIDIA driver detected, would query NVML for PID {}", pid);
        }
        
        Ok(None)
    }
}

impl GpuDetector for NvidiaDetector {
    fn detect_allocations(&self, pid: u32) -> Result<DetectionResult> {
        info!("Starting NVIDIA GPU detection for PID {}", pid);
        
        let mut result = DetectionResult::new(pid, GpuVendor::Nvidia);
        
        // Parse memory maps
        let regions = MemoryMapParser::parse_maps(pid)?;
        
        // Check file descriptors
        let fds = ProcessScanner::scan_file_descriptors(pid)?;
        let gpu_fds: Vec<_> = fds.iter()
            .filter_map(|fd| ProcessScanner::classify_fd(fd))
            .filter(|info| matches!(
                info.device_type, 
                GpuDeviceType::NvidiaDevice | 
                GpuDeviceType::NvidiaControl | 
                GpuDeviceType::NvidiaUvm
            ))
            .collect();
        
        if gpu_fds.is_empty() && !ProcessScanner::has_gpu_environment(pid)? {
            debug!("No NVIDIA GPU usage detected for PID {}", pid);
            return Ok(result);
        }
        
        // Detect different allocation types
        let uvm_allocs = self.detect_uvm_allocations(&regions);
        let ipc_allocs = self.detect_ipc_allocations(&regions);
        let bar_allocs = self.detect_bar_mappings(&regions);
        
        // Add device IDs from file descriptors
        for alloc in uvm_allocs {
            result.add_allocation(alloc);
        }
        for alloc in ipc_allocs {
            result.add_allocation(alloc);
        }
        for alloc in bar_allocs {
            result.add_allocation(alloc);
        }
        
        // Try to get additional info from NVML
        if let Ok(Some(nvml_info)) = self.check_nvidia_ml(pid) {
            debug!("NVML reports {} bytes GPU memory for PID {}", 
                   nvml_info.gpu_memory_used, pid);
        }
        
        info!("NVIDIA detection complete for PID {}: found {} allocations, {} problematic",
              pid, result.allocations.len(), 
              result.allocations.iter().filter(|a| a.is_problematic()).count());
        
        Ok(result)
    }
    
    fn is_gpu_process(&self, pid: u32) -> Result<bool> {
        // Quick check for NVIDIA GPU usage
        let fds = ProcessScanner::scan_file_descriptors(pid)?;
        
        for fd in &fds {
            if let Some(gpu_info) = ProcessScanner::classify_fd(fd) {
                if matches!(gpu_info.device_type, 
                    GpuDeviceType::NvidiaDevice | 
                    GpuDeviceType::NvidiaControl | 
                    GpuDeviceType::NvidiaUvm) {
                    return Ok(true);
                }
            }
        }
        
        // Also check environment
        ProcessScanner::has_gpu_environment(pid)
    }
    
    fn get_vendor(&self) -> GpuVendor {
        GpuVendor::Nvidia
    }
}

#[derive(Debug)]
struct NvmlInfo {
    gpu_memory_used: u64,
    device_id: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nvidia_detector_creation() {
        let detector = NvidiaDetector::new();
        assert_eq!(detector.get_vendor(), GpuVendor::Nvidia);
    }
}