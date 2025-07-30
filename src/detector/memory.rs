use crate::Result;
#[cfg(target_os = "linux")]
use crate::GpuCheckpointError;
use crate::detector::types::{GpuAllocation, AllocationType, AllocationMetadata};
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::{BufRead, BufReader};
#[allow(unused_imports)]
use std::path::Path;
use tracing::debug;
#[cfg(target_os = "linux")]
use tracing::trace;

#[derive(Debug)]
pub struct MemoryRegion {
    pub start: u64,
    pub end: u64,
    pub perms: String,
    pub offset: u64,
    pub dev: String,
    pub inode: u64,
    pub pathname: Option<String>,
}

pub struct MemoryMapParser;

impl MemoryMapParser {
    pub fn parse_maps(pid: u32) -> Result<Vec<MemoryRegion>> {
        #[cfg(target_os = "linux")]
        {
            let maps_path = format!("/proc/{}/maps", pid);
            let file = File::open(&maps_path).map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    GpuCheckpointError::ProcessNotFound(pid)
                } else if e.kind() == std::io::ErrorKind::PermissionDenied {
                    GpuCheckpointError::PermissionDenied
                } else {
                    GpuCheckpointError::IoError(e)
                }
            })?;
            
            let reader = BufReader::new(file);
            let mut regions = Vec::new();
            
            for line in reader.lines() {
                let line = line?;
                if let Some(region) = Self::parse_line(&line) {
                    trace!("Parsed memory region: {:?}", region);
                    regions.push(region);
                }
            }
            
            debug!("Parsed {} memory regions for PID {}", regions.len(), pid);
            Ok(regions)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            let _ = pid;
            debug!("Memory map parsing not supported on this platform");
            Ok(Vec::new())
        }
    }
    
    fn parse_line(line: &str) -> Option<MemoryRegion> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            return None;
        }
        
        // Parse address range (e.g., "7f1234567000-7f1234568000")
        let addr_parts: Vec<&str> = parts[0].split('-').collect();
        if addr_parts.len() != 2 {
            return None;
        }
        
        let start = u64::from_str_radix(addr_parts[0], 16).ok()?;
        let end = u64::from_str_radix(addr_parts[1], 16).ok()?;
        
        // Parse permissions (e.g., "rw-p")
        let perms = parts[1].to_string();
        
        // Parse offset
        let offset = u64::from_str_radix(parts[2], 16).ok()?;
        
        // Parse device (e.g., "00:00" or "fd:01")
        let dev = parts[3].to_string();
        
        // Parse inode
        let inode = parts[4].parse::<u64>().ok()?;
        
        // Parse pathname (optional)
        let pathname = if parts.len() > 5 {
            Some(parts[5..].join(" "))
        } else {
            None
        };
        
        Some(MemoryRegion {
            start,
            end,
            perms,
            offset,
            dev,
            inode,
            pathname,
        })
    }
    
    pub fn classify_region(region: &MemoryRegion) -> Option<GpuAllocation> {
        let pathname = region.pathname.as_ref()?;
        
        // NVIDIA GPU memory patterns
        if pathname.contains("/dev/nvidia") {
            let alloc_type = if pathname.contains("nvidia-uvm") {
                AllocationType::Uvm
            } else {
                AllocationType::Standard
            };
            
            let mut allocation = GpuAllocation::new(region.start, region.end, alloc_type);
            allocation.metadata = AllocationMetadata {
                backing_file: Some(pathname.clone()),
                protection: region.perms.clone(),
                is_shared: region.perms.contains('s'),
                ..Default::default()
            };
            
            return Some(allocation);
        }
        
        // CUDA managed memory (often shows as anonymous mappings with specific patterns)
        if pathname == "[heap]" || pathname.starts_with("[anon:") {
            // Check for CUDA-specific anonymous mapping patterns
            if region.end - region.start >= 1024 * 1024 * 64 { // >= 64MB
                // Large anonymous mappings might be CUDA managed memory
                let mut allocation = GpuAllocation::new(
                    region.start, 
                    region.end, 
                    AllocationType::Unknown
                );
                allocation.metadata.protection = region.perms.clone();
                return Some(allocation);
            }
        }
        
        // Check for GPU BAR mappings (PCIe memory-mapped regions)
        if pathname.contains("/sys/bus/pci/devices/") && pathname.contains("resource") {
            let mut allocation = GpuAllocation::new(
                region.start, 
                region.end, 
                AllocationType::BarMapped
            );
            allocation.metadata.backing_file = Some(pathname.clone());
            allocation.metadata.protection = region.perms.clone();
            return Some(allocation);
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_maps_line() {
        let line = "7f1234567000-7f1234568000 rw-p 00000000 00:00 0 /dev/nvidia0";
        let region = MemoryMapParser::parse_line(line).unwrap();
        
        assert_eq!(region.start, 0x7f1234567000);
        assert_eq!(region.end, 0x7f1234568000);
        assert_eq!(region.perms, "rw-p");
        assert_eq!(region.pathname, Some("/dev/nvidia0".to_string()));
    }
    
    #[test]
    fn test_classify_nvidia_uvm() {
        let region = MemoryRegion {
            start: 0x7f0000000000,
            end: 0x7f0001000000,
            perms: "rw-s".to_string(),
            offset: 0,
            dev: "00:00".to_string(),
            inode: 0,
            pathname: Some("/dev/nvidia-uvm".to_string()),
        };
        
        let allocation = MemoryMapParser::classify_region(&region).unwrap();
        assert_eq!(allocation.alloc_type, AllocationType::Uvm);
        assert_eq!(allocation.size, 0x1000000); // 16MB
    }
}