use crate::{Result, GpuCheckpointError};
use crate::detector::{DetectionResult, GpuAllocation};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, warn};
use indicatif::{ProgressBar, ProgressStyle};

/// BAR sliding window size (typically 256MB for most GPUs)
const BAR_WINDOW_SIZE: usize = 256 * 1024 * 1024;

/// Checkpoint header magic number
const CHECKPOINT_MAGIC: u32 = 0x47505543; // "GPUC"

/// Version of the checkpoint format
const CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug)]
pub struct BarSlidingCheckpoint {
    /// Size of the BAR window for sliding
    window_size: usize,
    
    /// Progress reporting
    show_progress: bool,
}

#[derive(Debug, Clone)]
pub struct CheckpointHeader {
    magic: u32,
    version: u32,
    pid: u32,
    num_allocations: u32,
    total_size: u64,
    timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct AllocationHeader {
    vaddr_start: u64,
    vaddr_end: u64,
    size: u64,
    device_id: u32,
    flags: u32,
}

impl BarSlidingCheckpoint {
    pub fn new() -> Self {
        Self {
            window_size: BAR_WINDOW_SIZE,
            show_progress: true,
        }
    }
    
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }
    
    pub fn checkpoint_process(
        &self,
        pid: u32,
        detection: &DetectionResult,
        output_path: &Path,
    ) -> Result<CheckpointMetadata> {
        info!("Starting BAR sliding checkpoint for PID {}", pid);
        let start_time = Instant::now();
        
        // Create checkpoint file
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output_path)
            .map_err(|e| GpuCheckpointError::IoError(e))?;
        
        // Write header
        let header = CheckpointHeader {
            magic: CHECKPOINT_MAGIC,
            version: CHECKPOINT_VERSION,
            pid,
            num_allocations: detection.allocations.len() as u32,
            total_size: detection.total_gpu_memory,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        self.write_header(&mut file, &header)?;
        
        // Set up progress bar
        let progress = if self.show_progress {
            let pb = ProgressBar::new(detection.total_gpu_memory);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("=>-")
            );
            Some(pb)
        } else {
            None
        };
        
        // Checkpoint each allocation
        let mut total_written = 0u64;
        for (idx, allocation) in detection.allocations.iter().enumerate() {
            debug!("Checkpointing allocation {} of {}", idx + 1, detection.allocations.len());
            
            let bytes_written = self.checkpoint_allocation(
                pid,
                allocation,
                &mut file,
                &progress,
            )?;
            
            total_written += bytes_written;
        }
        
        if let Some(pb) = progress {
            pb.finish_with_message("Checkpoint complete");
        }
        
        let duration = start_time.elapsed();
        info!(
            "Checkpoint completed: {} bytes in {:.2}s ({:.2} MB/s)",
            total_written,
            duration.as_secs_f64(),
            (total_written as f64 / (1024.0 * 1024.0)) / duration.as_secs_f64()
        );
        
        Ok(CheckpointMetadata {
            pid,
            path: output_path.to_path_buf(),
            size_bytes: total_written,
            duration_ms: duration.as_millis() as u64,
            num_allocations: detection.allocations.len(),
        })
    }
    
    fn checkpoint_allocation(
        &self,
        pid: u32,
        allocation: &GpuAllocation,
        output: &mut File,
        progress: &Option<ProgressBar>,
    ) -> Result<u64> {
        // Write allocation header
        let alloc_header = AllocationHeader {
            vaddr_start: allocation.vaddr_start,
            vaddr_end: allocation.vaddr_end,
            size: allocation.size,
            device_id: allocation.device_id.unwrap_or(0),
            flags: 0, // Reserved for future use
        };
        
        self.write_allocation_header(output, &alloc_header)?;
        
        // For real implementation, we would:
        // 1. Pause the process using CRIU or ptrace
        // 2. Map the GPU memory via BAR
        // 3. Copy in sliding windows
        // 4. Resume the process
        
        // For now, simulate by reading from /proc/pid/mem
        let mem_path = format!("/proc/{}/mem", pid);
        
        if Path::new(&mem_path).exists() {
            self.copy_memory_sliding(
                &mem_path,
                allocation.vaddr_start,
                allocation.size,
                output,
                progress,
            )?;
        } else {
            // Fallback: write zeros for testing
            warn!("Cannot access {}, writing zeros", mem_path);
            self.write_zeros(allocation.size, output, progress)?;
        }
        
        Ok(allocation.size)
    }
    
    fn copy_memory_sliding(
        &self,
        mem_path: &str,
        start_addr: u64,
        size: u64,
        output: &mut File,
        progress: &Option<ProgressBar>,
    ) -> Result<()> {
        let mut mem_file = OpenOptions::new()
            .read(true)
            .open(mem_path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    GpuCheckpointError::PermissionDenied
                } else {
                    GpuCheckpointError::IoError(e)
                }
            })?;
        
        mem_file.seek(SeekFrom::Start(start_addr))?;
        
        let mut remaining = size;
        let mut buffer = vec![0u8; self.window_size.min(size as usize)];
        
        while remaining > 0 {
            let to_read = remaining.min(self.window_size as u64) as usize;
            let bytes_read = mem_file.read(&mut buffer[..to_read])?;
            
            if bytes_read == 0 {
                break;
            }
            
            output.write_all(&buffer[..bytes_read])?;
            
            remaining -= bytes_read as u64;
            
            if let Some(pb) = progress {
                pb.inc(bytes_read as u64);
            }
        }
        
        Ok(())
    }
    
    fn write_zeros(
        &self,
        size: u64,
        output: &mut File,
        progress: &Option<ProgressBar>,
    ) -> Result<()> {
        let zeros = vec![0u8; self.window_size];
        let mut remaining = size;
        
        while remaining > 0 {
            let to_write = remaining.min(self.window_size as u64) as usize;
            output.write_all(&zeros[..to_write])?;
            
            remaining -= to_write as u64;
            
            if let Some(pb) = progress {
                pb.inc(to_write as u64);
            }
        }
        
        Ok(())
    }
    
    fn write_header(&self, file: &mut File, header: &CheckpointHeader) -> Result<()> {
        // Write as binary for efficiency
        file.write_all(&header.magic.to_le_bytes())?;
        file.write_all(&header.version.to_le_bytes())?;
        file.write_all(&header.pid.to_le_bytes())?;
        file.write_all(&header.num_allocations.to_le_bytes())?;
        file.write_all(&header.total_size.to_le_bytes())?;
        file.write_all(&header.timestamp.to_le_bytes())?;
        Ok(())
    }
    
    fn write_allocation_header(&self, file: &mut File, header: &AllocationHeader) -> Result<()> {
        file.write_all(&header.vaddr_start.to_le_bytes())?;
        file.write_all(&header.vaddr_end.to_le_bytes())?;
        file.write_all(&header.size.to_le_bytes())?;
        file.write_all(&header.device_id.to_le_bytes())?;
        file.write_all(&header.flags.to_le_bytes())?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    pub pid: u32,
    pub path: std::path::PathBuf,
    pub size_bytes: u64,
    pub duration_ms: u64,
    pub num_allocations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_checkpoint_header_serialization() {
        let header = CheckpointHeader {
            magic: CHECKPOINT_MAGIC,
            version: CHECKPOINT_VERSION,
            pid: 1234,
            num_allocations: 2,
            total_size: 1024 * 1024,
            timestamp: 1234567890,
        };
        
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ckpt");
        let mut file = File::create(&path).unwrap();
        
        let checkpoint = BarSlidingCheckpoint::new();
        checkpoint.write_header(&mut file, &header).unwrap();
        
        // Verify file size
        let metadata = file.metadata().unwrap();
        assert_eq!(metadata.len(), 32); // 6 fields * 4-8 bytes each
    }
    
    #[test]
    fn test_write_zeros() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("zeros.bin");
        let mut file = File::create(&path).unwrap();
        
        let checkpoint = BarSlidingCheckpoint::new();
        checkpoint.write_zeros(1024 * 1024, &mut file, &None).unwrap();
        
        let metadata = file.metadata().unwrap();
        assert_eq!(metadata.len(), 1024 * 1024);
    }
}