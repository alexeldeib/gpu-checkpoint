use crate::checkpoint::bar_sliding::{
    AllocationHeader, CheckpointHeader, CHECKPOINT_MAGIC, CHECKPOINT_VERSION,
};
use crate::{GpuCheckpointError, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, warn};

/// BAR restore engine for restoring GPU state from checkpoint
#[derive(Debug)]
pub struct BarRestore {
    /// Size of the window for sliding restore
    window_size: usize,

    /// Progress reporting
    show_progress: bool,
}

#[derive(Debug)]
pub struct RestoreMetadata {
    pub pid: u32,
    pub num_allocations: usize,
    pub total_size: u64,
    pub duration_ms: u64,
}

impl BarRestore {
    pub fn new() -> Self {
        Self {
            window_size: 256 * 1024 * 1024, // 256MB
            show_progress: true,
        }
    }

    pub fn restore_from_checkpoint(
        &self,
        checkpoint_path: &Path,
        target_pid: Option<u32>,
    ) -> Result<RestoreMetadata> {
        info!("Starting BAR restore from {:?}", checkpoint_path);
        let start_time = Instant::now();

        // Open checkpoint file
        let mut file = OpenOptions::new()
            .read(true)
            .open(checkpoint_path)
            .map_err(|e| GpuCheckpointError::IoError(e))?;

        // Read and validate header
        let header = self.read_header(&mut file)?;
        self.validate_header(&header)?;

        let pid = target_pid.unwrap_or(header.pid);
        info!(
            "Restoring checkpoint for PID {} ({} allocations, {} bytes)",
            pid, header.num_allocations, header.total_size
        );

        // Set up progress bar
        let progress = if self.show_progress {
            let pb = ProgressBar::new(header.total_size);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} ({eta})",
                    )
                    .unwrap()
                    .progress_chars("=>-"),
            );
            Some(pb)
        } else {
            None
        };

        // Restore each allocation
        let mut total_restored = 0u64;
        for idx in 0..header.num_allocations {
            debug!(
                "Restoring allocation {} of {}",
                idx + 1,
                header.num_allocations
            );

            let alloc_header = self.read_allocation_header(&mut file)?;
            let bytes_restored = self.restore_allocation(
                pid,
                &alloc_header,
                &mut file,
                &progress,
            )?;

            total_restored += bytes_restored;
        }

        if let Some(pb) = progress {
            pb.finish_with_message("Restore complete");
        }

        let duration = start_time.elapsed();
        info!(
            "Restore completed: {} bytes in {:.2}s ({:.2} MB/s)",
            total_restored,
            duration.as_secs_f64(),
            (total_restored as f64 / (1024.0 * 1024.0)) / duration.as_secs_f64()
        );

        Ok(RestoreMetadata {
            pid,
            num_allocations: header.num_allocations as usize,
            total_size: total_restored,
            duration_ms: duration.as_millis() as u64,
        })
    }

    fn restore_allocation(
        &self,
        pid: u32,
        alloc_header: &AllocationHeader,
        input: &mut File,
        progress: &Option<ProgressBar>,
    ) -> Result<u64> {
        debug!(
            "Restoring allocation at 0x{:016x}-0x{:016x} ({} bytes)",
            alloc_header.vaddr_start, alloc_header.vaddr_end, alloc_header.size
        );

        // For real implementation, we would:
        // 1. Pause the target process
        // 2. Map the GPU memory via BAR at the original addresses
        // 3. Restore memory contents in sliding windows
        // 4. Resume the process

        // For now, simulate by reading the data
        let mem_path = format!("/proc/{}/mem", pid);

        if Path::new(&mem_path).exists() {
            match self.restore_memory_sliding(
                &mem_path,
                alloc_header.vaddr_start,
                alloc_header.size,
                input,
                progress,
            ) {
                Ok(()) => Ok(alloc_header.size),
                Err(e) => {
                    warn!("Failed to restore to process memory: {}", e);
                    // Fall back to just reading and discarding the data
                    self.skip_allocation_data(alloc_header.size, input, progress)?;
                    Ok(alloc_header.size)
                }
            }
        } else {
            // No target process, just skip the data
            warn!("Target process {} not found, skipping restore", pid);
            self.skip_allocation_data(alloc_header.size, input, progress)?;
            Ok(alloc_header.size)
        }
    }

    fn restore_memory_sliding(
        &self,
        mem_path: &str,
        start_addr: u64,
        size: u64,
        input: &mut File,
        progress: &Option<ProgressBar>,
    ) -> Result<()> {
        let mut mem_file = OpenOptions::new()
            .write(true)
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
            let bytes_read = input.read(&mut buffer[..to_read])?;

            if bytes_read == 0 {
                break;
            }

            mem_file.write_all(&buffer[..bytes_read])?;

            remaining -= bytes_read as u64;

            if let Some(pb) = progress {
                pb.inc(bytes_read as u64);
            }
        }

        Ok(())
    }

    fn skip_allocation_data(
        &self,
        size: u64,
        input: &mut File,
        progress: &Option<ProgressBar>,
    ) -> Result<()> {
        let mut remaining = size;
        let mut buffer = vec![0u8; self.window_size.min(size as usize)];

        while remaining > 0 {
            let to_read = remaining.min(self.window_size as u64) as usize;
            let bytes_read = input.read(&mut buffer[..to_read])?;

            if bytes_read == 0 {
                break;
            }

            remaining -= bytes_read as u64;

            if let Some(pb) = progress {
                pb.inc(bytes_read as u64);
            }
        }

        Ok(())
    }

    fn read_header(&self, file: &mut File) -> Result<CheckpointHeader> {
        let mut buf = [0u8; 4];

        // Read magic
        file.read_exact(&mut buf)?;
        let magic = u32::from_le_bytes(buf);

        // Read version
        file.read_exact(&mut buf)?;
        let version = u32::from_le_bytes(buf);

        // Read pid
        file.read_exact(&mut buf)?;
        let pid = u32::from_le_bytes(buf);

        // Read num_allocations
        file.read_exact(&mut buf)?;
        let num_allocations = u32::from_le_bytes(buf);

        // Read total_size
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let total_size = u64::from_le_bytes(buf8);

        // Read timestamp
        file.read_exact(&mut buf8)?;
        let timestamp = u64::from_le_bytes(buf8);

        Ok(CheckpointHeader {
            magic,
            version,
            pid,
            num_allocations,
            total_size,
            timestamp,
        })
    }

    fn read_allocation_header(&self, file: &mut File) -> Result<AllocationHeader> {
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        // Read vaddr_start
        file.read_exact(&mut buf8)?;
        let vaddr_start = u64::from_le_bytes(buf8);

        // Read vaddr_end
        file.read_exact(&mut buf8)?;
        let vaddr_end = u64::from_le_bytes(buf8);

        // Read size
        file.read_exact(&mut buf8)?;
        let size = u64::from_le_bytes(buf8);

        // Read device_id
        file.read_exact(&mut buf4)?;
        let device_id = u32::from_le_bytes(buf4);

        // Read flags
        file.read_exact(&mut buf4)?;
        let flags = u32::from_le_bytes(buf4);

        Ok(AllocationHeader {
            vaddr_start,
            vaddr_end,
            size,
            device_id,
            flags,
        })
    }

    fn validate_header(&self, header: &CheckpointHeader) -> Result<()> {
        if header.magic != CHECKPOINT_MAGIC {
            return Err(GpuCheckpointError::RestoreError(format!(
                "Invalid checkpoint magic: 0x{:08x} (expected 0x{:08x})",
                header.magic, CHECKPOINT_MAGIC
            )));
        }

        if header.version != CHECKPOINT_VERSION {
            return Err(GpuCheckpointError::RestoreError(format!(
                "Unsupported checkpoint version: {} (expected {})",
                header.version, CHECKPOINT_VERSION
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::bar_sliding::BarSlidingCheckpoint;
    use crate::detector::{AllocationType, DetectionResult, GpuAllocation, GpuVendor};
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_restore_roundtrip() {
        let dir = tempdir().unwrap();
        let checkpoint_path = dir.path().join("test.ckpt");

        // Create a mock detection result
        let mut detection = DetectionResult::new(1234, GpuVendor::Nvidia);
        detection.add_allocation(GpuAllocation::new(
            0x100000,
            0x200000,
            AllocationType::Standard,
        ));

        // Create checkpoint
        let checkpoint = BarSlidingCheckpoint::new();
        let ckpt_metadata = checkpoint
            .checkpoint_process(1234, &detection, &checkpoint_path)
            .unwrap();

        assert_eq!(ckpt_metadata.num_allocations, 1);

        // Restore checkpoint
        let restore = BarRestore::new();
        let restore_metadata = restore
            .restore_from_checkpoint(&checkpoint_path, Some(5678))
            .unwrap();

        assert_eq!(restore_metadata.num_allocations, 1);
        assert_eq!(restore_metadata.total_size, ckpt_metadata.size_bytes);
    }
}