#[allow(unused_imports)]
use crate::detector::types::{AllocationType, GpuAllocation};
#[cfg(target_os = "linux")]
use crate::GpuCheckpointError;
use crate::Result;
use std::fs;
#[allow(unused_imports)]
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[allow(unused_imports)]
use std::path::Path;
use tracing::debug;
#[cfg(target_os = "linux")]
use tracing::trace;

pub struct ProcessScanner;

impl ProcessScanner {
    pub fn scan_file_descriptors(pid: u32) -> Result<Vec<FileDescriptor>> {
        #[cfg(target_os = "linux")]
        {
            let fd_dir = format!("/proc/{pid}/fd");
            let mut descriptors = Vec::new();

            let entries = fs::read_dir(&fd_dir).map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    GpuCheckpointError::ProcessNotFound(pid)
                } else if e.kind() == std::io::ErrorKind::PermissionDenied {
                    GpuCheckpointError::PermissionDenied
                } else {
                    GpuCheckpointError::IoError(e)
                }
            })?;

            for entry in entries {
                let entry = entry?;
                let fd_path = entry.path();

                if let Ok(target) = fs::read_link(&fd_path) {
                    if let Some(fd_num) = fd_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .and_then(|s| s.parse::<i32>().ok())
                    {
                        let fd_info = FileDescriptor {
                            fd: fd_num,
                            target: target.to_string_lossy().to_string(),
                            metadata: fs::metadata(&fd_path).ok(),
                        };

                        trace!("Found fd {} -> {}", fd_num, fd_info.target);
                        descriptors.push(fd_info);
                    }
                }
            }

            debug!(
                "Found {} file descriptors for PID {}",
                descriptors.len(),
                pid
            );
            Ok(descriptors)
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = pid;
            debug!("File descriptor scanning not supported on this platform");
            Ok(Vec::new())
        }
    }

    pub fn classify_fd(fd: &FileDescriptor) -> Option<GpuFdInfo> {
        // NVIDIA GPU device files
        if fd.target.starts_with("/dev/nvidia") {
            let device_type = if fd.target.contains("nvidia-uvm") {
                GpuDeviceType::NvidiaUvm
            } else if fd.target.contains("nvidiactl") {
                GpuDeviceType::NvidiaControl
            } else if let Some(captures) = regex::Regex::new(r"/dev/nvidia(\d+)")
                .ok()
                .and_then(|re| re.captures(&fd.target))
            {
                let device_id = captures[1].parse::<u32>().ok();
                return Some(GpuFdInfo {
                    fd: fd.fd,
                    device_type: GpuDeviceType::NvidiaDevice,
                    device_id,
                    path: fd.target.clone(),
                });
            } else {
                GpuDeviceType::Unknown
            };

            return Some(GpuFdInfo {
                fd: fd.fd,
                device_type,
                device_id: None,
                path: fd.target.clone(),
            });
        }

        // AMD GPU device files
        if fd.target.starts_with("/dev/dri/") || fd.target.starts_with("/dev/kfd") {
            return Some(GpuFdInfo {
                fd: fd.fd,
                device_type: GpuDeviceType::AmdGpu,
                device_id: None,
                path: fd.target.clone(),
            });
        }

        // Shared memory that might be GPU-related
        if fd.target.starts_with("/dev/shm/") && fd.target.contains("cuda") {
            return Some(GpuFdInfo {
                fd: fd.fd,
                device_type: GpuDeviceType::SharedMemory,
                device_id: None,
                path: fd.target.clone(),
            });
        }

        None
    }

    pub fn check_process_cmdline(pid: u32) -> Result<String> {
        #[cfg(target_os = "linux")]
        {
            let cmdline_path = format!("/proc/{pid}/cmdline");
            let cmdline = fs::read_to_string(&cmdline_path)?;

            // Replace null bytes with spaces for readability
            Ok(cmdline.replace('\0', " ").trim().to_string())
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = pid;
            Ok("<cmdline not available on this platform>".to_string())
        }
    }

    pub fn check_process_environ(pid: u32) -> Result<Vec<(String, String)>> {
        #[cfg(target_os = "linux")]
        {
            let environ_path = format!("/proc/{pid}/environ");
            let environ = fs::read_to_string(&environ_path)?;

            let mut env_vars = Vec::new();
            for var in environ.split('\0') {
                if let Some((key, value)) = var.split_once('=') {
                    env_vars.push((key.to_string(), value.to_string()));
                }
            }

            Ok(env_vars)
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = pid;
            Ok(Vec::new())
        }
    }

    pub fn has_gpu_environment(pid: u32) -> Result<bool> {
        let env_vars = Self::check_process_environ(pid)?;

        let gpu_env_patterns = [
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "LD_LIBRARY_PATH",
            "ROCR_VISIBLE_DEVICES", // AMD
            "HIP_VISIBLE_DEVICES",  // AMD
        ];

        for (key, value) in &env_vars {
            if gpu_env_patterns.iter().any(|pattern| key.contains(pattern)) {
                debug!("Found GPU environment variable: {}={}", key, value);
                return Ok(true);
            }

            // Check for CUDA/GPU libraries in LD_LIBRARY_PATH
            if key == "LD_LIBRARY_PATH"
                && (value.contains("cuda") || value.contains("nvidia") || value.contains("rocm"))
            {
                debug!("Found GPU libraries in LD_LIBRARY_PATH");
                return Ok(true);
            }
        }

        Ok(false)
    }
}

#[derive(Debug)]
pub struct FileDescriptor {
    pub fd: i32,
    pub target: String,
    pub metadata: Option<fs::Metadata>,
}

#[derive(Debug, Clone)]
pub struct GpuFdInfo {
    pub fd: i32,
    pub device_type: GpuDeviceType,
    pub device_id: Option<u32>,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuDeviceType {
    NvidiaDevice,
    NvidiaControl,
    NvidiaUvm,
    AmdGpu,
    SharedMemory,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_nvidia_fd() {
        let fd = FileDescriptor {
            fd: 10,
            target: "/dev/nvidia0".to_string(),
            metadata: None,
        };

        let info = ProcessScanner::classify_fd(&fd).unwrap();
        assert_eq!(info.device_type, GpuDeviceType::NvidiaDevice);
        assert_eq!(info.device_id, Some(0));
    }

    #[test]
    fn test_classify_nvidia_uvm_fd() {
        let fd = FileDescriptor {
            fd: 11,
            target: "/dev/nvidia-uvm".to_string(),
            metadata: None,
        };

        let info = ProcessScanner::classify_fd(&fd).unwrap();
        assert_eq!(info.device_type, GpuDeviceType::NvidiaUvm);
    }
}
