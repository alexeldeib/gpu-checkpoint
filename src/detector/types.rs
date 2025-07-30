use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::SystemTime;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationType {
    /// Standard GPU memory allocation (cudaMalloc)
    Standard,

    /// Unified Virtual Memory allocation
    Uvm,

    /// Managed memory (cudaMallocManaged)
    Managed,

    /// IPC shared memory
    Ipc,

    /// Distributed training allocation (NCCL, etc)
    Distributed,

    /// Memory-mapped BAR region
    BarMapped,

    /// Host-pinned memory
    HostPinned,

    /// Unknown allocation type
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocation {
    /// Virtual address range start
    pub vaddr_start: u64,

    /// Virtual address range end
    pub vaddr_end: u64,

    /// Size in bytes
    pub size: u64,

    /// Allocation type
    pub alloc_type: AllocationType,

    /// GPU device ID
    pub device_id: Option<u32>,

    /// File descriptor (if memory-mapped)
    pub fd: Option<i32>,

    /// Additional metadata
    pub metadata: AllocationMetadata,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AllocationMetadata {
    /// Is this allocation part of a distributed setup?
    pub is_distributed: bool,

    /// NUMA node if applicable
    pub numa_node: Option<u32>,

    /// Backing file path if memory-mapped
    pub backing_file: Option<String>,

    /// Memory protection flags
    pub protection: String,

    /// Is this a shared mapping?
    pub is_shared: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Process ID
    pub pid: u32,

    /// GPU vendor
    pub vendor: GpuVendor,

    /// All detected allocations
    pub allocations: Vec<GpuAllocation>,

    /// Total GPU memory used
    pub total_gpu_memory: u64,

    /// Detection timestamp
    pub timestamp: SystemTime,

    /// Summary statistics
    pub stats: DetectionStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectionStats {
    pub standard_allocations: usize,
    pub uvm_allocations: usize,
    pub managed_allocations: usize,
    pub ipc_allocations: usize,
    pub distributed_allocations: usize,
    pub total_size: u64,
    pub largest_allocation: u64,
}

impl GpuAllocation {
    pub fn new(start: u64, end: u64, alloc_type: AllocationType) -> Self {
        Self {
            vaddr_start: start,
            vaddr_end: end,
            size: end - start,
            alloc_type,
            device_id: None,
            fd: None,
            metadata: AllocationMetadata::default(),
        }
    }

    pub fn is_problematic(&self) -> bool {
        matches!(
            self.alloc_type,
            AllocationType::Uvm
                | AllocationType::Managed
                | AllocationType::Ipc
                | AllocationType::Distributed
        )
    }
}

impl fmt::Display for AllocationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocationType::Standard => write!(f, "Standard"),
            AllocationType::Uvm => write!(f, "UVM"),
            AllocationType::Managed => write!(f, "Managed"),
            AllocationType::Ipc => write!(f, "IPC"),
            AllocationType::Distributed => write!(f, "Distributed"),
            AllocationType::BarMapped => write!(f, "BAR-Mapped"),
            AllocationType::HostPinned => write!(f, "Host-Pinned"),
            AllocationType::Unknown => write!(f, "Unknown"),
        }
    }
}

impl DetectionResult {
    pub fn new(pid: u32, vendor: GpuVendor) -> Self {
        Self {
            pid,
            vendor,
            allocations: Vec::new(),
            total_gpu_memory: 0,
            timestamp: SystemTime::now(),
            stats: DetectionStats::default(),
        }
    }

    pub fn add_allocation(&mut self, allocation: GpuAllocation) {
        self.total_gpu_memory += allocation.size;
        self.stats.total_size += allocation.size;

        if allocation.size > self.stats.largest_allocation {
            self.stats.largest_allocation = allocation.size;
        }

        match allocation.alloc_type {
            AllocationType::Standard => self.stats.standard_allocations += 1,
            AllocationType::Uvm => self.stats.uvm_allocations += 1,
            AllocationType::Managed => self.stats.managed_allocations += 1,
            AllocationType::Ipc => self.stats.ipc_allocations += 1,
            AllocationType::Distributed => self.stats.distributed_allocations += 1,
            _ => {}
        }

        self.allocations.push(allocation);
    }

    pub fn has_problematic_allocations(&self) -> bool {
        self.allocations.iter().any(|a| a.is_problematic())
    }
}
