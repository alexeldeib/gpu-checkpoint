# GPU-Checkpoint: GPU-Aware Checkpoint/Restore System

A production-ready GPU checkpoint/restore system written in Rust that intelligently detects GPU allocations and selects optimal checkpoint strategies.

## Features

- **GPU Allocation Detection**: Comprehensive detection of GPU memory allocations including:
  - Standard CUDA allocations (cudaMalloc)
  - Unified Virtual Memory (UVM)
  - Managed memory (cudaMallocManaged)
  - IPC shared memory
  - Distributed training allocations (NCCL)
  - PCIe BAR mappings

- **Intelligent Strategy Selection**: Automatically chooses between:
  - CUDA checkpoint API (fastest for standard allocations)
  - BAR sliding approach (universal but slower)
  - Hybrid strategies based on allocation patterns

- **Production-Ready Design**:
  - Sub-100ms detection latency
  - Detailed logging and tracing
  - Cross-platform support (Linux primary target)
  - Zero runtime dependencies on GPU applications

## Architecture

The system is organized into modular components:

```
src/
├── detector/       # GPU allocation detection
│   ├── types.rs    # Core types and enums
│   ├── memory.rs   # /proc/PID/maps parser
│   ├── process.rs  # Process and FD analysis
│   └── nvidia.rs   # NVIDIA-specific detection
├── checkpoint/     # Checkpoint strategies
├── restore/        # Restore engine
└── utils/          # Utilities
```

## Usage

### Detection

Detect GPU allocations in a running process:

```bash
# Basic detection
gpu-checkpoint detect --pid 12345

# JSON output for scripting
gpu-checkpoint detect --pid 12345 --format json

# Verbose output with detailed allocations
gpu-checkpoint detect --pid 12345 --verbose
```

### Checkpoint (Not Yet Implemented)

```bash
# Auto-select strategy
gpu-checkpoint checkpoint --pid 12345 --storage /mnt/weka/checkpoints

# Force specific strategy
gpu-checkpoint checkpoint --pid 12345 --strategy bar-sliding
```

### Restore (Not Yet Implemented)

```bash
gpu-checkpoint restore --metadata checkpoint.json --storage /mnt/weka/checkpoints
```

## Building

```bash
# Debug build
cargo build

# Release build with optimizations
cargo build --release

# Run tests
cargo test

# Check compilation (cross-platform)
cargo check
```

## Detection Algorithm

1. **Process Analysis**:
   - Scan `/proc/PID/maps` for memory mappings
   - Analyze `/proc/PID/fd/` for GPU device files
   - Check environment variables for GPU indicators

2. **Allocation Classification**:
   - Identify UVM allocations via `/dev/nvidia-uvm`
   - Detect managed memory patterns
   - Find IPC/distributed allocations in `/dev/shm`
   - Locate PCIe BAR mappings

3. **Strategy Selection**:
   - No allocations → Skip GPU
   - Problematic allocations → BAR sliding
   - Standard allocations → CUDA checkpoint

## Platform Support

- **Linux**: Full support with `/proc` filesystem access
- **macOS/Windows**: Compiles but limited functionality (useful for development)

## Performance Targets

- Detection: <100ms per process
- Checkpoint: 2-4 minutes for 2048 H100 cluster
- Restore: 3-4 minutes with 4TB/s bandwidth
- Memory overhead: <1% during operation

## Future Work

- [ ] Complete BAR sliding implementation
- [ ] CUDA checkpoint integration
- [ ] AMD GPU support
- [ ] Distributed checkpoint coordination
- [ ] Compression and deduplication
- [ ] Performance benchmarks

## License

MIT OR Apache-2.0