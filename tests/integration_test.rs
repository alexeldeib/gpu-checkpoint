use gpu_checkpoint::{
    checkpoint::{CheckpointEngine, CheckpointStrategy},
    detector::{AllocationType, CompositeDetector, GpuAllocation},
};

#[test]
fn test_composite_detector_no_gpu() {
    // Test that detector handles processes without GPU gracefully
    let detector = CompositeDetector::new();

    // Use current process PID for testing
    let pid = std::process::id();
    let results = detector.detect_all(pid).expect("Detection should not fail");

    // On non-GPU systems or processes, results should be empty
    // This test will pass on CI where no GPUs are present
    assert!(results.is_empty() || results.iter().all(|r| r.allocations.is_empty()));
}

#[test]
fn test_strategy_selection() {
    use gpu_checkpoint::detector::{DetectionResult, GpuVendor};

    // Test with no allocations
    let empty_result = DetectionResult::new(1234, GpuVendor::Nvidia);
    assert_eq!(
        CheckpointEngine::select_strategy(&empty_result),
        CheckpointStrategy::SkipGpu
    );

    // Test with standard allocations only
    let mut standard_result = DetectionResult::new(1234, GpuVendor::Nvidia);
    standard_result.add_allocation(GpuAllocation::new(
        0x100000000,
        0x200000000,
        AllocationType::Standard,
    ));
    assert_eq!(
        CheckpointEngine::select_strategy(&standard_result),
        CheckpointStrategy::CudaCheckpoint
    );

    // Test with problematic allocations
    let mut problematic_result = DetectionResult::new(1234, GpuVendor::Nvidia);
    problematic_result.add_allocation(GpuAllocation::new(
        0x100000000,
        0x200000000,
        AllocationType::Uvm,
    ));
    assert_eq!(
        CheckpointEngine::select_strategy(&problematic_result),
        CheckpointStrategy::BarSliding
    );
}

#[test]
fn test_allocation_classification() {
    // Test that allocations are properly classified as problematic
    let standard = GpuAllocation::new(0x1000, 0x2000, AllocationType::Standard);
    assert!(!standard.is_problematic());

    let uvm = GpuAllocation::new(0x1000, 0x2000, AllocationType::Uvm);
    assert!(uvm.is_problematic());

    let managed = GpuAllocation::new(0x1000, 0x2000, AllocationType::Managed);
    assert!(managed.is_problematic());

    let ipc = GpuAllocation::new(0x1000, 0x2000, AllocationType::Ipc);
    assert!(ipc.is_problematic());

    let distributed = GpuAllocation::new(0x1000, 0x2000, AllocationType::Distributed);
    assert!(distributed.is_problematic());
}
