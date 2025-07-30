use std::fs::{File, OpenOptions};
use std::io::Write;
use std::thread;
use std::time::Duration;

fn main() {
    println!("Mock GPU process starting (PID: {})", std::process::id());
    
    // Create mock GPU device files for testing
    let mock_files = vec![
        "/tmp/mock_nvidia0",
        "/tmp/mock_nvidia-uvm",
    ];
    
    // Create and hold open file descriptors to simulate GPU usage
    let mut fds = Vec::new();
    
    for path in &mock_files {
        match OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
        {
            Ok(mut file) => {
                writeln!(file, "Mock GPU device").ok();
                println!("Created mock device: {}", path);
                fds.push(file);
            }
            Err(e) => {
                eprintln!("Failed to create {}: {}", path, e);
            }
        }
    }
    
    // Allocate some memory to simulate GPU allocations
    let size = 256 * 1024 * 1024; // 256MB
    let mut buffer = vec![0u8; size];
    
    // Touch the memory
    for i in (0..size).step_by(4096) {
        buffer[i] = (i % 256) as u8;
    }
    
    println!("Allocated {} MB of memory", size / (1024 * 1024));
    println!("Mock GPU process ready. Press Ctrl+C to exit.");
    
    // Keep the process alive
    loop {
        thread::sleep(Duration::from_secs(1));
        print!(".");
        std::io::stdout().flush().ok();
    }
}