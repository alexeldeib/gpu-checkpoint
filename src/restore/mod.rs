pub mod bar_restore;

use crate::checkpoint::CheckpointMetadata;
use crate::Result;

pub use bar_restore::{BarRestore, RestoreMetadata};

pub struct RestoreEngine {
    _storage_path: String,
}

impl RestoreEngine {
    pub fn new(storage_path: String) -> Self {
        Self {
            _storage_path: storage_path,
        }
    }

    pub async fn restore(&self, _metadata: &CheckpointMetadata) -> Result<u32> {
        // This would implement the actual restore
        todo!("Implement restore")
    }
}
