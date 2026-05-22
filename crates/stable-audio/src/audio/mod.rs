pub mod wav;

use std::path::Path;

use crate::Result;

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl AudioBuffer {
    pub fn write_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        wav::write_f32_as_i16(path, self)
    }
}
