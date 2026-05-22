use std::path::Path;

use crate::{AudioBuffer, Result};

pub fn write_f32_as_i16(path: impl AsRef<Path>, audio: &AudioBuffer) -> Result<()> {
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in &audio.samples {
        let clipped = sample.clamp(-1.0, 1.0);
        writer.write_sample((clipped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
