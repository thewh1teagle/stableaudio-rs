use std::path::Path;

use crate::{AudioBuffer, Error, Result};

pub fn read_f32_stereo_44k(path: impl AsRef<Path>) -> Result<AudioBuffer> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 2 || spec.sample_rate != 44_100 {
        return Err(Error::Incomplete(format!(
            "input WAV must be stereo 44.1 kHz for now, got {} channels at {} Hz",
            spec.channels, spec.sample_rate
        )));
    }
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|sample| sample as f32 / i16::MAX as f32))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        _ => {
            return Err(Error::Incomplete(format!(
                "unsupported input WAV format: {:?} {} bits",
                spec.sample_format, spec.bits_per_sample
            )));
        }
    };
    Ok(AudioBuffer {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        samples,
    })
}

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
