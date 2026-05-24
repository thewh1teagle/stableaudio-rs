use std::path::PathBuf;

use llama_rs_sys as ffi;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("WAV error: {0}")]
    Wav(#[from] hound::Error),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("failed to open GGUF model: {0}")]
    GgufOpen(PathBuf),

    #[error("path contains a NUL byte: {0}")]
    NulPath(PathBuf),

    #[error("missing GGUF metadata key: {0}")]
    MissingMetadata(String),

    #[error("invalid GGUF metadata key: {0}")]
    InvalidMetadataKey(String),

    #[error("missing GGUF tensor: {0}")]
    MissingTensor(String),

    #[error("missing GGUF tensor name at index {0}")]
    MissingTensorName(i64),

    #[error("invalid GGUF tensor index: {0}")]
    TensorIndex(i64),

    #[error("invalid tensor range in {path}: offset={offset} size={size}")]
    InvalidTensorRange {
        path: PathBuf,
        offset: usize,
        size: usize,
    },

    #[error("unsupported tensor type for {name}: {tensor_type}")]
    UnsupportedTensorType {
        name: String,
        tensor_type: ffi::ggml_type,
    },

    #[error("ggml error: {0}")]
    Ggml(String),

    #[error("SA3 runtime is incomplete: {0}")]
    Incomplete(String),
}

pub type Result<T> = std::result::Result<T, Error>;
