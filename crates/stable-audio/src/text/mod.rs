use tokenizers::Tokenizer as HfTokenizer;

use crate::ggml_runtime::gguf::GgufModel;
use crate::{Error, Result};

pub struct Tokenizer {
    inner: HfTokenizer,
}

#[derive(Debug, Clone)]
pub struct TokenizedPrompt {
    pub ids: Vec<u32>,
    pub mask: Vec<u32>,
    pub max_len: usize,
}

impl Tokenizer {
    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let json = model
            .get_string("tokenizer.huggingface.json")?
            .ok_or_else(|| Error::MissingMetadata("tokenizer.huggingface.json".into()))?;
        let inner = HfTokenizer::from_bytes(json.as_bytes())
            .map_err(|err| Error::Tokenizer(err.to_string()))?;
        Ok(Self { inner })
    }

    pub fn encode_sa3_prompt(&self, prompt: &str, max_len: usize) -> Result<TokenizedPrompt> {
        let encoding = self
            .inner
            .encode(prompt, false)
            .map_err(|err| Error::Tokenizer(err.to_string()))?;
        let mut ids = vec![0_u32; max_len];
        let mut mask = vec![0_u32; max_len];
        for (idx, id) in encoding.get_ids().iter().copied().take(max_len).enumerate() {
            ids[idx] = id;
            mask[idx] = 1;
        }
        Ok(TokenizedPrompt { ids, mask, max_len })
    }
}
