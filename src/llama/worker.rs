use crate::llama::model::{Cache, Config, Llama};
use byteorder::{LittleEndian, ReadBytesExt};
use candle::{DType, Device, Error, IndexOp, Result, Shape, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use serde::{Deserialize, Serialize};
use std::io::Read;
use tokenizers::Tokenizer;

// Communication structures
#[derive(Serialize, Deserialize, Clone)]
pub struct ModelData {
    pub tokenizer: Vec<u8>,
    pub model: Vec<u8>,
}

// Helper functions for loading models
pub fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

pub fn read_tensor<R: Read, S: Into<Shape>>(r: &mut R, shape: S, dev: &Device) -> Result<Tensor> {
    let shape = shape.into();
    let mut data_t = vec![0f32; shape.elem_count()];
    r.read_f32_into::<LittleEndian>(&mut data_t)?;
    let tensor = Tensor::from_vec(data_t, shape, dev)?;
    Ok(tensor)
}

// Model container
pub struct Model {
    pub cache: Cache,
    pub config: Config,
    pub llama: Llama,
    pub tokenizer: Tokenizer,
}

impl Model {
    // Function to load the model
    pub fn load(md: ModelData) -> Result<Self> {
        let dev = Device::Cpu;
        let mut model = std::io::Cursor::new(md.model);
        let config = Config::from_reader(&mut model)?;
        let weights = TransformerWeights::from_reader(&mut model, &config, &dev)?;
        let vb = weights.var_builder(&config, &dev)?;
        let cache = Cache::new(true, &config, vb.pp("rot"))?;
        let llama = Llama::load(vb, &cache, &config)?;
        let tokenizer =
            Tokenizer::from_bytes(&md.tokenizer).map_err(|m| Error::Msg(m.to_string()))?;
        Ok(Self {
            cache,
            config,
            llama,
            tokenizer,
        })
    }

    // Add generate method that matches the expected signature
    pub fn generate(&self, prompt: &str, temperature: f64, top_p: f64) -> Result<String> {
        let mut result = String::new();

        // Use the existing generate_text method but capture the output
        self.generate_text(temperature, top_p, prompt.to_string(), |text| {
            result.push_str(&text)
        })?;

        Ok(result)
    }

    // Function to generate text with callback
    pub fn generate_text(
        &self,
        temp: f64,
        top_p: f64,
        prompt: String,
        callback: impl FnMut(String),
    ) -> Result<()> {
        let dev = Device::Cpu;
        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1.0 {
            None
        } else {
            Some(top_p)
        };

        let mut logits_processor = LogitsProcessor::new(299792458, temp, top_p);
        let mut index_pos = 0;
        let mut tokens = self
            .tokenizer
            .encode(prompt.to_string(), true)
            .map_err(|m| Error::Msg(m.to_string()))?
            .get_ids()
            .to_vec();

        let mut callback = callback;
        callback(prompt);

        for index in 0.. {
            if tokens.len() >= self.config.seq_len {
                break;
            }
            let context_size = if self.cache.use_kv_cache && index > 0 {
                1
            } else {
                tokens.len()
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &dev)?.unsqueeze(0)?;
            let logits = self.llama.forward(&input, index_pos)?;
            let logits = logits.squeeze(0)?;
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                callback(text);
            }
        }
        Ok(())
    }
}

// Transformer weights structure and implementation
pub struct TransformerWeights {
    token_embedding_table: Tensor, // (vocab_size, dim)
    rms_att_weight: Tensor,        // (layer, dim) rmsnorm weights
    rms_ffn_weight: Tensor,        // (layer, dim)
    wq: Tensor,                    // (layer, dim, dim)
    wk: Tensor,                    // (layer, dim, dim)
    wv: Tensor,                    // (layer, dim, dim)
    wo: Tensor,                    // (layer, dim, dim)
    w1: Tensor,                    // (layer, hidden_dim, dim)
    w2: Tensor,                    // (layer, dim, hidden_dim)
    w3: Tensor,                    // (layer, hidden_dim, dim)
    rms_final_weight: Tensor,      // (dim,)
    freq_cis_real: Tensor,         // (seq_len, head_size/2)
    freq_cis_imag: Tensor,         // (seq_len, head_size/2)
}

impl TransformerWeights {
    fn from_reader<R: Read>(r: &mut R, c: &Config, dev: &Device) -> Result<Self> {
        let token_embedding_table = read_tensor(r, (c.vocab_size, c.dim), dev)?;
        let rms_att_weight = read_tensor(r, (c.n_layers, c.dim), dev)?;
        let wq = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wk = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wv = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wo = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let rms_ffn_weight = read_tensor(r, (c.n_layers, c.dim), dev)?;
        let w1 = read_tensor(r, (c.n_layers, c.hidden_dim, c.dim), dev)?;
        let w2 = read_tensor(r, (c.n_layers, c.dim, c.hidden_dim), dev)?;
        let w3 = read_tensor(r, (c.n_layers, c.hidden_dim, c.dim), dev)?;
        let rms_final_weight = read_tensor(r, c.dim, dev)?;
        let head_size = c.head_size();
        let freq_cis_real = read_tensor(r, (c.seq_len, head_size / 2), dev)?;
        let freq_cis_imag = read_tensor(r, (c.seq_len, head_size / 2), dev)?;
        Ok(Self {
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
        })
    }

    fn var_builder(&self, cfg: &Config, device: &Device) -> Result<VarBuilder<'_>> {
        let mut ws = std::collections::HashMap::new();
        let mut insert = |name: &str, t: Tensor| {
            ws.insert(name.to_string(), t);
        };
        insert("rot.freq_cis_real", self.freq_cis_real.clone());
        insert("rot.freq_cis_imag", self.freq_cis_imag.clone());
        insert(
            "model.embed_tokens.weight",
            self.token_embedding_table.clone(),
        );
        insert("lm_head.weight", self.token_embedding_table.clone());
        insert("model.norm.weight", self.rms_final_weight.clone());
        for layer in 0..cfg.n_layers {
            ws.insert(
                format!("model.layers.{layer}.self_attn.q_proj.weight"),
                self.wq.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.self_attn.k_proj.weight"),
                self.wk.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.self_attn.v_proj.weight"),
                self.wv.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.self_attn.o_proj.weight"),
                self.wo.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.mlp.gate_proj.weight"),
                self.w1.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.mlp.down_proj.weight"),
                self.w2.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.mlp.up_proj.weight"),
                self.w3.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.input_layernorm.weight"),
                self.rms_att_weight.i(layer)?,
            );
            ws.insert(
                format!("model.layers.{layer}.post_attention_layernorm.weight"),
                self.rms_ffn_weight.i(layer)?,
            );
        }
        let vb = VarBuilder::from_tensors(ws, DType::F32, device);
        Ok(vb)
    }
}

impl Config {
    fn from_reader<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let dim = read_i32(r)? as usize;
        let hidden_dim = read_i32(r)? as usize;
        let n_layers = read_i32(r)? as usize;
        let n_heads = read_i32(r)? as usize;
        let n_kv_heads = read_i32(r)? as usize;
        let vocab_size = read_i32(r)? as usize;
        let seq_len = read_i32(r)? as usize;
        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            norm_eps: 1e-5,
        })
    }
}
