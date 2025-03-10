// This file contains the embedded model data
use crate::llama::ModelData;

// Include the model.bin file as a byte array
const EMBEDDED_MODEL: &[u8] = include_bytes!("../../assets/model.bin");

// Include the tokenizer.json file as a byte array
const EMBEDDED_TOKENIZER: &[u8] = include_bytes!("../../assets/tokenizer.json");

// Function to create ModelData from embedded constants
pub fn get_embedded_model_data() -> ModelData {
    ModelData {
        model: EMBEDDED_MODEL.to_vec(),
        tokenizer: EMBEDDED_TOKENIZER.to_vec(),
    }
}
