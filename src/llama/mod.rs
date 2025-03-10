// Import the modules
mod model;
mod worker;
mod embedded_model;

// Re-export everything from model
pub use model::{Cache, Config, Llama, TokenCallback};

// Re-export from worker
pub use worker::{Model, ModelData, TransformerWeights};

// Re-export embedded model functionality
pub use embedded_model::get_embedded_model_data;

// Export LlamaConfig and LlamaModel for compatibility with existing code
pub use candle_transformers::models::llama::LlamaConfig;
pub type LlamaModel = model::Llama; // Type alias for backward compatibility

// Define model type for UI selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LlamaModelType {
    Default,
    Embedded, // Added for embedded model support
}
