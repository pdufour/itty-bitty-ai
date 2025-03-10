use crate::llama::{LlamaModelType, Model};
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use egui::{
    text::CCursorRange, Color32, Id, Key, KeyboardShortcut, Modifiers, RichText, ScrollArea,
    Stroke, TextEdit, TextStyle, Ui,
};
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// Define shared types for the LLaMA model
#[derive(Clone, Default)]
pub struct LlamaResult {
    pub text: String,
    pub status: String,
    pub loaded: bool,
    pub generating: bool,
    pub error: Option<String>,
}

// Type alias for shared result
type SharedLlamaResult = Arc<Mutex<LlamaResult>>;

// Type definitions used across all platforms
pub type LocalLlamaConfig = (LlamaModelType, f64, Option<String>);

#[derive(Clone, Debug, Default)]
struct SuggestionState {
    pub active_suggestion: String, // The current active suggestion text
    pub active_suggestion_pos: Option<usize>, // Position where the active suggestion starts
    pub trigger_word: String,      // The word that triggered the suggestion
    pub last_update_time: f64,     // Time when the suggestion was last updated
}

// Define LlamaConfig struct
// Removed duplicate LocalLlamaConfig struct

// Define a struct for text analysis results
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct TextAnalysisResult {
    pub text: String,
    pub status: String,
    pub loaded: bool,
    pub generating: bool,
    pub error: Option<String>,
    pub progress: f32, // Add progress field for tracking percentage
}

impl Default for TextAnalysisResult {
    fn default() -> Self {
        Self {
            text: String::new(),
            status: "Analysis not started".to_string(),
            loaded: false,
            generating: false,
            error: None,
            progress: 0.0, // Default progress value
        }
    }
}

type SharedTextAnalysisResult = Arc<Mutex<TextAnalysisResult>>;

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct EasyMarkEditor {
    code: String,
    highlight_editor: bool,
    show_rendered: bool,

    #[cfg_attr(feature = "serde", serde(skip))]
    highlighter: crate::easy_mark::MemoizedEasymarkHighlighter,
    #[cfg_attr(feature = "serde", serde(skip))]
    log_messages: Arc<Mutex<Vec<String>>>, // Thread-safe log messages

    // Llama fields
    #[cfg_attr(feature = "serde", serde(skip))]
    llama_result: SharedLlamaResult,
    #[cfg_attr(feature = "serde", serde(skip))]
    ai_prompt: String,
    #[cfg_attr(feature = "serde", serde(skip))]
    loading_model: bool,
    #[cfg_attr(feature = "serde", serde(skip))]
    model: Arc<Mutex<Option<Model>>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    loading_started: Arc<AtomicBool>,
    
    // Text analysis fields
    #[cfg_attr(feature = "serde", serde(skip))]
    text_analysis_result: SharedTextAnalysisResult,
    #[cfg_attr(feature = "serde", serde(skip))]
    last_analyzed_text: String,
    #[cfg_attr(feature = "serde", serde(skip))]
    analysis_scheduled: bool,
    
    // Auto-analysis after typing
    #[cfg_attr(feature = "serde", serde(skip))]
    last_typing_time: f64,
    
    // Track if this is the first frame to auto-focus
    #[cfg_attr(feature = "serde", serde(skip))]
    first_frame: bool,
}

impl PartialEq for EasyMarkEditor {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
            && self.highlight_editor == other.highlight_editor
            && self.show_rendered == other.show_rendered
            && self.ai_prompt == other.ai_prompt
            && self.loading_model == other.loading_model
            && self.loading_started.load(Ordering::SeqCst)
                == other.loading_started.load(Ordering::SeqCst)
    }
}

impl Eq for EasyMarkEditor {}

impl Default for EasyMarkEditor {
    fn default() -> Self {
        Self {
            code: DEFAULT_CODE.trim().to_owned(),
            highlight_editor: true,
            show_rendered: true,
            highlighter: Default::default(),
            log_messages: Arc::new(Mutex::new(Vec::new())),
            llama_result: Arc::new(Mutex::new(LlamaResult::default())),
            ai_prompt: "Complete this text:".to_string(),
            loading_model: false,
            model: Arc::new(Mutex::new(None)),
            loading_started: Arc::new(AtomicBool::new(false)),
            text_analysis_result: Arc::new(Mutex::new(TextAnalysisResult::default())),
            last_analyzed_text: String::new(),
            analysis_scheduled: false,
            last_typing_time: 0.0,
            first_frame: true,
        }
    }
}

impl EasyMarkEditor {
    pub fn new() -> Self {
        Self::default()
    }

    fn load_model(&mut self, _model_type: LlamaModelType) {
        self.log_message("Starting model load...".to_string());

        // Update UI to indicate loading state
        if let Ok(mut result) = self.llama_result.lock() {
            result.status = "Loading embedded model...".to_string();
            result.loaded = false;
            result.generating = true; // Mark as busy
        }

        // Clone necessary data for processing in a separate context
        let llama_result_clone = self.llama_result.clone();
        let model_clone = self.model.clone();
        let ai_prompt_clone = self.ai_prompt.clone();
        let log_messages_clone = self.log_messages.clone();

        self.log_message("Spawning model load thread...".to_string());

        // Use a cross-platform approach to load the model
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen_futures::spawn_local;
            spawn_local(async move {
                Self::load_model_impl(
                    llama_result_clone,
                    model_clone,
                    ai_prompt_clone,
                    log_messages_clone,
                );
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                Self::load_model_impl(
                    llama_result_clone,
                    model_clone,
                    ai_prompt_clone,
                    log_messages_clone,
                );
            });
        }
    }

    // Shared implementation for model loading that works on all platforms
    fn load_model_impl(
        llama_result: SharedLlamaResult,
        model: Arc<Mutex<Option<Model>>>,
        ai_prompt: String,
        log_messages: Arc<Mutex<Vec<String>>>,
    ) {
        // Helper function to log in the thread
        let log = |msg: String| {
            if let Ok(mut messages) = log_messages.lock() {
                messages.push(msg);
            }
        };

        // Mark as generating in progress
        if let Ok(mut result) = llama_result.lock() {
            result.status = "Starting model load...".to_string();
            result.loaded = false;
            result.generating = true;
            result.error = None;
        }

        log("Initializing model load...".to_string());

        // Get the embedded model data in a separate thread to avoid blocking
        log("Loading embedded model data from assets...".to_string());
        let model_data = match std::panic::catch_unwind(|| crate::llama::get_embedded_model_data())
        {
            Ok(data) => {
                log("Successfully loaded embedded model data".to_string());
                data
            }
            Err(e) => {
                let error_msg = format!("Failed to load embedded model data: {:?}", e);
                log(error_msg.clone());
                if let Ok(mut result) = llama_result.lock() {
                    result.status = error_msg;
                    result.loaded = false;
                    result.generating = false;
                    result.error = Some("Model data loading failed".to_string());
                }
                return;
            }
        };

        // Update status
        if let Ok(mut result) = llama_result.lock() {
            result.status = "Initializing model...".to_string();
        }

        log("Initializing model from data...".to_string());
        // Try to load the model and update state
        match Model::load(model_data) {
            Ok(loaded_model) => {
                log("Model initialized successfully".to_string());
                // Store the model
                if let Ok(mut model_guard) = model.lock() {
                    *model_guard = Some(loaded_model);

                    if let Ok(mut result) = llama_result.lock() {
                        result.status = "Model ready".to_string();
                        result.loaded = true;
                        result.generating = false;
                        result.error = None;

                        // Check if we have a pending generation request
                        if !ai_prompt.is_empty() {
                            result.text =
                                format!("Model loaded successfully. Prompt: {}", ai_prompt);
                        }
                    }
                    log("Model is now ready for use".to_string());
                } else {
                    log("Failed to store model - mutex lock failed".to_string());
                    if let Ok(mut result) = llama_result.lock() {
                        result.status = "Failed to store model - mutex lock failed".to_string();
                        result.loaded = false;
                        result.generating = false;
                        result.error = Some("Mutex lock failed".to_string());
                    }
                }
            }
            Err(e) => {
                let error_msg = format!("Failed to initialize model: {:?}", e);
                log(error_msg.clone());
                if let Ok(mut result) = llama_result.lock() {
                    result.status = error_msg;
                    result.loaded = false;
                    result.generating = false;
                    result.error = Some("Model initialization failed".to_string());
                }
            }
        }
    }

    fn handle_suggestions(
        &mut self,
        ui: &mut egui::Ui,
        response: egui::Response,
        code: &mut String,
        tab_pressed_for_suggestion: bool,
    ) {
        // Get model readiness check
        let model_ready = if let Ok(result) = self.llama_result.lock() {
            result.loaded && !result.generating
        } else {
            false
        };

        // Get current state from memory
        let suggestion_key = ui.id().with("ghost_suggestion");
        let mut suggestion_state = ui
            .memory_mut(|mem| mem.data.get_temp::<SuggestionState>(suggestion_key))
            .unwrap_or_default();

        // If tab was pressed, accept the suggestion
        if tab_pressed_for_suggestion && !suggestion_state.active_suggestion.is_empty() {
            if let Some(pos) = suggestion_state.active_suggestion_pos {
                if pos <= code.len() {
                    // Get the text before cursor - make an owned copy to avoid borrowing issues
                    let text_before = code[..pos].to_string();
                    
                    // Find the word at cursor
                    let word_before = find_word_at_cursor(&text_before);
                    
                    // Get the suggestion text
                    let suggested_text = suggestion_state.active_suggestion.clone();
                    
                    // Calculate how much of the current word we should keep
                    // We only want to append the part of the suggestion that hasn't been typed yet
                    let completion_part = if suggested_text.starts_with(&word_before) {
                        // Extract only the part of suggestion that comes after what user already typed
                        suggested_text[word_before.len()..].to_string()
                    } else {
                        // Fallback: If suggestion doesn't start with what user typed (unusual case)
                        // just use the full suggestion
                        suggested_text.clone()
                    };
                    
                    // Only need to insert the completion part at cursor position
                    if pos <= code.len() && code.is_char_boundary(pos) {
                        // Insert the completion
                        code.insert_str(pos, &completion_part);
                        
                        // Calculate the new cursor position (at the end of the inserted completion)
                        let new_cursor_pos = pos + completion_part.len();
                        
                        // Update cursor position
                        if let Some(mut state) = TextEdit::load_state(ui.ctx(), response.id) {
                            if let Some(mut ccursor_range) = state.cursor.char_range() {
                                // Only update if it's a valid position
                                if new_cursor_pos <= code.len() && code.is_char_boundary(new_cursor_pos) {
                                    ccursor_range.primary.index = new_cursor_pos;
                                    ccursor_range.secondary.index = new_cursor_pos;
                                    state.cursor.set_char_range(Some(ccursor_range));
                                    TextEdit::store_state(ui.ctx(), response.id, state);
                                    self.log_message(format!(
                                        "Updated cursor to position {}",
                                        new_cursor_pos
                                    ));
                                }
                            }
                        }
                    }

                    // Clear the suggestion
                    suggestion_state.active_suggestion = String::new();
                    suggestion_state.active_suggestion_pos = None;
                    suggestion_state.trigger_word = String::new();

                    // Update the state in memory
                    ui.memory_mut(|mem| mem.data.insert_temp(suggestion_key, suggestion_state));
                    
                    // Request focus to keep the editor focused
                    response.request_focus();
                }
            }
            return; // Exit early after handling tab
        }

        // Try to get cursor position from TextEdit state for suggestions
        if let Some(state) = TextEdit::load_state(ui.ctx(), response.id) {
            if let Some(ccursor_range) = state.cursor.char_range() {
                let [primary, _] = ccursor_range.sorted();
                let cursor_pos = primary.index;

                // Check if we have cursor position in valid range
                if cursor_pos > code.len() {
                    // Clear suggestions if cursor position is invalid
                    suggestion_state.active_suggestion = String::new();
                    suggestion_state.active_suggestion_pos = None;
                    ui.memory_mut(|mem| {
                        mem.data
                            .insert_temp(suggestion_key, suggestion_state.clone());
                    });
                    return;
                }

                // Get the text before cursor
                let text_before = code[..cursor_pos].to_string();

                // Find the word at cursor
                let word_before = find_word_at_cursor(&text_before);

                // Clear suggestion if we've deleted text or moved the cursor away from suggestion position
                if let Some(pos) = suggestion_state.active_suggestion_pos {
                    if cursor_pos < pos
                        || (word_before.len() < suggestion_state.trigger_word.len()
                            && !word_before.is_empty() // Allow empty word for tab completion at start of line
                            && cursor_pos > 0)         // Allow cursor at beginning of text
                        || (!word_before.starts_with(&suggestion_state.trigger_word) 
                            && !suggestion_state.trigger_word.starts_with(&word_before)) // Check both directions
                    {
                        // Text was likely deleted or cursor moved away
                        suggestion_state.active_suggestion = String::new();
                        suggestion_state.active_suggestion_pos = None;
                        suggestion_state.trigger_word = String::new(); // Reset trigger_word so suggestion can reappear
                        suggestion_state.last_update_time = 0.0;       // Reset timer to allow immediate suggestion
                        ui.memory_mut(|mem| {
                            mem.data
                                .insert_temp(suggestion_key, suggestion_state.clone());
                        });
                    }
                }


                if !model_ready && suggestion_state.last_update_time == 0.0 {
                    self.log_message(
                        "Model not ready".to_string(),
                    );
                }

                // Increase debounce time to 1 second for better performance
                let current_time = ui.input(|i| i.time);
                let time_since_last_check = current_time - suggestion_state.last_update_time;

                // Check if user is in the middle of typing a word or at end of word
                let is_at_end_of_word = cursor_pos == code.len() || 
                    (cursor_pos < code.len() && code[cursor_pos..].chars().next().map_or(false, |c| c.is_whitespace()));
                
                // Only get suggestions if:
                // 1. Enough time has passed (1 second)
                // 2. Word is non-empty and at least 3 chars
                // 3. Model is ready
                // 4. Word has changed since last suggestion
                // 5. User is not in the middle of typing a word (cursor is at the end of text or followed by whitespace)
                if time_since_last_check > 1.0
                    && word_before.len() >= 3
                    && model_ready
                    && word_before != suggestion_state.trigger_word
                    && is_at_end_of_word
                {
                    self.log_message(format!("New word to check: {}", word_before));
                    suggestion_state.last_update_time = current_time;

                    // Get suggestions from Llama
                    if let Some(completion_text) = self.get_completion_from_llama(&word_before) {
                        self.log_message(format!(
                            "Got suggestion for '{}': {}",
                            word_before, completion_text
                        ));
                        suggestion_state.active_suggestion = completion_text;
                        suggestion_state.trigger_word = word_before.clone();
                        suggestion_state.active_suggestion_pos = Some(cursor_pos);

                        // Update state in memory
                        ui.memory_mut(|mem| {
                            mem.data
                                .insert_temp(suggestion_key, suggestion_state.clone());
                        });
                    }
                }
            }
        }
    }

    // Render the input panel with editor and suggestions
    fn render_input_panel(&mut self, ui: &mut egui::Ui) {
        let id = ui.make_persistent_id("code_editor");
        let is_focused = ui.memory(|mem| mem.has_focus(id));
        let mut code_clone = self.code.clone();

        // Request focus on first frame
        if self.first_frame {
            ui.memory_mut(|mem| mem.request_focus(id));
            self.first_frame = false;
            self.log_message("Requesting focus on editor".to_string());
        }

        ui.vertical(|ui| {
            // Panel header with improved styling
            ui.add_space(8.0);
            
            // Get the suggestion state to check if there's an active suggestion (for TAB button)
            let suggestion_key = ui.id().with("ghost_suggestion");
            let active_suggestion = ui
                .memory(|mem| mem.data.get_temp::<SuggestionState>(suggestion_key))
                .map_or(false, |state| !state.active_suggestion.is_empty());
            
            // Create a frame for the header without border
            let header_frame = egui::Frame::none()
                .fill(egui::Color32::BLACK)
                .inner_margin(egui::vec2(0.0, 0.0))
                .outer_margin(egui::vec2(0.0, 0.0));
                
            header_frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Left side with title
                    ui.horizontal(|ui| {
                        // Red square icon
                        let (square_rect, _) =
                            ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                        ui.painter()
                            .rect_filled(square_rect, 0.0, egui::Color32::from_rgb(255, 50, 50));
                        ui.add_space(8.0);
            
                        // NEURAL INPUT text with improved styling
                        ui.label(
                            egui::RichText::new("NEURAL INPUT")
                                .size(18.0)
                                .text_style(egui::TextStyle::Monospace)
                                .color(egui::Color32::from_rgb(0, 240, 100))
                                .strong(),
                        );
                    });
                    
                    // Expand to push TAB button to the right
                    ui.add_space(ui.available_width() - 40.0); // Reserve space for TAB button
                    
                    // Add TAB button on right side when suggestions are active
                    if active_suggestion {
                        // Create a fixed size button area
                        let tab_size = egui::vec2(36.0, 20.0);
                        let tab_rect = egui::Rect::from_min_size(
                            ui.cursor().min,
                            tab_size
                        );
                        
                        // Draw TAB button background with improved visibility
                        ui.painter().rect(
                            tab_rect,
                            2.0,
                            egui::Color32::from_rgb(0, 100, 40),
                            egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 240, 100))
                        );
                        
                        // Draw TAB text
                        ui.painter().text(
                            tab_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "TAB",
                            egui::FontId::monospace(11.0),
                            egui::Color32::from_rgb(0, 255, 100)
                        );
                        
                        // Allocate space for the button to prevent layout issues
                        ui.allocate_rect(tab_rect, egui::Sense::hover());
                    }
                });
            });
            
            // Add a bit more space before separator
            ui.add_space(5.0);

            // Separator line with consistent styling
            let rect = ui.max_rect();
            let line_rect = egui::Rect::from_min_max(
                egui::pos2(rect.left(), ui.min_rect().bottom() + 2.0),
                egui::pos2(rect.right(), ui.min_rect().bottom() + 2.5),
            );
            ui.painter()
                .rect_filled(line_rect, 0.0, egui::Color32::from_rgb(0, 120, 50));
            ui.add_space(16.0);

            // Get the suggestion state for the layouter
            let suggestion_state = ui
                .memory_mut(|mem| mem.data.get_temp::<SuggestionState>(suggestion_key))
                .unwrap_or_default();

            // Create a custom layouter that shows ghost text
            let mut layouter = |ui: &egui::Ui, text: &str, wrap_width: f32| {
                let mut job = self.highlighter.highlight(ui.style(), text);

                // Add ghost text for suggestions
                if let Some(pos) = suggestion_state.active_suggestion_pos {
                    if pos <= text.len()
                        && !suggestion_state.active_suggestion.is_empty()
                    {
                        let suggestion = &suggestion_state.active_suggestion;

                        // Create a ghost text section to append at the cursor position
                        let ghost_format = egui::TextFormat {
                            font_id: egui::FontId::monospace(14.0), // Changed from 12.0 to match main text size
                            color: egui::Color32::from_rgb(150, 150, 180),
                            background: egui::Color32::from_rgba_unmultiplied(
                                70, 70, 120, 100,
                            ),
                            italics: true,
                            ..Default::default()
                        };

                        // First, append the suggestion text to the job's text
                        let original_text_len = job.text.len();
                        job.text = format!("{}{}", job.text, suggestion);
                        
                        // Then create a section for the ghost text that starts at the end of the
                        // original text and extends to the end of the combined text
                        job.sections.push(egui::text::LayoutSection {
                            leading_space: 0.0,
                            byte_range: original_text_len..job.text.len(),
                            format: ghost_format,
                        });
                    }
                }

                job.wrap.max_width = wrap_width;
                ui.fonts(|f| f.layout_job(job))
            };

            // Intercept tab key when suggestion is active
            let mut tab_pressed_for_suggestion = false;
            if !suggestion_state.active_suggestion.is_empty() {
                let tab_key_pressed = ui.input(|i| i.key_pressed(egui::Key::Tab));
                if tab_key_pressed {
                    ui.input_mut(|i| i.events = vec![]);
                    tab_pressed_for_suggestion = true;
                }
            }

            // Add text editor with custom styling
            let text_edit = egui::TextEdit::multiline(&mut code_clone)
                .id(id)  // Associate with the persistent ID
                .font(egui::TextStyle::Monospace)
                .code_editor()
                .desired_rows(20)
                .lock_focus(true)
                .desired_width(f32::INFINITY)
                .frame(false)
                .text_color(egui::Color32::from_rgb(0, 230, 100))
                .layouter(&mut layouter);

            let editor_response = ui.add(text_edit);

            // Handle suggestions
            self.handle_suggestions(
                ui,
                editor_response.clone(),
                &mut code_clone,
                tab_pressed_for_suggestion,
            );

            // Show placeholder text if empty (BEGIN TRANSMISSION...)
            if code_clone.is_empty() {
                let placeholder_rect = ui.max_rect();
                let text_pos = placeholder_rect.left_top() + egui::vec2(24.0, 100.0);
                
                // Create a pulsing effect (similar to animate-pulse in React)
                let time = ui.ctx().input(|i| i.time);
                let pulse_alpha = 0.4 + (time.sin().abs() * 0.2) as f32; // Pulse between 40% and 60% opacity
                
                // Create a container with flex-like layout
                let prompt_color = egui::Color32::from_rgb(0, 180, 60).linear_multiply(pulse_alpha);
                
                // First draw the arrow with proper spacing
                ui.painter().text(
                    text_pos,
                    egui::Align2::LEFT_TOP,
                    ">",
                    egui::FontId::monospace(14.0),
                    prompt_color,
                );
                
                // Then the text with proper spacing
                ui.painter().text(
                    text_pos + egui::vec2(15.0, 0.0),
                    egui::Align2::LEFT_TOP,
                    "BEGIN TRANSMISSION...",
                    egui::FontId::monospace(14.0),
                    prompt_color,
                );
                
                // No more blinking cursor after the text
            }

            // Update main code if changed
            if code_clone != self.code {
                self.code = code_clone;
                
                // Update last typing time when code changes
                self.last_typing_time = ui.ctx().input(|i| i.time);
                self.log_message(format!("Text updated, last_typing_time: {}", self.last_typing_time));
            }
            
            // Use a more moderate debounce time - 2 seconds
            let debounce_time = 2.0;
            
            // Only debug log when approaching the threshold to avoid spam
            let current_time = ui.ctx().input(|i| i.time);
            let idle_duration = current_time - self.last_typing_time;
            
            if idle_duration > debounce_time - 0.1 && idle_duration < debounce_time + 0.1 {
                self.log_message(format!(
                    "Auto-analysis check: idle_duration={:.2}, scheduled={}, same_text={}", 
                    idle_duration, 
                    self.analysis_scheduled,
                    self.code == self.last_analyzed_text
                ));
            }
        });
    }

    pub fn panels(&mut self, ctx: &egui::Context) {
        // Start loading the model if not already started
        if !self.loading_started.load(Ordering::SeqCst) {
            self.loading_started.store(true, Ordering::SeqCst);
            self.load_model(LlamaModelType::Default);
        }

        // Apply cyberpunk theme
        self.apply_cyberpunk_theme(ctx);

        // Reset analysis scheduled flag when text is empty
        if self.code.is_empty() {
            self.analysis_scheduled = false;
        }
        
        // Check if we should reset the analysis_scheduled flag
        self.check_and_reset_analysis_scheduled();
        
        // Check if we should show log popup
        if ctx.memory(|mem| {
            mem.data
                .get_temp::<bool>(Id::new("log_popup_open"))
                .unwrap_or(false)
        }) {
            self.show_log_popup(ctx);
        }
        
        // SIMPLE ANALYSIS TRIGGER
        // Only analyze if not already analyzing and text has changed
        if !self.analysis_scheduled && !self.code.is_empty() && self.code != self.last_analyzed_text {
            // Check if debounce time has passed
            let current_time = ctx.input(|i| i.time);
            let idle_duration = current_time - self.last_typing_time;
            let debounce_time = 2.0;
            
            if idle_duration >= debounce_time && self.is_model_ready_for_analysis() {
                self.log_message("Triggering analysis from panels method".to_string());
                self.analyze_text();
            }
        }

        // Main content - full cyberpunk UI
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(0, 0, 0)))
            .show(ctx, |ui| {
                // Full column layout for the app
                ui.vertical(|ui| {
                    self.render_top_bar(ui);

                    // Main content area with split panels - FIXED LAYOUT
                    let available_height = ui.available_height() - 30.0; // Reserve space for status bar
                    
                    // Calculate exact panel widths to avoid pixel rounding issues
                    let total_width = ui.available_width();
                    let panel_width = (total_width - 2.0) / 2.0; // Account for 2px separator
                    
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0); // Remove spacing between items
                        
                        // LEFT PANEL - Neural Input
                        let left_frame = egui::Frame::none()
                            .inner_margin(8.0)
                            .fill(egui::Color32::BLACK);
                            
                        left_frame.show(ui, |ui| {
                            ui.set_min_width(panel_width);
                            ui.set_max_width(panel_width);
                            ui.set_min_height(available_height);
                            self.render_input_panel(ui);
                        });

                        // SEPARATOR
                        let separator_width = 2.0;
                        let separator_rect = egui::Rect::from_min_max(
                            ui.cursor().min,
                            ui.cursor().min + egui::vec2(separator_width, available_height),
                        );
                        ui.painter().rect_filled(
                            separator_rect,
                            0.0,
                            egui::Color32::from_rgb(0, 140, 70),
                        );
                        ui.add_space(separator_width);

                        // RIGHT PANEL - Neural Output
                        let right_frame = egui::Frame::none()
                            .inner_margin(8.0)
                            .fill(egui::Color32::BLACK);
                            
                        right_frame.show(ui, |ui| {
                            ui.set_min_width(panel_width);
                            ui.set_max_width(panel_width);
                            ui.set_min_height(available_height);
                            self.render_output_panel(ui);
                        });
                    });

                    self.render_status_bar(ui);
                });
            });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // This method is now empty since we handle all UI in panels()
    }

    // Apply cyberpunk theme to the UI context
    fn apply_cyberpunk_theme(&self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();

        // Set dark background with green accent colors
        let mut visuals = style.visuals.clone();
        visuals.dark_mode = true;

        // Text colors - brighter green for main text
        visuals.override_text_color = Some(egui::Color32::from_rgb(0, 240, 100));

        // Widget colors - pure black background
        visuals.widgets.noninteractive.bg_fill = egui::Color32::BLACK;
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(5, 15, 5);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(10, 30, 15);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(20, 60, 30);
        visuals.widgets.noninteractive.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 180, 70));
        visuals.widgets.inactive.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 200, 80));
        visuals.widgets.hovered.fg_stroke =
            egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 240, 100));
        visuals.widgets.active.fg_stroke =
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 255, 120));

        // Selection - more vibrant green
        visuals.selection.bg_fill = egui::Color32::from_rgb(0, 120, 60);
        visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 220, 100));

        // Window & panel colors - pure black for backgrounds
        visuals.window_fill = egui::Color32::BLACK;
        visuals.panel_fill = egui::Color32::BLACK;
        visuals.faint_bg_color = egui::Color32::BLACK;
        visuals.extreme_bg_color = egui::Color32::BLACK;

        // Customize spacing and sizes
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(8.0, 4.0);
        
        // Text styles - ensure all text has correct monospace font
        // Create a monospace font for all text styles
        let font_id = egui::FontId::monospace(14.0);
        style.text_styles.insert(egui::TextStyle::Body, font_id.clone());
        style.text_styles.insert(egui::TextStyle::Button, font_id.clone());
        style.text_styles.insert(egui::TextStyle::Small, egui::FontId::monospace(10.0));
        style.text_styles.insert(egui::TextStyle::Heading, egui::FontId::monospace(18.0));
        style.text_styles.insert(egui::TextStyle::Monospace, font_id);
        
        style.visuals = visuals;
        
        // Set the style
        ctx.set_style(style);
    }

    // Render the top bar with SYNC text and status indicators
    fn render_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.style_mut().visuals.widgets.noninteractive.bg_fill = egui::Color32::BLACK;

            // Create a frame for the top bar
            egui::Frame::none()
                .fill(egui::Color32::BLACK)
                .inner_margin(egui::vec2(16.0, 8.0))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 100, 40)))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.style_mut().spacing.item_spacing = egui::vec2(8.0, 0.0);

                        // Left side with SYNC and dots
                        ui.horizontal(|ui| {
                            // SYNC text with tracking (letter spacing)
                            let mut sync_text = egui::RichText::new("SYNC")
                                .color(egui::Color32::from_rgb(0, 255, 100))
                                .size(20.0)
                                .text_style(egui::TextStyle::Monospace)
                                .strong();
                            
                            ui.label(sync_text);

                            ui.add_space(8.0);
                            
                            // Green dots with proper spacing
                            ui.horizontal(|ui| {
                                ui.style_mut().spacing.item_spacing = egui::vec2(4.0, 0.0);
                                for _ in 0..3 {
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(4.0, 4.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().circle_filled(
                                        rect.center(),
                                        1.5,
                                        egui::Color32::from_rgb(0, 255, 100),
                                    );
                                }
                            });
                        });

                        // Status text on right side
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.style_mut().spacing.item_spacing = egui::vec2(16.0, 0.0);

                            // SHOW LOGS button - prominent in top right
                            let logs_button = ui.add(
                                egui::Button::new(
                                    egui::RichText::new("SHOW LOGS")
                                        .text_style(egui::TextStyle::Monospace)
                                        .color(egui::Color32::from_rgb(0, 255, 100))  // Bright green
                                        .size(12.0)
                                )
                                .fill(egui::Color32::from_rgb(0, 40, 20))             // Dark green background
                                .rounding(2.0)                                        // Rounded corners
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 150, 60))) // Border
                            );
                            
                            if logs_button.clicked() {
                                ui.memory_mut(|mem| {
                                    mem.data.insert_temp(Id::new("log_popup_open"), true);
                                });
                            }

                            // Get processing status
                            let is_processing = if let Ok(result) = self.llama_result.lock() {
                                result.generating
                            } else {
                                false
                            };

                            let status_text = if is_processing { "PROCESSING" } else { "READY" };
                            
                            // Neural engine status - smaller text with better color
                            ui.label(
                                egui::RichText::new("NEURAL ENGINE: ACTIVE")
                                    .color(egui::Color32::from_rgb(0, 160, 60))
                                    .text_style(egui::TextStyle::Monospace)
                                    .size(11.0),
                            );
                            
                            // Status indicator - smaller text with better color
                            ui.label(
                                egui::RichText::new(format!("STATUS: {}", status_text))
                                    .color(egui::Color32::from_rgb(0, 160, 60))
                                    .text_style(egui::TextStyle::Monospace)
                                    .size(11.0),
                            );
                        });
                    });
                });
        });

        // Bottom border - thinner and more subtle
        let rect = ui.max_rect();
        ui.painter().line_segment(
            [
                egui::pos2(rect.left(), rect.bottom()),
                egui::pos2(rect.right(), rect.bottom()),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 100, 40)),
        );
    }

    // Render the output panel with text analysis results
    fn render_output_panel(&mut self, ui: &mut egui::Ui) {
        // Get text analysis status
        let (has_analysis, is_analyzing, analysis_text, error, progress) = if let Ok(result) = self.text_analysis_result.lock() {
            (
                !result.text.is_empty(),
                result.generating,
                result.text.clone(),
                result.error.clone(),
                result.progress,
            )
        } else {
            (false, false, String::new(), None, 0.0)
        };

        // Track when analysis started to ensure yellow messages show for minimum time
        static mut ANALYSIS_START_TIME: f64 = 0.0;
        static mut LAST_ANALYSIS_TIME: f64 = 0.0;
        let current_time = ui.ctx().input(|i| i.time);
        
        // If just started analyzing, record the start time
        if is_analyzing {
            unsafe {
                if ANALYSIS_START_TIME == 0.0 {
                    ANALYSIS_START_TIME = current_time;
                }
            }
        }
        
        // Determine if we should show yellow messages
        // Show for at least 3 seconds regardless of analysis completion
        let force_show_yellow = unsafe {
            let elapsed = current_time - ANALYSIS_START_TIME;
            if elapsed < 3.0 && ANALYSIS_START_TIME > 0.0 {
                true
            } else {
                if !is_analyzing && ANALYSIS_START_TIME > 0.0 {
                    // Reset the timer when analysis is complete and time has passed
                    LAST_ANALYSIS_TIME = ANALYSIS_START_TIME;
                    ANALYSIS_START_TIME = 0.0;
                }
                false
            }
        };
        
        // Determine the mode to show
        let show_mode = if is_analyzing || force_show_yellow {
            "analyzing"
        } else if has_analysis {
            "result"
        } else if error.is_some() {
            "error"
        } else {
            "empty"
        };

        ui.vertical(|ui| {
            // Panel header
            ui.add_space(8.0);
            
            ui.horizontal(|ui| {
                // Blue square icon
                let (square_rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter().rect_filled(square_rect, 0.0, egui::Color32::from_rgb(50, 100, 255));
                ui.add_space(8.0);
    
                // NEURAL OUTPUT text
                ui.heading("NEURAL OUTPUT");
            });
            
            // Add separator
            ui.add_space(5.0);
            ui.separator();
            ui.add_space(8.0);

            // Remove the indicator above the scroll area which could be confusing
            // Show cyberpunk-style yellow messages in the main content instead

            ui.add_space(5.0);
            
            // Create a scrolling area for the analysis text
            egui::ScrollArea::vertical().show(ui, |ui| {
                if show_mode == "analyzing" {
                    // Show processing indicator with more cyberpunk messages in yellow - use a VERY BRIGHT yellow
                    let yellow_color = egui::Color32::from_rgb(255, 255, 0);  // Pure bright yellow
                    
                    ui.add_space(10.0); // Add some space at the top
                    
                    // Text is smaller and lowercase to match screenshot
                    ui.label(
                        egui::RichText::new("Analyzing input patterns...")
                            .color(yellow_color)
                            .text_style(egui::TextStyle::Monospace)
                            .size(14.0) // Smaller text size to match reference
                    );
                    
                    ui.add_space(4.0); // Reduced spacing between lines
                    
                    ui.label(
                        egui::RichText::new("Processing language context...")
                            .color(yellow_color)
                            .text_style(egui::TextStyle::Monospace)
                            .size(14.0) // Smaller text size to match reference
                    );
                    
                    ui.add_space(4.0); // Reduced spacing between lines
                    
                    ui.label(
                        egui::RichText::new("Generating strategic response...")
                            .color(yellow_color)
                            .text_style(egui::TextStyle::Monospace)
                            .size(14.0) // Smaller text size to match reference
                    );
                    
                    ui.add_space(8.0); // Reduced spacing between lines
                    
                    ui.label(
                        egui::RichText::new("Searching neural pathways...")
                            .color(yellow_color)
                            .text_style(egui::TextStyle::Monospace)
                            .size(14.0) // Smaller text size to match reference
                    );
                    
                    ui.add_space(12.0);
                    
                    // Add progress percentage text like in the React reference
                    ui.add_space(8.0);
                    
                    // Get current time for animation
                    let current_time = ui.ctx().input(|i| i.time);
                    
                    // Track when analysis was first shown to ensure minimum display time
                    if let Ok(mut result) = self.text_analysis_result.lock() {
                        // If we're at 90% or higher but not yet at 100%, and we've been showing for at least 3 seconds,
                        // auto-complete the progress
                        if result.progress >= 0.9 && result.progress < 1.0 && result.generating {
                            // Auto-complete progress to 100% after appropriate time
                            // Use a smooth transition from 90% to 100%
                            let progress_increment = 0.02; // 2% increment per frame
                            result.progress = (result.progress + progress_increment).min(1.0);
                            
                            // If we've reached 100%, mark as not generating anymore
                            if result.progress >= 1.0 {
                                result.generating = false;
                            }
                        }
                        
                        // Ensure progress bar shows at least a minimal value even at 0%
                        let display_progress = if result.progress < 0.01 && result.generating {
                            0.01 // Always show at least a tiny bit of progress
                        } else {
                            result.progress
                        };
                        
                        // Display the progress percentage with lowercase text to match reference
                        ui.label(
                            egui::RichText::new(format!("Processing neural patterns: {}%", (display_progress * 100.0) as i32))
                                .color(egui::Color32::from_rgb(0, 255, 100))
                                .text_style(egui::TextStyle::Monospace)
                                .size(14.0)
                        );
                    } else {
                        // Fallback if we can't lock the result
                        ui.label(
                            egui::RichText::new(format!("Processing neural patterns: {}%", (progress * 100.0) as i32))
                                .color(egui::Color32::from_rgb(0, 255, 100))
                                .text_style(egui::TextStyle::Monospace)
                                .size(14.0)
                        );
                    }
                    
                    ui.add_space(8.0);
                    
                    // Improved progress bar with custom styling
                    // Background for progress bar
                    let progress_rect = ui.available_rect_before_wrap();
                    let progress_rect = egui::Rect::from_min_size(
                        progress_rect.min,
                        egui::vec2(progress_rect.width(), 5.0)
                    );
                    ui.painter().rect_filled(
                        progress_rect, 
                        0.0,
                        egui::Color32::from_rgb(0, 80, 30) // Dark green background for progress bar
                    );
                    
                    // Use smoothed progress for display
                    let display_progress = if let Ok(result) = self.text_analysis_result.lock() {
                        // Ensure progress bar shows at least a minimal value even at 0%
                        if result.progress < 0.01 && result.generating {
                            0.01 // Always show at least a tiny bit of progress
                        } else {
                            result.progress
                        }
                    } else {
                        progress
                    };
                    
                    // Actual progress indicator
                    let fill_width = progress_rect.width() * display_progress;
                    let fill_rect = egui::Rect::from_min_size(
                        progress_rect.min,
                        egui::vec2(fill_width, progress_rect.height())
                    );
                    ui.painter().rect_filled(
                        fill_rect,
                        0.0,
                        egui::Color32::from_rgb(0, 255, 100) // Bright green fill
                    );
                    
                    ui.add_space(16.0);
                    
                    // Add random characters in a grid layout similar to React
                    ui.add_space(16.0);
                    
                    // Use a centered layout approach to ensure symmetrical padding
                    let available_width = ui.available_width();
                    let actual_grid_width = available_width * 0.95; // Use 95% of available width
                    let left_margin = (available_width - actual_grid_width) / 2.0; // Calculate exact left margin
                    
                    ui.add_space(left_margin); // Add explicit left margin
                    
                    // Create a container for the grid with exact width
                    ui.scope(|ui| {
                        ui.set_width(actual_grid_width);
                        
                        // Create a grid-like layout for random characters (10 columns)
                        let time = ui.ctx().input(|i| i.time);
                        let grid_columns = 10;
                        let total_chars = 50;
                        let rows = (total_chars + grid_columns - 1) / grid_columns;
                        
                        // Calculate column width based on fixed grid width
                        let column_width = actual_grid_width / grid_columns as f32;
                        
                        for row in 0..rows {
                            // Start a new row with exact sizing to ensure even distribution
                            ui.horizontal(|ui| {
                                ui.set_width(actual_grid_width);
                                
                                for col in 0..grid_columns {
                                    let idx = row * grid_columns + col;
                                    if idx < total_chars {
                                        let char_code = 33 + (time as usize * 10 + rand::random::<usize>()) % 93;
                                        let random_char = std::char::from_u32(char_code as u32).unwrap_or('?');
                                        
                                        // Random color for each character, matching React implementation colors
                                        let color = match rand::random::<u8>() % 3 {
                                            0 => egui::Color32::from_rgb(0, 255, 100), // bright green
                                            1 => egui::Color32::from_rgb(255, 100, 50),  // bright orange/red
                                            _ => egui::Color32::from_rgb(50, 150, 255), // bright blue
                                        };
                                        
                                        // Allocate exact space for each column
                                        let (rect, _) = ui.allocate_exact_size(
                                            egui::vec2(column_width, 24.0),
                                            egui::Sense::hover()
                                        );
                                        
                                        // Draw the character centered in its cell
                                        ui.painter().text(
                                            rect.center(),
                                            egui::Align2::CENTER_CENTER,
                                            random_char.to_string(),
                                            egui::FontId::monospace(14.0), // Smaller font size to match React's text-xs
                                            color
                                        );
                                    }
                                }
                            });
                            
                            // Add a small gap between rows
                            ui.add_space(2.0);
                        }
                    });
                
                } else if show_mode == "result" {
                    // Show the yellow processing text headers first, then the result
                    // This matches the screenshot where the yellow status text remains visible
                    let yellow_color = egui::Color32::from_rgb(255, 255, 0);  // Pure bright yellow
                    
                    ui.add_space(10.0); // Add some space at the top
                    
                    // Text is smaller and lowercase to match screenshot
                    ui.label(
                        egui::RichText::new("Analyzing input patterns...")
                            .color(yellow_color)
                            .text_style(egui::TextStyle::Monospace)
                            .size(14.0) // Smaller text size to match reference
                    );
                    
                    ui.add_space(4.0); // Reduced spacing between lines
                    
                    ui.label(
                        egui::RichText::new("Processing language context...")
                            .color(yellow_color)
                            .text_style(egui::TextStyle::Monospace)
                            .size(14.0) // Smaller text size to match reference
                    );
                    
                    ui.add_space(4.0); // Reduced spacing between lines
                    
                    ui.label(
                        egui::RichText::new("Generating strategic response...")
                            .color(yellow_color)
                            .text_style(egui::TextStyle::Monospace)
                            .size(14.0) // Smaller text size to match reference
                    );
                    
                    ui.add_space(16.0);
                    
                    // Now display the analysis text in green
                    let green_text_color = egui::Color32::from_rgb(0, 220, 100);
                    let lines = analysis_text.split('\n').collect::<Vec<&str>>();
                    
                    for line in lines {
                        if !line.is_empty() {
                            ui.label(
                                egui::RichText::new(line)
                                    .color(green_text_color)
                                    .text_style(egui::TextStyle::Monospace)
                                    .size(14.0)
                            );
                            ui.add_space(4.0);
                        }
                    }
                } else if show_mode == "error" {
                    // Show error
                    if let Some(err) = error {
                        ui.colored_label(egui::Color32::RED, format!("Error: {}", err));
                    } else {
                        ui.colored_label(egui::Color32::RED, "Unknown error occurred");
                    }
                } else {
                    // Empty state
                    ui.weak("Awaiting neural processing...");
                }
            });
        });
    }

    // Render the status bar at the bottom
    fn render_status_bar(&mut self, ui: &mut egui::Ui) {
        // Top border - match exact color from reference
        let rect = ui.max_rect();
        ui.painter().line_segment(
            [
                egui::pos2(rect.left(), rect.top()),
                egui::pos2(rect.right(), rect.top()),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 80, 30)),
        );

        ui.horizontal(|ui| {
            ui.style_mut().spacing.item_spacing = egui::vec2(16.0, 0.0);

            // Create a frame for consistent padding - match reference padding
            egui::Frame::none()
                .inner_margin(egui::vec2(16.0, 4.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Left side - stats with proper color
                        let time = ui.input(|i| i.time);
                        // Random units between 30-100
                        let random_units = ((time.sin().abs() * 70.0) as u32 + 30) as usize;
                        let token_count = if self.code.is_empty() {
                            0
                        } else {
                            self.code.split_whitespace().count() * 4 + 18
                        };

                        ui.label(
                            egui::RichText::new(format!("UNITS: {}", random_units))
                                .text_style(egui::TextStyle::Monospace)
                                .color(egui::Color32::from_rgb(0, 150, 60))
                                .size(11.0),
                        );

                        ui.add_space(16.0);

                        ui.label(
                            egui::RichText::new(format!("TOKENS: {}", token_count))
                                .text_style(egui::TextStyle::Monospace)
                                .color(egui::Color32::from_rgb(0, 150, 60))
                                .size(11.0),
                        );

                        // Right side - signal indicators
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.horizontal(|ui| {
                                ui.style_mut().spacing.item_spacing = egui::vec2(2.0, 0.0);
                                
                                // Exactly 10 signal bars with consistent width and spacing
                                for i in 0..10 {
                                    // Vary the heights but keep consistent with reference design
                                    let random_active = {
                                        let t = ui.input(|inp| inp.time) * 5.0 + i as f64;
                                        t.sin().abs() > 0.3
                                    };
                                    
                                    ui.painter().rect_filled(
                                        egui::Rect::from_min_max(
                                            ui.next_widget_position(),
                                            ui.next_widget_position() + egui::vec2(4.0, 12.0 * if random_active { 1.0 } else { 0.4 }),
                                        ),
                                        0.0,
                                        if random_active {
                                            egui::Color32::from_rgb(0, 150, 60)
                                        } else {
                                            egui::Color32::from_rgb(0, 80, 30)
                                        },
                                    );
                                    
                                    ui.add_space(2.0);
                                }
                            });
                        });
                    });
                });
        });
    }

    // Log popup dialog - simplified to only show the popup
    pub fn show_log_popup(&mut self, ctx: &egui::Context) {
        // Create a window for the log messages with better styling
        let mut open = true;
        egui::Window::new("NEURAL LOG TERMINAL")
            .open(&mut open) // Track if window is open
            .collapsible(false)
            .resizable(true)
            .min_width(400.0)
            .min_height(300.0)
            .default_pos(egui::pos2(100.0, 100.0))
            .frame(egui::Frame::window(&ctx.style()).fill(egui::Color32::from_rgb(0, 20, 10)))
            .show(ctx, |ui| {
                ui.style_mut().visuals.override_text_color = Some(egui::Color32::from_rgb(0, 220, 100));
                
                // Add a clear button with cyberpunk styling
                if ui.add(egui::Button::new(
                    egui::RichText::new("CLEAR LOG DATA")
                        .text_style(egui::TextStyle::Monospace)
                        .color(egui::Color32::from_rgb(0, 255, 100))
                )
                .fill(egui::Color32::from_rgb(0, 40, 20)))
                .clicked() {
                    if let Ok(mut messages) = self.log_messages.lock() {
                        messages.clear();
                    }
                }

                ui.add_space(8.0);

                // Display each log message in a scroll area
                egui::ScrollArea::vertical()
                    .max_height(f32::INFINITY)
                    .show(ui, |ui| {
                        if let Ok(messages) = self.log_messages.lock() {
                            if messages.is_empty() {
                                ui.label("No log messages available.");
                            } else {
                                for message in messages.iter() {
                                    ui.label(
                                        egui::RichText::new(message)
                                            .text_style(egui::TextStyle::Monospace)
                                            .size(12.0)
                                    );
                                }
                            }
                        } else {
                            ui.label("Error: Could not access log messages.");
                        }
                    });
            });
            
        // If window was closed, update the memory
        if !open {
            ctx.memory_mut(|mem| {
                mem.data.insert_temp(Id::new("log_popup_open"), false);
            });
        }
    }

    // Thread-safe logging function
    fn log_message(&self, message: String) {
        if let Ok(mut messages) = self.log_messages.lock() {
            messages.push(message);
        }
    }

    // Get completion suggestions from the llama model
    fn get_completion_from_llama(&self, prompt: &str) -> Option<String> {
        if !self.is_model_ready_for_completion() {
            self.log_message("Completion attempted but model not ready".to_string());
            return None;
        }

        if prompt.len() < 2 {
            self.log_message(format!("Word too short for completion: {}", prompt));
            return None;
        }

        self.log_message(format!("Attempting completion for word: {}", prompt));

        // Reset KV cache before each completion to avoid position issues
        self.reset_kv_cache();

        if let Ok(model_guard) = self.model.lock() {
            if let Some(model) = &*model_guard {
                let dev = Device::Cpu;
                let temp = Some(0.3);
                let top_p = Some(0.9);
                let mut logits_processor = LogitsProcessor::new(299792458, temp, top_p);

                // Tokenize the input prompt
                match model.tokenizer.encode(prompt, true) {
                    Ok(encoded) => {
                        let mut tokens = encoded.get_ids().to_vec();
                        let mut generated_text = String::new();
                        let mut index_pos = 0;

                        // Generate up to 10 tokens or until we hit a natural stopping point
                        for _ in 0..10 {
                            // Create input tensor for the current token(s)
                            let context_size = if model.cache.use_kv_cache && index_pos > 0 {
                                1 // Use only the last token if using KV cache
                            } else {
                                tokens.len() // Use all tokens for first pass
                            };

                            let ctxt_range =
                                tokens.len().saturating_sub(context_size)..tokens.len();
                            let ctxt = tokens[ctxt_range].to_vec();

                            let input = match Tensor::new(ctxt.as_slice(), &dev)
                                .and_then(|t| t.unsqueeze(0))
                            {
                                Ok(t) => t,
                                Err(e) => {
                                    self.log_message(format!(
                                        "Error creating input tensor: {:?}",
                                        e
                                    ));
                                    return if generated_text.is_empty() {
                                        None
                                    } else {
                                        Some(generated_text)
                                    };
                                }
                            };

                            // Forward pass through the model
                            match model.llama.forward(&input, index_pos) {
                                Ok(logits) => {
                                    // Update index_pos for next iteration if using KV cache
                                    if model.cache.use_kv_cache {
                                        index_pos += ctxt.len();
                                    }

                                    // Sample next token
                                    match logits.squeeze(0) {
                                        Ok(squeezed_logits) => {
                                            match logits_processor.sample(&squeezed_logits) {
                                                Ok(next_token) => {
                                                    // Convert token to text
                                                    if let Some(text) =
                                                        model.tokenizer.id_to_token(next_token)
                                                    {
                                                        let token_text = text
                                                            .replace('▁', " ")
                                                            .replace("<0x0A>", "\n");

                                                        self.log_message(format!(
                                                            "Generated token: {}",
                                                            token_text
                                                        ));

                                                        // If we get a space or punctuation after generating something,
                                                        // it's a good place to stop
                                                        if !generated_text.is_empty()
                                                            && (token_text.trim().is_empty()
                                                                || token_text.contains(
                                                                    |c: char| {
                                                                        c.is_ascii_punctuation()
                                                                    },
                                                                ))
                                                        {
                                                            return Some(generated_text);
                                                        }

                                                        // Add to result and continue generating
                                                        generated_text.push_str(&token_text);
                                                        tokens.push(next_token);

                                                        // Stop if we hit a newline
                                                        if token_text.contains('\n') {
                                                            break;
                                                        }
                                                    } else {
                                                        self.log_message(format!(
                                                            "Could not decode token: {}",
                                                            next_token
                                                        ));
                                                        break;
                                                    }
                                                }
                                                Err(e) => {
                                                    self.log_message(format!(
                                                        "Error sampling token: {:?}",
                                                        e
                                                    ));
                                                    break;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            self.log_message(format!(
                                                "Error squeezing logits: {:?}",
                                                e
                                            ));
                                            break;
                                        }
                                    }
                                }
                                Err(e) => {
                                    self.log_message(format!("Error in forward pass: {:?}", e));
                                    break;
                                }
                            }
                        }

                        // Return the generated text if we have any
                        if !generated_text.is_empty() {
                            self.log_message(format!("Generated completion: {}", generated_text));
                            return Some(generated_text);
                        }
                    }
                    Err(e) => {
                        self.log_message(format!("Error tokenizing prompt: {:?}", e));
                    }
                }
            } else {
                self.log_message("Model not loaded".to_string());
            }
        } else {
            self.log_message("Could not lock model".to_string());
        }

        None
    }

    // Check if the model is loaded and ready to generate completions
    fn is_model_ready_for_completion(&self) -> bool {
        if let Ok(result) = self.llama_result.lock() {
            return result.loaded && !result.generating;
        }
        false
    }

    // Helper method to check if model is ready for text analysis
    fn is_model_ready_for_analysis(&self) -> bool {
        if let Ok(model_guard) = self.model.lock() {
            let model_loaded = model_guard.is_some();
            
            if let Ok(result) = self.text_analysis_result.lock() {
                // Consider it ready if model is loaded and not currently generating
                let is_ready = model_loaded && !result.generating;
                
                if !is_ready {
                    self.log_message(format!(
                        "Model not ready for analysis: loaded={}, generating={}", 
                        model_loaded, 
                        result.generating
                    ));
                }
                
                return is_ready;
            }
        }
        
        self.log_message("Failed to lock model or results for analysis readiness check".to_string());
        false
    }

    // Reset KV cache for both completion and analysis
    fn reset_kv_cache(&self) {
        if let Ok(model_guard) = self.model.lock() {
            if let Some(model) = &*model_guard {
                // Reset the KV cache by clearing all entries
                if let Ok(mut kvs) = model.cache.kvs.lock() {
                    for kv in kvs.iter_mut() {
                        *kv = None;
                    }
                }
            }
        }
    }

    // Analyze text content in the editor
    pub fn analyze_text(&mut self) {
        let current_text = self.code.clone();
        
        // Don't re-analyze if text hasn't changed since last analysis
        if !self.analysis_scheduled && current_text == self.last_analyzed_text && self.is_model_ready_for_analysis() {
            self.log_message("Skipping analysis: text hasn't changed since last analysis".to_string());
            return;
        }
        
        // Reset KV cache before analyzing
        self.reset_kv_cache();
        
        // Update state to show we're analyzing
        if let Ok(mut result) = self.text_analysis_result.lock() {
            result.text = "Analyzing text...".to_string();
            result.generating = true;
            result.status = "Analysis in progress...".to_string();
            result.progress = 0.0; // Reset progress at start
            // Make sure loaded is set to true if we're starting an analysis
            result.loaded = true;
        }
        
        self.analysis_scheduled = true;
        self.last_analyzed_text = current_text.clone();
        
        // Log the analysis attempt
        self.log_message(format!("Starting text analysis with scheduled={}", self.analysis_scheduled));
        
        // Prepare analysis prompt
        let analysis_prompt = format!(
            "Analyze the following text and provide insights about its structure, style, and content: \n\n{}",
            current_text
        );
        
        // Create the model instance and analyze text using the embedded model data
        let model_data = crate::llama::get_embedded_model_data();
        let temperature = 0.5; // Lower temperature for more focused analysis
        let top_p = 0.9;
        
        // Clone necessary data for async processing
        let text_analysis_result_clone = self.text_analysis_result.clone();
        let prompt_clone = analysis_prompt.clone();
        
        // Handle different platforms
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen_futures::spawn_local;
            spawn_local(async move {
                Self::process_text_analysis(
                    model_data,
                    prompt_clone,
                    temperature,
                    top_p,
                    text_analysis_result_clone,
                );
            });
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                Self::process_text_analysis(
                    model_data,
                    prompt_clone,
                    temperature,
                    top_p,
                    text_analysis_result_clone,
                );
            });
        }
    }
    
    // Shared implementation for text analysis processing
    fn process_text_analysis(
        model_data: crate::llama::ModelData,
        prompt: String,
        temperature: f64,
        top_p: f64,
        text_analysis_result: SharedTextAnalysisResult,
    ) {
        // Make sure progress starts at 0%
        if let Ok(mut result) = text_analysis_result.lock() {
            result.progress = 0.0; // Explicitly start at 0%
        }
        
        // Try to load the model and generate analysis
        match crate::llama::Model::load(model_data) {
            Ok(model) => {
                // Log success
                if let Ok(mut result) = text_analysis_result.lock() {
                    result.status = "Model loaded, analyzing text...".to_string();
                    result.loaded = true;
                    // Still keep progress at very low value
                    result.progress = 0.05; // Just 5% progress after model loads
                }
                
                // Platform-specific progress animation handling
                #[cfg(not(target_arch = "wasm32"))]
                {
                    // Start time tracking for the minimum 3 second animation
                    let start_time = std::time::Instant::now();
                    
                    // Use the generate method from the Model
                    let generation_result = model.generate(&prompt, temperature, top_p);
                    
                    // Update progress gradually over time to ensure minimum 3 second animation
                    let min_animation_duration = std::time::Duration::from_secs(3);
                    let elapsed = start_time.elapsed();
                    
                    // If the generation was faster than our minimum duration, gradually animate progress
                    if elapsed < min_animation_duration {
                        // Calculate how much more time we need to wait
                        let remaining_time = min_animation_duration - elapsed;
                        let steps = 8; // Number of progress steps
                        let step_duration = remaining_time.as_millis() as u64 / steps as u64;
                        
                        // Gradually update progress
                        for step in 1..=steps {
                            // Sleep briefly between progress updates
                            std::thread::sleep(std::time::Duration::from_millis(step_duration));
                            
                            // Update progress to show animation
                            if let Ok(mut result) = text_analysis_result.lock() {
                                if result.generating { // Only update if still generating
                                    // Progress from 10% to 90% over steps
                                    result.progress = 0.1 + (0.8 * (step as f32 / steps as f32));
                                }
                            }
                        }
                    }
                    
                    // Process the generation result with full completion
                    match generation_result {
                        Ok(analysis_text) => {
                            // Update with successful analysis
                            if let Ok(mut result) = text_analysis_result.lock() {
                                result.text = analysis_text;
                                result.status = "Analysis complete".to_string();
                                result.generating = false;
                                result.progress = 1.0; // Set to 100% when complete
                            }
                        }
                        Err(e) => {
                            // Handle analysis error
                            if let Ok(mut result) = text_analysis_result.lock() {
                                result.text = format!("Error analyzing text: {}", e);
                                result.status = "Analysis failed".to_string();
                                result.generating = false;
                                result.error = Some(format!("Analysis error: {}", e));
                            }
                        }
                    }
                }
                
                #[cfg(target_arch = "wasm32")]
                {
                    // For WebAssembly, use a simpler approach with predefined progress steps
                    // Update progress to 10% before starting
                    if let Ok(mut result) = text_analysis_result.lock() {
                        result.progress = 0.1;
                    }
                    
                    // Generate the analysis text
                    let generation_result = model.generate(&prompt, temperature, top_p);
                    
                    // Set progress to intermediate value to show work happening
                    if let Ok(mut result) = text_analysis_result.lock() {
                        result.progress = 0.5; // 50% progress after generation completes
                    }
                    
                    // In WASM, we'll use setTimeout callbacks for the animation
                    // This value gets shared with JS to control progress
                    // The animation will be driven by the UI refresh cycle
                    // We just need to set a few key progress points
                    
                    // Process the generation result
                    match generation_result {
                        Ok(analysis_text) => {
                            // Update with successful analysis but keep progress at 90%
                            // to allow the UI animation to complete naturally
                            if let Ok(mut result) = text_analysis_result.lock() {
                                result.text = analysis_text;
                                result.status = "Analysis complete".to_string();
                                // Don't set generating=false yet to let animation complete
                                result.progress = 0.9; // 90% progress - UI will animate to 100%
                            }
                            
                            // Final progress update to 100% will happen in UI rendering
                        }
                        Err(e) => {
                            // Handle analysis error
                            if let Ok(mut result) = text_analysis_result.lock() {
                                result.text = format!("Error analyzing text: {}", e);
                                result.status = "Analysis failed".to_string();
                                result.generating = false;
                                result.error = Some(format!("Analysis error: {}", e));
                            }
                        }
                    }
                }
            }
            Err(e) => {
                // Handle error in model loading
                if let Ok(mut result) = text_analysis_result.lock() {
                    result.text = format!("Error loading model: {}", e);
                    result.status = "Analysis failed".to_string();
                    result.generating = false;
                    result.error = Some(format!("Model loading error: {}", e));
                }
            }
        }
    }

    // UI panel method
    pub fn llama_ui_panel(&mut self, ui: &mut egui::Ui, response_id: egui::Id) {
        // Get current status
        let (status, has_response, is_generating) = if let Ok(result) = self.llama_result.lock() {
            (
                result.status.clone(),
                !result.text.is_empty(),
                result.generating,
            )
        } else {
            ("Error accessing Llama result".to_string(), false, false)
        };

        ui.vertical(|ui| {
            // Model loading button
            ui.horizontal(|ui| {
                if ui.button("Load Llama Model").clicked() {
                    self.load_model(LlamaModelType::Default);
                }

                // Show status
                ui.label(&status);
            });

            // Prompt input
            ui.horizontal(|ui| {
                ui.label("Prompt:");
                ui.text_edit_singleline(&mut self.ai_prompt);
            });

            // Generate button
            ui.horizontal(|ui| {
                if !is_generating {
                    if ui.button("Generate").clicked() {
                        self.generate_text(self.ai_prompt.clone());
                    }
                } else {
                    ui.spinner();
                    ui.label("Generating...");
                }

                if has_response && !is_generating {
                    if ui.button("Insert at cursor").clicked() {
                        self.insert_ai_text(ui, response_id);
                    }
                }
            });

            // Response preview (if available)
            if has_response {
                ui.collapsing("AI Response", |ui| {
                    if let Ok(result) = self.llama_result.lock() {
                        ui.label(&result.text);
                    }
                });
            }
        });
    }

    // Insert AI text method
    pub fn insert_ai_text(&mut self, ui: &mut egui::Ui, response_id: egui::Id) {
        let text_to_insert = if let Ok(result) = self.llama_result.lock() {
            result.text.clone()
        } else {
            String::new()
        };

        if !text_to_insert.is_empty() {
            if let Some(mut state) = TextEdit::load_state(ui.ctx(), response_id) {
                if let Some(ccursor_range) = state.cursor.char_range() {
                    let [primary, _secondary] = ccursor_range.sorted();
                    let cursor_pos = primary.index;

                    // Insert the AI response at cursor position
                    self.code.insert_str(cursor_pos, &text_to_insert);

                    // Clear the response after inserting
                    if let Ok(mut result) = self.llama_result.lock() {
                        result.text.clear();
                    }
                }
            }
        }
    }

    // New method to check if model is loaded (works on all platforms)
    fn is_model_loaded_web(&self) -> bool {
        if let Ok(result) = self.llama_result.lock() {
            result.loaded
        } else {
            false
        }
    }

    // Public method to generate text
    pub fn generate_text(&mut self, prompt: String) {
        // Reset KV cache before generating new text
        self.reset_kv_cache();
        
        // Directly run the generation with the embedded model
        self.run_generation(prompt);
    }
    
    // Generate text with the Llama model
    fn run_generation(&mut self, prompt: String) {
        // Update the UI to show we're generating
        if let Ok(mut result) = self.llama_result.lock() {
            result.text = format!("Generating: {}", prompt);
            result.generating = true;
            result.status = "Generation in progress...".to_string();
        }
        
        // Log the generation attempt
        self.log_message(format!("Generating text for prompt: {}", prompt));
        
        // Create the actual model instance and generate text using the embedded model data
        let model_data = crate::llama::get_embedded_model_data();
        let temperature = 0.7; // Default temperature
        let top_p = 0.9; // Default top_p
        
        // Clone necessary data for async processing
        let llama_result_clone = self.llama_result.clone();
        let prompt_clone = prompt.clone();
        
        // Handle different platforms
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen_futures::spawn_local;
            spawn_local(async move {
                Self::process_generation(
                    model_data,
                    prompt_clone,
                    temperature,
                    top_p,
                    llama_result_clone,
                );
            });
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                Self::process_generation(
                    model_data,
                    prompt_clone,
                    temperature,
                    top_p,
                    llama_result_clone,
                );
            });
        }
    }
    
    // Shared implementation for both WASM and non-WASM targets
    fn process_generation(
        model_data: crate::llama::ModelData,
        prompt: String,
        temperature: f64,
        top_p: f64,
        llama_result: SharedLlamaResult,
    ) {
        // Try to load the model and generate text
        match Model::load(model_data) {
            Ok(model) => {
                // Log success
                if let Ok(mut result) = llama_result.lock() {
                    result.status = "Model loaded, generating text...".to_string();
                }
                
                // Use the generate method from the Model
                match model.generate(&prompt, temperature, top_p) {
                    Ok(generated_text) => {
                        // Update with successful generation
                        if let Ok(mut result) = llama_result.lock() {
                            result.text = generated_text;
                            result.status = "Generation complete".to_string();
                            result.generating = false;
                        }
                    }
                    Err(e) => {
                        // Handle generation error
                        if let Ok(mut result) = llama_result.lock() {
                            result.text = format!("Error generating text: {}", e);
                            result.status = "Generation failed".to_string();
                            result.generating = false;
                        }
                    }
                }
            }
            Err(e) => {
                // Handle error in model loading
                if let Ok(mut result) = llama_result.lock() {
                    result.text = format!("Error loading model for generation: {}", e);
                    result.status = "Model loading failed".to_string();
                    result.generating = false;
                }
            }
        }
    }

    // Check if analysis has completed and reset the scheduled flag
    fn check_and_reset_analysis_scheduled(&mut self) {
        // If analysis is scheduled but model isn't generating, reset the flag
        if self.analysis_scheduled {
            let should_reset = if let Ok(analysis_result) = self.text_analysis_result.lock() {
                // Use our tracking state to check if analysis is complete
                !analysis_result.generating 
            } else {
                false // Can't lock, don't reset
            };

            if should_reset {
                self.log_message("Resetting analysis_scheduled flag".to_string());
                self.analysis_scheduled = false;
                // We'll update last_typing_time in the UI thread since we need context
            }
        }
    }
}

fn find_word_at_cursor(text: &str) -> String {
    // Use the approach from the working example:
    // 1. Split by whitespace
    // 2. Take the last part (the word before cursor)
    // 3. Trim non-alphanumeric characters from the start
    text.split(|c: char| c.is_whitespace() || c == '\n')
        .last()
        .unwrap_or("")
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

const DEFAULT_CODE: &str = r#"

"#;
