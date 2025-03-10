# Itty Bitty AI

A lightweight AI-powered text editor built with Rust, egui, and llama.cpp.

## Features

- Modern, easy-to-use text editor
- Integrated AI text generation and analysis
- Cross-platform (desktop and web)
- Offline capable

## Cross-Platform Architecture

Itty Bitty AI is built with a true cross-platform architecture:

- **Single Codebase**: Written in Rust and compiled to both native binaries and WebAssembly
- **Desktop Native**: Runs as a native application on Windows, macOS, and Linux with native performance and UI rendering
- **Web Enabled**: Compiles to WebAssembly and runs in any modern browser using Canvas-based rendering
- **Consistent Experience**: The same features and interface on all platforms thanks to egui's abstraction layer

The application use pure Rust immediate mode GUI framework that:
- Provides native-feeling interfaces on desktop platforms
- Renders via Canvas API when running in browsers
- Maintains the same look, feel, and functionality across all environments

## Running the Application

### Prerequisites

Make sure you have the latest version of stable Rust by running:

```bash
rustup update stable
```

### Running on Web

You can compile and run the application as a web page:

1. Install the required target:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

2. Install Trunk:
   ```bash
   cargo install --locked trunk
   ```

3. Build and serve:
   ```bash
   trunk serve
   ```

4. Open `http://127.0.0.1:8080/index.html#dev` in your browser.
   - Adding `#dev` skips caching, allowing you to see the latest changes during development.

### Running on Desktop

To run the application locally:

```bash
# Build and run in debug mode
cargo run

# Build and run in release mode (recommended for better performance)
cargo run --release
```

#### Linux Dependencies

On Ubuntu/Debian:

```bash
sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev
```


On Fedora:

```bash
dnf install clang clang-devel clang-tools-extra libxkbcommon-devel pkg-config openssl-devel libxcb-devel gtk3-devel atk fontconfig-devel
```

## Deployment

To build for web deployment:

1. Run `trunk build --release`
2. Deploy the generated `dist` directory to your preferred hosting service.

## Development

This application is built with:
- [egui](https://github.com/emilk/egui/) for the UI framework - a portable immediate mode GUI framework written in Rust
- [eframe-template](https://github.com/emilk/eframe_template) for the eframe boilerplate
- [candle](https://github.com/huggingface/candle) for the machine learning backend

## License

This project is licensed under the MIT License.
