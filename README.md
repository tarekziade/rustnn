<div align="center">
  <img src="logo/rustnn.png" alt="rustnn logo" width="200"/>

  # rustnn

  A Rust implementation of the W3C WebNN specification for neural network graph validation and backend conversion.
</div>

---

## [WARNING] EXPERIMENTAL - DO NOT USE IN PRODUCTION

This is an early-stage experimental implementation for research and exploration. Many features are incomplete, untested, or may change significantly.

---

## What is rustnn?

rustnn is a Rust library that provides:

- **WebNN Graph Validation**: Validates WebNN graph structures against the W3C specification
- **Backend Conversion**: Converts WebNN graphs to ONNX and CoreML formats
- **Runtime Backends**: Executes graphs on CPU, GPU, or Neural Engine
- **Shape Inference**: Automatic tensor shape computation
- **Operation Support**: 88 WebNN operations (84% spec coverage)

## Python Bindings

Python users should use **[pywebnn](https://github.com/rustnn/pywebnn)** - a separate package that provides full W3C WebNN API Python bindings using rustnn as the core library.

**Install Python package:**
```bash
pip install pywebnn
```

See the [pywebnn repository](https://github.com/rustnn/pywebnn) for Python documentation and examples.

## Rust Library Installation

Add rustnn to your `Cargo.toml`:

```toml
[dependencies]
rustnn = { git = "https://github.com/rustnn/rustnn" }

# Optional: Enable runtime backends
rustnn = { git = "https://github.com/rustnn/rustnn", features = ["onnx-runtime"] }
```

**Features:**
- `onnx-runtime` - ONNX Runtime execution (CPU/GPU)
- `coreml-runtime` - CoreML execution (macOS only)
- `trtx-runtime` - TensorRT execution (Linux/Windows with NVIDIA GPU)

## Quick Start (Rust)

```rust
use rustnn::graph::GraphInfo;
use rustnn::converters::{GraphConverter, OnnxConverter};
use rustnn::validator::GraphValidator;

// Load a WebNN graph from JSON
let graph: GraphInfo = serde_json::from_str(&json_string)?;

// Validate the graph
let validator = GraphValidator::new();
let artifacts = validator.validate(&graph)?;

// Convert to ONNX
let converter = OnnxConverter;
let onnx_model = converter.convert(&graph)?;

// Save ONNX model
std::fs::write("model.onnx", onnx_model.data)?;
```

**For Python examples**, see the [pywebnn repository](https://github.com/rustnn/pywebnn).

## Backend Selection

Following the [W3C WebNN Device Selection spec](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md), backends are selected via hints:

```python
# CPU-only execution
context = ml.create_context(accelerated=False)

# Request GPU/NPU (platform selects best available)
context = ml.create_context(accelerated=True)

# Request high-performance (prefers GPU)
context = ml.create_context(accelerated=True, power_preference="high-performance")

# Request low-power (prefers NPU/Neural Engine)
context = ml.create_context(accelerated=True, power_preference="low-power")
```

**Platform-Specific Backends:**
- NPU: CoreML Neural Engine (Apple Silicon macOS only)
- GPU: ONNX Runtime GPU (cross-platform) or CoreML GPU (macOS)
- CPU: ONNX Runtime CPU (cross-platform)

## Examples

### Complete MobileNetV2 Image Classification

```bash
# Download pretrained weights (first time only)
bash scripts/download_mobilenet_weights.sh

# Run on different backends
python examples/mobilenetv2_complete.py examples/images/test.jpg --backend cpu
python examples/mobilenetv2_complete.py examples/images/test.jpg --backend gpu
python examples/mobilenetv2_complete.py examples/images/test.jpg --backend coreml
```

**Output:**
```
Top 5 Predictions (Real ImageNet Labels):
  1. lesser panda                                        99.60%
  2. polecat                                              0.20%
  3. weasel                                               0.09%

Performance: 74.41ms (CPU) / 77.14ms (GPU) / 51.93ms (CoreML)
```

### Text Generation with Transformer Attention

```bash
# Run generation with attention
make text-gen-demo

# Train on custom text
make text-gen-train

# Generate with trained weights
make text-gen-trained
```

See [examples/](examples/) for more samples.

## Documentation

- **[Getting Started](docs/user-guide/getting-started.md)** - Installation and first steps
- **[API Reference](docs/user-guide/api-reference.md)** - Complete Python API documentation
- **[Examples](docs/user-guide/examples.md)** - Code examples and tutorials
- **[Architecture](docs/architecture/overview.md)** - Design principles and structure
- **[Development Guide](docs/development/setup.md)** - Building and contributing

## Implementation Status

- 85 of ~95 WebNN operations (89% spec coverage)
- Shape inference: 85/85 (100%)
- Python API: 85/85 (100%)
- ONNX Backend: 85/85 (100%)
- CoreML MLProgram: 85/85 (100%)
- 1350+ WPT conformance tests passing

See [docs/development/implementation-status.md](docs/development/implementation-status.md) for complete details.

## Rust CLI Usage

```bash
# Validate a graph
cargo run -- examples/sample_graph.json

# Visualize a graph (requires graphviz)
cargo run -- examples/sample_graph.json --export-dot graph.dot
dot -Tpng graph.dot -o graph.png

# Convert to ONNX
cargo run -- examples/sample_graph.json --convert onnx --convert-output model.onnx

# Execute with ONNX Runtime
cargo run --features onnx-runtime -- examples/sample_graph.json --convert onnx --run-onnx
```

See `make help` for all available targets.

## Contributing

Contributions welcome! Please see:

- [AGENTS.md](AGENTS.md) - Project architecture and conventions
- [docs/development/contributing.md](docs/development/contributing.md) - How to add features
- [TODO.txt](TODO.txt) - Feature requests and known issues

**Quick Contribution Guide:**

1. Fork and create feature branch: `git checkout -b feature/my-feature`
2. Install hooks (optional): `./scripts/install-git-hooks.sh`
3. Make changes and test: `make test && make python-test`
4. Format code: `make fmt`
5. Commit and push

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Links

- **GitHub**: [https://github.com/tarekziade/rustnn](https://github.com/tarekziade/rustnn)
- **PyPI**: [https://pypi.org/project/pywebnn/](https://pypi.org/project/pywebnn/)
- **Documentation**: [https://tarekziade.github.io/rustnn/](https://tarekziade.github.io/rustnn/)
- **W3C WebNN Spec**: [https://www.w3.org/TR/webnn/](https://www.w3.org/TR/webnn/)

## Acknowledgments

- W3C WebNN Community Group for the specification
- Chromium WebNN implementation for reference
- PyO3 and Maturin projects for excellent Python-Rust integration

---

**Made with Rust by [Tarek Ziade](https://github.com/tarekziade)**
