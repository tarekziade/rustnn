# rustnn / PyWebNN

A Rust implementation of WebNN graph handling with Python bindings that implement the W3C WebNN API specification.

**Features:**
- 🦀 **Rust Library**: Validates WebNN graphs and converts to ONNX/CoreML formats
- 🐍 **Python API**: Complete W3C WebNN API implementation via PyO3 bindings
- 📊 **Format Conversion**: Export graphs to ONNX (cross-platform) and CoreML (macOS)
- 🚀 **Model Execution**: Run converted models on CPU, GPU, and Neural Engine (macOS)
- 🔍 **Graph Visualization**: Generate Graphviz diagrams of your neural networks
- ✅ **Validation**: Comprehensive graph validation matching Chromium's WebNN implementation

---

## 📦 Installation

### Python Package (PyWebNN)

Install from PyPI:

```bash
pip install pywebnn
```

Or install from source with maturin:

```bash
# Clone the repository
git clone https://github.com/tarekziade/rustnn.git
cd rustnn

# Install in development mode
pip install maturin
maturin develop --features python
```

**Requirements:** Python 3.8+, NumPy 1.20+

### Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
rustnn = "0.1"
```

Or use directly from this repository.

---

## 🚀 Quick Start

### Python API

```python
import webnn
import numpy as np

# Create ML context
ml = webnn.ML()
context = ml.create_context(device_type="cpu")
builder = context.create_graph_builder()

# Build a simple neural network: output = relu(matmul(input, weights) + bias)
input_tensor = builder.input("input", [1, 4], "float32")

# Define weights and bias as constants
weights = np.array([[0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                    [1.0, 1.1, 1.2]], dtype=np.float32)
bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)

weights_const = builder.constant(weights)
bias_const = builder.constant(bias)

# Build computation graph
matmul_result = builder.matmul(input_tensor, weights_const)
add_result = builder.add(matmul_result, bias_const)
output = builder.relu(add_result)

# Compile the graph
graph = builder.build({"output": output})

# Convert to ONNX
context.convert_to_onnx(graph, "model.onnx")
print(f"✓ ONNX model saved: model.onnx")

# Convert to CoreML (macOS only)
context.convert_to_coreml(graph, "model.mlmodel")
print(f"✓ CoreML model saved: model.mlmodel")
```

### Rust Library

```rust
use rustnn::{GraphInfo, GraphValidator, ContextProperties};
use rustnn::converters::{ConverterRegistry, OnnxConverter};

// Load graph from JSON
let graph_info: GraphInfo = serde_json::from_str(&json_data)?;

// Validate the graph
let validator = GraphValidator::new(&graph_info, ContextProperties::default());
let artifacts = validator.validate()?;

// Convert to ONNX
let mut registry = ConverterRegistry::new();
registry.register(Box::new(OnnxConverter));
let converted = registry.convert("onnx", &graph_info)?;

// Save to file
std::fs::write("model.onnx", &converted.data)?;
```

---

## 📚 Documentation

- **Python API**: Full documentation at [https://tarekziade.github.io/rustnn/](https://tarekziade.github.io/rustnn/)
- **Getting Started Guide**: [docs/getting-started.md](docs/getting-started.md)
- **API Reference**: [docs/api-reference.md](docs/api-reference.md)
- **Examples**: [docs/examples.md](docs/examples.md)
- **Advanced Topics**: [docs/advanced.md](docs/advanced.md)
- **Python-Specific**: [README_PYTHON.md](README_PYTHON.md)
- **Project Guide**: [CLAUDE.md](CLAUDE.md)

---

## 🎯 Python API Overview

The Python API implements the [W3C WebNN specification](https://www.w3.org/TR/webnn/):

### Core Classes

- **`webnn.ML`** - Entry point for creating ML contexts
- **`webnn.MLContext`** - Manages graph builders and model conversion
- **`webnn.MLGraphBuilder`** - Builds computational graphs with operations
- **`webnn.MLGraph`** - Compiled graph ready for execution or conversion
- **`webnn.MLOperand`** - Represents tensors in the computation graph

### Supported Operations

**Binary Operations:** add, sub, mul, div, matmul

**Activations:** relu, sigmoid, tanh, softmax

**Shape Operations:** reshape

**Constants:** Define weights and biases from NumPy arrays

### Format Support

| Format | Status | Platform | Features |
|--------|--------|----------|----------|
| **ONNX** | ✅ Full | All | All operations, cross-platform |
| **CoreML** | ⚠️ Basic | macOS | Basic ops (add, matmul) - activations coming soon |

See [TODO.txt](TODO.txt) for planned features and improvements.

---

## 🦀 Rust CLI Usage

The Rust library includes a powerful CLI tool for working with WebNN graphs.

### Validate a Graph

```bash
cargo run -- examples/sample_graph.json
```

### Visualize a Graph

```bash
# Generate DOT file
cargo run -- examples/sample_graph.json --export-dot graph.dot

# Convert to PNG (requires graphviz)
dot -Tpng graph.dot -o graph.png

# Or use the Makefile shortcut (macOS)
make viz
```

### Convert to ONNX

```bash
cargo run -- examples/sample_graph.json \
    --convert onnx \
    --convert-output model.onnx
```

### Convert to CoreML

```bash
cargo run -- examples/sample_graph.json \
    --convert coreml \
    --convert-output model.mlmodel
```

### Execute Models

**ONNX Runtime** (cross-platform):

```bash
cargo run --features onnx-runtime -- \
    examples/sample_graph.json \
    --convert onnx \
    --run-onnx
```

**CoreML Runtime** (macOS only):

```bash
cargo run --features coreml-runtime -- \
    examples/sample_graph.json \
    --convert coreml \
    --run-coreml \
    --device gpu  # or 'cpu', 'ane' for Neural Engine
```

### Makefile Targets

```bash
make help              # Show all available targets
make build             # Build Rust project
make test              # Run Rust tests
make python-dev        # Install Python package in dev mode
make python-test       # Run Python tests
make docs-serve        # Serve documentation locally
make validate-all-env  # Run full test pipeline
```

---

## 🏗️ Project Structure

```
rustnn/
├── src/
│   ├── lib.rs              # Public Rust API
│   ├── main.rs             # CLI tool
│   ├── graph.rs            # WebNN graph data structures
│   ├── validator.rs        # Graph validation logic
│   ├── converters/         # ONNX and CoreML converters
│   ├── executors/          # Runtime execution
│   └── python/             # Python bindings (PyO3)
│       ├── context.rs      # ML and MLContext classes
│       ├── graph_builder.rs # MLGraphBuilder class
│       ├── graph.rs        # MLGraph class
│       └── operand.rs      # MLOperand class
│
├── python/webnn/           # Python package
│   ├── __init__.py         # Public API exports
│   └── __init__.pyi        # Type stubs for IDEs
│
├── docs/                   # Documentation (MkDocs)
│   ├── index.md
│   ├── getting-started.md
│   ├── api-reference.md
│   ├── examples.md
│   └── advanced.md
│
├── tests/                  # Python tests
│   ├── test_python_api.py
│   ├── test_integration.py
│   └── test_coreml_basic.py
│
├── examples/               # Example code and graphs
│   ├── sample_graph.json
│   ├── python_simple.py
│   └── python_matmul.py
│
├── pyproject.toml          # Python package configuration
├── Cargo.toml              # Rust package configuration
└── Makefile                # Build automation
```

---

## 🔧 Development

### Prerequisites

- **Rust**: 1.70+ (install from [rustup.rs](https://rustup.rs/))
- **Python**: 3.8+ with pip
- **Maturin**: `pip install maturin`
- **Optional**: Graphviz for visualization (`brew install graphviz` on macOS)

### Building from Source

```bash
# Clone repository
git clone https://github.com/tarekziade/rustnn.git
cd rustnn

# Build Rust library
cargo build --release

# Build Python package
maturin develop --features python

# Run tests
cargo test                    # Rust tests
python -m pytest tests/       # Python tests

# Build documentation
mkdocs serve                  # Live preview at http://127.0.0.1:8000
mkdocs build                  # Build static site
```

### Running Examples

**Python:**

```bash
# Install package first
maturin develop --features python

# Run examples
python examples/python_simple.py
python examples/python_matmul.py

# Run integration tests
python tests/test_integration.py
python tests/test_coreml_basic.py --cleanup
```

**Rust:**

```bash
cargo run -- examples/sample_graph.json --export-dot graph.dot
```

---

## 🧪 Testing

### Python Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_python_api.py -v

# Run integration tests with cleanup
python tests/test_integration.py --cleanup
```

### Rust Tests

```bash
# All tests
cargo test

# Specific module
cargo test converters

# With features
cargo test --features onnx-runtime,coreml-runtime
```

---

## 📋 Roadmap

See [TODO.txt](TODO.txt) for a comprehensive list of planned features.

**High Priority:**
- ✅ Python WebNN API implementation
- ✅ ONNX conversion with full operation support
- ✅ Comprehensive documentation
- ⬜ CoreML support for activation functions (relu, sigmoid, tanh, softmax)
- ⬜ Actual tensor execution in `MLContext.compute()`
- ⬜ PyPI package publishing automation

**Medium Priority:**
- ⬜ More operations (conv2d, pooling, normalization)
- ⬜ Graph optimization passes
- ⬜ Multi-platform wheel building (manylinux, Windows)
- ⬜ Performance benchmarks

---

## 🤝 Contributing

Contributions are welcome! Please see:

- [CLAUDE.md](CLAUDE.md) - Project architecture and conventions
- [docs/README.md](docs/README.md) - Documentation guide
- [TODO.txt](TODO.txt) - Feature requests and known limitations

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `cargo test && pytest tests/`
5. Format code: `cargo fmt`
6. Commit: `git commit -m "Add my feature"`
7. Push and create a pull request

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **GitHub**: [https://github.com/tarekziade/rustnn](https://github.com/tarekziade/rustnn)
- **PyPI**: [https://pypi.org/project/pywebnn/](https://pypi.org/project/pywebnn/)
- **Documentation**: [https://tarekziade.github.io/rustnn/](https://tarekziade.github.io/rustnn/)
- **W3C WebNN Spec**: [https://www.w3.org/TR/webnn/](https://www.w3.org/TR/webnn/)
- **Issues**: [https://github.com/tarekziade/rustnn/issues](https://github.com/tarekziade/rustnn/issues)

---

## 🙏 Acknowledgments

- W3C WebNN Community Group for the specification
- Chromium WebNN implementation for reference
- PyO3 project for excellent Python-Rust bindings
- Maturin for seamless Python package building

---

**Made with ❤️ by [Tarek Ziade](https://github.com/tarekziade)**
