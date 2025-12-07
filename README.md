<div align="center">
  <img src="logo/rustnn.png" alt="rustnn logo" width="200"/>

  # rustnn / PyWebNN

  A Rust implementation of WebNN graph handling with Python bindings that implement the W3C WebNN API specification.
</div>

---

## ⚠️ **EXPERIMENTAL - DO NOT USE IN PRODUCTION**

**This project is a proof-of-concept and experimental implementation. It is NOT ready for production use.**

This is an early-stage experiment to explore WebNN graph handling and format conversion. Many features are incomplete, untested, or may change significantly. Use at your own risk for research and experimentation only.

---

**Features:**
- 🦀 **Rust Library**: Validates WebNN graphs and converts to ONNX/CoreML formats
- 🐍 **Python API**: Complete W3C WebNN API implementation via PyO3 bindings
- 🎯 **Runtime Backend Selection**: Choose CPU, GPU, or NPU execution at context creation
- 📊 **Format Conversion**: Export graphs to ONNX (cross-platform) and CoreML (macOS)
- 🚀 **Model Execution**: Run converted models on CPU, GPU, and Neural Engine (macOS)
- ⚡ **Async Support**: Non-blocking execution with Python asyncio integration
- 🔍 **Graph Visualization**: Generate Graphviz diagrams of your neural networks
- ✅ **Validation**: Comprehensive graph validation matching Chromium's WebNN implementation
- 📐 **Shape Inference**: Automatic shape computation with NumPy-style broadcasting

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

# With optional runtime features
maturin develop --features python,onnx-runtime,coreml-runtime
```

**Requirements:** Python 3.11+, NumPy 1.20+

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

# Create ML context - use hints for device selection
ml = webnn.ML()
context = ml.create_context(accelerated=False)  # CPU-only execution
# Or: context = ml.create_context(accelerated=True)  # Request GPU/NPU if available

# Create graph builder
builder = context.create_graph_builder()

# Define a simple graph: z = relu(x + y)
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
z = builder.add(x, y)
output = builder.relu(z)

# Compile the graph (creates backend-agnostic representation)
graph = builder.build({"output": output})

# Prepare input data
x_data = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
y_data = np.array([[-1, 2, -3], [-4, 5, -6]], dtype=np.float32)

# Execute: converts to backend-specific format and runs
results = context.compute(graph, {"x": x_data, "y": y_data})
print(results["output"])  # Actual computed values from ONNX Runtime

# Optional: Export the ONNX model to file (for deployment, inspection, etc.)
context.convert_to_onnx(graph, "model.onnx")
```

### Backend Selection

Following the [W3C WebNN Device Selection spec](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md), device selection uses **hints** rather than explicit device types:

```python
# Request GPU/NPU acceleration (default)
context = ml.create_context(accelerated=True, power_preference="default")
print(f"Accelerated: {context.accelerated}")  # Check if acceleration is available

# Request low-power execution (prefers NPU over GPU)
context = ml.create_context(accelerated=True, power_preference="low-power")

# Request high-performance execution (prefers GPU)
context = ml.create_context(accelerated=True, power_preference="high-performance")

# CPU-only execution (no acceleration)
context = ml.create_context(accelerated=False)
```

**Device Selection Logic:**
- `accelerated=True` + `power_preference="low-power"` → **NPU** > GPU > CPU
- `accelerated=True` + `power_preference="high-performance"` → **GPU** > NPU > CPU
- `accelerated=True` + `power_preference="default"` → **GPU** > NPU > CPU
- `accelerated=False` → **CPU only**

**Platform-Specific Backends:**
- **NPU**: CoreML Neural Engine (Apple Silicon macOS only)
- **GPU**: ONNX Runtime GPU (cross-platform) or CoreML GPU (macOS)
- **CPU**: ONNX Runtime CPU (cross-platform)

**Important:** The `accelerated` property indicates **platform capability**, not a guarantee. Query `context.accelerated` after creation to check if GPU/NPU resources are available. The platform controls actual device allocation based on runtime conditions.

The graph compilation (`builder.build()`) creates a **backend-agnostic representation**. Backend-specific conversion happens automatically during `compute()` based on the context's selected backend.

### Async Execution

WebNN supports asynchronous execution following the W3C specification. Use `AsyncMLContext` for non-blocking operations:

```python
import asyncio
import numpy as np
import webnn

async def main():
    # Create context
    ml = webnn.ML()
    context = ml.create_context(accelerated=False)
    async_context = webnn.AsyncMLContext(context)

    # Build graph
    builder = async_context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)
    output = builder.relu(z)
    graph = builder.build({"output": output})

    # Async dispatch (non-blocking execution)
    x_data = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
    y_data = np.array([[-1, 2, -3], [-4, 5, -6]], dtype=np.float32)
    await async_context.dispatch(graph, {"x": x_data, "y": y_data})

    print("Graph executed asynchronously!")

asyncio.run(main())
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

## 📚 Python API Reference

The Python API implements the [W3C WebNN specification](https://www.w3.org/TR/webnn/).

### Core Classes

#### `webnn.ML`
Entry point for the WebNN API.

**Methods:**
- `create_context(accelerated=True, power_preference="default")`: Create an execution context
  - `accelerated`: Request GPU/NPU acceleration (default: True)
  - `power_preference`: Hint for power/performance tradeoffs ("default", "high-performance", "low-power")

```python
ml = webnn.ML()
context = ml.create_context(accelerated=False, power_preference="default")
```

#### `webnn.MLContext`
Execution context for neural network operations.

**Properties:**
- `accelerated` (readonly): Boolean indicating if GPU/NPU acceleration is available
- `power_preference` (readonly): Power preference hint ("default", "high-performance", "low-power")

**Methods:**
- `create_graph_builder()`: Create a new graph builder
- `compute(graph, inputs, outputs=None)`: Execute a compiled graph with actual computation
  - Uses the backend selected at context creation (based on accelerated + power_preference)
  - Automatically converts backend-agnostic graph to backend-specific format
  - accelerated=False → ONNX Runtime CPU backend
  - accelerated=True + low-power → CoreML NPU (macOS) or ONNX Runtime GPU
  - accelerated=True + high-performance → ONNX Runtime GPU or CoreML GPU
  - Accepts numpy arrays as inputs
  - Returns dictionary of numpy arrays as outputs
- `convert_to_onnx(graph, output_path)`: Export graph to ONNX file (for deployment/inspection)
- `convert_to_coreml(graph, output_path)`: Export graph to CoreML file (macOS only, for deployment)
- `create_tensor(shape, data_type, readable=True, writable=True, exportable_to_gpu=False)`: Create a tensor for explicit memory management ([MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md))
- `read_tensor(tensor)`: Read tensor data (synchronous)
- `write_tensor(tensor, data)`: Write tensor data (synchronous)
- `dispatch(graph, inputs, outputs)`: Dispatch graph execution with MLTensor inputs/outputs ([MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md))

#### `webnn.AsyncMLContext`
Async wrapper for MLContext providing WebNN-compliant asynchronous execution.

**Creation:**
```python
context = ml.create_context(accelerated=False)
async_context = webnn.AsyncMLContext(context)
```

**Async Methods:**
- `async dispatch(graph, inputs, outputs)`: Execute graph asynchronously with MLTensor inputs/outputs ([MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md))
  - Returns immediately, execution happens in background
  - Use for non-blocking computation
- `async read_tensor_async(tensor)`: Read tensor data asynchronously
- `async write_tensor_async(tensor, data)`: Write tensor data asynchronously

**Synchronous Methods (pass-through):**
- `create_graph_builder()`: Create a new graph builder
- `create_tensor(shape, data_type, readable=True, writable=True, exportable_to_gpu=False)`: Create a tensor
- `compute(graph, inputs, outputs=None)`: Synchronous execution
- `read_tensor(tensor)`: Synchronous tensor read
- `write_tensor(tensor, data)`: Synchronous tensor write

**Properties:**
- `accelerated`: Boolean indicating if GPU/NPU acceleration is available
- `power_preference`: Power preference setting

#### `webnn.MLGraphBuilder`
Builder for constructing computational graphs.

**Input/Constant Operations:**
- `input(name, shape, data_type="float32")`: Create an input operand
- `constant(value, shape=None, data_type=None)`: Create a constant from NumPy array

**Binary Operations:**
- `add(a, b)`: Element-wise addition (with broadcasting)
- `sub(a, b)`: Element-wise subtraction (with broadcasting)
- `mul(a, b)`: Element-wise multiplication (with broadcasting)
- `div(a, b)`: Element-wise division (with broadcasting)
- `matmul(a, b)`: Matrix multiplication (with batched matmul support)

**Shape Inference:**
Binary operations automatically compute output shapes using NumPy-style broadcasting rules:
- Dimensions are aligned from right to left
- Two dimensions are compatible if they are equal or one is 1
- Output shape is the maximum of each dimension
- Incompatible shapes raise `ValueError` at graph build time

Matrix multiplication follows proper matmul shape rules:
- For 2D: `[M, K] @ [K, N] -> [M, N]`
- For batched: batch dimensions are broadcasted
- Inner dimensions must match or a `ValueError` is raised

**Unary Operations:**
- `relu(x)`: ReLU activation
- `sigmoid(x)`: Sigmoid activation
- `tanh(x)`: Tanh activation
- `softmax(x)`: Softmax activation

**Shape Operations:**
- `reshape(x, new_shape)`: Reshape operand

**Graph Building:**
- `build(outputs)`: Compile the graph with specified outputs

#### `webnn.MLOperand`
Represents an operand (tensor) in the computational graph.

**Properties:**
- `data_type`: Data type string (e.g., "float32")
- `shape`: List of dimensions
- `name`: Optional name of the operand

#### `webnn.MLGraph`
Compiled computational graph.

**Properties:**
- `operand_count`: Number of operands in the graph
- `operation_count`: Number of operations in the graph

**Methods:**
- `get_input_names()`: Get list of input names
- `get_output_names()`: Get list of output names

#### `webnn.MLTensor`
Explicit tensor for memory management.

Following the [W3C WebNN MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md).

**Properties:**
- `data_type`: Data type string
- `shape`: Tensor dimensions
- `size`: Total number of elements
- `readable`: Whether tensor data can be read back to CPU
- `writable`: Whether tensor data can be written from CPU
- `exportable_to_gpu`: Whether tensor can be exported as GPU texture

**Methods:**
- `destroy()`: Explicitly release tensor resources

### Supported Data Types

- `"float32"`: 32-bit floating point
- `"float16"`: 16-bit floating point
- `"int32"`: 32-bit signed integer
- `"uint32"`: 32-bit unsigned integer
- `"int8"`: 8-bit signed integer
- `"uint8"`: 8-bit unsigned integer

### Supported Operations

**Binary Operations:** add, sub, mul, div, matmul

**Activations:** relu, sigmoid, tanh, softmax

**Shape Operations:** reshape

**Constants:** Define weights and biases from NumPy arrays

### Format Support

| Format | Status | Platform | Features |
|--------|--------|----------|----------|
| **ONNX** | ✅ Full | All | All operations, cross-platform |
| **CoreML** | ✅ Good | macOS | add, matmul, relu, sigmoid, tanh, softmax |

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

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│ CLI (main.rs) / Library API (lib.rs) / Python API (PyO3)    │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┬──────────────┬─────────────────┐
    ▼                     ▼              ▼                 ▼
┌────────┐     ┌──────────────┐   ┌──────────┐    ┌──────────────┐
│Loader  │────▶│  Validator   │──▶│ Context  │───▶│  Backend     │
│(JSON)  │     │(graph.rs)    │   │(selects) │    │  Selection   │
└────────┘     └──────────────┘   └────┬─────┘    └──────┬───────┘
                                        │                 │
                                        ▼                 ▼
                                  ┌──────────┐    ┌──────────────┐
                                  │ Builder  │    │  Converter   │
                                  │(backend- │    │  (Runtime)   │
                                  │agnostic) │    │              │
                                  └────┬─────┘    └──────┬───────┘
                                       │                 │
                                       ▼                 ▼
                              ┌─────────────┐   ┌────────────────┐
                              │  MLGraph    │   │ ONNX / CoreML  │
                              │(immutable)  │   │   Execution    │
                              └─────────────┘   └────────────────┘
```

### Key Principles

**1. Backend-Agnostic Graph Representation**
- `builder.build()` creates an immutable, platform-independent `GraphInfo` structure
- Contains operands, operations, inputs, outputs, and constant data
- No backend-specific artifacts at this stage

**2. Runtime Backend Selection (WebNN Spec-Compliant)**
- Backend selection happens at **context creation** via `accelerated` and `power_preference` hints
- `accelerated=False` → ONNX Runtime CPU
- `accelerated=True` + `power="high-performance"` → GPU preferred (ONNX or CoreML)
- `accelerated=True` + `power="low-power"` → NPU preferred (CoreML Neural Engine on Apple Silicon)
- Platform autonomously selects actual device based on availability and runtime conditions
- Selection logic in `PyMLContext::select_backend()`

**3. Lazy Backend Conversion**
- Backend-specific conversion happens during `compute()`, not `build()`
- `compute()` routes to appropriate backend method:
  - `compute_onnx()` for ONNX Runtime
  - `compute_coreml()` for CoreML
  - `compute_fallback()` when no backend available
- Same graph can be executed on different backends via different contexts

**4. Rust-First Architecture**
- All core functionality in pure Rust (validation, conversion, execution)
- Python bindings are thin wrappers exposing Rust functionality
- Rust library usable independently without Python
- Design principle: "Rust is the implementation, Python is the interface"

### File Organization

```
src/
├── lib.rs              # Public Rust API exports
├── main.rs             # CLI entry point
├── graph.rs            # Core data structures (backend-agnostic)
├── error.rs            # Error types
├── validator.rs        # Graph validation
├── loader.rs           # JSON loading
├── graphviz.rs         # DOT export
├── protos.rs           # Protobuf module setup
├── converters/
│   ├── mod.rs          # Registry and trait
│   ├── onnx.rs         # ONNX converter
│   └── coreml.rs       # CoreML converter
├── executors/
│   ├── mod.rs          # Conditional compilation
│   ├── onnx.rs         # ONNX runtime
│   └── coreml.rs       # CoreML runtime
└── python/             # Python bindings (PyO3)
    ├── mod.rs          # Python module definition
    ├── context.rs      # ML and MLContext classes (backend selection)
    ├── graph_builder.rs # MLGraphBuilder class
    ├── graph.rs        # MLGraph class
    ├── operand.rs      # MLOperand class
    └── tensor.rs       # MLTensor class

python/webnn/           # Python package
├── __init__.py         # Package exports (AsyncMLContext)
└── __init__.pyi        # Type stubs

tests/
└── test_python_api.py  # Python API tests (45 tests)

examples/
├── python_simple.py    # Basic Python example
└── python_matmul.py    # Matrix multiplication example
```

---

## 🔧 Development

### Prerequisites

- **Rust**: 1.70+ (install from [rustup.rs](https://rustup.rs/))
- **Python**: 3.11+ with pip
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

**Completed:**
- ✅ Python WebNN API implementation
- ✅ Runtime backend selection (WebNN spec-compliant)
- ✅ ONNX conversion with full operation support
- ✅ Actual tensor execution with ONNX Runtime
- ✅ Async execution support (AsyncMLContext)
- ✅ Shape inference and broadcasting
- ✅ Comprehensive documentation

**High Priority:**
- ⬜ PyPI package publishing automation
- ⬜ More operations (conv2d, pooling, normalization)
- ⬜ CoreML execution with actual tensor I/O

**Medium Priority:**
- ⬜ Graph optimization passes
- ⬜ Multi-platform wheel building (manylinux, Windows)
- ⬜ Performance benchmarks

---

## 🤝 Contributing

Contributions are welcome! Please see:

- [CLAUDE.md](CLAUDE.md) - Project architecture and conventions
- [TODO.txt](TODO.txt) - Feature requests and known limitations

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. **Install git hooks** (optional but recommended):
   ```bash
   ./scripts/install-git-hooks.sh
   ```
   This installs a pre-commit hook that automatically checks code formatting before each commit.
4. Make your changes
5. Run tests: `cargo test && pytest tests/`
6. Format code: `cargo fmt` (or let the pre-commit hook handle it)
7. Commit: `git commit -m "Add my feature"`
8. Push and create a pull request

**Note:** The pre-commit hook will prevent commits with unformatted code. If needed, you can bypass it with `git commit --no-verify`, but this is not recommended.

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
