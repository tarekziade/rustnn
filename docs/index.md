<div align="center">
  <img src="../logo/rustnn.png" alt="rustnn logo" width="200"/>
</div>

# WebNN Python API Documentation

Welcome to the WebNN Python API documentation. This library provides Python bindings for the [W3C WebNN (Web Neural Network) API](https://www.w3.org/TR/webnn/), enabling you to build, validate, and execute neural network graphs in Python.

## Overview

The WebNN Python API allows you to:

- **Build neural network graphs** using a simple, intuitive Python API
- **Validate graphs** using the same validation logic as web browsers
- **Convert graphs** to ONNX and CoreML formats
- **Execute models** on CPU, GPU, or Neural Engine (macOS)
- **Integrate seamlessly** with NumPy for tensor operations

## Key Features

✅ **W3C Standard Compliant** - Implements the official WebNN specification
✅ **Type-Safe** - Full type hints for IDE autocomplete
✅ **NumPy Integration** - Seamless conversion between NumPy arrays and graph operands
✅ **Multiple Backends** - ONNX and CoreML export support
✅ **Fast** - Built with Rust and PyO3 for maximum performance
✅ **Cross-Platform** - Works on Linux, macOS, and Windows

## Quick Example

```python
import webnn
import numpy as np

# Create ML context
ml = webnn.ML()
context = ml.create_context(device_type="cpu")
builder = context.create_graph_builder()

# Build a simple neural network
input_tensor = builder.input("input", [1, 784], "float32")
weights = builder.constant(np.random.randn(784, 10).astype('float32'))
bias = builder.constant(np.zeros(10, dtype='float32'))

# Forward pass: output = relu(input @ weights + bias)
matmul_result = builder.matmul(input_tensor, weights)
add_result = builder.add(matmul_result, bias)
output = builder.relu(add_result)

# Compile the graph
graph = builder.build({"output": output})

# Export to ONNX
context.convert_to_onnx(graph, "model.onnx")
print(f"Graph compiled: {graph.operand_count} operands, {graph.operation_count} operations")
```

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/rust-webnn-graph.git
cd rust-webnn-graph

# Install maturin
pip install maturin

# Build and install
maturin develop --features python
```

### From PyPI (Coming Soon)

```bash
pip install webnn
```

## Documentation Structure

- **[Getting Started](getting-started.md)** - Installation and first steps
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Examples](examples.md)** - Code examples and tutorials
- **[Advanced Topics](advanced.md)** - Advanced usage patterns

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-org/rust-webnn-graph/issues)
- **Specification**: [W3C WebNN Spec](https://www.w3.org/TR/webnn/)

## License

Apache-2.0 License - See [LICENSE](https://github.com/your-org/rust-webnn-graph/blob/main/LICENSE) for details.
