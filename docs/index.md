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

[OK] **W3C Standard Compliant** - Implements the official WebNN specification
[OK] **85 Operations** - 89% coverage of WebNN spec operations
[OK] **Type-Safe** - Full type hints for IDE autocomplete
[OK] **NumPy Integration** - Seamless conversion between NumPy arrays
[OK] **Multiple Backends** - ONNX Runtime (CPU/GPU) and CoreML (macOS)
[OK] **Actual Execution** - Run models with real tensor inputs/outputs
[OK] **Async Support** - Non-blocking execution with Python asyncio
[OK] **Fast** - Built with Rust and PyO3 for maximum performance
[OK] **Cross-Platform** - Works on Linux, macOS, and Windows

## Quick Example

```python
import webnn
import numpy as np

# Create ML context with device hints
ml = webnn.ML()
context = ml.create_context(accelerated=True)  # Request GPU/NPU if available
builder = context.create_graph_builder()

# Build a simple computation: z = relu(x + y)
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
z = builder.add(x, y)
output = builder.relu(z)

# Compile the graph (backend-agnostic)
graph = builder.build({"output": output})

# Execute with actual data
x_data = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
y_data = np.array([[-1, 2, -3], [-4, 5, -6]], dtype=np.float32)
results = context.compute(graph, {"x": x_data, "y": y_data})

print(results["output"])  # [[0. 0. 0.] [0. 0. 0.]]

# Export to ONNX for deployment
context.convert_to_onnx(graph, "model.onnx")
```

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/rustnn.git
cd rustnn

# Install maturin
pip install maturin

# Build and install
maturin develop --features python
```

### From PyPI

```bash
pip install pywebnn
```

## Documentation Structure

- **[Getting Started](getting-started.md)** - Installation and first steps
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Examples](examples.md)** - Code examples and tutorials
- **[Advanced Topics](advanced.md)** - Advanced usage patterns

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-org/rustnn/issues)
- **Specification**: [W3C WebNN Spec](https://www.w3.org/TR/webnn/)

## License

Apache-2.0 License - See [LICENSE](https://github.com/your-org/rustnn/blob/main/LICENSE) for details.
