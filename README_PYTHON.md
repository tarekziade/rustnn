# WebNN Python API

Python bindings for the WebNN (Web Neural Network) API, providing a high-level interface for building, validating, and executing neural network graphs.

## Installation

### From Source

You'll need:
- Python 3.8 or later
- Rust toolchain (install from https://rustup.rs/)
- maturin: `pip install maturin`

Build and install:

```bash
# Development installation (editable)
maturin develop --features python

# Or build a wheel
maturin build --release --features python
pip install target/wheels/webnn-*.whl
```

### Optional Features

Build with optional runtime features:

```bash
# With ONNX runtime support
maturin develop --features python,onnx-runtime

# With CoreML runtime support (macOS only)
maturin develop --features python,coreml-runtime

# With all features
maturin develop --features python,onnx-runtime,coreml-runtime
```

## Quick Start

```python
import numpy as np
import webnn

# Create ML context
ml = webnn.ML()
context = ml.create_context(device_type="cpu")

# Create graph builder
builder = context.create_graph_builder()

# Define a simple graph: z = relu(x + y)
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
z = builder.add(x, y)
output = builder.relu(z)

# Compile the graph
graph = builder.build({"output": output})

# Prepare input data
x_data = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
y_data = np.array([[-1, 2, -3], [-4, 5, -6]], dtype=np.float32)

# Execute the graph with actual computation
results = context.compute(graph, {"x": x_data, "y": y_data})
print(results["output"])  # Contains actual computed values

# Convert to ONNX for deployment
context.convert_to_onnx(graph, "model.onnx")
```

## API Overview

### Main Classes

#### `webnn.ML`
Entry point for the WebNN API.

```python
ml = webnn.ML()
context = ml.create_context(device_type="cpu", power_preference="default")
```

#### `webnn.MLContext`
Execution context for neural network operations.

**Properties:**
- `device_type`: The device type ("cpu", "gpu", "npu")
- `power_preference`: Power preference setting

**Methods:**
- `create_graph_builder()`: Create a new graph builder
- `compute(graph, inputs, outputs=None)`: Execute a compiled graph with actual computation
  - Converts graph to ONNX format internally
  - Executes using ONNX runtime (if available)
  - Accepts numpy arrays as inputs
  - Returns dictionary of numpy arrays as outputs
  - Falls back to zeros if ONNX runtime not available
- `convert_to_onnx(graph, output_path)`: Convert graph to ONNX format
- `convert_to_coreml(graph, output_path)`: Convert graph to CoreML format (macOS only)

#### `webnn.MLGraphBuilder`
Builder for constructing computational graphs.

**Methods:**

**Input/Constant Operations:**
- `input(name, shape, data_type="float32")`: Create an input operand
- `constant(value, shape=None, data_type=None)`: Create a constant from NumPy array

**Binary Operations:**
- `add(a, b)`: Element-wise addition
- `sub(a, b)`: Element-wise subtraction
- `mul(a, b)`: Element-wise multiplication
- `div(a, b)`: Element-wise division
- `matmul(a, b)`: Matrix multiplication

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

## Supported Data Types

- `"float32"`: 32-bit floating point
- `"float16"`: 16-bit floating point
- `"int32"`: 32-bit signed integer
- `"uint32"`: 32-bit unsigned integer
- `"int8"`: 8-bit signed integer
- `"uint8"`: 8-bit unsigned integer

## Examples

### Simple Addition with Activation

```python
import webnn
import numpy as np

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Build graph
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
sum_result = builder.add(x, y)
output = builder.relu(sum_result)

graph = builder.build({"output": output})
print(f"Graph has {graph.operation_count} operations")
```

### Matrix Multiplication with Bias

```python
import webnn
import numpy as np

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Define network: output = relu(input @ weights + bias)
input_tensor = builder.input("input", [2, 4], "float32")
weights = builder.constant(np.random.randn(4, 3).astype(np.float32))
bias = builder.constant(np.zeros(3, dtype=np.float32))

matmul_result = builder.matmul(input_tensor, weights)
add_result = builder.add(matmul_result, bias)
output = builder.relu(add_result)

graph = builder.build({"output": output})
```

### Using Constants from NumPy

```python
import webnn
import numpy as np

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Create constant from NumPy array
data = np.array([[1, 2], [3, 4]], dtype=np.float32)
const_operand = builder.constant(data)

x = builder.input("x", [2, 2], "float32")
result = builder.add(x, const_operand)

graph = builder.build({"result": result})
```

### Converting to Different Formats

```python
import webnn

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Build a simple graph
x = builder.input("x", [1, 3, 224, 224], "float32")
output = builder.relu(x)
graph = builder.build({"output": output})

# Convert to ONNX
context.convert_to_onnx(graph, "model.onnx")

# Convert to CoreML (macOS only)
# context.convert_to_coreml(graph, "model.mlmodel")
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio numpy

# Build the package
maturin develop --features python

# Run tests
pytest tests/test_python_api.py -v
```

## Current Limitations

1. **Execution**: The `compute()` method currently returns placeholder results. Full execution support requires integration with ONNX/CoreML runtimes.

2. **Shape Inference**: Broadcasting and automatic shape inference are not fully implemented. Shapes must be compatible manually.

3. **Operations**: Only a subset of WebNN operations are currently implemented. More operations can be added following the existing pattern.

4. **Async Support**: The API is currently synchronous. The WebNN spec uses async operations which could be added using Python's `asyncio`.

## Architecture

The Python API is built using:
- **PyO3**: Rust ↔ Python bindings
- **maturin**: Build system for Rust/Python packages
- **NumPy**: Tensor data interchange format

The bindings wrap the existing Rust implementation:
```
Python API (webnn)
    ↓
PyO3 Bindings (src/python/)
    ↓
Rust Core (src/graph.rs, src/validators.rs, etc.)
    ↓
Converters (ONNX, CoreML)
    ↓
Executors (ONNX Runtime, CoreML)
```

## Contributing

To add new operations:

1. Add the operation method to `PyMLGraphBuilder` in `src/python/graph_builder.rs`
2. Follow the pattern of existing operations (unary/binary helpers)
3. Update type stubs in `python/webnn/__init__.pyi`
4. Add tests in `tests/test_python_api.py`
5. Update this README with examples

## License

Apache-2.0 (same as the parent project)

## See Also

- [WebNN Specification](https://www.w3.org/TR/webnn/)
- [Main Project README](README.md)
- [CLAUDE.md](CLAUDE.md) - Project development guide
