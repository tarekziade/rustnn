# Getting Started

This guide will help you get started with the WebNN Python API.

## Installation

### Prerequisites

- Python 3.8 or later
- Rust toolchain (for building from source)
- NumPy (automatically installed as a dependency)

### Building from Source

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/rustnn.git
   cd rustnn
   ```

3. **Install maturin**:
   ```bash
   pip install maturin
   ```

4. **Build and install**:
   ```bash
   # Development mode (editable install)
   maturin develop --features python

   # Or build a release wheel
   maturin build --release --features python
   pip install target/wheels/webnn-*.whl
   ```

### Optional Features

Build with additional runtime support:

```bash
# With ONNX runtime support
maturin develop --features python,onnx-runtime

# With CoreML runtime support (macOS only)
maturin develop --features python,coreml-runtime

# With all features
maturin develop --features python,onnx-runtime,coreml-runtime
```

## Your First Graph

Let's build a simple computational graph that adds two tensors and applies ReLU activation.

### Step 1: Import and Setup

```python
import webnn
import numpy as np

# Create the ML namespace and context
ml = webnn.ML()
context = ml.create_context(accelerated=False, power_preference="default")
```

The `MLContext` represents the execution environment. Following the [W3C WebNN Device Selection spec](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md), you provide hints:
- `accelerated`: `True` to request GPU/NPU, `False` for CPU-only
- `power_preference`: "default", "high-performance", or "low-power"

The platform autonomously selects the actual device based on availability.

### Step 2: Create a Graph Builder

```python
# Create a graph builder
builder = context.create_graph_builder()
```

The graph builder is used to construct computational graphs using a declarative API.

### Step 3: Define Inputs

```python
# Define two input operands
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
```

Each input has:
- A **name** for identification
- A **shape** (list of dimensions)
- A **data type** ("float32", "float16", "int32", etc.)

### Step 4: Build Operations

```python
# Add the inputs
sum_result = builder.add(x, y)

# Apply ReLU activation
output = builder.relu(sum_result)
```

Operations are chained to build the computational graph.

### Step 5: Compile the Graph

```python
# Compile the graph with named outputs
graph = builder.build({"output": output})

# Inspect the compiled graph
print(f"Graph has {graph.operand_count} operands")
print(f"Graph has {graph.operation_count} operations")
print(f"Inputs: {graph.get_input_names()}")
print(f"Outputs: {graph.get_output_names()}")
```

The `build()` method:
- Validates the graph structure
- Returns a compiled `MLGraph` object
- Takes a dictionary mapping output names to operands

### Step 6: Convert to Other Formats

```python
# Convert to ONNX
context.convert_to_onnx(graph, "my_model.onnx")
print("✓ ONNX model saved")

# Convert to CoreML (macOS only, basic operations)
try:
    context.convert_to_coreml(graph, "my_model.mlmodel")
    print("✓ CoreML model saved")
except Exception as e:
    print(f"CoreML conversion: {e}")
```

## Complete Example

Here's the complete code:

```python
import webnn
import numpy as np

def main():
    # Setup
    ml = webnn.ML()
    context = ml.create_context(accelerated=False)
    builder = context.create_graph_builder()

    # Build graph
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    sum_result = builder.add(x, y)
    output = builder.relu(sum_result)

    # Compile
    graph = builder.build({"output": output})

    # Export
    context.convert_to_onnx(graph, "model.onnx")

    print(f"✓ Graph compiled: {graph.operand_count} operands, "
          f"{graph.operation_count} operations")
    print(f"✓ Model exported to model.onnx")

if __name__ == "__main__":
    main()
```

## Next Steps

- Learn about all available operations in the [API Reference](api-reference.md)
- Explore more complex examples in [Examples](examples.md)
- Read about advanced topics in [Advanced Topics](advanced.md)

## Common Issues

### Import Error

If you get `ModuleNotFoundError: No module named 'webnn'`:
- Make sure you ran `maturin develop` successfully
- Verify you're using the correct Python environment

### Build Errors

If maturin build fails:
- Ensure Rust is installed: `rustc --version`
- Update maturin: `pip install -U maturin`
- Check that you have the required features: `cargo check --features python`

### NumPy Compatibility

The library requires NumPy >= 1.20.0. Update if needed:
```bash
pip install -U numpy
```
