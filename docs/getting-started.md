# Getting Started

This guide will help you get started with the WebNN Python API.

## Installation

### From PyPI (Quick Start)

For validation and conversion only:

```bash
pip install pywebnn
```

For full execution support, add ONNX Runtime:

```bash
pip install pywebnn onnxruntime
```

### Building from Source (Recommended for Full Features)

#### Prerequisites

- Python 3.11 or later
- Rust toolchain
- NumPy (automatically installed)
- ONNX Runtime 1.23+ (for execution support)

#### Quick Setup with Makefile (Easiest)

The Makefile handles everything automatically:

```bash
# Clone the repository
git clone https://github.com/tarekziade/rustnn.git
cd rustnn

# Install with ONNX Runtime support (downloads ONNX Runtime automatically)
make python-dev

# Run tests to verify
make python-test
```

This creates a `.venv-webnn` virtual environment with everything configured.

#### Manual Setup with Maturin

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone https://github.com/tarekziade/rustnn.git
   cd rustnn
   pip install maturin
   ```

3. **Build with features**:
   ```bash
   # With ONNX Runtime support (requires ONNX Runtime 1.23+)
   maturin develop --features python,onnx-runtime

   # macOS: Add CoreML support
   maturin develop --features python,onnx-runtime,coreml-runtime

   # Basic (validation/conversion only, no execution)
   maturin develop --features python
   ```

**Note:** When building with `onnx-runtime` feature, you need ONNX Runtime libraries available. The Makefile handles this automatically. For manual setup, see the [development guide](development.md).

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

### Step 6: Execute the Graph

```python
import numpy as np

# Prepare input data
x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
y_data = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)

# Execute the graph with actual inputs
results = context.compute(graph, {"x": x_data, "y": y_data})

print("Input x:")
print(x_data)
print("\nInput y:")
print(y_data)
print("\nOutput (relu(x + y)):")
print(results["output"])
# [[2. 3. 4.]
#  [5. 6. 7.]]
```

### Step 7: Export to Other Formats (Optional)

```python
# Export to ONNX for deployment
context.convert_to_onnx(graph, "my_model.onnx")
print("✓ ONNX model saved")

# Export to CoreML (macOS only)
try:
    context.convert_to_coreml(graph, "my_model.mlmodel")
    print("✓ CoreML model saved")
except Exception as e:
    print(f"CoreML conversion: {e}")
```

## Complete Example

Here's the complete code with execution:

```python
import webnn
import numpy as np

def main():
    # Setup
    ml = webnn.ML()
    context = ml.create_context(accelerated=False)
    builder = context.create_graph_builder()

    # Build graph: output = relu(x + y)
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    sum_result = builder.add(x, y)
    output = builder.relu(sum_result)

    # Compile
    graph = builder.build({"output": output})

    print(f"✓ Graph compiled: {graph.operand_count} operands, "
          f"{graph.operation_count} operations")

    # Execute with real data
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y_data = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data, "y": y_data})

    print(f"✓ Computed output:\n{results['output']}")

    # Optional: Export to ONNX
    context.convert_to_onnx(graph, "model.onnx")
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
