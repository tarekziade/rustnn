# Advanced Topics

Advanced usage patterns and best practices for the WebNN Python API.

## Performance Optimization

### Graph Compilation

Compile graphs once and reuse them:

```python
import webnn

class ModelCache:
    def __init__(self):
        self.ml = webnn.ML()
        self.context = self.ml.create_context()
        self.graphs = {}

    def get_or_build_graph(self, name, builder_fn):
        """Cache compiled graphs for reuse."""
        if name not in self.graphs:
            builder = self.context.create_graph_builder()
            output = builder_fn(builder)
            self.graphs[name] = builder.build({name: output})
        return self.graphs[name]

# Usage
cache = ModelCache()

def build_relu(builder):
    x = builder.input("x", [100], "float32")
    return builder.relu(x)

# First call: compiles the graph
graph1 = cache.get_or_build_graph("relu", build_relu)

# Second call: returns cached graph (fast!)
graph2 = cache.get_or_build_graph("relu", build_relu)
assert graph1 is graph2
```

### Memory-Efficient Constants

For large constant tensors, use the most memory-efficient data type:

```python
import webnn
import numpy as np

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

# Use float16 instead of float32 to halve memory usage
large_weights = np.random.randn(1000, 1000).astype('float16')
weights_op = builder.constant(large_weights)

print(f"Memory saved: {large_weights.nbytes / 1024 / 1024:.2f} MB vs "
      f"{(large_weights.nbytes * 2) / 1024 / 1024:.2f} MB for float32")
```

## Integration with Other Libraries

### NumPy Integration with Execution

Seamless conversion between NumPy and WebNN:

```python
import webnn
import numpy as np

ml = webnn.ML()
context = ml.create_context(accelerated=False)
builder = context.create_graph_builder()

# Build a simple matmul with NumPy weights
x = builder.input("x", [1, 100], "float32")
weights = np.random.randn(100, 50).astype('float32') * 0.01
bias = np.zeros(50, dtype='float32')

w_op = builder.constant(weights)
b_op = builder.constant(bias)

output = builder.add(builder.matmul(x, w_op), b_op)
graph = builder.build({"output": output})

# Execute with NumPy input
x_data = np.random.randn(1, 100).astype('float32')
results = context.compute(graph, {"x": x_data})

print(f"Input shape: {x_data.shape}")
print(f"Output shape: {results['output'].shape}")
print(f"Result is NumPy array: {isinstance(results['output'], np.ndarray)}")
```

### ONNX Integration

Load existing ONNX models and convert them:

```python
import webnn
import numpy as np
# Note: This is a conceptual example. Full ONNX loading
# would require parsing the ONNX protobuf format.

def load_onnx_weights(onnx_path):
    """
    Conceptual example of loading ONNX weights.
    In practice, you'd use onnx.load() to parse the model.
    """
    # This is a simplified example
    weights = {
        'fc1': np.random.randn(784, 128).astype('float32'),
        'fc1_bias': np.zeros(128, dtype='float32'),
        'fc2': np.random.randn(128, 10).astype('float32'),
        'fc2_bias': np.zeros(10, dtype='float32'),
    }
    return weights

def build_from_onnx_weights(weights):
    ml = webnn.ML()
    context = ml.create_context()
    builder = context.create_graph_builder()

    # Build graph using ONNX weights
    x = builder.input("input", [1, 784], "float32")

    w1 = builder.constant(weights['fc1'])
    b1 = builder.constant(weights['fc1_bias'])
    h1 = builder.matmul(x, w1)
    h1 = builder.add(h1, b1)
    h1 = builder.relu(h1)

    w2 = builder.constant(weights['fc2'])
    b2 = builder.constant(weights['fc2_bias'])
    output = builder.matmul(h1, w2)
    output = builder.add(output, b2)

    return builder.build({"output": output})

weights = load_onnx_weights("model.onnx")
graph = build_from_onnx_weights(weights)
```

## Graph Introspection and Execution

Inspect and analyze compiled graphs, then execute them:

```python
import webnn
import numpy as np

ml = webnn.ML()
context = ml.create_context(accelerated=False)
builder = context.create_graph_builder()

# Build a complex graph
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [3, 4], "float32")
z = builder.matmul(x, y)
w = builder.relu(z)
output = builder.sigmoid(w)

graph = builder.build({"final": output})

# Inspect the graph
print("Graph Analysis:")
print(f"  Inputs: {graph.get_input_names()}")
print(f"  Outputs: {graph.get_output_names()}")
print(f"  Total operands: {graph.operand_count}")
print(f"  Total operations: {graph.operation_count}")

# Execute the graph
x_data = np.random.randn(2, 3).astype('float32')
y_data = np.random.randn(3, 4).astype('float32')
results = context.compute(graph, {"x": x_data, "y": y_data})

print(f"\nExecution:")
print(f"  Output shape: {results['final'].shape}")
print(f"  Output range: [{results['final'].min():.4f}, {results['final'].max():.4f}]")
```

## Custom Graph Patterns

### Residual Connections

```python
import webnn
import numpy as np

def residual_block(builder, x, hidden_size):
    """Create a residual block: output = relu(x + fc(x))"""

    # Linear transformation
    w = builder.constant(np.random.randn(hidden_size, hidden_size).astype('float32') * 0.01)
    transformed = builder.matmul(x, w)

    # Add residual connection
    residual = builder.add(x, transformed)

    # Activation
    output = builder.relu(residual)

    return output

ml = webnn.ML()
context = ml.create_context()
builder = context.create_graph_builder()

x = builder.input("x", [1, 128], "float32")
y = residual_block(builder, x, 128)
graph = builder.build({"output": y})

context.convert_to_onnx(graph, "residual.onnx")
```

### Attention Mechanism (Simplified)

```python
import webnn
import numpy as np

def scaled_dot_product_attention(builder, query, key, value, d_k):
    """
    Simplified attention mechanism (without softmax for now).
    attention = (query @ key.T) @ value
    """
    # Transpose key (conceptually)
    key_t = key  # In practice, you'd need to handle transposition

    # Attention scores: query @ key.T
    scores = builder.matmul(query, key_t)

    # Apply scaling factor (as a constant multiply)
    scale = 1.0 / np.sqrt(d_k)
    scale_tensor = builder.constant(np.full_like(scores, scale))
    scaled_scores = builder.mul(scores, scale_tensor)

    # Attention output: scores @ value
    output = builder.matmul(scaled_scores, value)

    return output
```

## Error Handling Strategies

### Comprehensive Error Handling

```python
import webnn
import sys
import traceback

def safe_graph_export(graph_fn, output_path):
    """
    Safely build and export a graph with comprehensive error handling.
    """
    try:
        ml = webnn.ML()
        context = ml.create_context()
        builder = context.create_graph_builder()

        # Build the graph
        try:
            output = graph_fn(builder)
            graph = builder.build({"output": output})
        except ValueError as e:
            print(f" Graph validation failed: {e}", file=sys.stderr)
            traceback.print_exc()
            return False

        # Export to ONNX
        try:
            context.convert_to_onnx(graph, output_path)
            print(f"[OK] Successfully exported to {output_path}")
            return True
        except IOError as e:
            print(f" File I/O error: {e}", file=sys.stderr)
            return False
        except RuntimeError as e:
            print(f" Conversion failed: {e}", file=sys.stderr)
            return False

    except Exception as e:
        print(f" Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        return False

# Usage
def my_graph(builder):
    x = builder.input("x", [10], "float32")
    return builder.relu(x)

success = safe_graph_export(my_graph, "model.onnx")
sys.exit(0 if success else 1)
```

## Testing Graphs

### Unit Testing WebNN Graphs

```python
import unittest
import webnn
import numpy as np
import os

class TestWebNNGraphs(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.ml = webnn.ML()
        self.context = self.ml.create_context()

    def test_simple_relu(self):
        """Test ReLU graph creation and export."""
        builder = self.context.create_graph_builder()
        x = builder.input("x", [10], "float32")
        y = builder.relu(x)
        graph = builder.build({"y": y})

        self.assertEqual(graph.operand_count, 2)
        self.assertEqual(graph.operation_count, 1)
        self.assertIn("x", graph.get_input_names())
        self.assertIn("y", graph.get_output_names())

    def test_onnx_export(self):
        """Test ONNX export functionality."""
        builder = self.context.create_graph_builder()
        x = builder.input("x", [10], "float32")
        y = builder.relu(x)
        graph = builder.build({"y": y})

        output_path = "test_model.onnx"
        try:
            self.context.convert_to_onnx(graph, output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_invalid_shape(self):
        """Test that invalid shapes raise errors."""
        builder = self.context.create_graph_builder()

        # This should work
        x = builder.input("x", [10, 20], "float32")

        # Empty shape is valid (scalar)
        scalar = builder.input("scalar", [], "float32")

    def test_multiple_outputs(self):
        """Test graphs with multiple outputs."""
        builder = self.context.create_graph_builder()
        x = builder.input("x", [10], "float32")

        y1 = builder.relu(x)
        y2 = builder.sigmoid(x)

        graph = builder.build({"relu": y1, "sigmoid": y2})

        outputs = graph.get_output_names()
        self.assertIn("relu", outputs)
        self.assertIn("sigmoid", outputs)

if __name__ == '__main__':
    unittest.main()
```

## Debugging Tips

### Verbose Graph Building

```python
import webnn

class VerboseBuilder:
    """Wrapper that logs all operations."""

    def __init__(self, context):
        self.context = context
        self.builder = context.create_graph_builder()
        self.op_count = 0

    def input(self, name, shape, dtype="float32"):
        result = self.builder.input(name, shape, dtype)
        print(f"[{self.op_count}] INPUT: {name} {shape} {dtype}")
        self.op_count += 1
        return result

    def constant(self, value, **kwargs):
        result = self.builder.constant(value, **kwargs)
        print(f"[{self.op_count}] CONSTANT: shape={value.shape}")
        self.op_count += 1
        return result

    def relu(self, x):
        result = self.builder.relu(x)
        print(f"[{self.op_count}] RELU")
        self.op_count += 1
        return result

    def matmul(self, a, b):
        result = self.builder.matmul(a, b)
        print(f"[{self.op_count}] MATMUL")
        self.op_count += 1
        return result

    # Add other operations as needed...

    def build(self, outputs):
        print(f"\nBuilding graph with {len(outputs)} output(s)...")
        return self.builder.build(outputs)

# Usage
ml = webnn.ML()
context = ml.create_context()
builder = VerboseBuilder(context)

x = builder.input("x", [10], "float32")
y = builder.relu(x)
graph = builder.build({"y": y})
```

Output:
```
[0] INPUT: x [10] float32
[1] RELU

Building graph with 1 output(s)...
```

## Platform-Specific Features

### Backend Selection and Execution

Choose the best backend for your platform and execute models:

```python
import webnn
import numpy as np
import platform

ml = webnn.ML()

# Try GPU/NPU acceleration first
context = ml.create_context(accelerated=True, power_preference="high-performance")
print(f"Platform: {platform.system()}")
print(f"Accelerated: {context.accelerated}")

# Build a simple graph
builder = context.create_graph_builder()
x = builder.input("x", [10], "float32")
y = builder.relu(x)
graph = builder.build({"y": y})

# Execute on selected backend
x_data = np.array([-5, -3, -1, 0, 1, 3, 5, 7, 9, 11], dtype=np.float32)
results = context.compute(graph, {"x": x_data})

print(f"Result: {results['y']}")

# Export for different platforms
context.convert_to_onnx(graph, "model.onnx")
print("[OK] Exported ONNX (cross-platform)")

if platform.system() == "Darwin":
    try:
        context.convert_to_coreml(graph, "model.mlmodel")
        print("[OK] Exported CoreML (macOS GPU/Neural Engine)")
    except Exception as e:
        print(f" CoreML export: {e}")
```

## Best Practices Summary

1. **Compile once, reuse**: Cache compiled graphs
2. **Use appropriate data types**: float16 for memory efficiency
3. **Handle errors gracefully**: Wrap operations in try-except blocks
4. **Test thoroughly**: Write unit tests for your graphs
5. **Validate shapes**: Check tensor dimensions before building
6. **Profile performance**: Measure compilation and export times
7. **Document graphs**: Add comments explaining graph structure
8. **Use type hints**: Leverage Python type hints for better IDE support

```python
from typing import Dict
import webnn
import numpy as np

def build_classifier(
    input_size: int,
    hidden_size: int,
    num_classes: int
) -> webnn.MLGraph:
    """
    Build a simple classifier graph.

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layer
        num_classes: Number of output classes

    Returns:
        Compiled MLGraph ready for export
    """
    ml = webnn.ML()
    context = ml.create_context()
    builder = context.create_graph_builder()

    # Build model...
    x = builder.input("input", [1, input_size], "float32")
    # ... rest of the model

    return graph
```
