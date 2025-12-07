# API Reference

Complete reference for the WebNN Python API.

## Module: `webnn`

The main module exports all public classes and types.

```python
import webnn
```

---

## Class: `ML`

Entry point for the WebNN API. Provides methods to create execution contexts.

### Constructor

```python
ml = webnn.ML()
```

Creates a new ML namespace instance.

### Methods

#### `create_context(accelerated=True, power_preference="default")`

Creates a new execution context following the [W3C WebNN Device Selection spec](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md).

**Parameters:**

- `accelerated` (bool): Request GPU/NPU acceleration. Default: `True`
  - `True`: Platform selects GPU or NPU if available
  - `False`: CPU-only execution
- `power_preference` (str): Power/performance hint. Options: `"default"`, `"high-performance"`, `"low-power"`. Default: `"default"`
  - `"low-power"`: Prefers NPU over GPU (Neural Engine on Apple Silicon)
  - `"high-performance"`: Prefers GPU over NPU
  - `"default"`: Platform decides (typically GPU > NPU > CPU)

**Returns:** `MLContext`

**Example:**

```python
ml = webnn.ML()

# Request acceleration (default)
context = ml.create_context(accelerated=True, power_preference="default")
print(f"Accelerated: {context.accelerated}")  # Check actual capability

# CPU-only execution
context = ml.create_context(accelerated=False)
```

**Note:** Per the WebNN Device Selection Explainer, `accelerated` is a hint. The platform autonomously selects the actual device based on availability and runtime conditions.

---

## Class: `MLContext`

Represents an execution context for neural network operations.

### Properties

#### `accelerated` (bool, read-only)

Indicates if GPU/NPU acceleration is available for this context.

- `True`: Platform can provide GPU or NPU resources
- `False`: Only CPU execution available

This represents platform capability, not a guarantee of specific device allocation.

#### `power_preference` (str, read-only)

The power preference hint for this context.

### Methods

#### `create_graph_builder()`

Creates a new graph builder for constructing computational graphs.

**Returns:** `MLGraphBuilder`

**Example:**

```python
builder = context.create_graph_builder()
```

#### `compute(graph, inputs, outputs=None)`

Executes the graph with given inputs (placeholder implementation).

**Parameters:**

- `graph` (MLGraph): The compiled graph to execute
- `inputs` (dict): Dictionary mapping input names to NumPy arrays
- `outputs` (dict, optional): Pre-allocated output arrays

**Returns:** dict - Dictionary mapping output names to result NumPy arrays

**Example:**

```python
results = context.compute(graph, {
    "input": np.array([[1, 2, 3]], dtype=np.float32)
})
```

#### `convert_to_onnx(graph, output_path)`

Converts the graph to ONNX format and saves it to a file.

**Parameters:**

- `graph` (MLGraph): The graph to convert
- `output_path` (str): Path where the ONNX model will be saved

**Example:**

```python
context.convert_to_onnx(graph, "model.onnx")
```

#### `convert_to_coreml(graph, output_path)`

Converts the graph to CoreML format (macOS only).

**Parameters:**

- `graph` (MLGraph): The graph to convert
- `output_path` (str): Path where the CoreML model will be saved

**Note:** Only available on macOS. Supports limited operations (add, matmul).

**Example:**

```python
context.convert_to_coreml(graph, "model.mlmodel")
```

#### `create_tensor(shape, data_type, readable=True, writable=True, exportable_to_gpu=False)`

Creates an MLTensor for explicit tensor management.

Following the [W3C WebNN MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md).

**Parameters:**

- `shape` (list[int]): Shape of the tensor
- `data_type` (str): Data type (e.g., "float32")
- `readable` (bool): If True, tensor data can be read back to CPU. Default: `True`
- `writable` (bool): If True, tensor data can be written from CPU. Default: `True`
- `exportable_to_gpu` (bool): If True, tensor can be exported for use as GPU texture. Default: `False`

**Returns:** `MLTensor`

**Example:**

```python
# Create default tensor (readable and writable)
tensor = context.create_tensor([2, 3], "float32")

# Create read-only tensor
ro_tensor = context.create_tensor([2, 3], "float32", readable=True, writable=False)

# Create write-only tensor
wo_tensor = context.create_tensor([2, 3], "float32", readable=False, writable=True)

# Create GPU-exportable tensor
gpu_tensor = context.create_tensor([2, 3], "float32", exportable_to_gpu=True)
```

#### `read_tensor(tensor)`

Reads data from an MLTensor into a numpy array.

**Parameters:**

- `tensor` (MLTensor): The tensor to read from (must have `readable=True`)

**Returns:** `numpy.ndarray`

**Raises:**

- `RuntimeError`: If tensor is not readable or has been destroyed

**Example:**

```python
tensor = context.create_tensor([2, 3], "float32")
result = context.read_tensor(tensor)
```

#### `write_tensor(tensor, data)`

Writes data from a numpy array into an MLTensor.

**Parameters:**

- `tensor` (MLTensor): The tensor to write to (must have `writable=True`)
- `data` (numpy.ndarray): Data to write

**Raises:**

- `RuntimeError`: If tensor is not writable or has been destroyed
- `ValueError`: If data shape doesn't match tensor shape

**Example:**

```python
tensor = context.create_tensor([2, 3], "float32")
data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
context.write_tensor(tensor, data)
```

#### `dispatch(graph, inputs, outputs)`

Dispatches graph execution asynchronously with MLTensor inputs/outputs.

Following the [W3C WebNN MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md) timeline model.

**Parameters:**

- `graph` (MLGraph): The compiled graph to execute
- `inputs` (dict): Dictionary mapping input names to MLTensor objects
- `outputs` (dict): Dictionary mapping output names to MLTensor objects

**Returns:** None (results are written to output tensors)

**Example:**

```python
# Create tensors
input_tensor = context.create_tensor([2, 3], "float32")
output_tensor = context.create_tensor([2, 3], "float32")

# Write input data
input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
context.write_tensor(input_tensor, input_data)

# Dispatch execution
context.dispatch(graph, {"x": input_tensor}, {"output": output_tensor})

# Read results
result = context.read_tensor(output_tensor)
```

---

## Class: `MLTensor`

Represents an opaque typed tensor for explicit resource management.

Following the [W3C WebNN MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md).

### Properties

#### `shape` (list[int], read-only)

The shape of the tensor.

#### `data_type` (str, read-only)

The data type of the tensor.

#### `size` (int, read-only)

The total number of elements in the tensor.

#### `readable` (bool, read-only)

Whether tensor data can be read back to CPU.

#### `writable` (bool, read-only)

Whether tensor data can be written from CPU.

#### `exportable_to_gpu` (bool, read-only)

Whether tensor can be exported for use as GPU texture.

### Methods

#### `destroy()`

Explicitly destroys the tensor and releases its resources.

After calling `destroy()`, the tensor cannot be used for any operations.

**Raises:**

- `RuntimeError`: If tensor is already destroyed

**Example:**

```python
tensor = context.create_tensor([2, 3], "float32")
# ... use tensor ...
tensor.destroy()  # Explicit cleanup
```

---

## Class: `MLGraphBuilder`

Builder for constructing computational graphs using a declarative API.

### Input/Constant Operations

#### `input(name, shape, data_type="float32")`

Creates an input operand.

**Parameters:**

- `name` (str): Name of the input
- `shape` (list[int]): Shape of the tensor
- `data_type` (str): Data type. Options: `"float32"`, `"float16"`, `"int32"`, `"uint32"`, `"int8"`, `"uint8"`

**Returns:** `MLOperand`

**Example:**

```python
x = builder.input("x", [1, 3, 224, 224], "float32")
```

#### `constant(value, shape=None, data_type=None)`

Creates a constant operand from a NumPy array or Python list.

**Parameters:**

- `value` (array-like): NumPy array or Python list
- `shape` (list[int], optional): Shape override
- `data_type` (str, optional): Data type override

**Returns:** `MLOperand`

**Example:**

```python
import numpy as np

weights = builder.constant(np.random.randn(784, 10).astype('float32'))
bias = builder.constant(np.zeros(10, dtype='float32'))
```

### Binary Operations

All binary operations take two operands and return a new operand.

#### `add(a, b)`

Element-wise addition: `a + b`

#### `sub(a, b)`

Element-wise subtraction: `a - b`

#### `mul(a, b)`

Element-wise multiplication: `a * b`

#### `div(a, b)`

Element-wise division: `a / b`

#### `matmul(a, b)`

Matrix multiplication: `a @ b`

**Example:**

```python
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")

sum_result = builder.add(x, y)
product = builder.mul(x, y)
```

### Unary Operations

All unary operations take one operand and return a new operand.

#### `relu(x)`

Rectified Linear Unit activation: `max(0, x)`

#### `sigmoid(x)`

Sigmoid activation: `1 / (1 + exp(-x))`

#### `tanh(x)`

Hyperbolic tangent activation

#### `softmax(x)`

Softmax activation (normalizes to probability distribution)

**Example:**

```python
x = builder.input("x", [1, 10], "float32")

relu_out = builder.relu(x)
sigmoid_out = builder.sigmoid(x)
tanh_out = builder.tanh(x)
softmax_out = builder.softmax(x)
```

### Shape Operations

#### `reshape(x, new_shape)`

Reshapes a tensor to a new shape.

**Parameters:**

- `x` (MLOperand): Input operand
- `new_shape` (list[int]): New shape

**Returns:** `MLOperand`

**Example:**

```python
x = builder.input("x", [1, 784], "float32")
reshaped = builder.reshape(x, [1, 28, 28, 1])
```

### Graph Building

#### `build(outputs)`

Compiles the graph and returns an immutable MLGraph.

**Parameters:**

- `outputs` (dict): Dictionary mapping output names to MLOperand objects

**Returns:** `MLGraph`

**Example:**

```python
x = builder.input("x", [2, 3], "float32")
y = builder.relu(x)

graph = builder.build({"output": y})
```

---

## Class: `MLOperand`

Represents a tensor operand in the computational graph.

### Properties

#### `data_type` (str, read-only)

The data type of the operand.

#### `shape` (list[int], read-only)

The shape of the operand.

#### `name` (str | None, read-only)

The name of the operand (if any).

**Example:**

```python
x = builder.input("x", [2, 3], "float32")

print(x.data_type)  # "float32"
print(x.shape)      # [2, 3]
print(x.name)       # "x"
```

---

## Class: `MLGraph`

Represents a compiled, immutable computational graph.

### Properties

#### `operand_count` (int, read-only)

The number of operands in the graph.

#### `operation_count` (int, read-only)

The number of operations in the graph.

### Methods

#### `get_input_names()`

Returns the names of all input operands.

**Returns:** list[str]

#### `get_output_names()`

Returns the names of all output operands.

**Returns:** list[str]

**Example:**

```python
graph = builder.build({"output": y})

print(f"Operands: {graph.operand_count}")
print(f"Operations: {graph.operation_count}")
print(f"Inputs: {graph.get_input_names()}")
print(f"Outputs: {graph.get_output_names()}")
```

---

## Data Types

Supported data types:

| Type | Description | Bytes per element |
|------|-------------|-------------------|
| `"float32"` | 32-bit floating point | 4 |
| `"float16"` | 16-bit floating point | 2 |
| `"int32"` | 32-bit signed integer | 4 |
| `"uint32"` | 32-bit unsigned integer | 4 |
| `"int8"` | 8-bit signed integer | 1 |
| `"uint8"` | 8-bit unsigned integer | 1 |

---

## Error Handling

All operations can raise Python exceptions:

```python
try:
    graph = builder.build({"output": invalid_operand})
except ValueError as e:
    print(f"Graph validation failed: {e}")

try:
    context.convert_to_onnx(graph, "/invalid/path.onnx")
except IOError as e:
    print(f"Failed to write file: {e}")

try:
    context.convert_to_coreml(graph, "model.mlmodel")
except RuntimeError as e:
    print(f"Conversion failed: {e}")
```

Common exceptions:
- `ValueError`: Invalid graph structure or parameters
- `IOError`: File I/O errors
- `RuntimeError`: Conversion or execution failures
