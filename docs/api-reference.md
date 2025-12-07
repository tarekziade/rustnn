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

### Convolution Operations

#### `conv2d(input, filter, strides=None, dilations=None, pads=None, groups=None, input_layout=None, filter_layout=None)`

2D convolution operation for neural networks.

**Parameters:**

- `input` (MLOperand): Input tensor (4D: batch, channels, height, width or batch, height, width, channels)
- `filter` (MLOperand): Filter/kernel weights (4D constant tensor)
- `strides` (list[int], optional): Stride along each spatial axis. Default: `[1, 1]`
- `dilations` (list[int], optional): Dilation along each spatial axis. Default: `[1, 1]`
- `pads` (list[int], optional): Padding `[begin_height, begin_width, end_height, end_width]`. Default: `[0, 0, 0, 0]`
- `groups` (int, optional): Number of groups for grouped/depthwise convolution. Default: `1`
- `input_layout` (str, optional): Input tensor layout, either `"nchw"` (channels-first) or `"nhwc"` (channels-last). Default: `"nchw"`
- `filter_layout` (str, optional): Filter tensor layout: `"oihw"`, `"hwio"`, `"ohwi"`, or `"ihwo"`. Default: `"oihw"`

**Returns:** MLOperand with output tensor

**Shape Inference:**

For NCHW input `[N, C_in, H_in, W_in]` and OIHW filter `[C_out, C_in/groups, K_h, K_w]`:

```
output_h = (H_in + pad_begin_h + pad_end_h - dilation_h * (K_h - 1) - 1) / stride_h + 1
output_w = (W_in + pad_begin_w + pad_end_w - dilation_w * (K_w - 1) - 1) / stride_w + 1
output_shape = [N, C_out, output_h, output_w]
```

**Example: Standard Convolution**

```python
# Input: [batch=1, channels=3, height=32, width=32] (RGB image)
input_op = builder.input("input", [1, 3, 32, 32], "float32")

# Filter: [out_channels=64, in_channels=3, height=3, width=3]
filter_weights = np.random.randn(64, 3, 3, 3).astype(np.float32)
filter_op = builder.constant(filter_weights)

# Apply conv2d with stride=2 and padding=1
output = builder.conv2d(
    input_op,
    filter_op,
    strides=[2, 2],
    pads=[1, 1, 1, 1]
)
# Output shape: [1, 64, 16, 16]
```

**Example: Depthwise Convolution**

```python
# Depthwise convolution: each input channel is convolved separately
input_op = builder.input("input", [1, 32, 28, 28], "float32")

# Filter: [out_channels=32, in_channels=1, height=3, width=3]
# groups=32 means 32 separate 1-channel convolutions
filter_weights = np.random.randn(32, 1, 3, 3).astype(np.float32)
filter_op = builder.constant(filter_weights)

output = builder.conv2d(
    input_op,
    filter_op,
    pads=[1, 1, 1, 1],
    groups=32  # Depthwise: groups = input channels
)
# Output shape: [1, 32, 28, 28]
```

**Example: Dilated Convolution**

```python
# Dilated (atrous) convolution increases receptive field
input_op = builder.input("input", [1, 3, 32, 32], "float32")
filter_weights = np.random.randn(64, 3, 3, 3).astype(np.float32)
filter_op = builder.constant(filter_weights)

output = builder.conv2d(
    input_op,
    filter_op,
    dilations=[2, 2],  # Dilation factor of 2
    pads=[2, 2, 2, 2]  # Larger padding for dilated kernels
)
# Effective kernel size: 3 + (3-1)*2 = 5x5
```

**Example: NHWC Layout (Channels-Last)**

```python
# Input in NHWC format: [batch, height, width, channels]
input_op = builder.input("input", [1, 32, 32, 3], "float32")
filter_weights = np.random.randn(64, 3, 3, 3).astype(np.float32)
filter_op = builder.constant(filter_weights)

output = builder.conv2d(
    input_op,
    filter_op,
    input_layout="nhwc",  # Channels-last input
    pads=[1, 1, 1, 1]
)
# Output shape: [1, 32, 32, 64] (also NHWC)
```

#### `conv_transpose2d(input, filter, strides=None, dilations=None, pads=None, output_padding=None, output_sizes=None, groups=None, input_layout=None, filter_layout=None)`

2D transposed convolution (deconvolution) operation for upsampling.

**Parameters:**

- `input` (MLOperand): Input tensor (4D)
- `filter` (MLOperand): Filter weights (4D constant tensor)
- `strides` (list[int], optional): Stride along each spatial axis. Default: `[1, 1]`
- `dilations` (list[int], optional): Dilation along each spatial axis. Default: `[1, 1]`
- `pads` (list[int], optional): Padding. Default: `[0, 0, 0, 0]`
- `output_padding` (list[int], optional): Additional output padding. Default: `[0, 0]`
- `output_sizes` (list[int], optional): Explicit output spatial dimensions. Default: `None` (computed)
- `groups` (int, optional): Number of groups. Default: `1`
- `input_layout` (str, optional): `"nchw"` or `"nhwc"`. Default: `"nchw"`
- `filter_layout` (str, optional): Filter layout. Default: `"oihw"`

**Returns:** MLOperand with upsampled output tensor

**Shape Inference:**

For NCHW input `[N, C_in, H_in, W_in]` and OIHW filter `[C_in, C_out/groups, K_h, K_w]`:

```
output_h = (H_in - 1) * stride_h + effective_kernel_h - pad_begin_h - pad_end_h + output_pad_h
output_w = (W_in - 1) * stride_w + effective_kernel_w - pad_begin_w - pad_end_w + output_pad_w
output_shape = [N, C_out, output_h, output_w]
```

**Example: Basic Upsampling**

```python
# Upsample 14x14 to 29x29 with stride=2
input_op = builder.input("input", [1, 64, 14, 14], "float32")
filter_weights = np.random.randn(64, 32, 3, 3).astype(np.float32)
filter_op = builder.constant(filter_weights)

output = builder.conv_transpose2d(input_op, filter_op, strides=[2, 2])
# Output shape: [1, 32, 29, 29]
```

**Example: With Output Padding**

```python
# Use output_padding to control exact output size
input_op = builder.input("input", [1, 64, 14, 14], "float32")
filter_weights = np.random.randn(64, 32, 3, 3).astype(np.float32)
filter_op = builder.constant(filter_weights)

output = builder.conv_transpose2d(
    input_op,
    filter_op,
    strides=[2, 2],
    output_padding=[1, 1]
)
# Output shape: [1, 32, 30, 30]
```

**Example: Explicit Output Sizes**

```python
# Specify exact output dimensions
input_op = builder.input("input", [1, 64, 14, 14], "float32")
filter_weights = np.random.randn(64, 32, 3, 3).astype(np.float32)
filter_op = builder.constant(filter_weights)

output = builder.conv_transpose2d(
    input_op,
    filter_op,
    strides=[2, 2],
    pads=[1, 1, 1, 1],
    output_sizes=[28, 28]
)
# Output shape: [1, 32, 28, 28]
```

### Pooling Operations

#### `average_pool2d(input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None)`

2D average pooling operation for downsampling by computing the average of values in a pooling window.

**Parameters:**

- `input` (MLOperand): Input tensor (4D)
- `window_dimensions` (list[int], optional): Pooling window size `[height, width]`. Default: `[1, 1]`
- `strides` (list[int], optional): Stride along each spatial axis. Default: `[1, 1]`
- `dilations` (list[int], optional): Dilation along each spatial axis. Default: `[1, 1]`
- `pads` (list[int], optional): Padding `[begin_height, begin_width, end_height, end_width]`. Default: `[0, 0, 0, 0]`
- `layout` (str, optional): `"nchw"` or `"nhwc"`. Default: `"nchw"`

**Returns:** `MLOperand` - Output tensor after pooling

**Shape Inference:**

For each spatial dimension:
```
output_size = floor((input_size + pad_begin + pad_end - effective_window_size) / stride) + 1
```

where `effective_window_size = (window_size - 1) * dilation + 1`

**Example: Basic Average Pooling**

```python
# Input: [1, 64, 28, 28]
input_op = builder.input("input", [1, 64, 28, 28], "float32")

# Apply 2x2 average pooling with stride 2
output = builder.average_pool2d(
    input_op,
    window_dimensions=[2, 2],
    strides=[2, 2]
)
# Output shape: [1, 64, 14, 14]
```

**Example: Average Pooling with Padding**

```python
input_op = builder.input("input", [1, 64, 28, 28], "float32")

output = builder.average_pool2d(
    input_op,
    window_dimensions=[3, 3],
    strides=[2, 2],
    pads=[1, 1, 1, 1]  # Padding on all sides
)
# Output shape: [1, 64, 14, 14]
```

**Example: NHWC Layout**

```python
# Input in NHWC format: [batch, height, width, channels]
input_op = builder.input("input", [1, 28, 28, 64], "float32")

output = builder.average_pool2d(
    input_op,
    window_dimensions=[2, 2],
    strides=[2, 2],
    layout="nhwc"
)
# Output shape: [1, 14, 14, 64] (also NHWC)
```

#### `max_pool2d(input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None)`

2D max pooling operation for downsampling by taking the maximum value in a pooling window.

**Parameters:**

- `input` (MLOperand): Input tensor (4D)
- `window_dimensions` (list[int], optional): Pooling window size `[height, width]`. Default: `[1, 1]`
- `strides` (list[int], optional): Stride along each spatial axis. Default: `[1, 1]`
- `dilations` (list[int], optional): Dilation along each spatial axis. Default: `[1, 1]`
- `pads` (list[int], optional): Padding `[begin_height, begin_width, end_height, end_width]`. Default: `[0, 0, 0, 0]`
- `layout` (str, optional): `"nchw"` or `"nhwc"`. Default: `"nchw"`

**Returns:** `MLOperand` - Output tensor after pooling

**Shape Inference:**

Same as `average_pool2d` - for each spatial dimension:
```
output_size = floor((input_size + pad_begin + pad_end - effective_window_size) / stride) + 1
```

**Example: Basic Max Pooling**

```python
# Input: [1, 64, 28, 28]
input_op = builder.input("input", [1, 64, 28, 28], "float32")

# Apply 2x2 max pooling with stride 2
output = builder.max_pool2d(
    input_op,
    window_dimensions=[2, 2],
    strides=[2, 2]
)
# Output shape: [1, 64, 14, 14]
```

**Example: Overlapping Max Pooling**

```python
input_op = builder.input("input", [1, 32, 14, 14], "float32")

# Window size 2x2, stride 1x1 (overlapping windows)
output = builder.max_pool2d(
    input_op,
    window_dimensions=[2, 2],
    strides=[1, 1]
)
# Output shape: [1, 32, 13, 13]
```

**Example: Max Pooling with Padding**

```python
input_op = builder.input("input", [1, 64, 28, 28], "float32")

output = builder.max_pool2d(
    input_op,
    window_dimensions=[3, 3],
    strides=[2, 2],
    pads=[1, 1, 1, 1]
)
# Output shape: [1, 64, 14, 14]
```

#### `global_average_pool(input, layout=None)`

Global average pooling operation that reduces spatial dimensions to 1x1 by averaging over all spatial locations.

**Parameters:**

- `input` (MLOperand): Input tensor (4D)
- `layout` (str, optional): `"nchw"` or `"nhwc"`. Default: `"nchw"`

**Returns:** `MLOperand` - Output tensor with spatial dimensions 1x1

**Shape Inference:**

- NCHW: `[N, C, H, W]` → `[N, C, 1, 1]`
- NHWC: `[N, H, W, C]` → `[N, 1, 1, C]`

**Example: Basic Global Average Pooling**

```python
# Input: [1, 64, 28, 28]
input_op = builder.input("input", [1, 64, 28, 28], "float32")

# Global average pool reduces spatial dimensions to 1x1
output = builder.global_average_pool(input_op)
# Output shape: [1, 64, 1, 1]
```

**Example: For Classification (Typical ResNet-style)**

```python
# After last conv layer: [1, 2048, 7, 7]
features = builder.input("features", [1, 2048, 7, 7], "float32")

# Global average pooling instead of flatten
pooled = builder.global_average_pool(features)
# Output shape: [1, 2048, 1, 1]

# Reshape for fully connected layer
flattened = builder.reshape(pooled, [1, 2048])
```

**Example: NHWC Layout**

```python
# Input in NHWC: [1, 28, 28, 64]
input_op = builder.input("input", [1, 28, 28, 64], "float32")

output = builder.global_average_pool(input_op, layout="nhwc")
# Output shape: [1, 1, 1, 64]
```

#### `global_max_pool(input, layout=None)`

Global max pooling operation that reduces spatial dimensions to 1x1 by taking the maximum value over all spatial locations.

**Parameters:**

- `input` (MLOperand): Input tensor (4D)
- `layout` (str, optional): `"nchw"` or `"nhwc"`. Default: `"nchw"`

**Returns:** `MLOperand` - Output tensor with spatial dimensions 1x1

**Shape Inference:**

Same as `global_average_pool`:
- NCHW: `[N, C, H, W]` → `[N, C, 1, 1]`
- NHWC: `[N, H, W, C]` → `[N, 1, 1, C]`

**Example: Basic Global Max Pooling**

```python
# Input: [2, 128, 7, 7]
input_op = builder.input("input", [2, 128, 7, 7], "float32")

# Global max pool reduces spatial dimensions to 1x1
output = builder.global_max_pool(input_op)
# Output shape: [2, 128, 1, 1]
```

**Example: Multi-scale Feature Extraction**

```python
# Extract features at different scales
input_op = builder.input("input", [1, 512, 14, 14], "float32")

# Global max pooling captures strongest activations
max_pooled = builder.global_max_pool(input_op)
# Output shape: [1, 512, 1, 1]

# Global average pooling captures average response
avg_pooled = builder.global_average_pool(input_op)
# Output shape: [1, 512, 1, 1]

# Can concatenate both for richer representation
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
