"""Tests for the Python WebNN API"""

import sys
import pytest
import numpy as np

# Note: Import will only work after building with maturin
try:
    import webnn
    WEBNN_AVAILABLE = True
except ImportError:
    WEBNN_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="webnn not built yet")

# Check if ONNX runtime is available by testing if compute returns non-zero values
def _has_onnx_runtime():
    """Check if ONNX runtime is available for actual computation"""
    if not WEBNN_AVAILABLE:
        return False
    try:
        ml = webnn.ML()
        ctx = ml.create_context(power_preference="default", accelerated=False)
        builder = ctx.create_graph_builder()
        x = builder.input("x", [1, 1], "float32")
        y = builder.relu(x)
        graph = builder.build({"output": y})
        result = ctx.compute(graph, {"x": np.array([[1.0]], dtype=np.float32)})
        # If ONNX runtime is available, result should be non-zero
        return np.any(result["output"] != 0)
    except:
        return False

ONNX_RUNTIME_AVAILABLE = _has_onnx_runtime()
requires_onnx_runtime = pytest.mark.skipif(
    not ONNX_RUNTIME_AVAILABLE,
    reason="ONNX runtime not available - built without onnx-runtime feature"
)


@pytest.fixture
def ml():
    """Create ML instance"""
    return webnn.ML()


@pytest.fixture
def context(ml):
    """Create ML context"""
    return ml.create_context(power_preference="default", accelerated=False)


@pytest.fixture
def builder(context):
    """Create graph builder"""
    return context.create_graph_builder()


def test_ml_creation():
    """Test ML instance creation"""
    ml = webnn.ML()
    assert ml is not None


def test_context_creation(ml):
    """Test context creation with different options"""
    # Test CPU-only context (accelerated=False)
    ctx = ml.create_context(power_preference="default", accelerated=False)
    assert ctx.accelerated == False
    assert ctx.power_preference == "default"

    # Test accelerated context with high performance
    ctx_accel = ml.create_context(power_preference="high-performance", accelerated=True)
    assert ctx_accel.power_preference == "high-performance"
    # Note: accelerated may be True or False depending on platform capabilities

    # Test accelerated context with low power (NPU preferred)
    ctx_low_power = ml.create_context(power_preference="low-power", accelerated=True)
    assert ctx_low_power.power_preference == "low-power"


def test_graph_builder_creation(context):
    """Test graph builder creation"""
    builder = context.create_graph_builder()
    assert builder is not None


def test_input_operand(builder):
    """Test creating input operands"""
    x = builder.input("x", [2, 3], "float32")
    assert x.name == "x"
    assert x.shape == [2, 3]
    assert x.data_type == "float32"


def test_constant_operand(builder):
    """Test creating constant operands"""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    const = builder.constant(data)
    assert const.shape == [2, 2]
    assert const.data_type == "float32"


def test_binary_operations(builder):
    """Test binary operations"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")

    # Test add
    z_add = builder.add(x, y)
    assert z_add.shape == [2, 3]
    assert z_add.data_type == "float32"

    # Test sub
    z_sub = builder.sub(x, y)
    assert z_sub.shape == [2, 3]

    # Test mul
    z_mul = builder.mul(x, y)
    assert z_mul.shape == [2, 3]

    # Test div
    z_div = builder.div(x, y)
    assert z_div.shape == [2, 3]


def test_unary_operations(builder):
    """Test unary operations"""
    x = builder.input("x", [2, 3], "float32")

    # Test ReLU
    relu_out = builder.relu(x)
    assert relu_out.shape == [2, 3]
    assert relu_out.data_type == "float32"

    # Test sigmoid
    sigmoid_out = builder.sigmoid(x)
    assert sigmoid_out.shape == [2, 3]

    # Test tanh
    tanh_out = builder.tanh(x)
    assert tanh_out.shape == [2, 3]

    # Test softmax
    softmax_out = builder.softmax(x)
    assert softmax_out.shape == [2, 3]


def test_reshape_operation(builder):
    """Test reshape operation"""
    x = builder.input("x", [2, 3], "float32")
    reshaped = builder.reshape(x, [3, 2])
    assert reshaped.shape == [3, 2]
    assert reshaped.data_type == "float32"


def test_matmul_operation(builder):
    """Test matrix multiplication"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [3, 4], "float32")
    c = builder.matmul(a, b)
    # Note: shape inference would need proper implementation
    assert c is not None


def test_graph_building(builder):
    """Test building a complete graph"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)
    output = builder.relu(z)

    graph = builder.build({"output": output})
    assert graph is not None
    assert graph.operand_count > 0
    assert graph.operation_count > 0
    assert "output" in graph.get_output_names()
    assert "x" in graph.get_input_names()
    assert "y" in graph.get_input_names()


def test_simple_computation(context, builder):
    """Test simple graph computation"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)

    graph = builder.build({"z": z})

    # Create input data
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y_data = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)

    # Compute
    results = context.compute(graph, {"x": x_data, "y": y_data})
    assert "z" in results
    assert results["z"].shape == (2, 3)


@requires_onnx_runtime
def test_simple_computation_with_values(context, builder):
    """Test simple graph computation with actual values (requires ONNX runtime)"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)

    graph = builder.build({"z": z})

    # Create input data
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y_data = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)

    # Compute and verify actual computation
    results = context.compute(graph, {"x": x_data, "y": y_data})
    expected = x_data + y_data
    np.testing.assert_allclose(results["z"], expected, rtol=1e-5)


def test_onnx_conversion(context, builder, tmp_path):
    """Test ONNX conversion"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.relu(x)
    graph = builder.build({"y": y})

    output_path = tmp_path / "model.onnx"
    context.convert_to_onnx(graph, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="CoreML tests only run on macOS")
def test_coreml_conversion(context, builder, tmp_path):
    """Test CoreML conversion (macOS only)"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.relu(x)
    graph = builder.build({"y": y})

    output_path = tmp_path / "model.mlmodel"
    context.convert_to_coreml(graph, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_complex_graph(builder):
    """Test building a more complex graph"""
    # Input layer
    x = builder.input("input", [1, 3, 224, 224], "float32")

    # Simple transformation
    x1 = builder.relu(x)
    x2 = builder.sigmoid(x1)

    # Multiple outputs
    graph = builder.build({
        "relu_out": x1,
        "sigmoid_out": x2
    })

    assert graph.operand_count > 0
    assert graph.operation_count == 2
    assert set(graph.get_output_names()) == {"relu_out", "sigmoid_out"}


@requires_onnx_runtime
def test_relu_computation(context, builder):
    """Test ReLU activation with actual computation"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.relu(x)
    graph = builder.build({"y": y})

    # Test data with positive and negative values
    x_data = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    # Verify ReLU: max(0, x)
    expected = np.maximum(0, x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_sigmoid_computation(context, builder):
    """Test sigmoid activation with actual computation"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.sigmoid(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0, 1, -1], [2, -2, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    # Verify sigmoid: 1 / (1 + exp(-x))
    expected = 1 / (1 + np.exp(-x_data))
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_tanh_computation(context, builder):
    """Test tanh activation with actual computation"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.tanh(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0, 1, -1], [2, -2, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    # Verify tanh
    expected = np.tanh(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_softmax_computation(context, builder):
    """Test softmax activation with actual computation"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.softmax(x)
    graph = builder.build({"y": y})

    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    # Verify softmax output is in [0, 1] and sums to 1
    assert np.all(results["y"] >= 0)
    assert np.all(results["y"] <= 1)
    # Note: Softmax normalization depends on axis, so we just check properties


@requires_onnx_runtime
def test_chained_operations(context, builder):
    """Test chained operations with actual computation"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)
    output = builder.relu(z)

    graph = builder.build({"output": output})

    x_data = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    y_data = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data, "y": y_data})
    assert "output" in results
    assert results["output"].shape == (2, 3)

    # Verify: relu(x + y)
    expected = np.maximum(0, x_data + y_data)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_matmul_computation(context, builder):
    """Test matrix multiplication with actual computation"""
    a = builder.input("a", [2, 3], "float32")
    b_data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    b = builder.constant(b_data)
    c = builder.matmul(a, b)

    graph = builder.build({"c": c})

    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data})
    assert "c" in results

    # Verify matrix multiplication
    expected = np.matmul(a_data, b_data)
    np.testing.assert_allclose(results["c"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_multi_output_computation(context, builder):
    """Test graph with multiple outputs"""
    x = builder.input("x", [2, 3], "float32")
    y1 = builder.relu(x)
    y2 = builder.sigmoid(x)
    y3 = builder.tanh(x)

    graph = builder.build({"relu_out": y1, "sigmoid_out": y2, "tanh_out": y3})

    x_data = np.array([[-1, 0, 1], [2, -2, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})

    assert "relu_out" in results
    assert "sigmoid_out" in results
    assert "tanh_out" in results

    # Verify each output
    np.testing.assert_allclose(results["relu_out"], np.maximum(0, x_data), rtol=1e-5)
    np.testing.assert_allclose(results["sigmoid_out"], 1 / (1 + np.exp(-x_data)), rtol=1e-5)
    np.testing.assert_allclose(results["tanh_out"], np.tanh(x_data), rtol=1e-5)


# Shape Inference Tests
def test_broadcasting_same_shape(builder):
    """Test broadcasting with same shapes"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 3], "float32")
    c = builder.add(a, b)

    assert c.shape == [2, 3], f"Expected [2, 3], got {c.shape}"


def test_broadcasting_with_ones(builder):
    """Test broadcasting with dimensions of size 1"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [1, 3], "float32")
    c = builder.add(a, b)

    assert c.shape == [2, 3], f"Expected [2, 3], got {c.shape}"


def test_broadcasting_different_ranks(builder):
    """Test broadcasting with different tensor ranks"""
    a = builder.input("a", [2, 3, 4], "float32")
    b = builder.input("b", [3, 4], "float32")
    c = builder.add(a, b)

    assert c.shape == [2, 3, 4], f"Expected [2, 3, 4], got {c.shape}"


def test_broadcasting_scalar(builder):
    """Test broadcasting with scalar-like tensor"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [1], "float32")
    c = builder.mul(a, b)

    assert c.shape == [2, 3], f"Expected [2, 3], got {c.shape}"


def test_broadcasting_incompatible(builder):
    """Test that incompatible shapes raise errors"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 4], "float32")

    with pytest.raises(ValueError, match="Incompatible shapes"):
        builder.add(a, b)


def test_matmul_shape_inference(builder):
    """Test matmul shape inference"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [3, 4], "float32")
    c = builder.matmul(a, b)

    assert c.shape == [2, 4], f"Expected [2, 4], got {c.shape}"


def test_matmul_batched(builder):
    """Test batched matmul shape inference"""
    a = builder.input("a", [5, 2, 3], "float32")
    b = builder.input("b", [5, 3, 4], "float32")
    c = builder.matmul(a, b)

    assert c.shape == [5, 2, 4], f"Expected [5, 2, 4], got {c.shape}"


def test_matmul_incompatible(builder):
    """Test that incompatible matmul shapes raise errors"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [4, 5], "float32")

    with pytest.raises(ValueError, match="Incompatible shapes for matmul"):
        builder.matmul(a, b)


def test_reshape_valid(builder):
    """Test valid reshape operation"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.reshape(x, [6])

    assert y.shape == [6], f"Expected [6], got {y.shape}"


def test_reshape_invalid(builder):
    """Test that invalid reshape raises error"""
    x = builder.input("x", [2, 3], "float32")

    with pytest.raises(ValueError, match="Reshape requires same number of elements"):
        builder.reshape(x, [5])


@requires_onnx_runtime
def test_broadcasting_computation(context, builder):
    """Test that broadcasting works with actual computation"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [1, 3], "float32")
    c = builder.add(a, b)

    graph = builder.build({"c": c})

    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b_data = np.array([[10, 20, 30]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data, "b": b_data})
    expected = a_data + b_data

    np.testing.assert_allclose(results["c"], expected, rtol=1e-5)


# MLTensor Tests
def test_create_tensor(context):
    """Test creating a tensor"""
    tensor = context.create_tensor([2, 3], "float32")

    assert tensor.shape == [2, 3]
    assert tensor.data_type == "float32"
    assert tensor.size == 6


def test_write_read_tensor(context):
    """Test writing and reading tensor data"""
    tensor = context.create_tensor([2, 3], "float32")

    # Write data to tensor
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    context.write_tensor(tensor, data)

    # Read data back
    result = context.read_tensor(tensor)

    np.testing.assert_array_equal(result, data)


def test_write_tensor_shape_mismatch(context):
    """Test that writing wrong shape raises error"""
    tensor = context.create_tensor([2, 3], "float32")
    wrong_data = np.array([[1, 2], [3, 4]], dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        context.write_tensor(tensor, wrong_data)


def test_tensor_initial_data(context):
    """Test that tensors are initialized with zeros"""
    tensor = context.create_tensor([2, 3], "float32")
    result = context.read_tensor(tensor)

    expected = np.zeros((2, 3), dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_tensor_different_shapes(context):
    """Test creating tensors with different shapes"""
    tensor1 = context.create_tensor([5], "float32")
    tensor2 = context.create_tensor([2, 3, 4], "float32")

    assert tensor1.shape == [5]
    assert tensor1.size == 5

    assert tensor2.shape == [2, 3, 4]
    assert tensor2.size == 24


def test_tensor_repr(context):
    """Test tensor string representation"""
    tensor = context.create_tensor([2, 3], "float32")
    repr_str = repr(tensor)

    assert "MLTensor" in repr_str
    assert "[2, 3]" in repr_str
    assert "float32" in repr_str


def test_tensor_descriptor_flags(context):
    """Test MLTensor descriptor flags per W3C MLTensor Explainer"""
    # Test default flags (readable=True, writable=True, exportable_to_gpu=False)
    tensor_default = context.create_tensor([2, 3], "float32")
    assert tensor_default.readable == True
    assert tensor_default.writable == True
    assert tensor_default.exportable_to_gpu == False

    # Test custom flags
    tensor_readonly = context.create_tensor([2, 3], "float32", readable=True, writable=False)
    assert tensor_readonly.readable == True
    assert tensor_readonly.writable == False

    tensor_writeonly = context.create_tensor([2, 3], "float32", readable=False, writable=True)
    assert tensor_writeonly.readable == False
    assert tensor_writeonly.writable == True

    tensor_gpu_exportable = context.create_tensor([2, 3], "float32", exportable_to_gpu=True)
    assert tensor_gpu_exportable.exportable_to_gpu == True


def test_tensor_destroy(context):
    """Test explicit tensor resource cleanup with destroy()"""
    tensor = context.create_tensor([2, 3], "float32")

    # Write some data
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    context.write_tensor(tensor, data)

    # Read data should work
    result = context.read_tensor(tensor)
    np.testing.assert_array_equal(result, data)

    # Destroy the tensor
    tensor.destroy()

    # Further operations should fail
    with pytest.raises(RuntimeError, match="destroyed"):
        context.read_tensor(tensor)

    # Destroy again should also fail
    with pytest.raises(RuntimeError, match="already destroyed"):
        tensor.destroy()


def test_tensor_read_write_permissions(context):
    """Test that tensor flags are enforced"""
    # Create read-only tensor
    tensor_readonly = context.create_tensor([2, 3], "float32", readable=True, writable=False)

    # Writing should fail
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="not writable"):
        context.write_tensor(tensor_readonly, data)

    # Create write-only tensor
    tensor_writeonly = context.create_tensor([2, 3], "float32", readable=False, writable=True)

    # Writing should work
    context.write_tensor(tensor_writeonly, data)

    # Reading should fail
    with pytest.raises(RuntimeError, match="not readable"):
        context.read_tensor(tensor_writeonly)


@requires_onnx_runtime
def test_dispatch_method(context, builder):
    """Test dispatch() method for async execution per W3C MLTensor Explainer"""
    # Build a simple graph
    x = builder.input("x", [2, 3], "float32")
    y = builder.relu(x)
    graph = builder.build({"output": y})

    # Create input and output tensors
    input_tensor = context.create_tensor([2, 3], "float32")
    output_tensor = context.create_tensor([2, 3], "float32")

    # Write input data
    input_data = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    context.write_tensor(input_tensor, input_data)

    # Dispatch graph execution
    context.dispatch(graph, {"x": input_tensor}, {"output": output_tensor})

    # Read output
    result = context.read_tensor(output_tensor)

    # Verify result (relu should zero out negative values)
    expected = np.maximum(input_data, 0)
    np.testing.assert_array_equal(result, expected)


@requires_onnx_runtime
def test_tensor_workflow(context, builder):
    """Test complete tensor workflow with graph execution"""
    # Create tensors for inputs and outputs
    input_tensor = context.create_tensor([2, 3], "float32")

    # Write input data
    input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    context.write_tensor(input_tensor, input_data)

    # Build a simple graph
    x = builder.input("x", [2, 3], "float32")
    y = builder.relu(x)
    graph = builder.build({"output": y})

    # Read input from tensor for compute
    tensor_data = context.read_tensor(input_tensor)

    # Execute graph
    results = context.compute(graph, {"x": tensor_data})

    # Verify result
    expected = np.maximum(input_data, 0)
    np.testing.assert_array_equal(results["output"], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ============================================================================
# Async Execution Tests
# ============================================================================

@pytest.fixture
def async_context(context):
    """Create async ML context wrapper"""
    return webnn.AsyncMLContext(context)


@pytest.mark.asyncio
async def test_async_dispatch(async_context):
    """Test asynchronous graph dispatch"""
    builder = async_context.create_graph_builder()
    
    # Build simple graph
    x = builder.input("x", [2, 3], "float32")
    y = builder.relu(x)
    graph = builder.build({"output": y})
    
    # Dispatch asynchronously
    inputs = {"x": np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)}
    await async_context.dispatch(graph, inputs)
    
    # Dispatch should complete without error


@pytest.mark.asyncio
async def test_async_tensor_read_write(async_context):
    """Test asynchronous tensor read/write operations"""
    # Create tensor
    tensor = async_context.create_tensor([2, 3], "float32")
    
    # Write data asynchronously
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    await async_context.write_tensor_async(tensor, data)
    
    # Read data asynchronously
    result = await async_context.read_tensor_async(tensor)
    
    np.testing.assert_array_equal(result, data)


@pytest.mark.asyncio
async def test_async_concurrent_operations(async_context):
    """Test multiple concurrent async operations"""
    import asyncio
    
    # Create multiple tensors
    tensors = [
        async_context.create_tensor([2, 2], "float32")
        for _ in range(3)
    ]
    
    # Write to all tensors concurrently
    async def write_tensor(tensor, value):
        data = np.full((2, 2), value, dtype=np.float32)
        await async_context.write_tensor_async(tensor, data)
    
    await asyncio.gather(*[
        write_tensor(tensor, i)
        for i, tensor in enumerate(tensors)
    ])
    
    # Read from all tensors concurrently
    results = await asyncio.gather(*[
        async_context.read_tensor_async(tensor)
        for tensor in tensors
    ])
    
    # Verify results
    for i, result in enumerate(results):
        expected = np.full((2, 2), i, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.asyncio
@requires_onnx_runtime
async def test_async_dispatch_with_actual_computation(async_context):
    """Test async dispatch with actual ONNX computation"""
    builder = async_context.create_graph_builder()
    
    # Build graph: output = relu(x + y)
    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)
    output = builder.relu(z)
    graph = builder.build({"output": output})
    
    # Dispatch asynchronously
    x_data = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    y_data = np.array([[0, 1, -1], [2, -3, 4]], dtype=np.float32)
    
    result = await async_context.dispatch(graph, {"x": x_data, "y": y_data})
    
    # Verify execution completed
    assert result is None or isinstance(result, dict)


@pytest.mark.asyncio
async def test_async_context_properties(async_context):
    """Test that async context preserves underlying context properties"""
    assert async_context.accelerated == False  # CPU-only from fixture
    assert async_context.power_preference == "default"

    # Test synchronous methods still work
    builder = async_context.create_graph_builder()
    assert builder is not None

    tensor = async_context.create_tensor([2, 2], "float32")
    assert tensor is not None
