"""
Integration tests for device-resident tensors (MLDeviceTensor)

These tests validate the device tensor API for zero-copy execution.
"""

import pytest
import numpy as np

try:
    import webnn
    HAS_WEBNN = True
except ImportError:
    HAS_WEBNN = False

# Skip all tests if webnn is not available
pytestmark = pytest.mark.skipif(not HAS_WEBNN, reason="webnn package not available")


@pytest.fixture
def context():
    """Create a WebNN context for testing"""
    ml = webnn.ML()
    return ml.create_context(device_type="cpu", accelerated=False)


@pytest.fixture
def simple_graph(context):
    """Build a simple graph: y = x + 1"""
    builder = context.create_graph_builder()

    x = builder.input("x", [2, 3], "float32")
    one = builder.constant(np.ones((2, 3), dtype=np.float32), [2, 3], "float32")
    y = builder.add(x, one)

    return builder.build({"y": y})


def test_create_device_tensor(context, simple_graph):
    """Test creating a device tensor"""
    tensor = context.create_device_tensor(simple_graph, [2, 3], "float32")

    assert tensor.shape == [2, 3]
    assert tensor.data_type == "float32"
    assert tensor.size == 6
    # Device should be cpu since we're using CPU backend
    assert "cpu" in tensor.device.lower() or "onnx" in tensor.backend.lower()


def test_device_tensor_write_read(context, simple_graph):
    """Test writing and reading device tensor data"""
    tensor = context.create_device_tensor(simple_graph, [2, 3], "float32")

    # Write data
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor.write(data)

    # Read data back
    result = tensor.read()

    np.testing.assert_array_almost_equal(result, data)


def test_device_tensor_shape_mismatch(context, simple_graph):
    """Test that writing wrong shape raises error"""
    tensor = context.create_device_tensor(simple_graph, [2, 3], "float32")

    # Try to write wrong shape
    data = np.array([1, 2, 3], dtype=np.float32)  # Wrong shape

    with pytest.raises(ValueError, match="Shape mismatch"):
        tensor.write(data)


def test_device_tensor_destroy(context, simple_graph):
    """Test explicit tensor destruction"""
    tensor = context.create_device_tensor(simple_graph, [2, 3], "float32")

    # Destroy tensor
    tensor.destroy()

    # Operations after destroy should fail
    with pytest.raises(RuntimeError, match="destroyed"):
        tensor.read()


def test_dispatch_with_device_tensors(context, simple_graph):
    """Test dispatch with device tensor inputs/outputs"""
    # Create device tensors
    x_device = context.create_device_tensor(simple_graph, [2, 3], "float32")
    y_device = context.create_device_tensor(simple_graph, [2, 3], "float32")

    # Write input data
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x_device.write(x_data)

    # Execute graph with device tensors
    context.dispatch(
        simple_graph,
        {"x": x_device},
        {"y": y_device}
    )

    # Read output
    result = y_device.read()

    # Verify result: y = x + 1
    expected = x_data + 1
    np.testing.assert_array_almost_equal(result, expected)


def test_dispatch_mixed_tensors(context, simple_graph):
    """Test dispatch with mix of host and device tensors"""
    # Host input tensor
    x_host = context.create_host_tensor([2, 3], "float32")
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    context.write_tensor(x_host, x_data)

    # Device output tensor
    y_device = context.create_device_tensor(simple_graph, [2, 3], "float32")

    # Execute graph
    context.dispatch(
        simple_graph,
        {"x": x_host},
        {"y": y_device}
    )

    # Read output from device tensor
    result = y_device.read()

    # Verify result
    expected = x_data + 1
    np.testing.assert_array_almost_equal(result, expected)


def test_kv_cache_pattern(context):
    """Test KV cache ping-pong pattern with device tensors"""
    # Build simple graph that passes through KV cache
    builder = context.create_graph_builder()

    past_k = builder.input("past_k", [1, 4, 8, 16], "float32")
    past_v = builder.input("past_v", [1, 4, 8, 16], "float32")

    # Simple identity transformation with small update
    scale = builder.constant(np.array([0.01], dtype=np.float32), [1], "float32")
    scale_broadcast = builder.reshape(scale, [1, 1, 1, 1])
    scale_tiled = builder.tile(scale_broadcast, [1, 4, 8, 16])

    present_k = builder.add(past_k, scale_tiled)
    present_v = builder.add(past_v, scale_tiled)

    graph = builder.build({
        "present_k": present_k,
        "present_v": present_v,
    })

    # Create device tensors for KV cache
    kv_shape = [1, 4, 8, 16]
    past_k_device = context.create_device_tensor(graph, kv_shape, "float32")
    past_v_device = context.create_device_tensor(graph, kv_shape, "float32")
    present_k_device = context.create_device_tensor(graph, kv_shape, "float32")
    present_v_device = context.create_device_tensor(graph, kv_shape, "float32")

    # Initialize
    past_k_device.write(np.zeros(kv_shape, dtype=np.float32))
    past_v_device.write(np.zeros(kv_shape, dtype=np.float32))

    # Run 10 decode steps
    for step in range(10):
        context.dispatch(
            graph,
            {"past_k": past_k_device, "past_v": past_v_device},
            {"present_k": present_k_device, "present_v": present_v_device}
        )

        # Swap ping-pong buffers (just swap references)
        past_k_device, present_k_device = present_k_device, past_k_device
        past_v_device, present_v_device = present_v_device, past_v_device

    # Read final values
    final_k = past_k_device.read()

    # After 10 steps with 0.01 increment, should be ~0.1
    assert final_k.mean() > 0.05 and final_k.mean() < 0.15

    # Cleanup
    past_k_device.destroy()
    past_v_device.destroy()
    present_k_device.destroy()
    present_v_device.destroy()


def test_device_tensor_repr(context, simple_graph):
    """Test device tensor string representation"""
    tensor = context.create_device_tensor(simple_graph, [2, 3], "float32")

    repr_str = repr(tensor)
    assert "MLDeviceTensor" in repr_str
    assert "shape" in repr_str
    assert "dtype" in repr_str


def test_multiple_graphs_same_context(context):
    """Test creating device tensors for multiple graphs"""
    builder = context.create_graph_builder()

    # Graph 1: y = x + 1
    x1 = builder.input("x", [2, 2], "float32")
    one = builder.constant(np.ones((2, 2), dtype=np.float32), [2, 2], "float32")
    y1 = builder.add(x1, one)
    graph1 = builder.build({"y": y1})

    # Graph 2: z = a * 2
    a2 = builder.input("a", [3, 3], "float32")
    two = builder.constant(np.full((3, 3), 2.0, dtype=np.float32), [3, 3], "float32")
    z2 = builder.mul(a2, two)
    graph2 = builder.build({"z": z2})

    # Create device tensors for each graph
    tensor1 = context.create_device_tensor(graph1, [2, 2], "float32")
    tensor2 = context.create_device_tensor(graph2, [3, 3], "float32")

    assert tensor1.shape == [2, 2]
    assert tensor2.shape == [3, 3]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
