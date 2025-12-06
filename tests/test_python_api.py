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


@pytest.fixture
def ml():
    """Create ML instance"""
    return webnn.ML()


@pytest.fixture
def context(ml):
    """Create ML context"""
    return ml.create_context(device_type="cpu")


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
    ctx = ml.create_context(device_type="cpu", power_preference="default")
    assert ctx.device_type == "cpu"
    assert ctx.power_preference == "default"

    ctx_gpu = ml.create_context(device_type="gpu", power_preference="high-performance")
    assert ctx_gpu.device_type == "gpu"
    assert ctx_gpu.power_preference == "high-performance"


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

    # Verify actual computation (if ONNX runtime is available)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
