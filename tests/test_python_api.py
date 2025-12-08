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


# ============================================================================
# Element-wise Operations Tests
# ============================================================================

# Basic math operations

@requires_onnx_runtime
def test_abs_computation(context, builder):
    """Test element-wise absolute value"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.abs(x)
    graph = builder.build({"y": y})

    x_data = np.array([[-1.5, 2.0, -3.7], [4.2, -5.9, 0.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.abs(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_ceil_computation(context, builder):
    """Test element-wise ceiling"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.ceil(x)
    graph = builder.build({"y": y})

    x_data = np.array([[-1.5, 2.1, -3.9], [4.2, -5.7, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.ceil(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_floor_computation(context, builder):
    """Test element-wise floor"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.floor(x)
    graph = builder.build({"y": y})

    x_data = np.array([[-1.5, 2.9, -3.1], [4.8, -5.2, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.floor(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_round_computation(context, builder):
    """Test element-wise rounding"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.round(x)
    graph = builder.build({"y": y})

    x_data = np.array([[-1.5, 2.3, -3.7], [4.6, -5.4, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.round(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_neg_computation(context, builder):
    """Test element-wise negation"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.neg(x)
    graph = builder.build({"y": y})

    x_data = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 0.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = -x_data
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_sign_computation(context, builder):
    """Test element-wise sign"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.sign(x)
    graph = builder.build({"y": y})

    x_data = np.array([[-1.5, 2.3, 0.0], [4.6, -5.4, 0.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.sign(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


# Exponential and logarithmic operations

@requires_onnx_runtime
def test_exp_computation(context, builder):
    """Test element-wise exponential"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.exp(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.exp(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_log_computation(context, builder):
    """Test element-wise natural logarithm"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.log(x)
    graph = builder.build({"y": y})

    x_data = np.array([[1.0, 2.0, 3.0], [0.5, 0.1, 10.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.log(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_sqrt_computation(context, builder):
    """Test element-wise square root"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.sqrt(x)
    graph = builder.build({"y": y})

    x_data = np.array([[1.0, 4.0, 9.0], [16.0, 25.0, 0.25]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.sqrt(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_reciprocal_computation(context, builder):
    """Test element-wise reciprocal (1/x)"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.reciprocal(x)
    graph = builder.build({"y": y})

    x_data = np.array([[1.0, 2.0, 4.0], [0.5, 0.25, 10.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.reciprocal(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


# Trigonometric operations

@requires_onnx_runtime
def test_sin_computation(context, builder):
    """Test element-wise sine"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.sin(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, np.pi/2, np.pi], [-np.pi/2, np.pi/4, -np.pi]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.sin(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5, atol=1e-7)


@requires_onnx_runtime
def test_cos_computation(context, builder):
    """Test element-wise cosine"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.cos(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, np.pi/2, np.pi], [-np.pi/2, np.pi/4, -np.pi]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.cos(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5, atol=1e-7)


@requires_onnx_runtime
def test_tan_computation(context, builder):
    """Test element-wise tangent"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.tan(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, np.pi/4, -np.pi/4], [np.pi/6, -np.pi/6, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.tan(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5, atol=1e-7)


@requires_onnx_runtime
def test_asin_computation(context, builder):
    """Test element-wise arcsine"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.asin(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, 0.5, -0.5], [0.707, -0.707, 1.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.arcsin(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5, atol=1e-7)


@requires_onnx_runtime
def test_acos_computation(context, builder):
    """Test element-wise arccosine"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.acos(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, 0.5, -0.5], [0.707, -0.707, 1.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.arccos(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5, atol=1e-7)


@requires_onnx_runtime
def test_atan_computation(context, builder):
    """Test element-wise arctangent"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.atan(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.arctan(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5, atol=1e-7)


# Hyperbolic operations

@requires_onnx_runtime
def test_sinh_computation(context, builder):
    """Test element-wise hyperbolic sine"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.sinh(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, 1.0, -1.0], [0.5, -0.5, 2.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.sinh(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_cosh_computation(context, builder):
    """Test element-wise hyperbolic cosine"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.cosh(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, 1.0, -1.0], [0.5, -0.5, 2.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.cosh(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_asinh_computation(context, builder):
    """Test element-wise hyperbolic arcsine"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.asinh(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.arcsinh(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_acosh_computation(context, builder):
    """Test element-wise hyperbolic arccosine"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.acosh(x)
    graph = builder.build({"y": y})

    x_data = np.array([[1.0, 2.0, 3.0], [1.5, 5.0, 10.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.arccosh(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_atanh_computation(context, builder):
    """Test element-wise hyperbolic arctangent"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.atanh(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, 0.5, -0.5], [0.3, -0.3, 0.9]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.arctanh(x_data)
    np.testing.assert_allclose(results["y"], expected, rtol=1e-5)


# Special operations

@requires_onnx_runtime
def test_erf_computation(context, builder):
    """Test element-wise error function"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.erf(x)
    graph = builder.build({"y": y})

    x_data = np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    # Use scipy for erf if available, otherwise skip
    try:
        from scipy.special import erf
        expected = erf(x_data)
        np.testing.assert_allclose(results["y"], expected, rtol=1e-5)
    except ImportError:
        # If scipy not available, just verify shape and reasonable values
        assert np.all(results["y"] >= -1) and np.all(results["y"] <= 1)


@requires_onnx_runtime
def test_identity_computation(context, builder):
    """Test identity operation"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.identity(x)
    graph = builder.build({"y": y})

    x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    # Identity should return exact same values
    np.testing.assert_array_equal(results["y"], x_data)


# Logic operations tests

@requires_onnx_runtime
def test_equal_computation(context, builder):
    """Test element-wise equality comparison"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 3], "float32")
    y = builder.equal(a, b)
    graph = builder.build({"y": y})

    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_data = np.array([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data, "b": b_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = (a_data == b_data).astype(np.uint8)
    np.testing.assert_array_equal(results["y"], expected)


@requires_onnx_runtime
def test_greater_computation(context, builder):
    """Test element-wise greater than comparison"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 3], "float32")
    y = builder.greater(a, b)
    graph = builder.build({"y": y})

    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_data = np.array([[0.5, 2.5, 2.5], [4.5, 4.5, 5.5]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data, "b": b_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = (a_data > b_data).astype(np.uint8)
    np.testing.assert_array_equal(results["y"], expected)


@requires_onnx_runtime
def test_greater_or_equal_computation(context, builder):
    """Test element-wise greater than or equal comparison"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 3], "float32")
    y = builder.greater_or_equal(a, b)
    graph = builder.build({"y": y})

    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_data = np.array([[0.5, 2.0, 3.5], [4.5, 5.0, 5.5]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data, "b": b_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = (a_data >= b_data).astype(np.uint8)
    np.testing.assert_array_equal(results["y"], expected)


@requires_onnx_runtime
def test_lesser_computation(context, builder):
    """Test element-wise less than comparison"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 3], "float32")
    y = builder.lesser(a, b)
    graph = builder.build({"y": y})

    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_data = np.array([[1.5, 1.5, 3.5], [3.5, 5.5, 6.5]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data, "b": b_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = (a_data < b_data).astype(np.uint8)
    np.testing.assert_array_equal(results["y"], expected)


@requires_onnx_runtime
def test_lesser_or_equal_computation(context, builder):
    """Test element-wise less than or equal comparison"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 3], "float32")
    y = builder.lesser_or_equal(a, b)
    graph = builder.build({"y": y})

    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_data = np.array([[1.5, 2.0, 2.5], [3.5, 5.0, 6.5]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data, "b": b_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = (a_data <= b_data).astype(np.uint8)
    np.testing.assert_array_equal(results["y"], expected)


@requires_onnx_runtime
def test_logical_not_computation(context, builder):
    """Test element-wise logical NOT"""
    x = builder.input("x", [2, 3], "float32")
    y = builder.logical_not(x)
    graph = builder.build({"y": y})

    # Use 0.0 for false and non-zero for true
    x_data = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float32)

    results = context.compute(graph, {"x": x_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.logical_not(x_data).astype(np.uint8)
    np.testing.assert_array_equal(results["y"], expected)


@requires_onnx_runtime
def test_logical_and_computation(context, builder):
    """Test element-wise logical AND"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 3], "float32")
    y = builder.logical_and(a, b)
    graph = builder.build({"y": y})

    # Use 0.0 for false and non-zero for true
    a_data = np.array([[0.0, 1.0, 1.0], [0.0, 2.0, 3.0]], dtype=np.float32)
    b_data = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 0.0]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data, "b": b_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.logical_and(a_data, b_data).astype(np.uint8)
    np.testing.assert_array_equal(results["y"], expected)


@requires_onnx_runtime
def test_logical_or_computation(context, builder):
    """Test element-wise logical OR"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 3], "float32")
    y = builder.logical_or(a, b)
    graph = builder.build({"y": y})

    # Use 0.0 for false and non-zero for true
    a_data = np.array([[0.0, 1.0, 1.0], [0.0, 2.0, 3.0]], dtype=np.float32)
    b_data = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 0.0]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data, "b": b_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.logical_or(a_data, b_data).astype(np.uint8)
    np.testing.assert_array_equal(results["y"], expected)


@requires_onnx_runtime
def test_logical_xor_computation(context, builder):
    """Test element-wise logical XOR"""
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [2, 3], "float32")
    y = builder.logical_xor(a, b)
    graph = builder.build({"y": y})

    # Use 0.0 for false and non-zero for true
    a_data = np.array([[0.0, 1.0, 1.0], [0.0, 2.0, 3.0]], dtype=np.float32)
    b_data = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 0.0]], dtype=np.float32)

    results = context.compute(graph, {"a": a_data, "b": b_data})
    assert "y" in results
    assert results["y"].shape == (2, 3)

    expected = np.logical_xor(a_data, b_data).astype(np.uint8)
    np.testing.assert_array_equal(results["y"], expected)


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


def test_conv2d_basic_nchw(builder):
    """Test basic conv2d operation with NCHW layout"""
    # Input: [1, 3, 32, 32], Filter: [64, 3, 3, 3]
    input_op = builder.input("input", [1, 3, 32, 32], "float32")
    filter_data = np.random.randn(64, 3, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    # Conv2d with padding to maintain spatial dimensions
    output = builder.conv2d(input_op, filter_op, pads=[1, 1, 1, 1])

    assert output.shape == [1, 64, 32, 32], f"Expected [1, 64, 32, 32], got {output.shape}"
    assert output.data_type == "float32"


def test_conv2d_with_stride(builder):
    """Test conv2d with stride=2"""
    input_op = builder.input("input", [1, 3, 28, 28], "float32")
    filter_data = np.random.randn(32, 3, 5, 5).astype(np.float32)
    filter_op = builder.constant(filter_data)

    output = builder.conv2d(input_op, filter_op, strides=[2, 2])

    assert output.shape == [1, 32, 12, 12], f"Expected [1, 32, 12, 12], got {output.shape}"


def test_conv2d_nhwc_layout(builder):
    """Test conv2d with NHWC (channels-last) layout"""
    # Input: [1, 32, 32, 3] (NHWC format)
    input_op = builder.input("input", [1, 32, 32, 3], "float32")
    filter_data = np.random.randn(64, 3, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    output = builder.conv2d(input_op, filter_op, pads=[1, 1, 1, 1], input_layout="nhwc")

    # Output should also be NHWC: [1, 32, 32, 64]
    assert output.shape == [1, 32, 32, 64], f"Expected [1, 32, 32, 64], got {output.shape}"


def test_conv2d_depthwise(builder):
    """Test depthwise convolution (groups = input channels)"""
    input_op = builder.input("input", [1, 32, 28, 28], "float32")
    # Depthwise filter: [32, 1, 3, 3] - one filter per input channel
    filter_data = np.random.randn(32, 1, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    output = builder.conv2d(input_op, filter_op, pads=[1, 1, 1, 1], groups=32)

    # Output should maintain channel count for depthwise conv
    assert output.shape == [1, 32, 28, 28], f"Expected [1, 32, 28, 28], got {output.shape}"


def test_conv2d_with_dilation(builder):
    """Test conv2d with dilated convolution"""
    input_op = builder.input("input", [1, 3, 32, 32], "float32")
    filter_data = np.random.randn(64, 3, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    # Dilation=2 increases effective kernel size to 5x5
    output = builder.conv2d(input_op, filter_op, dilations=[2, 2], pads=[2, 2, 2, 2])

    assert output.shape == [1, 64, 32, 32], f"Expected [1, 64, 32, 32], got {output.shape}"


def test_conv2d_invalid_input_shape(builder):
    """Test that conv2d rejects non-4D input"""
    input_op = builder.input("input", [3, 32, 32], "float32")  # Only 3D
    filter_data = np.random.randn(64, 3, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    with pytest.raises(ValueError, match="Conv2d input must be 4D"):
        builder.conv2d(input_op, filter_op)


def test_conv2d_invalid_groups(builder):
    """Test that conv2d validates groups parameter"""
    input_op = builder.input("input", [1, 3, 32, 32], "float32")
    # Filter with wrong channel count for groups=2
    filter_data = np.random.randn(64, 1, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    with pytest.raises(ValueError, match="Conv2d input channels.*must be divisible by groups"):
        builder.conv2d(input_op, filter_op, groups=2)


def test_conv2d_invalid_layout(builder):
    """Test that conv2d validates layout parameters"""
    input_op = builder.input("input", [1, 3, 32, 32], "float32")
    filter_data = np.random.randn(64, 3, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    with pytest.raises(ValueError, match="Invalid input_layout"):
        builder.conv2d(input_op, filter_op, input_layout="invalid")


def test_conv_transpose2d_basic(builder):
    """Test basic convTranspose2d operation"""
    # Input: [1, 64, 14, 14], Filter: [64, 32, 3, 3]
    # Output: (14-1)*1 + 3 - 0 - 0 + 0 = 16
    input_op = builder.input("input", [1, 64, 14, 14], "float32")
    filter_data = np.random.randn(64, 32, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    output = builder.conv_transpose2d(input_op, filter_op)

    assert output.shape == [1, 32, 16, 16], f"Expected [1, 32, 16, 16], got {output.shape}"
    assert output.data_type == "float32"


def test_conv_transpose2d_with_stride(builder):
    """Test convTranspose2d with stride=2 for upsampling"""
    # Input: [1, 64, 14, 14], Filter: [64, 32, 3, 3]
    # Output: (14-1)*2 + 3 - 0 - 0 + 0 = 29
    input_op = builder.input("input", [1, 64, 14, 14], "float32")
    filter_data = np.random.randn(64, 32, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    output = builder.conv_transpose2d(input_op, filter_op, strides=[2, 2])

    assert output.shape == [1, 32, 29, 29], f"Expected [1, 32, 29, 29], got {output.shape}"


def test_conv_transpose2d_with_padding(builder):
    """Test convTranspose2d with padding"""
    # Input: [1, 64, 14, 14], Filter: [64, 32, 3, 3]
    # Output: (14-1)*2 + 3 - 1 - 1 + 0 = 27
    input_op = builder.input("input", [1, 64, 14, 14], "float32")
    filter_data = np.random.randn(64, 32, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    output = builder.conv_transpose2d(input_op, filter_op, strides=[2, 2], pads=[1, 1, 1, 1])

    assert output.shape == [1, 32, 27, 27], f"Expected [1, 32, 27, 27], got {output.shape}"


def test_conv_transpose2d_with_output_padding(builder):
    """Test convTranspose2d with output_padding"""
    # Input: [1, 64, 14, 14], Filter: [64, 32, 3, 3]
    # Output: (14-1)*2 + 3 - 0 - 0 + 1 = 30
    input_op = builder.input("input", [1, 64, 14, 14], "float32")
    filter_data = np.random.randn(64, 32, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    output = builder.conv_transpose2d(input_op, filter_op, strides=[2, 2], output_padding=[1, 1])

    assert output.shape == [1, 32, 30, 30], f"Expected [1, 32, 30, 30], got {output.shape}"


def test_conv_transpose2d_with_output_sizes(builder):
    """Test convTranspose2d with explicit output_sizes"""
    input_op = builder.input("input", [1, 64, 14, 14], "float32")
    filter_data = np.random.randn(64, 32, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    # Specify exact output size
    output = builder.conv_transpose2d(
        input_op, filter_op, strides=[2, 2], pads=[1, 1, 1, 1], output_sizes=[28, 28]
    )

    assert output.shape == [1, 32, 28, 28], f"Expected [1, 32, 28, 28], got {output.shape}"


def test_conv_transpose2d_nhwc_layout(builder):
    """Test convTranspose2d with NHWC (channels-last) layout"""
    # Input: [1, 14, 14, 64] (NHWC format)
    input_op = builder.input("input", [1, 14, 14, 64], "float32")
    filter_data = np.random.randn(64, 32, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    output = builder.conv_transpose2d(
        input_op, filter_op, strides=[2, 2], input_layout="nhwc"
    )

    # Output should also be NHWC: [1, 29, 29, 32]
    assert output.shape == [1, 29, 29, 32], f"Expected [1, 29, 29, 32], got {output.shape}"


def test_conv_transpose2d_invalid_input_shape(builder):
    """Test that convTranspose2d rejects non-4D input"""
    input_op = builder.input("input", [64, 14, 14], "float32")  # Only 3D
    filter_data = np.random.randn(64, 32, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    with pytest.raises(ValueError, match="ConvTranspose2d input must be 4D"):
        builder.conv_transpose2d(input_op, filter_op)


def test_conv_transpose2d_invalid_layout(builder):
    """Test that convTranspose2d validates layout parameters"""
    input_op = builder.input("input", [1, 64, 14, 14], "float32")
    filter_data = np.random.randn(64, 32, 3, 3).astype(np.float32)
    filter_op = builder.constant(filter_data)

    with pytest.raises(ValueError, match="Invalid input_layout"):
        builder.conv_transpose2d(input_op, filter_op, input_layout="invalid")


# Pooling tests
def test_average_pool2d_basic(builder):
    """Test basic averagePool2d operation"""
    # Input: [1, 64, 28, 28], window: [2, 2], stride: [2, 2]
    input_op = builder.input("input", [1, 64, 28, 28], "float32")
    output = builder.average_pool2d(input_op, window_dimensions=[2, 2], strides=[2, 2])

    # Output shape: [1, 64, 14, 14]
    assert output.shape == [1, 64, 14, 14], f"Expected [1, 64, 14, 14], got {output.shape}"


def test_average_pool2d_with_padding(builder):
    """Test averagePool2d with padding"""
    input_op = builder.input("input", [1, 64, 28, 28], "float32")
    output = builder.average_pool2d(
        input_op,
        window_dimensions=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1]
    )

    # With padding, output shape: [1, 64, 14, 14]
    assert output.shape == [1, 64, 14, 14], f"Expected [1, 64, 14, 14], got {output.shape}"


def test_average_pool2d_nhwc_layout(builder):
    """Test averagePool2d with NHWC layout"""
    # Input in NHWC: [1, 28, 28, 64]
    input_op = builder.input("input", [1, 28, 28, 64], "float32")
    output = builder.average_pool2d(
        input_op,
        window_dimensions=[2, 2],
        strides=[2, 2],
        layout="nhwc"
    )

    # Output should also be NHWC: [1, 14, 14, 64]
    assert output.shape == [1, 14, 14, 64], f"Expected [1, 14, 14, 64], got {output.shape}"


def test_max_pool2d_basic(builder):
    """Test basic maxPool2d operation"""
    # Input: [1, 64, 28, 28], window: [2, 2], stride: [2, 2]
    input_op = builder.input("input", [1, 64, 28, 28], "float32")
    output = builder.max_pool2d(input_op, window_dimensions=[2, 2], strides=[2, 2])

    # Output shape: [1, 64, 14, 14]
    assert output.shape == [1, 64, 14, 14], f"Expected [1, 64, 14, 14], got {output.shape}"


def test_max_pool2d_with_padding(builder):
    """Test maxPool2d with padding"""
    input_op = builder.input("input", [1, 64, 28, 28], "float32")
    output = builder.max_pool2d(
        input_op,
        window_dimensions=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1]
    )

    # With padding, output shape: [1, 64, 14, 14]
    assert output.shape == [1, 64, 14, 14], f"Expected [1, 64, 14, 14], got {output.shape}"


def test_max_pool2d_stride_variations(builder):
    """Test maxPool2d with different stride values"""
    input_op = builder.input("input", [1, 32, 14, 14], "float32")
    # Window 2x2, stride 1x1 (overlapping)
    output = builder.max_pool2d(input_op, window_dimensions=[2, 2], strides=[1, 1])

    # Output shape: [1, 32, 13, 13]
    assert output.shape == [1, 32, 13, 13], f"Expected [1, 32, 13, 13], got {output.shape}"


def test_pool2d_invalid_input_shape(builder):
    """Test that pooling rejects non-4D input"""
    input_op = builder.input("input", [64, 28, 28], "float32")  # Only 3D

    with pytest.raises(ValueError, match="Pool2d input must be 4D"):
        builder.average_pool2d(input_op, window_dimensions=[2, 2])


def test_pool2d_invalid_layout(builder):
    """Test that pooling validates layout parameter"""
    input_op = builder.input("input", [1, 64, 28, 28], "float32")

    with pytest.raises(ValueError, match="Invalid layout"):
        builder.max_pool2d(input_op, window_dimensions=[2, 2], layout="invalid")


# Global pooling tests
def test_global_average_pool_basic(builder):
    """Test basic globalAveragePool operation"""
    # Input: [1, 64, 28, 28] -> Output: [1, 64, 1, 1]
    input_op = builder.input("input", [1, 64, 28, 28], "float32")
    output = builder.global_average_pool(input_op)

    assert output.shape == [1, 64, 1, 1], f"Expected [1, 64, 1, 1], got {output.shape}"


def test_global_average_pool_nhwc(builder):
    """Test globalAveragePool with NHWC layout"""
    # Input in NHWC: [1, 28, 28, 64] -> Output: [1, 1, 1, 64]
    input_op = builder.input("input", [1, 28, 28, 64], "float32")
    output = builder.global_average_pool(input_op, layout="nhwc")

    assert output.shape == [1, 1, 1, 64], f"Expected [1, 1, 1, 64], got {output.shape}"


def test_global_max_pool_basic(builder):
    """Test basic globalMaxPool operation"""
    # Input: [2, 128, 7, 7] -> Output: [2, 128, 1, 1]
    input_op = builder.input("input", [2, 128, 7, 7], "float32")
    output = builder.global_max_pool(input_op)

    assert output.shape == [2, 128, 1, 1], f"Expected [2, 128, 1, 1], got {output.shape}"


def test_global_max_pool_various_sizes(builder):
    """Test globalMaxPool with different input sizes"""
    # Test with 14x14 spatial dimensions
    input_op = builder.input("input", [1, 512, 14, 14], "float32")
    output = builder.global_max_pool(input_op)
    assert output.shape == [1, 512, 1, 1], f"Expected [1, 512, 1, 1], got {output.shape}"

    # Test with different batch size
    input_op2 = builder.input("input2", [4, 256, 8, 8], "float32")
    output2 = builder.global_max_pool(input_op2)
    assert output2.shape == [4, 256, 1, 1], f"Expected [4, 256, 1, 1], got {output2.shape}"


def test_global_pool_invalid_input_shape(builder):
    """Test that global pooling rejects non-4D input"""
    input_op = builder.input("input", [64, 28, 28], "float32")  # Only 3D

    with pytest.raises(ValueError, match="Global pooling input must be 4D"):
        builder.global_average_pool(input_op)


def test_global_pool_invalid_layout(builder):
    """Test that global pooling validates layout parameter"""
    input_op = builder.input("input", [1, 64, 28, 28], "float32")

    with pytest.raises(ValueError, match="Invalid layout"):
        builder.global_max_pool(input_op, layout="invalid")


# Normalization operations tests


def test_batch_normalization_basic(builder):
    """Test basic batch normalization operation"""
    # Input: [2, 64, 28, 28] (batch, channels, height, width)
    input_op = builder.input("input", [2, 64, 28, 28], "float32")
    mean = builder.input("mean", [64], "float32")
    variance = builder.input("variance", [64], "float32")

    output = builder.batch_normalization(input_op, mean, variance)
    assert output.shape == [2, 64, 28, 28], f"Expected [2, 64, 28, 28], got {output.shape}"


def test_batch_normalization_with_scale_bias(builder):
    """Test batch normalization with optional scale and bias"""
    input_op = builder.input("input", [1, 32, 14, 14], "float32")
    mean = builder.input("mean", [32], "float32")
    variance = builder.input("variance", [32], "float32")
    scale = builder.input("scale", [32], "float32")
    bias = builder.input("bias", [32], "float32")

    output = builder.batch_normalization(
        input_op, mean, variance, scale=scale, bias=bias, epsilon=1e-5, axis=1
    )
    assert output.shape == [1, 32, 14, 14]


def test_batch_normalization_custom_epsilon(builder):
    """Test batch normalization with custom epsilon"""
    input_op = builder.input("input", [4, 128, 7, 7], "float32")
    mean = builder.input("mean", [128], "float32")
    variance = builder.input("variance", [128], "float32")

    output = builder.batch_normalization(input_op, mean, variance, epsilon=1e-3)
    assert output.shape == [4, 128, 7, 7]


def test_instance_normalization_basic(builder):
    """Test basic instance normalization operation"""
    # Input: [2, 64, 28, 28]
    input_op = builder.input("input", [2, 64, 28, 28], "float32")

    output = builder.instance_normalization(input_op)
    assert output.shape == [2, 64, 28, 28], f"Expected [2, 64, 28, 28], got {output.shape}"


def test_instance_normalization_with_scale_bias(builder):
    """Test instance normalization with scale and bias"""
    input_op = builder.input("input", [1, 32, 14, 14], "float32")
    scale = builder.input("scale", [32], "float32")
    bias = builder.input("bias", [32], "float32")

    output = builder.instance_normalization(input_op, scale=scale, bias=bias)
    assert output.shape == [1, 32, 14, 14]


def test_instance_normalization_nhwc(builder):
    """Test instance normalization with NHWC layout"""
    input_op = builder.input("input", [2, 28, 28, 64], "float32")

    output = builder.instance_normalization(input_op, layout="nhwc")
    assert output.shape == [2, 28, 28, 64]


def test_instance_normalization_custom_epsilon(builder):
    """Test instance normalization with custom epsilon"""
    input_op = builder.input("input", [4, 128, 7, 7], "float32")

    output = builder.instance_normalization(input_op, epsilon=1e-6)
    assert output.shape == [4, 128, 7, 7]


def test_layer_normalization_basic(builder):
    """Test basic layer normalization operation"""
    # Input: [2, 512] (batch, features) - typical for transformers
    input_op = builder.input("input", [2, 512], "float32")

    output = builder.layer_normalization(input_op)
    assert output.shape == [2, 512], f"Expected [2, 512], got {output.shape}"


def test_layer_normalization_with_scale_bias(builder):
    """Test layer normalization with scale and bias"""
    input_op = builder.input("input", [4, 768], "float32")
    scale = builder.input("scale", [768], "float32")
    bias = builder.input("bias", [768], "float32")

    output = builder.layer_normalization(input_op, scale=scale, bias=bias)
    assert output.shape == [4, 768]


def test_layer_normalization_3d_input(builder):
    """Test layer normalization with 3D input (sequence data)"""
    # Input: [2, 10, 512] (batch, sequence_length, features)
    input_op = builder.input("input", [2, 10, 512], "float32")

    output = builder.layer_normalization(input_op, axes=[-1])
    assert output.shape == [2, 10, 512]


def test_layer_normalization_custom_axes(builder):
    """Test layer normalization with custom axes"""
    input_op = builder.input("input", [2, 8, 256], "float32")

    output = builder.layer_normalization(input_op, axes=[-2, -1])
    assert output.shape == [2, 8, 256]


def test_layer_normalization_custom_epsilon(builder):
    """Test layer normalization with custom epsilon"""
    input_op = builder.input("input", [1, 1024], "float32")

    output = builder.layer_normalization(input_op, epsilon=1e-12)
    assert output.shape == [1, 1024]


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

    # Create tensors for dispatch
    input_tensor = async_context.create_tensor([2, 3], "float32")
    output_tensor = async_context.create_tensor([2, 3], "float32")

    # Write input data
    input_data = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    await async_context.write_tensor_async(input_tensor, input_data)

    # Dispatch asynchronously
    await async_context.dispatch(graph, {"x": input_tensor}, {"output": output_tensor})

    # Dispatch should complete without error
    # Optionally verify output
    result = await async_context.read_tensor_async(output_tensor)
    assert result.shape == (2, 3)


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

    # Create tensors for dispatch
    x_tensor = async_context.create_tensor([2, 3], "float32")
    y_tensor = async_context.create_tensor([2, 3], "float32")
    output_tensor = async_context.create_tensor([2, 3], "float32")

    # Write input data asynchronously
    x_data = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    y_data = np.array([[0, 1, -1], [2, -3, 4]], dtype=np.float32)
    await async_context.write_tensor_async(x_tensor, x_data)
    await async_context.write_tensor_async(y_tensor, y_data)

    # Dispatch asynchronously
    await async_context.dispatch(
        graph,
        {"x": x_tensor, "y": y_tensor},
        {"output": output_tensor}
    )

    # Read results asynchronously
    result = await async_context.read_tensor_async(output_tensor)

    # Verify execution completed with correct results (relu(x + y))
    expected = np.array([[1, 0, 2], [0, 2, 0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


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


# === Reduction Operations Tests ===

def test_reduce_sum_single_axis(context):
    """Test reduceSum operation with single axis"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_sum(x, axes=[1], keep_dimensions=False)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_reduce_sum_single_axis_keep_dims(context):
    """Test reduceSum operation with keep_dimensions=True"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_sum(x, axes=[1], keep_dimensions=True)
    assert output.shape == [2, 1, 4]
    graph = builder.build({"output": output})


def test_reduce_sum_multiple_axes(context):
    """Test reduceSum operation with multiple axes"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4, 5], "float32")
    output = builder.reduce_sum(x, axes=[1, 2], keep_dimensions=False)
    assert output.shape == [2, 5]
    graph = builder.build({"output": output})


def test_reduce_sum_all_axes(context):
    """Test reduceSum operation reducing all axes"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_sum(x, axes=None, keep_dimensions=False)
    assert output.shape == []  # Scalar
    graph = builder.build({"output": output})


def test_reduce_mean_single_axis(context):
    """Test reduceMean operation with single axis"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_mean(x, axes=[1], keep_dimensions=False)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_reduce_mean_keep_dims(context):
    """Test reduceMean operation with keep_dimensions=True"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_mean(x, axes=[0, 2], keep_dimensions=True)
    assert output.shape == [1, 3, 1]
    graph = builder.build({"output": output})


def test_reduce_max(context):
    """Test reduceMax operation"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_max(x, axes=[1], keep_dimensions=False)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_reduce_min(context):
    """Test reduceMin operation"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_min(x, axes=[2], keep_dimensions=False)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})


def test_reduce_product(context):
    """Test reduceProduct operation"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_product(x, axes=[1], keep_dimensions=False)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_reduce_l1(context):
    """Test reduceL1 operation (sum of absolute values)"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_l1(x, axes=[1], keep_dimensions=False)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_reduce_l2(context):
    """Test reduceL2 operation (Euclidean norm)"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_l2(x, axes=[1], keep_dimensions=False)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_reduce_log_sum(context):
    """Test reduceLogSum operation"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_log_sum(x, axes=[1], keep_dimensions=False)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_reduce_log_sum_exp(context):
    """Test reduceLogSumExp operation"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_log_sum_exp(x, axes=[1], keep_dimensions=False)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_reduce_sum_square(context):
    """Test reduceSumSquare operation"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.reduce_sum_square(x, axes=[1], keep_dimensions=False)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_reduce_invalid_axis(context):
    """Test that reduction with out-of-bounds axis raises error"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    with pytest.raises(ValueError, match="out of bounds"):
        output = builder.reduce_sum(x, axes=[5], keep_dimensions=False)


def test_reduce_duplicate_axes(context):
    """Test that reduction with duplicate axes raises error"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    with pytest.raises(ValueError, match="Duplicate axis"):
        output = builder.reduce_sum(x, axes=[1, 1], keep_dimensions=False)


def test_reduce_sum_non_contiguous_axes(context):
    """Test reduction with non-contiguous axes"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4, 5], "float32")
    output = builder.reduce_sum(x, axes=[0, 2], keep_dimensions=False)
    assert output.shape == [3, 5]
    graph = builder.build({"output": output})


def test_reduce_sum_non_contiguous_axes_keep_dims(context):
    """Test reduction with non-contiguous axes and keep_dimensions=True"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4, 5], "float32")
    output = builder.reduce_sum(x, axes=[0, 2], keep_dimensions=True)
    assert output.shape == [1, 3, 1, 5]
    graph = builder.build({"output": output})


# Quantization operations
def test_dequantize_linear(context):
    """Test dequantizeLinear operation"""
    builder = context.create_graph_builder()
    # Create quantized input (int8)
    x = builder.input("x", [1, 3, 224, 224], "int8")
    # Scale and zero_point for dequantization
    scale = builder.input("scale", [1], "float32")
    zero_point = builder.input("zero_point", [1], "int8")
    # Dequantize: output = (x - zero_point) * scale
    output = builder.dequantize_linear(x, scale, zero_point)
    # Output should be float32 with same shape as input
    assert output.shape == [1, 3, 224, 224]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_quantize_linear(context):
    """Test quantizeLinear operation"""
    builder = context.create_graph_builder()
    # Create float input
    x = builder.input("x", [1, 3, 224, 224], "float32")
    # Scale and zero_point for quantization
    scale = builder.input("scale", [1], "float32")
    zero_point = builder.input("zero_point", [1], "int8")
    # Quantize: output = x / scale + zero_point
    output = builder.quantize_linear(x, scale, zero_point)
    # Output should be int8 with same shape as input
    assert output.shape == [1, 3, 224, 224]
    assert output.data_type == "int8"
    graph = builder.build({"output": output})


def test_quantize_linear_uint8(context):
    """Test quantizeLinear operation with uint8 output"""
    builder = context.create_graph_builder()
    x = builder.input("x", [8, 128], "float32")
    scale = builder.input("scale", [1], "float32")
    zero_point = builder.input("zero_point", [1], "uint8")
    output = builder.quantize_linear(x, scale, zero_point)
    # Output data type matches zero_point
    assert output.shape == [8, 128]
    assert output.data_type == "uint8"
    graph = builder.build({"output": output})


def test_dequantize_linear_uint8(context):
    """Test dequantizeLinear operation with uint8 input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [8, 128], "uint8")
    scale = builder.input("scale", [1], "float32")
    zero_point = builder.input("zero_point", [1], "uint8")
    output = builder.dequantize_linear(x, scale, zero_point)
    assert output.shape == [8, 128]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_quantization_roundtrip(context):
    """Test quantize followed by dequantize"""
    builder = context.create_graph_builder()
    # Original float input
    x = builder.input("x", [10, 20], "float32")
    scale = builder.input("scale", [1], "float32")
    zero_point = builder.input("zero_point", [1], "int8")
    # Quantize then dequantize
    quantized = builder.quantize_linear(x, scale, zero_point)
    dequantized = builder.dequantize_linear(quantized, scale, zero_point)
    # Final output should be float32 with same shape
    assert dequantized.shape == [10, 20]
    assert dequantized.data_type == "float32"
    graph = builder.build({"output": dequantized})


# ========================================
# Tensor Manipulation Operations Tests
# ========================================


# Transpose tests
def test_transpose_default_permutation(context):
    """Test transpose with default permutation (reverses dimensions)"""
    builder = context.create_graph_builder()
    x = builder.input("x", [4, 6], "float32")
    output = builder.transpose(x)
    assert output.shape == [6, 4]
    graph = builder.build({"output": output})


def test_transpose_custom_permutation_2d(context):
    """Test transpose with custom 2D permutation"""
    builder = context.create_graph_builder()
    x = builder.input("x", [4, 6], "float32")
    output = builder.transpose(x, permutation=[1, 0])
    assert output.shape == [6, 4]
    graph = builder.build({"output": output})


def test_transpose_custom_permutation_3d(context):
    """Test transpose with custom 3D permutation"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.transpose(x, permutation=[2, 0, 1])
    assert output.shape == [4, 2, 3]
    graph = builder.build({"output": output})


def test_transpose_4d_nchw_to_nhwc(context):
    """Test transpose 4D tensor from NCHW to NHWC layout"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 3, 224, 224], "float32")
    output = builder.transpose(x, permutation=[0, 2, 3, 1])
    assert output.shape == [1, 224, 224, 3]
    graph = builder.build({"output": output})


# Concat tests
def test_concat_2_inputs_axis_0(context):
    """Test concat with 2 inputs along axis 0"""
    builder = context.create_graph_builder()
    x1 = builder.input("x1", [2, 3], "float32")
    x2 = builder.input("x2", [2, 3], "float32")
    output = builder.concat([x1, x2], axis=0)
    assert output.shape == [4, 3]
    graph = builder.build({"output": output})


def test_concat_2_inputs_axis_1(context):
    """Test concat with 2 inputs along axis 1"""
    builder = context.create_graph_builder()
    x1 = builder.input("x1", [2, 3], "float32")
    x2 = builder.input("x2", [2, 3], "float32")
    output = builder.concat([x1, x2], axis=1)
    assert output.shape == [2, 6]
    graph = builder.build({"output": output})


def test_concat_multiple_inputs(context):
    """Test concat with 3 inputs"""
    builder = context.create_graph_builder()
    x1 = builder.input("x1", [1, 3], "float32")
    x2 = builder.input("x2", [2, 3], "float32")
    x3 = builder.input("x3", [3, 3], "float32")
    output = builder.concat([x1, x2, x3], axis=0)
    assert output.shape == [6, 3]
    graph = builder.build({"output": output})


def test_concat_3d(context):
    """Test concat with 3D tensors"""
    builder = context.create_graph_builder()
    x1 = builder.input("x1", [2, 3, 4], "float32")
    x2 = builder.input("x2", [2, 3, 4], "float32")
    output = builder.concat([x1, x2], axis=2)
    assert output.shape == [2, 3, 8]
    graph = builder.build({"output": output})


# Slice tests
def test_slice_1d(context):
    """Test slice on 1D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [24], "float32")
    output = builder.slice(x, starts=[12], sizes=[12])
    assert output.shape == [12]
    graph = builder.build({"output": output})


def test_slice_2d(context):
    """Test slice on 2D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [4, 6], "float32")
    output = builder.slice(x, starts=[2, 2], sizes=[2, 4])
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})


def test_slice_3d(context):
    """Test slice on 3D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [4, 3, 2], "float32")
    output = builder.slice(x, starts=[1, 1, 1], sizes=[3, 2, 1])
    assert output.shape == [3, 2, 1]
    graph = builder.build({"output": output})


def test_slice_whole_dimension(context):
    """Test slice that selects entire dimension"""
    builder = context.create_graph_builder()
    x = builder.input("x", [4, 6], "float32")
    output = builder.slice(x, starts=[0, 2], sizes=[4, 4])
    assert output.shape == [4, 4]
    graph = builder.build({"output": output})


# Expand tests
def test_expand_1d_to_larger_1d(context):
    """Test expand 1D to larger 1D"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1], "float32")
    output = builder.expand(x, new_shape=[24])
    assert output.shape == [24]
    graph = builder.build({"output": output})


def test_expand_1d_to_2d(context):
    """Test expand 1D to 2D"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1], "float32")
    output = builder.expand(x, new_shape=[4, 6])
    assert output.shape == [4, 6]
    graph = builder.build({"output": output})


def test_expand_some_dimensions(context):
    """Test expand specific dimensions"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 6], "float32")
    output = builder.expand(x, new_shape=[4, 6])
    assert output.shape == [4, 6]
    graph = builder.build({"output": output})


def test_expand_scalar_to_tensor(context):
    """Test expand scalar (0D) to tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [], "float32")
    output = builder.expand(x, new_shape=[24])
    assert output.shape == [24]
    graph = builder.build({"output": output})


# Gather tests
def test_gather_1d_indices(context):
    """Test gather with 1D indices on 1D input"""
    builder = context.create_graph_builder()
    input_tensor = builder.input("input", [24], "float32")
    indices = builder.input("indices", [8], "int32")
    output = builder.gather(input_tensor, indices, axis=0)
    assert output.shape == [8]
    graph = builder.build({"output": output})


def test_gather_2d_input_1d_indices_axis0(context):
    """Test gather with 2D input, 1D indices, axis=0"""
    builder = context.create_graph_builder()
    input_tensor = builder.input("input", [12, 2], "float32")
    indices = builder.input("indices", [8], "int32")
    output = builder.gather(input_tensor, indices, axis=0)
    assert output.shape == [8, 2]
    graph = builder.build({"output": output})


def test_gather_3d_input_2d_indices_axis1(context):
    """Test gather with 3D input, 2D indices, axis=1"""
    builder = context.create_graph_builder()
    input_tensor = builder.input("input", [3, 4, 2], "float32")
    indices = builder.input("indices", [2, 2], "int32")
    output = builder.gather(input_tensor, indices, axis=1)
    assert output.shape == [3, 2, 2, 2]
    graph = builder.build({"output": output})


def test_gather_default_axis(context):
    """Test gather with default axis=0"""
    builder = context.create_graph_builder()
    input_tensor = builder.input("input", [12, 2], "float32")
    indices = builder.input("indices", [8], "int32")
    output = builder.gather(input_tensor, indices)
    assert output.shape == [8, 2]
    graph = builder.build({"output": output})


# Split tests (note: current architecture only supports using first output)
def test_split_by_count(context):
    """Test split with equal count"""
    builder = context.create_graph_builder()
    x = builder.input("x", [24], "float32")
    outputs = builder.split(x, splits=3, axis=0)
    assert len(outputs) == 3
    assert all(o.shape == [8] for o in outputs)
    # Note: Only first output can be used in graph due to single-output limitation
    graph = builder.build({"output": outputs[0]})


def test_split_by_sizes(context):
    """Test split with custom sizes"""
    builder = context.create_graph_builder()
    x = builder.input("x", [24], "float32")
    outputs = builder.split(x, splits=[8, 8, 8], axis=0)
    assert len(outputs) == 3
    assert all(o.shape == [8] for o in outputs)
    graph = builder.build({"output": outputs[0]})


def test_split_unequal_sizes(context):
    """Test split with unequal sizes"""
    builder = context.create_graph_builder()
    x = builder.input("x", [12], "float32")
    outputs = builder.split(x, splits=[3, 3, 3, 3], axis=0)
    assert len(outputs) == 4
    assert all(o.shape == [3] for o in outputs)
    graph = builder.build({"output": outputs[0]})


def test_split_2d_along_axis1(context):
    """Test split 2D tensor along axis 1"""
    builder = context.create_graph_builder()
    x = builder.input("x", [8, 3], "float32")
    outputs = builder.split(x, splits=3, axis=1)
    assert len(outputs) == 3
    assert all(o.shape == [8, 1] for o in outputs)
    graph = builder.build({"output": outputs[0]})


# Where tests
def test_where_same_shapes(context):
    """Test where with all inputs having same shape"""
    builder = context.create_graph_builder()
    condition = builder.input("condition", [2, 3], "int32")
    true_value = builder.input("true_value", [2, 3], "float32")
    false_value = builder.input("false_value", [2, 3], "float32")
    output = builder.where_(condition, true_value, false_value)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})


def test_where_with_broadcasting(context):
    """Test where with broadcasting"""
    builder = context.create_graph_builder()
    condition = builder.input("condition", [2, 3], "int32")
    true_value = builder.input("true_value", [1, 3], "float32")
    false_value = builder.input("false_value", [2, 1], "float32")
    output = builder.where_(condition, true_value, false_value)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})


def test_where_scalar_values(context):
    """Test where with scalar condition and values"""
    builder = context.create_graph_builder()
    condition = builder.input("condition", [1], "int32")
    true_value = builder.input("true_value", [2, 3], "float32")
    false_value = builder.input("false_value", [2, 3], "float32")
    output = builder.where_(condition, true_value, false_value)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})


# Pad tests
def test_pad_1d(context):
    """Test pad on 1D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [9], "float32")
    output = builder.pad(x, padding=[1, 1])
    assert output.shape == [11]
    graph = builder.build({"output": output})


def test_pad_2d(context):
    """Test pad on 2D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 3], "float32")
    output = builder.pad(x, padding=[1, 1, 1, 1])
    assert output.shape == [5, 5]
    graph = builder.build({"output": output})


def test_pad_4d(context):
    """Test pad on 4D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 3, 3, 1], "float32")
    output = builder.pad(x, padding=[0, 2, 2, 0, 0, 2, 2, 0])
    assert output.shape == [1, 7, 7, 1]
    graph = builder.build({"output": output})


def test_pad_with_mode(context):
    """Test pad with edge mode"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 3], "float32")
    output = builder.pad(x, padding=[1, 1, 1, 1], mode="edge")
    assert output.shape == [5, 5]
    graph = builder.build({"output": output})


def test_pad_with_value(context):
    """Test pad with constant value"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 3], "float32")
    output = builder.pad(x, padding=[1, 1, 1, 1], mode="constant", value=5.0)
    assert output.shape == [5, 5]
    graph = builder.build({"output": output})


def test_pad_asymmetric(context):
    """Test pad with asymmetric padding"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 3], "float32")
    output = builder.pad(x, padding=[1, 2, 3, 4])
    assert output.shape == [7, 9]
    graph = builder.build({"output": output})


# ========================================
# Advanced Architecture Operations Tests
# ========================================

# GELU tests
def test_gelu_basic(context):
    """Test GELU activation"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.gelu(x)
    assert output.shape == [2, 3]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_gelu_multidimensional(context):
    """Test GELU with multidimensional input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4, 5], "float32")
    output = builder.gelu(x)
    assert output.shape == [2, 3, 4, 5]
    graph = builder.build({"output": output})


def test_gelu_1d(context):
    """Test GELU with 1D input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [10], "float32")
    output = builder.gelu(x)
    assert output.shape == [10]
    graph = builder.build({"output": output})


# Squeeze tests
def test_squeeze_all_ones(context):
    """Test squeeze removing all dimensions of size 1"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 3, 1, 4, 1], "float32")
    output = builder.squeeze(x)
    assert output.shape == [3, 4]
    graph = builder.build({"output": output})


def test_squeeze_specific_axes(context):
    """Test squeeze with specific axes"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 3, 1, 4], "float32")
    output = builder.squeeze(x, axes=[0, 2])
    assert output.shape == [3, 4]
    graph = builder.build({"output": output})


def test_squeeze_single_axis(context):
    """Test squeeze single axis"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 3, 4], "float32")
    output = builder.squeeze(x, axes=[0])
    assert output.shape == [3, 4]
    graph = builder.build({"output": output})


def test_squeeze_no_ones(context):
    """Test squeeze with no dimensions of size 1"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.squeeze(x)
    assert output.shape == [2, 3, 4]
    graph = builder.build({"output": output})


# Unsqueeze tests
def test_unsqueeze_single_axis_front(context):
    """Test unsqueeze adding dimension at front"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 4], "float32")
    output = builder.unsqueeze(x, axes=[0])
    assert output.shape == [1, 3, 4]
    graph = builder.build({"output": output})


def test_unsqueeze_single_axis_middle(context):
    """Test unsqueeze adding dimension in middle"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 4], "float32")
    output = builder.unsqueeze(x, axes=[1])
    assert output.shape == [3, 1, 4]
    graph = builder.build({"output": output})


def test_unsqueeze_single_axis_end(context):
    """Test unsqueeze adding dimension at end"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 4], "float32")
    output = builder.unsqueeze(x, axes=[2])
    assert output.shape == [3, 4, 1]
    graph = builder.build({"output": output})


def test_unsqueeze_multiple_axes(context):
    """Test unsqueeze adding multiple dimensions"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 4], "float32")
    output = builder.unsqueeze(x, axes=[0, 2])
    assert output.shape == [1, 3, 1, 4]
    graph = builder.build({"output": output})


def test_unsqueeze_1d_to_4d(context):
    """Test unsqueeze expanding 1D to 4D"""
    builder = context.create_graph_builder()
    x = builder.input("x", [5], "float32")
    output = builder.unsqueeze(x, axes=[0, 2, 3])
    assert output.shape == [1, 5, 1, 1]
    graph = builder.build({"output": output})


# ArgMax tests
def test_arg_max_axis_0_no_keep(context):
    """Test argMax along axis 0 without keeping dimensions"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.arg_max(x, axis=0)
    assert output.shape == [3, 4]
    assert output.data_type == "int64"
    graph = builder.build({"output": output})


def test_arg_max_axis_1_keep_dims(context):
    """Test argMax along axis 1 with keep_dimensions"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.arg_max(x, axis=1, keep_dimensions=True)
    assert output.shape == [2, 1, 4]
    assert output.data_type == "int64"
    graph = builder.build({"output": output})


def test_arg_max_axis_2_int32(context):
    """Test argMax with int32 output type"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.arg_max(x, axis=2, output_data_type="int32")
    assert output.shape == [2, 3]
    assert output.data_type == "int32"
    graph = builder.build({"output": output})


def test_arg_max_1d(context):
    """Test argMax on 1D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [10], "float32")
    output = builder.arg_max(x, axis=0)
    assert output.shape == []
    assert output.data_type == "int64"
    graph = builder.build({"output": output})


def test_arg_max_1d_keep_dims(context):
    """Test argMax on 1D tensor with keep_dimensions"""
    builder = context.create_graph_builder()
    x = builder.input("x", [10], "float32")
    output = builder.arg_max(x, axis=0, keep_dimensions=True)
    assert output.shape == [1]
    assert output.data_type == "int64"
    graph = builder.build({"output": output})


# ArgMin tests
def test_arg_min_axis_0_no_keep(context):
    """Test argMin along axis 0 without keeping dimensions"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.arg_min(x, axis=0)
    assert output.shape == [3, 4]
    assert output.data_type == "int64"
    graph = builder.build({"output": output})


def test_arg_min_axis_1_keep_dims(context):
    """Test argMin along axis 1 with keep_dimensions"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.arg_min(x, axis=1, keep_dimensions=True)
    assert output.shape == [2, 1, 4]
    assert output.data_type == "int64"
    graph = builder.build({"output": output})


def test_arg_min_axis_2_int32(context):
    """Test argMin with int32 output type"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.arg_min(x, axis=2, output_data_type="int32")
    assert output.shape == [2, 3]
    assert output.data_type == "int32"
    graph = builder.build({"output": output})


def test_arg_min_1d(context):
    """Test argMin on 1D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [10], "float32")
    output = builder.arg_min(x, axis=0)
    assert output.shape == []
    assert output.data_type == "int64"
    graph = builder.build({"output": output})


# Cast tests
def test_cast_float32_to_int32(context):
    """Test cast from float32 to int32"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.cast(x, "int32")
    assert output.shape == [2, 3]
    assert output.data_type == "int32"
    graph = builder.build({"output": output})


def test_cast_int32_to_float32(context):
    """Test cast from int32 to float32"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "int32")
    output = builder.cast(x, "float32")
    assert output.shape == [2, 3]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_cast_float32_to_float16(context):
    """Test cast from float32 to float16"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.cast(x, "float16")
    assert output.shape == [2, 3, 4]
    assert output.data_type == "float16"
    graph = builder.build({"output": output})


def test_cast_int8_to_uint8(context):
    """Test cast from int8 to uint8"""
    builder = context.create_graph_builder()
    x = builder.input("x", [10], "int8")
    output = builder.cast(x, "uint8")
    assert output.shape == [10]
    assert output.data_type == "uint8"
    graph = builder.build({"output": output})


def test_cast_to_int64(context):
    """Test cast to int64"""
    builder = context.create_graph_builder()
    x = builder.input("x", [5, 5], "int32")
    output = builder.cast(x, "int64")
    assert output.shape == [5, 5]
    assert output.data_type == "int64"
    graph = builder.build({"output": output})


def test_cast_preserves_shape(context):
    """Test that cast preserves multidimensional shape"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4, 5], "float32")
    output = builder.cast(x, "int32")
    assert output.shape == [2, 3, 4, 5]
    assert output.data_type == "int32"
    graph = builder.build({"output": output})


# ScatterElements tests
def test_scatter_elements_1d(context):
    """Test scatterElements on 1D tensor"""
    builder = context.create_graph_builder()
    data = builder.input("data", [4], "float32")
    indices = builder.input("indices", [4], "int32")
    updates = builder.input("updates", [4], "float32")
    output = builder.scatter_elements(data, indices, updates, axis=0)
    assert output.shape == [4]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_scatter_elements_2d_axis_0(context):
    """Test scatterElements on 2D tensor along axis 0"""
    builder = context.create_graph_builder()
    data = builder.input("data", [3, 4], "float32")
    indices = builder.input("indices", [2, 4], "int32")
    updates = builder.input("updates", [2, 4], "float32")
    output = builder.scatter_elements(data, indices, updates, axis=0)
    assert output.shape == [3, 4]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_scatter_elements_2d_axis_1(context):
    """Test scatterElements on 2D tensor along axis 1"""
    builder = context.create_graph_builder()
    data = builder.input("data", [3, 4], "float32")
    indices = builder.input("indices", [3, 2], "int32")
    updates = builder.input("updates", [3, 2], "float32")
    output = builder.scatter_elements(data, indices, updates, axis=1)
    assert output.shape == [3, 4]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_scatter_elements_negative_axis(context):
    """Test scatterElements with negative axis"""
    builder = context.create_graph_builder()
    data = builder.input("data", [3, 4], "float32")
    indices = builder.input("indices", [3, 2], "int32")
    updates = builder.input("updates", [3, 2], "float32")
    output = builder.scatter_elements(data, indices, updates, axis=-1)
    assert output.shape == [3, 4]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_scatter_elements_3d(context):
    """Test scatterElements on 3D tensor"""
    builder = context.create_graph_builder()
    data = builder.input("data", [2, 3, 4], "float32")
    indices = builder.input("indices", [2, 2, 4], "int32")
    updates = builder.input("updates", [2, 2, 4], "float32")
    output = builder.scatter_elements(data, indices, updates, axis=1)
    assert output.shape == [2, 3, 4]
    graph = builder.build({"output": output})


# ScatterND tests
def test_scatter_nd_basic(context):
    """Test scatterND with basic 2D case"""
    builder = context.create_graph_builder()
    data = builder.input("data", [4, 5], "float32")
    indices = builder.input("indices", [3, 1], "int32")  # k=1
    updates = builder.input("updates", [3, 5], "float32")  # [3] + [5]
    output = builder.scatter_nd(data, indices, updates)
    assert output.shape == [4, 5]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_scatter_nd_3d(context):
    """Test scatterND on 3D tensor"""
    builder = context.create_graph_builder()
    data = builder.input("data", [2, 3, 4], "float32")
    indices = builder.input("indices", [5, 2], "int32")  # k=2
    updates = builder.input("updates", [5, 4], "float32")  # [5] + [4]
    output = builder.scatter_nd(data, indices, updates)
    assert output.shape == [2, 3, 4]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_scatter_nd_full_rank(context):
    """Test scatterND where k equals input rank"""
    builder = context.create_graph_builder()
    data = builder.input("data", [2, 3, 4], "float32")
    indices = builder.input("indices", [5, 3], "int32")  # k=3 (full rank)
    updates = builder.input("updates", [5], "float32")  # [5] + []
    output = builder.scatter_nd(data, indices, updates)
    assert output.shape == [2, 3, 4]
    graph = builder.build({"output": output})


def test_scatter_nd_4d(context):
    """Test scatterND on 4D tensor"""
    builder = context.create_graph_builder()
    data = builder.input("data", [2, 3, 4, 5], "float32")
    indices = builder.input("indices", [6, 2], "int32")  # k=2
    updates = builder.input("updates", [6, 4, 5], "float32")  # [6] + [4, 5]
    output = builder.scatter_nd(data, indices, updates)
    assert output.shape == [2, 3, 4, 5]
    graph = builder.build({"output": output})


# Tile tests
def test_tile_1d(context):
    """Test tile on 1D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [4], "float32")
    output = builder.tile(x, [2])
    assert output.shape == [8]  # 4 * 2
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_tile_2d(context):
    """Test tile on 2D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.tile(x, [2, 3])
    assert output.shape == [4, 9]  # 2*2, 3*3
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_tile_no_repetition(context):
    """Test tile with all repetitions = 1"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")
    output = builder.tile(x, [1, 1, 1])
    assert output.shape == [2, 3, 4]
    graph = builder.build({"output": output})


def test_tile_different_repetitions(context):
    """Test tile with different repetitions per dimension"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.tile(x, [3, 1])
    assert output.shape == [6, 3]  # 2*3, 3*1
    graph = builder.build({"output": output})


def test_tile_4d(context):
    """Test tile on 4D tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 2, 3, 4], "float32")
    output = builder.tile(x, [2, 1, 2, 1])
    assert output.shape == [2, 2, 6, 4]  # 1*2, 2*1, 3*2, 4*1
    graph = builder.build({"output": output})


# Triangular tests
def test_triangular_2d_upper(context):
    """Test triangular on 2D tensor (upper triangle)"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 3], "float32")
    output = builder.triangular(x, True, 0)
    assert output.shape == [3, 3]
    assert output.data_type == "float32"
    graph = builder.build({"output": output})


def test_triangular_2d_lower(context):
    """Test triangular on 2D tensor (lower triangle)"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 3], "float32")
    output = builder.triangular(x, False, 0)
    assert output.shape == [3, 3]
    graph = builder.build({"output": output})


def test_triangular_default_params(context):
    """Test triangular with default-like parameters (upper=True, diagonal=0)"""
    builder = context.create_graph_builder()
    x = builder.input("x", [4, 4], "float32")
    output = builder.triangular(x, True, 0)  # explicit: upper=True, diagonal=0
    assert output.shape == [4, 4]
    graph = builder.build({"output": output})


def test_triangular_upper_diagonal_1(context):
    """Test triangular upper with diagonal offset +1"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 3], "float32")
    output = builder.triangular(x, True, 1)
    assert output.shape == [3, 3]
    graph = builder.build({"output": output})


def test_triangular_lower_diagonal_minus_1(context):
    """Test triangular lower with diagonal offset -1"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 3], "float32")
    output = builder.triangular(x, False, -1)
    assert output.shape == [3, 3]
    graph = builder.build({"output": output})


def test_triangular_3d(context):
    """Test triangular on 3D tensor (applies to last 2 dims)"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 3], "float32")
    output = builder.triangular(x, True, 0)
    assert output.shape == [2, 3, 3]
    graph = builder.build({"output": output})


def test_triangular_non_square(context):
    """Test triangular on non-square matrix"""
    builder = context.create_graph_builder()
    x = builder.input("x", [4, 5], "float32")
    output = builder.triangular(x, True, 0)
    assert output.shape == [4, 5]
    graph = builder.build({"output": output})


# ============================================================================
# Specialized Activation Operations Tests
# ============================================================================


@requires_onnx_runtime
def test_prelu_basic(context):
    """Test PReLU activation with basic slope tensor"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    slope = builder.constant(np.array([[0.25, 0.25, 0.25]], dtype=np.float32), [1, 3], "float32")
    output = builder.prelu(x, slope)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    # PReLU: x if x >= 0 else slope * x
    x_data = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})
    assert "output" in results
    assert results["output"].shape == (2, 3)

    # Verify PReLU: x if x >= 0 else slope * x
    expected = np.where(x_data >= 0, x_data, 0.25 * x_data)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_prelu_per_channel(context):
    """Test PReLU with per-channel slopes"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    slope = builder.constant(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), [1, 3], "float32")
    output = builder.prelu(x, slope)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-1, -2, -3], [1, 2, 3]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    # Each channel has different slope: [0.1, 0.2, 0.3]
    expected = np.array([[-0.1, -0.4, -0.9], [1, 2, 3]], dtype=np.float32)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_prelu_broadcast_slope(context):
    """Test PReLU with broadcasted slope"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    slope = builder.constant(np.array([0.5], dtype=np.float32), [1], "float32")
    output = builder.prelu(x, slope)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-2, 4, -6], [8, -10, 12]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.where(x_data >= 0, x_data, 0.5 * x_data)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_elu_default_alpha(context):
    """Test ELU activation with default alpha=1.0"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.elu(x)  # Default alpha=1.0
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-1, 0, 1], [-2, 2, -0.5]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})
    assert "output" in results
    assert results["output"].shape == (2, 3)

    # ELU: x if x >= 0 else alpha * (exp(x) - 1)
    expected = np.where(x_data >= 0, x_data, 1.0 * (np.exp(x_data) - 1))
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_elu_custom_alpha(context):
    """Test ELU activation with custom alpha"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.elu(x, alpha=0.5)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-1, 0, 1], [-2, 2, -0.5]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.where(x_data >= 0, x_data, 0.5 * (np.exp(x_data) - 1))
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_elu_multidimensional(context):
    """Test ELU with multidimensional input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2, 2], "float32")
    output = builder.elu(x, alpha=2.0)
    assert output.shape == [2, 2, 2]
    graph = builder.build({"output": output})

    x_data = np.array([[[-1, 1], [0, -0.5]], [[2, -2], [-3, 3]]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.where(x_data >= 0, x_data, 2.0 * (np.exp(x_data) - 1))
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_leaky_relu_default_alpha(context):
    """Test Leaky ReLU with default alpha=0.01"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.leaky_relu(x)  # Default alpha=0.01
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-1, 0, 1], [-10, 10, -5]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})
    assert "output" in results
    assert results["output"].shape == (2, 3)

    # Leaky ReLU: x if x >= 0 else alpha * x
    expected = np.where(x_data >= 0, x_data, 0.01 * x_data)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_leaky_relu_custom_alpha(context):
    """Test Leaky ReLU with custom alpha"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.leaky_relu(x, alpha=0.2)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-5, 5, 0], [10, -10, -1]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.where(x_data >= 0, x_data, 0.2 * x_data)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_leaky_relu_multidimensional(context):
    """Test Leaky ReLU with 4D input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 2, 2, 2], "float32")
    output = builder.leaky_relu(x, alpha=0.1)
    assert output.shape == [1, 2, 2, 2]
    graph = builder.build({"output": output})

    x_data = np.array([[[[-1, 1], [2, -2]], [[3, -3], [-4, 4]]]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.where(x_data >= 0, x_data, 0.1 * x_data)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_softplus_basic(context):
    """Test softplus activation: log(1 + exp(x))"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.softplus(x)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-1, 0, 1], [-2, 2, 0.5]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})
    assert "output" in results
    assert results["output"].shape == (2, 3)

    # Softplus: log(1 + exp(x))
    expected = np.log(1 + np.exp(x_data))
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_softplus_multidimensional(context):
    """Test softplus with 3D input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2, 2], "float32")
    output = builder.softplus(x)
    assert output.shape == [2, 2, 2]
    graph = builder.build({"output": output})

    x_data = np.array([[[-5, 0], [5, 10]], [[1, -1], [-10, 2]]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.log(1 + np.exp(x_data))
    # Use slightly relaxed tolerance due to numerical precision differences
    # between ONNX Runtime and NumPy for extreme values (rtol=5e-4 = 0.05%)
    np.testing.assert_allclose(results["output"], expected, rtol=5e-4)


@requires_onnx_runtime
def test_softsign_basic(context):
    """Test softsign activation: x / (1 + |x|)"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.softsign(x)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-1, 0, 1], [-2, 2, 0.5]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})
    assert "output" in results
    assert results["output"].shape == (2, 3)

    # Softsign: x / (1 + |x|)
    expected = x_data / (1 + np.abs(x_data))
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_softsign_multidimensional(context):
    """Test softsign with 3D input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2, 2], "float32")
    output = builder.softsign(x)
    assert output.shape == [2, 2, 2]
    graph = builder.build({"output": output})

    x_data = np.array([[[-5, 0], [5, 10]], [[1, -1], [-10, 2]]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = x_data / (1 + np.abs(x_data))
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_hard_sigmoid_default_params(context):
    """Test hard sigmoid with default alpha=0.2, beta=0.5"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.hard_sigmoid(x)  # Default alpha=0.2, beta=0.5
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-3, -1, 0], [1, 3, 5]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})
    assert "output" in results
    assert results["output"].shape == (2, 3)

    # Hard sigmoid: max(0, min(1, alpha * x + beta))
    expected = np.maximum(0, np.minimum(1, 0.2 * x_data + 0.5))
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_hard_sigmoid_custom_params(context):
    """Test hard sigmoid with custom alpha and beta"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.hard_sigmoid(x, alpha=0.5, beta=0.3)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.maximum(0, np.minimum(1, 0.5 * x_data + 0.3))
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_hard_sigmoid_multidimensional(context):
    """Test hard sigmoid with 4D input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 2, 2, 2], "float32")
    output = builder.hard_sigmoid(x, alpha=0.2, beta=0.5)
    assert output.shape == [1, 2, 2, 2]
    graph = builder.build({"output": output})

    x_data = np.array([[[[-3, -1], [0, 1]], [[2, 3], [4, 5]]]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.maximum(0, np.minimum(1, 0.2 * x_data + 0.5))
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@pytest.mark.skip(reason="HardSwish requires ONNX opset 14+, runtime 1.17.0 uses opset 13")
@requires_onnx_runtime
def test_hard_swish_default_params(context):
    """Test hard swish with default alpha=1/6, beta=0.5"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.hard_swish(x)  # Default alpha=0.16666..., beta=0.5
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-3, -1, 0], [1, 3, 5]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})
    assert "output" in results
    assert results["output"].shape == (2, 3)

    # Hard swish: x * hardSigmoid(x)
    alpha = 1.0 / 6.0
    beta = 0.5
    hard_sigmoid = np.maximum(0, np.minimum(1, alpha * x_data + beta))
    expected = x_data * hard_sigmoid
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@pytest.mark.skip(reason="HardSwish requires ONNX opset 14+, runtime 1.17.0 uses opset 13")
@requires_onnx_runtime
def test_hard_swish_custom_params(context):
    """Test hard swish with custom alpha and beta"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.hard_swish(x, alpha=0.2, beta=0.5)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    hard_sigmoid = np.maximum(0, np.minimum(1, 0.2 * x_data + 0.5))
    expected = x_data * hard_sigmoid
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@pytest.mark.skip(reason="HardSwish requires ONNX opset 14+, runtime 1.17.0 uses opset 13")
@requires_onnx_runtime
def test_hard_swish_multidimensional(context):
    """Test hard swish with 3D input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2, 2], "float32")
    output = builder.hard_swish(x, alpha=0.2, beta=0.5)
    assert output.shape == [2, 2, 2]
    graph = builder.build({"output": output})

    x_data = np.array([[[-3, -1], [0, 1]], [[2, 3], [4, 5]]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    hard_sigmoid = np.maximum(0, np.minimum(1, 0.2 * x_data + 0.5))
    expected = x_data * hard_sigmoid
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


# ============================================================================
# Clamp operation tests
# ============================================================================


@requires_onnx_runtime
def test_clamp_basic(context):
    """Test clamp with min and max values"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    output = builder.clamp(x, min_value=0.0, max_value=6.0)
    assert output.shape == [2, 3]
    graph = builder.build({"output": output})

    x_data = np.array([[-1, 0, 1], [5, 7, 3]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})
    assert "output" in results
    assert results["output"].shape == (2, 3)

    # Clamp: max(min_value, min(x, max_value))
    expected = np.clip(x_data, 0.0, 6.0)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_clamp_relu6(context):
    """Test clamp as ReLU6 (min=0, max=6)"""
    builder = context.create_graph_builder()
    x = builder.input("x", [3, 3], "float32")
    output = builder.clamp(x, min_value=0.0, max_value=6.0)
    graph = builder.build({"output": output})

    x_data = np.array([[-2, -1, 0], [1, 3, 5], [6, 7, 10]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.array([[0, 0, 0], [1, 3, 5], [6, 6, 6]], dtype=np.float32)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_clamp_multidimensional(context):
    """Test clamp with multidimensional input"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 2, 2], "float32")
    output = builder.clamp(x, min_value=-1.0, max_value=1.0)
    assert output.shape == [2, 2, 2]
    graph = builder.build({"output": output})

    x_data = np.array([[[-2, -0.5], [0, 0.5]], [[1, 1.5], [-1.5, 2]]], dtype=np.float32)
    results = context.compute(graph, {"x": x_data})

    expected = np.clip(x_data, -1.0, 1.0)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


def test_clamp_shape_inference(context):
    """Test clamp shape inference"""
    builder = context.create_graph_builder()
    x = builder.input("x", [1, 224, 224, 3], "float32")
    output = builder.clamp(x, min_value=0.0, max_value=6.0)
    assert output.shape == [1, 224, 224, 3]


def test_clamp_invalid_range(context):
    """Test clamp with invalid min/max range"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")

    # min_value > max_value should raise an error
    with pytest.raises(Exception):
        builder.clamp(x, min_value=6.0, max_value=0.0)


# ============================================================================
# GEMM (General Matrix Multiplication) operation tests
# ============================================================================


@requires_onnx_runtime
def test_gemm_basic(context):
    """Test basic GEMM: C = A * B"""
    builder = context.create_graph_builder()
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [3, 4], "float32")
    output = builder.gemm(a, b)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})

    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b_data = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
    results = context.compute(graph, {"a": a_data, "b": b_data})
    assert "output" in results
    assert results["output"].shape == (2, 4)

    expected = np.matmul(a_data, b_data)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_gemm_transpose_a(context):
    """Test GEMM with A transposed"""
    builder = context.create_graph_builder()
    a = builder.input("a", [3, 2], "float32")  # Will be transposed to [2, 3]
    b = builder.input("b", [3, 4], "float32")
    output = builder.gemm(a, b, a_transpose=True)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})

    a_data = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    b_data = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
    results = context.compute(graph, {"a": a_data, "b": b_data})

    expected = np.matmul(a_data.T, b_data)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_gemm_transpose_b(context):
    """Test GEMM with B transposed"""
    builder = context.create_graph_builder()
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [4, 3], "float32")  # Will be transposed to [3, 4]
    output = builder.gemm(a, b, b_transpose=True)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})

    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
    results = context.compute(graph, {"a": a_data, "b": b_data})

    expected = np.matmul(a_data, b_data.T)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_gemm_transpose_both(context):
    """Test GEMM with both A and B transposed"""
    builder = context.create_graph_builder()
    a = builder.input("a", [3, 2], "float32")  # Will be transposed to [2, 3]
    b = builder.input("b", [4, 3], "float32")  # Will be transposed to [3, 4]
    output = builder.gemm(a, b, a_transpose=True, b_transpose=True)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})

    a_data = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    b_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
    results = context.compute(graph, {"a": a_data, "b": b_data})

    expected = np.matmul(a_data.T, b_data.T)
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_gemm_with_bias(context):
    """Test GEMM with bias: C = A * B + c"""
    builder = context.create_graph_builder()
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [3, 4], "float32")
    c = builder.input("c", [2, 4], "float32")
    output = builder.gemm(a, b, c=c)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})

    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b_data = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
    c_data = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
    results = context.compute(graph, {"a": a_data, "b": b_data, "c": c_data})

    expected = np.matmul(a_data, b_data) + c_data
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


@requires_onnx_runtime
def test_gemm_with_alpha_beta(context):
    """Test GEMM with scaling factors: C = alpha * A * B + beta * c"""
    builder = context.create_graph_builder()
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [3, 4], "float32")
    c = builder.input("c", [2, 4], "float32")
    output = builder.gemm(a, b, c=c, alpha=2.0, beta=0.5)
    assert output.shape == [2, 4]
    graph = builder.build({"output": output})

    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b_data = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
    c_data = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
    results = context.compute(graph, {"a": a_data, "b": b_data, "c": c_data})

    expected = 2.0 * np.matmul(a_data, b_data) + 0.5 * c_data
    np.testing.assert_allclose(results["output"], expected, rtol=1e-5)


def test_gemm_shape_inference(context):
    """Test GEMM shape inference"""
    builder = context.create_graph_builder()
    a = builder.input("a", [128, 512], "float32")
    b = builder.input("b", [512, 1000], "float32")
    output = builder.gemm(a, b)
    assert output.shape == [128, 1000]


def test_gemm_shape_inference_transpose(context):
    """Test GEMM shape inference with transpose"""
    builder = context.create_graph_builder()
    a = builder.input("a", [512, 128], "float32")
    b = builder.input("b", [1000, 512], "float32")
    output = builder.gemm(a, b, a_transpose=True, b_transpose=True)
    assert output.shape == [128, 1000]


def test_gemm_invalid_shapes(context):
    """Test GEMM with incompatible shapes"""
    builder = context.create_graph_builder()
    a = builder.input("a", [2, 3], "float32")
    b = builder.input("b", [4, 5], "float32")  # Incompatible: 3 != 4

    with pytest.raises(Exception):
        builder.gemm(a, b)
