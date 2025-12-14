"""
WPT WebNN Conformance Tests

This module runs W3C Web Platform Tests (WPT) for WebNN conformance against
the rustnn implementation. It loads test data converted from the official WPT
test suite and validates that our implementation produces correct results.

Test data is loaded from tests/wpt_data/conformance/*.json files.

Usage:
    # Run all WPT conformance tests
    pytest tests/test_wpt_conformance.py -v

    # Run tests for specific operation
    pytest tests/test_wpt_conformance.py -k "reduce_sum" -v

    # Run with detailed failure output
    pytest tests/test_wpt_conformance.py -vv --tb=short
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import wpt_utils


# Directory containing WPT test data
WPT_DATA_DIR = Path(__file__).parent / "wpt_data" / "conformance"


def discover_wpt_operations() -> List[str]:
    """Discover all operations that have WPT test data available."""
    if not WPT_DATA_DIR.exists():
        return []

    operations = []
    for json_file in WPT_DATA_DIR.glob("*.json"):
        operations.append(json_file.stem)

    return sorted(operations)


def load_test_cases_for_operation(operation: str) -> List[Dict[str, Any]]:
    """Load all test cases for a given operation."""
    try:
        test_data = wpt_utils.load_wpt_test_data(operation, "conformance")
        return test_data.get("tests", [])
    except FileNotFoundError:
        pytest.skip(f"No WPT test data for {operation}")
        return []


def execute_wpt_test_case(context, test_case: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Execute a single WPT test case using the WebNN API.

    Args:
        context: MLContext instance
        test_case: Test case dictionary from WPT data

    Returns:
        Dictionary mapping output names to NumPy arrays
    """
    builder = context.create_graph_builder()

    # Create operands dictionary to track created operands by name
    operands: Dict[str, Any] = {}

    # Extract graph description from test case
    graph_desc = test_case.get("graph", {})

    # Step 1: Create input operands
    inputs_data = graph_desc.get("inputs", {})
    for input_name, input_spec in inputs_data.items():
        descriptor = input_spec.get("descriptor", input_spec)
        shape = descriptor["shape"]
        dtype = descriptor.get("dataType", "float32")
        operands[input_name] = builder.input(input_name, shape, dtype)

    # Step 2: Execute operators in order
    operators = graph_desc.get("operators", [])
    for op_spec in operators:
        op_name = op_spec["name"]
        op_args_raw = op_spec.get("arguments", {})
        op_output = op_spec.get("outputs", op_spec.get("output", "output"))

        # Convert arguments from list of dicts to single dict if needed
        if isinstance(op_args_raw, list):
            op_args = {}
            for arg_dict in op_args_raw:
                op_args.update(arg_dict)
        else:
            op_args = op_args_raw

        # Flatten 'options' dict if present (used by conv2d, conv_transpose2d, etc.)
        if "options" in op_args and isinstance(op_args["options"], dict):
            options = op_args.pop("options")
            op_args.update(options)

        # Resolve operand references in arguments
        resolved_args = {}
        for arg_name, arg_value in op_args.items():
            if isinstance(arg_value, str) and arg_value in operands:
                # This is a reference to another operand
                resolved_args[arg_name] = operands[arg_value]
            elif isinstance(arg_value, list):
                # This is a list of operand references (e.g., concat inputs)
                resolved_list = []
                for item in arg_value:
                    if isinstance(item, str) and item in operands:
                        resolved_list.append(operands[item])
                    else:
                        resolved_list.append(item)
                resolved_args[arg_name] = resolved_list
            else:
                # This is a literal value
                resolved_args[arg_name] = arg_value

        # Call the appropriate builder method
        result = call_builder_method(builder, op_name, resolved_args)

        # Store result operand(s)
        # Some operations (like split) return multiple outputs
        if isinstance(op_output, list):
            # Multiple outputs - result should also be a list
            if isinstance(result, (list, tuple)):
                for idx, output_name in enumerate(op_output):
                    operands[output_name] = result[idx]
            else:
                # Single result but multiple output names - store same result for all
                for output_name in op_output:
                    operands[output_name] = result
        else:
            # Single output
            operands[op_output] = result

    # Step 3: Build graph with outputs
    expected_outputs = graph_desc.get("expectedOutputs", {})
    graph_outputs = {}
    for output_name in expected_outputs.keys():
        if output_name in operands:
            graph_outputs[output_name] = operands[output_name]

    if not graph_outputs:
        raise ValueError("No outputs specified in test case")

    # Build the graph
    graph = builder.build(graph_outputs)

    # Step 4: Prepare input data for compute()
    compute_inputs = {}
    for input_name, input_spec in inputs_data.items():
        input_array = wpt_utils.numpy_array_from_test_data(input_spec)
        compute_inputs[input_name] = input_array

    # Step 5: Execute graph with compute()
    compute_results = context.compute(graph, compute_inputs)

    # Step 6: Convert results to NumPy arrays
    results = {}
    for output_name in graph_outputs.keys():
        if output_name in compute_results:
            results[output_name] = compute_results[output_name]

    return results


def call_builder_method(builder, op_name: str, args: Dict[str, Any]) -> Any:
    """
    Call a builder method by name with the given arguments.

    Args:
        builder: MLGraphBuilder instance
        op_name: Operation name (e.g., "reduce_sum", "relu")
        args: Arguments dictionary

    Returns:
        Resulting MLOperand
    """
    # Map WPT parameter names to Python API parameter names (camelCase to snake_case)
    param_name_map = {
        # General
        "newShape": "new_shape",  # expand operation
        "type": "data_type",  # cast operation
        "keepDimensions": "keep_dimensions",  # reduction operations
        # Clamp operation
        "minValue": "min_value",
        "maxValue": "max_value",
        # Conv2d/ConvTranspose2d
        "padding": "pads",
        "inputLayout": "input_layout",
        "filterLayout": "filter_layout",
        "outputSizes": "output_sizes",
        "outputPadding": "output_padding",
    }

    # Remap parameter names
    mapped_args = {}
    for key, value in args.items():
        mapped_key = param_name_map.get(key, key)
        mapped_args[mapped_key] = value
    args = mapped_args

    # Special handling for operations that use positional arguments
    if op_name == "reshape":
        # reshape(input, new_shape)
        input_op = args.get("input", args.get("a"))
        new_shape = args.get("new_shape")
        method = getattr(builder, "reshape")
        return method(input_op, new_shape)

    elif op_name == "softmax":
        # softmax(input, axis=None)
        input_op = args.get("input", args.get("x"))
        axis = args.get("axis")
        method = getattr(builder, "softmax")
        if axis is not None:
            # axis parameter not yet supported, skip for now
            raise NotImplementedError(f"softmax with axis={axis} not yet supported")
        return method(input_op)

    # Map operation names to builder method names (WPT uses camelCase)
    method_name_map = {
        # Reduction operations
        "reduceSum": "reduce_sum",
        "reduceMean": "reduce_mean",
        "reduceMax": "reduce_max",
        "reduceMin": "reduce_min",
        "reduceProduct": "reduce_product",
        "reduceL1": "reduce_l1",
        "reduceL2": "reduce_l2",
        "reduceLogSum": "reduce_log_sum",
        "reduceLogSumExp": "reduce_log_sum_exp",
        "reduceSumSquare": "reduce_sum_square",
        # Activation functions
        "relu": "relu",
        "sigmoid": "sigmoid",
        "tanh": "tanh",
        "softmax": "softmax",
        "leakyRelu": "leaky_relu",
        "hardSigmoid": "hard_sigmoid",
        "hardSwish": "hard_swish",
        "elu": "elu",
        "gelu": "gelu",
        "prelu": "prelu",
        "softplus": "softplus",
        "softsign": "softsign",
        # Normalization operations
        "batchNormalization": "batch_normalization",
        "instanceNormalization": "instance_normalization",
        "layerNormalization": "layer_normalization",
        # Convolution operations
        "conv2d": "conv2d",
        "convTranspose2d": "conv_transpose2d",
        # Pooling operations
        "averagePool2d": "average_pool2d",
        "maxPool2d": "max_pool2d",
        "globalAveragePool": "global_average_pool",
        "globalMaxPool": "global_max_pool",
        # Binary operations
        "add": "add",
        "sub": "sub",
        "mul": "mul",
        "div": "div",
        "matmul": "matmul",
        # Comparison operations
        "equal": "equal",
        "greater": "greater",
        "greaterOrEqual": "greater_or_equal",
        "lesser": "lesser",
        "lesserOrEqual": "lesser_or_equal",
        # Logical operations
        "logicalAnd": "logical_and",
        "logicalOr": "logical_or",
        "logicalNot": "logical_not",
        "logicalXor": "logical_xor",
        # Unary operations
        "abs": "abs",
        "ceil": "ceil",
        "cos": "cos",
        "exp": "exp",
        "floor": "floor",
        "log": "log",
        "neg": "neg",
        "reciprocal": "reciprocal",
        "sign": "sign",
        "sin": "sin",
        "sqrt": "sqrt",
        "tan": "tan",
        "acos": "acos",
        "asin": "asin",
        "atan": "atan",
        "acosh": "acosh",
        "asinh": "asinh",
        "atanh": "atanh",
        "cosh": "cosh",
        "sinh": "sinh",
        "erf": "erf",
        "round": "round",
        # Shape operations
        "reshape": "reshape",
        "transpose": "transpose",
        "concat": "concat",
        "expand": "expand",
        "gather": "gather",
        "pad": "pad",
        "slice": "slice",
        "split": "split",
        "squeeze": "squeeze",
        "unsqueeze": "unsqueeze",
        "tile": "tile",
        # Other operations
        "cast": "cast",
        "clamp": "clamp",
        "gemm": "gemm",
        "where": "where_",
        "identity": "identity",
        "quantizeLinear": "quantize_linear",
        "dequantizeLinear": "dequantize_linear",
        "scatterElements": "scatter_elements",
        "scatterND": "scatter_nd",
        "triangular": "triangular",
        "argMax": "arg_max",
        "argMin": "arg_min",
    }

    method_name = method_name_map.get(op_name, op_name)

    if not hasattr(builder, method_name):
        pytest.skip(f"Operation {op_name} not implemented")

    method = getattr(builder, method_name)

    # Handle different argument patterns
    # For operations with a single input operand
    if "input" in args and len(args) == 1:
        return method(args["input"])

    # For unary operations with 'a' parameter (like logical_not)
    if "a" in args and len(args) == 1:
        return method(args["a"])

    # For operations with options (like reduction ops, clamp)
    if "input" in args:
        input_operand = args["input"]
        # Filter out None values (they mean "use default")
        options = {k: v for k, v in args.items() if k != "input" and v is not None}

        # Handle special option name mappings
        if "keepDimensions" in options:
            options["keep_dimensions"] = options.pop("keepDimensions")

        return method(input_operand, **options)

    # For binary operations
    if "a" in args and "b" in args:
        remaining = {k: v for k, v in args.items() if k not in ["a", "b"]}
        return method(args["a"], args["b"], **remaining)

    # Fallback: try calling with all args as kwargs
    return method(**args)


def generate_test_id(operation: str, test_case: Dict[str, Any]) -> str:
    """Generate a pytest test ID for a test case."""
    test_name = test_case.get("name", "unnamed")
    # Sanitize name for pytest
    test_id = f"{operation}::{test_name}".replace(" ", "_")
    return test_id


# Pytest fixtures and test generation
@pytest.fixture(scope="module")
def available_operations():
    """Fixture providing list of operations with WPT test data."""
    operations = discover_wpt_operations()
    if not operations:
        pytest.skip("No WPT test data found. Run: ./scripts/update_wpt_tests.sh")
    return operations


def pytest_generate_tests(metafunc):
    """
    Dynamically generate tests from WPT test data.

    This hook is called by pytest to parameterize test functions.
    Backend parametrization is handled automatically by the context fixture.
    """
    if "wpt_test_case" in metafunc.fixturenames:
        # Discover all operations
        operations = discover_wpt_operations()

        if not operations:
            # No test data - create a single skip test
            metafunc.parametrize(
                "wpt_test_case,operation",
                [(None, None)],
                ids=["no_wpt_data"]
            )
            return

        # Generate test parameters for all operations and their test cases
        test_params = []
        test_ids = []

        for operation in operations:
            test_cases = load_test_cases_for_operation(operation)

            if not test_cases:
                # Operation file exists but has no test cases
                test_params.append((None, operation))
                test_ids.append(f"{operation}::no_tests")
                continue

            for test_case in test_cases:
                test_params.append((test_case, operation))
                test_ids.append(generate_test_id(operation, test_case))

        metafunc.parametrize(
            "wpt_test_case,operation",
            test_params,
            ids=test_ids
        )


def test_wpt_conformance(context, backend_name, wpt_test_case, operation):
    """
    Run a single WPT conformance test.

    This test is parameterized by pytest_generate_tests to run all WPT test cases.
    """
    if wpt_test_case is None and operation is None:
        pytest.skip("No WPT test data found. Run: ./scripts/update_wpt_tests.sh")

    if wpt_test_case is None:
        pytest.skip(f"No test cases for {operation} (may require manual conversion)")

    # All data types and tensor sizes should be supported
    test_name = wpt_test_case.get("name", "")

    # Skip Float16 tensors with large arrays on CoreML (CoreML/ANE platform limitation)
    # See: FLOAT16_STATUS.md - CoreML crashes with Float16 inputs/constants/outputs >4 elements
    # Chromium also skips Float16 tests on Mac: #if !BUILDFLAG(IS_MAC)
    if backend_name == "coreml":
        graph_desc = wpt_test_case.get("graph", {})

        # Check inputs
        inputs_data = graph_desc.get("inputs", {})
        for input_name, input_spec in inputs_data.items():
            descriptor = input_spec.get("descriptor", input_spec)
            dtype = descriptor.get("dataType", "float32")
            shape = descriptor.get("shape", [])
            element_count = 1
            for dim in shape:
                element_count *= dim
            if dtype == "float16" and element_count > 4:
                pytest.skip(f"CoreML limitation: Float16 input with >{4} elements crash (ANE memory alignment issue)")

        # Check constants
        constants = graph_desc.get("constants", {})
        for const_name, const_spec in constants.items():
            descriptor = const_spec.get("descriptor", const_spec)
            dtype = descriptor.get("dataType", "float32")
            shape = descriptor.get("shape", [])
            element_count = 1
            for dim in shape:
                element_count *= dim
            if dtype == "float16" and element_count > 4:
                pytest.skip(f"CoreML limitation: Float16 constant with >{4} elements crash (ANE memory alignment issue)")

        # Check expected outputs
        expected_outputs = wpt_test_case.get("expectedOutputs", {})
        for output_name, output_spec in expected_outputs.items():
            dtype = output_spec.get("dataType", "float32")
            shape = output_spec.get("shape", [])
            element_count = 1
            for dim in shape:
                element_count *= dim
            if dtype == "float16" and element_count > 4:
                pytest.skip(f"CoreML limitation: Float16 output with >{4} elements crash (ANE memory alignment issue)")

    # Skip expand 0D scalar operations on CoreML (CoreML tile doesn't support scalar inputs)
    # CoreML tile requires input rank to match reps rank
    if backend_name == "coreml" and operation == "expand":
        # Check if this is expanding a 0D scalar
        graph_desc = wpt_test_case.get("graph", {})
        inputs_data = graph_desc.get("inputs", {})
        for input_name, input_spec in inputs_data.items():
            descriptor = input_spec.get("descriptor", input_spec)
            shape = descriptor.get("shape", [])
            if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1 and "0D" in test_name):
                pytest.skip("CoreML limitation: tile operation doesn't support scalar (0D) inputs")

    # Skip hard_swish 0D scalar operations on CoreML (CoreML mul doesn't support scalar outputs correctly)
    if backend_name == "coreml" and operation == "hard_swish":
        # Check if this is a 0D scalar
        graph_desc = wpt_test_case.get("graph", {})
        inputs_data = graph_desc.get("inputs", {})
        for input_name, input_spec in inputs_data.items():
            descriptor = input_spec.get("descriptor", input_spec)
            shape = descriptor.get("shape", [])
            if len(shape) == 0 or "0D" in test_name:
                pytest.skip("CoreML limitation: hard_swish decomposition (sigmoid_hard + mul) doesn't support scalar (0D) tensors")

    # Skip neg 0D scalar operations on CoreML (CoreML mul doesn't support scalar outputs correctly)
    if backend_name == "coreml" and operation == "neg":
        # Check if this is a 0D scalar
        graph_desc = wpt_test_case.get("graph", {})
        inputs_data = graph_desc.get("inputs", {})
        for input_name, input_spec in inputs_data.items():
            descriptor = input_spec.get("descriptor", input_spec)
            shape = descriptor.get("shape", [])
            if len(shape) == 0 or "0D" in test_name:
                pytest.skip("CoreML limitation: neg decomposition (mul with -1) doesn't support scalar (0D) tensors")

    # Skip CoreML unsupported data types
    # CoreML feature descriptions only support: DOUBLE, FLOAT32, FLOAT16, INT32
    # Int8, Uint8, Uint32, Int64 are not supported (even though Int8 exists in protobuf)
    if backend_name == "coreml":
        graph_desc = wpt_test_case.get("graph", {})

        # Check inputs for unsupported types
        inputs_data = graph_desc.get("inputs", {})
        for input_name, input_spec in inputs_data.items():
            descriptor = input_spec.get("descriptor", input_spec)
            data_type = descriptor.get("dataType", "").lower()
            if data_type in ["int8", "uint8", "uint32", "int64"]:
                pytest.skip(f"CoreML limitation: {data_type} not supported in feature descriptions (only DOUBLE, FLOAT32, FLOAT16, INT32)")

        # Check expected outputs for unsupported types
        expected_outputs = wpt_test_case.get("expectedOutputs", {})
        for output_name, output_spec in expected_outputs.items():
            data_type = output_spec.get("dataType", "").lower()
            if data_type in ["int8", "uint8", "uint32", "int64"]:
                pytest.skip(f"CoreML limitation: {data_type} not supported in feature descriptions (only DOUBLE, FLOAT32, FLOAT16, INT32)")

    # Skip known architectural limitations (validated against Chromium reference implementation)
    # See: docs/implementation-status.md - Chromium Reference Implementation Comparison

    # These are the 32 tests that represent architectural limitations also present in Chromium
    skip_patterns = [
        # instance_normalization NHWC (8 tests) - Not supported in Chromium
        ("instance_normalization", "nhwc"),  # Matches "layout='nhwc'" and "all options" with nhwc
        ("instance_normalization", "all options"),
        # layer_normalization non-consecutive axes (12 tests) - Requires complex emulation in Chromium
        ("layer_normalization", "axes=[0, 2]"),  # Note: space after comma in WPT test names
        ("layer_normalization", "all options"),
        ("layer_normalization", "options.scale"),  # This catches both "options.scale" and "options.scale and"
        # batch_normalization 1D/NHWC (12 tests) - Semantic mismatches with ONNX
        ("batch_normalization", "1d tensor"),  # Note: space, not underscore
        ("batch_normalization", "nhwc tensor"),  # Note: space, not underscore
    ]

    for op_pattern, name_pattern in skip_patterns:
        if operation == op_pattern and name_pattern.lower() in test_name.lower():
            pytest.skip(f"{operation} architectural limitation (see docs/implementation-status.md)")

    # Skip CoreML operations that only support fp32/fp16 (not int32)
    if backend_name == "coreml":
        int32_unsupported_ops = ["clamp", "relu"]
        if operation in int32_unsupported_ops:
            graph_desc = wpt_test_case.get("graph", {})
            inputs_data = graph_desc.get("inputs", {})
            for input_name, input_spec in inputs_data.items():
                descriptor = input_spec.get("descriptor", input_spec)
                data_type = descriptor.get("dataType", "").lower()
                if data_type == "int32":
                    pytest.skip(f"CoreML limitation: {operation} only supports fp32/fp16, not int32")

    # Skip CoreML 0D (scalar) tensor operations that don't support rank-0
    # TODO: Implement reshape workaround (0D->1D->op->0D) like Chromium does
    # See: crbug.com/391672283 - Chromium handles this for gather with 5D input
    # by reshaping indices from 0D to [1], running gather, then reshaping output back
    if backend_name == "coreml":
        scalar_unsupported_ops = ["transpose", "slice", "gather"]
        if operation in scalar_unsupported_ops and "0D" in test_name:
            pytest.skip(f"CoreML limitation: {operation} doesn't support 0D (scalar) tensors")

    # Skip CoreML max rank limitation (5 dimensions)
    if backend_name == "coreml":
        if "5D" in test_name and operation == "reshape":
            pytest.skip("CoreML limitation: maximum tensor rank is 5 dimensions")

    # Skip CoreML identifier naming limitations
    # CoreML MIL requires valid identifiers (cannot start with digits)
    if backend_name == "coreml":
        if "special character" in test_name or "special_character" in test_name:
            pytest.skip("CoreML limitation: operand names must be valid identifiers (cannot start with digits)")

    # Execute test case and get results
    try:
        results = execute_wpt_test_case(context, wpt_test_case)
    except NotImplementedError as e:
        pytest.skip(f"Not implemented: {e}")
    except (ValueError, RuntimeError) as e:
        error_str = str(e)
        if "Unsupported data type" in error_str or "Unsupported feature data type" in error_str:
            pytest.skip(f"Unsupported data type: {e}")
        # Skip CoreML normalization tests with non-constant parameters
        # These are expected validation failures with clear error messages
        if backend_name == "coreml":
            if ("requires mean parameter to be a constant" in error_str or
                "requires variance parameter to be a constant" in error_str or
                "requires scale (gamma) parameter to be a constant" in error_str or
                "requires bias (beta) parameter to be a constant" in error_str):
                pytest.skip(f"CoreML limitation: normalization parameters must be constants - {e}")
            # Skip CoreML layer_norm with empty axes (requires multi-operation emulation)
            if "layer_norm with empty axes requires special handling" in error_str:
                pytest.skip(f"CoreML limitation: layer_norm with empty axes not yet implemented - {e}")
        raise

    # Validate results against expected outputs
    expected_outputs = wpt_test_case.get("expectedOutputs", {})
    for output_name, expected_spec in expected_outputs.items():
        # Get actual output
        if output_name not in results:
            pytest.fail(f"Output '{output_name}' not found in results")

        actual = results[output_name]
        expected = wpt_utils.numpy_array_from_test_data(expected_spec)

        # Get tolerance and validate
        tolerance = wpt_utils.get_operation_tolerance(operation, wpt_test_case)
        dtype = expected_spec.get("dataType", "float32")

        passed, message, failures = wpt_utils.validate_result(
            actual, expected, tolerance, dtype
        )

        if not passed:
            failure_msg = wpt_utils.format_test_failure(
                wpt_test_case.get("name", "unnamed"),
                failures
            )
            pytest.fail(f"{message}\n{failure_msg}")


# Mark all tests in this module
pytestmark = pytest.mark.wpt
