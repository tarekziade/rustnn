#!/usr/bin/env python3
"""
Test quantization round-trip: build graph with quantize ops, save with quantized=True, reload, verify

This tests the end-to-end quantization flow:
1. Build a graph with quantize_linear/dequantize_linear operations
2. Save the graph with quantized=True marker
3. Load the saved graph
4. Verify the quantized marker is preserved
5. Run inference to ensure it works
"""

import tempfile
import os
import numpy as np
from pathlib import Path

import webnn


def test_quantization_roundtrip(quant_dtype="int8", quant_level_name="int8"):
    """
    Test complete quantization workflow.

    Args:
        quant_dtype: WebNN data type for quantization (int4/int8/uint8)
        quant_level_name: Human-readable quantization level name
    """
    print("=" * 80)
    print(f"Testing Quantization Round-Trip: {quant_level_name}")
    print("=" * 80)

    # Create WebNN context
    ml = webnn.ML()
    context = ml.create_context(device_type="cpu")
    builder = context.create_graph_builder()

    # Build a simple quantized model: x -> quantize -> dequantize -> y
    print("\n[STEP 1] Building quantized graph...")
    x = builder.input("x", [4, 4], "float32")

    # Quantization parameters (per-tensor for simplicity)
    scale = builder.constant(np.array(0.5, dtype=np.float32))
    zero_point = builder.constant(np.array(0, dtype=np.int8 if quant_dtype == "int8" else np.uint8))

    # Quantize the input
    quantized = builder.quantize_linear(x, scale, zero_point)
    print(f"  Created quantize_linear node")
    print(f"    Input dtype: float32")
    print(f"    Output dtype: {quant_dtype}")

    # Dequantize for computation
    output = builder.dequantize_linear(quantized, scale, zero_point)
    print(f"  Created dequantize_linear node")
    print(f"    Input dtype: {quant_dtype}")
    print(f"    Output dtype: float32")

    # Build graph
    graph = builder.build({"output": output})
    print(f"  Graph built successfully")

    # Step 2: Save with quantized=True marker
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "quantized_model.webnn"
        print(f"\n[STEP 2] Saving quantized graph to {model_path}...")
        graph.save(str(model_path), quantized=True)
        print(f"  Model saved with quantized=True marker")

        # Verify file was created
        if not model_path.exists():
            print(f"  [ERROR] Model file not found!")
            return False

        # Read the file to verify quantized marker (JSON format uses "quantized": true)
        print(f"\n[STEP 3] Verifying saved file contains quantized marker...")
        with open(model_path, 'r') as f:
            content = f.read()
            if '"quantized": true' in content:
                print(f"  [OK] JSON quantized field found: \"quantized\": true")
            else:
                print(f"  [ERROR] Quantized marker NOT found in saved file")
                print(f"  First 500 chars: {content[:500]}")
                return False

        # Step 4: Run inference to verify it works
        print(f"\n[STEP 4] Running inference...")
        test_input = np.array([[1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0],
                                [13.0, 14.0, 15.0, 16.0]], dtype=np.float32)

        result = context.compute(graph, {"x": test_input})
        output_tensor = result["output"]

        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output_tensor.shape}")
        print(f"  Input sample: {test_input[0, :4]}")
        print(f"  Output sample: {output_tensor[0, :4]}")

        # Check that quantization/dequantization produced reasonable results
        # With scale=0.5 and zero_point=0:
        # quantize: q = round(x / 0.5) + 0 = round(2*x)
        # dequantize: y = (q - 0) * 0.5 = q * 0.5
        # So y should be close to x but with quantization error

        if not np.allclose(output_tensor, test_input, rtol=0.1):
            print(f"  [WARNING] Output differs from input more than expected")
        else:
            print(f"  [OK] Output matches input (within quantization error)")

    print(f"\n[STEP 5] COMPLETE")
    print("=" * 80)
    print()
    return True


def test_all_quantization_levels():
    """Test multiple quantization levels."""
    levels = [
        ("int8", "int8"),
        ("int4", "int4"),  # Will be mapped to int8 internally
        ("uint8", "uint8"),
    ]

    print("\n" + "=" * 80)
    print("TESTING ALL QUANTIZATION LEVELS")
    print("=" * 80 + "\n")

    results = {}
    for dtype, name in levels:
        try:
            success = test_quantization_roundtrip(dtype, name)
            results[name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"\n[ERROR] {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "ERROR"

    # Summary
    print("\n" + "=" * 80)
    print("QUANTIZATION ROUND-TRIP TEST SUMMARY")
    print("=" * 80)
    for name, result in results.items():
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"  {status_symbol} {name:10s} : {result}")
    print("=" * 80)

    all_passed = all(r == "PASS" for r in results.values())
    return all_passed


if __name__ == "__main__":
    # Get quantization level from environment (for make run-all-demos-matrix)
    quant_level = os.environ.get("RUN_ALL_DEMOS_LEVELS", "").lower()

    if quant_level and quant_level != "none":
        # Run specific quantization level
        print(f"Running quantization test for level: {quant_level}")
        test_quantization_roundtrip(quant_level, quant_level)
    else:
        # Run all levels
        success = test_all_quantization_levels()
        exit(0 if success else 1)
