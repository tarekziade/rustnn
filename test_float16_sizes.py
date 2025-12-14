#!/usr/bin/env python3
"""Test different Float16 input sizes to find crash threshold"""
import numpy as np
import webnn
import sys

ml = webnn.ML()
context = ml.create_context(device_type="npu")

# Test multiple sizes
test_sizes = [2, 3, 4, 8, 12, 16, 20, 24]

for size in test_sizes:
    print(f"\nTesting size [{size}]...")
    sys.stdout.flush()

    builder = context.create_graph_builder()
    x_input = builder.input("x", [size], "float16")
    output = builder.leaky_relu(x_input, alpha=0.01)
    graph = builder.build({"output": output})

    input_data = np.arange(size, dtype=np.float16)

    try:
        results = context.compute(graph, {"x": input_data})
        print(f"  [OK] size={size} passed")
    except Exception as e:
        print(f"  [FAILED] size={size}: {e}")
        break

print("\nTest complete")
