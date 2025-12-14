#!/usr/bin/env python3
"""Test with Float16 constant only (no input)"""
import numpy as np
import webnn

print("Test 1: Float16 constant with leakyRelu...")
ml = webnn.ML()
context = ml.create_context(device_type="npu")
builder = context.create_graph_builder()

# Create Float16 constant
const_data = np.array([1.0, 2.0, 3.0, -1.0, -2.0], dtype=np.float16)
const = builder.constant(const_data)

# leakyRelu on constant
output = builder.leaky_relu(const, alpha=0.01)

graph = builder.build({"output": output})

try:
    results = context.compute(graph, {})
    print(f"SUCCESS: {results}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
