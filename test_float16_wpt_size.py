#!/usr/bin/env python3
"""Test with WPT test configuration: 24 elements"""
import numpy as np
import webnn
import sys

print("Step 1: Creating ML context...")
ml = webnn.ML()
context = ml.create_context(device_type="npu")
builder = context.create_graph_builder()

print("Step 2: Creating Float16 input [24]...")
x_input = builder.input("x", [24], "float16")
output = builder.leaky_relu(x_input, alpha=0.01)

print("Step 3: Building graph...")
graph = builder.build({"output": output})

print("Step 4: Preparing Float16 input data [24 elements]...")
input_data = np.arange(24, dtype=np.float16)
print(f"  Input data shape: {input_data.shape}, dtype: {input_data.dtype}")

print("Step 5: Computing...")
sys.stdout.flush()
try:
    results = context.compute(graph, {"x": input_data})
    print(f"SUCCESS: output shape = {results['output'].shape}")
    print(f"  First 5 elements: {results['output'][:5]}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
