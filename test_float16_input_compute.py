#!/usr/bin/env python3
"""Test compute with Float16 input"""
import numpy as np
import webnn
import sys

print("Step 1: Creating ML context...")
ml = webnn.ML()
context = ml.create_context(device_type="npu")
builder = context.create_graph_builder()

print("Step 2: Creating Float16 input...")
x_input = builder.input("x", [3], "float16")
output = builder.relu(x_input)

print("Step 3: Building graph...")
graph = builder.build({"output": output})

print("Step 4: Preparing Float16 input data...")
input_data = np.array([1.0, -2.0, 3.0], dtype=np.float16)
print(f"  Input data: {input_data}, dtype: {input_data.dtype}")

print("Step 5: Computing (this is where it may crash)...")
sys.stdout.flush()
try:
    results = context.compute(graph, {"x": input_data})
    print(f"SUCCESS: {results}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
