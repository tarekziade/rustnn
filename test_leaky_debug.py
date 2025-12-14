#!/usr/bin/env python3
"""Debug script to replicate WPT leakyRelu test"""
import numpy as np
import webnn
import sys

print("Step 1: Creating ML context...")
ml = webnn.ML()
context = ml.create_context(device_type="npu")  # Force CoreML
builder = context.create_graph_builder()

print("Step 2: Creating Float16 input (not constant)...")
# WPT test uses input, not constant
x_input = builder.input("x", [24], "float16")

print("Step 3: Creating leakyRelu operation...")
output = builder.leaky_relu(x_input, alpha=0.01)

print("Step 4: Building graph...")
graph = builder.build({"output": output})

print("Step 5: Preparing input data...")
input_data = np.arange(24, dtype=np.float16)

print("Step 6: Computing (this is where WPT crashes)...")
sys.stdout.flush()
try:
    results = context.compute(graph, {"x": input_data})
    print(f"SUCCESS: {results}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
