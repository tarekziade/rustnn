#!/usr/bin/env python3
"""Debug script to test Float16 constant handling"""
import numpy as np
import webnn

print("Creating ML context...")
ml = webnn.ML()
context = ml.create_context(device_type="npu")  # Force CoreML
builder = context.create_graph_builder()

print("Creating Float16 constant...")
# Create a simple Float16 constant [3]
const_data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
const = builder.constant(const_data)

print("Creating output...")
output = builder.relu(const)

print("Building graph...")
graph = builder.build({"output": output})

print("Converting to CoreML...")
try:
    context.convert_to_coreml(graph, "/tmp/test_float16.mlmodel")
    print("SUCCESS: Conversion completed")
    print("Check /tmp/test_float16.mlmodel or /tmp/test_float16.mlpackage")
except Exception as e:
    print(f"ERROR during conversion: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to compute...")
try:
    results = context.compute(graph, {})
    print(f"SUCCESS: Compute completed: {results}")
except Exception as e:
    print(f"ERROR during compute: {e}")
    import traceback
    traceback.print_exc()
