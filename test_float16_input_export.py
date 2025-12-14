#!/usr/bin/env python3
"""Export CoreML model with Float16 input to examine the protobuf"""
import numpy as np
import webnn

print("Creating ML context...")
ml = webnn.ML()
context = ml.create_context(device_type="npu")
builder = context.create_graph_builder()

print("Creating Float16 input...")
x_input = builder.input("x", [3], "float16")
output = builder.relu(x_input)

print("Building graph...")
graph = builder.build({"output": output})

print("Exporting to CoreML...")
try:
    context.convert_to_coreml(graph, "/tmp/test_float16_input.mlmodel")
    print("SUCCESS: CoreML model exported to /tmp/test_float16_input.mlmodel")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
