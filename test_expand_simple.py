#!/usr/bin/env python3
"""Simple test for expand operation to debug CoreML conversion."""

import webnn
import numpy as np

# Create context and builder
ml = webnn.ML()
context = ml.create_context(device_type="npu")
builder = context.create_graph_builder()

# 1D input [3] to 2D output [2, 3]
x = builder.input("x", [3], "float32")
output = builder.expand(x, [2, 3])

# Build graph
graph = builder.build({"output": output})

# Export to CoreML to see the model
try:
    context.convert_to_coreml(graph, "/tmp/expand_test.mlpackage")
    print("CoreML model exported to /tmp/expand_test.mlpackage")

    # Try to run
    inputs = {"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    results = context.compute(graph, inputs)
    print("Results:", results)
    print("SUCCESS!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
