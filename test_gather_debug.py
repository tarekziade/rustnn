import webnn
import numpy as np

# Simple gather test
ml = webnn.ML()
context = ml.create_context(device_type="npu")
builder = context.create_graph_builder()

# 1D input, 1D indices
x = builder.input("x", [5], "float32")
indices = builder.input("indices", [2], "int32")  # 1D
output = builder.gather(x, indices)

graph = builder.build({"output": output})

# Export to see the models
try:
    context.convert_to_onnx(graph, "/tmp/gather_test.onnx")
    print("ONNX model exported to /tmp/gather_test.onnx")

    context.convert_to_coreml(graph, "/tmp/gather_test.mlmodel")
    print("CoreML model exported to /tmp/gather_test.mlmodel")

    # Try to run with ONNX first
    onnx_context = ml.create_context(device_type="gpu")  # ONNX
    onnx_graph = onnx_context.create_graph_builder().input("x", [5], "float32")
    onnx_indices = onnx_context.create_graph_builder().input("indices", [2], "int32")

    # Just test CoreML for now
    inputs = {
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "indices": np.array([1, 3], dtype=np.int32)  # 1D
    }
    results = context.compute(graph, inputs)
    print("Results:", results)
    print("SUCCESS!")
except Exception as e:
    print(f"CoreML Error: {e}")
    import traceback
    traceback.print_exc()
