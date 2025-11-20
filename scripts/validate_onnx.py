#!/usr/bin/env python3
"""
Validate and optionally execute the generated ONNX JSON graph.

Usage:
    python scripts/validate_onnx.py target/graph.onnx.json

Steps:
1) Reconstruct a real ONNX ModelProto from the JSON emitted by the converter.
2) Run onnx.checker on the rebuilt model and write a .onnx file alongside it.
3) If onnxruntime is installed, execute a CPU inference pass with zeroed inputs.
"""
import base64
import json
import sys
from pathlib import Path
from typing import List

import onnx
from onnx import TensorProto, helper


def parse_shape(shape_json: List) -> List[int]:
    dims = []
    for dim in shape_json:
        if isinstance(dim, dict) and "dim_value" in dim:
            dims.append(int(dim["dim_value"]))
        else:
            dims.append(int(dim))
    return dims


def make_value_info(entry):
    name = entry["name"]
    tt = entry["type"]["tensor_type"]
    elem_type = int(tt["elem_type"])
    shape = parse_shape(tt["shape"].get("dim", []))
    return helper.make_tensor_value_info(name, elem_type, shape or None)


def build_model(data: dict) -> onnx.ModelProto:
    graph_json = data["graph"]

    inputs = [make_value_info(inp) for inp in graph_json["input"]]
    outputs = [make_value_info(out) for out in graph_json["output"]]
    value_info = [make_value_info(v) for v in graph_json.get("value_info", [])]

    initializers = []
    for init in graph_json.get("initializer", []):
        name = init["name"]
        data_type = int(init["data_type"])
        dims = parse_shape(init.get("dims", []))
        raw = base64.b64decode(init["raw_data"])
        tensor = helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=dims,
            vals=raw,
            raw=True,
        )
        initializers.append(tensor)

    nodes = []
    for node_json in graph_json["node"]:
        op_type = node_json["op_type"]
        if op_type.islower():
            op_type = op_type.capitalize()
        node = helper.make_node(
            op_type,
            inputs=node_json["input"],
            outputs=node_json["output"],
            name=node_json.get("name", ""),
        )
        nodes.append(node)

    graph = helper.make_graph(
        nodes=nodes,
        name=graph_json.get("name", "webnn_graph"),
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
        value_info=value_info,
    )

    model = helper.make_model(
        graph,
        producer_name=data.get("producer_name", "rust-webnn-graph"),
        ir_version=int(data.get("ir_version", onnx.IR_VERSION)),
    )
    model.opset_import.clear()
    model.opset_import.extend([helper.make_operatorsetid("", 13)])
    return model


def try_run_inference(model_path: Path) -> None:
    try:
        import numpy as np  # type: ignore
        import onnxruntime as rt  # type: ignore
    except ImportError:
        print("onnxruntime/numpy not installed; skipping inference run.")
        return

    sess = rt.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])

    type_map = {
        TensorProto.FLOAT: "float32",
        TensorProto.UINT8: "uint8",
        TensorProto.INT8: "int8",
        TensorProto.INT32: "int32",
        TensorProto.FLOAT16: "float16",
        TensorProto.UINT32: "uint32",
        "tensor(float)": "float32",
        "tensor(uint8)": "uint8",
        "tensor(int8)": "int8",
        "tensor(int32)": "int32",
        "tensor(float16)": "float16",
        "tensor(uint32)": "uint32",
    }

    feeds = {}
    for inp in sess.get_inputs():
        dtype = type_map.get(inp.type) or type_map.get(inp.type)
        if dtype is None:
            raise RuntimeError(f"Unsupported input dtype for inference: {inp.type}")
        shape = [dim if dim is not None else 1 for dim in inp.shape]
        feeds[inp.name] = np.zeros(shape, dtype=dtype)

    outputs = [o.name for o in sess.get_outputs()]
    result = sess.run(outputs, feeds)
    print("ONNX runtime inference succeeded:")
    for name, arr in zip(outputs, result):
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    json_path = Path(sys.argv[1])
    data = json.loads(json_path.read_text())
    model = build_model(data)
    onnx.checker.check_model(model)
    print("ONNX model structure is valid.")
    model_path = json_path.with_suffix(".onnx")
    onnx.save(model, model_path)
    print(f"Wrote {model_path}")
    try_run_inference(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
