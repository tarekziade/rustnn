#!/usr/bin/env python3
"""
Validate and optionally compile a generated CoreML JSON graph.

Usage:
    python scripts/validate_coreml.py target/graph.coreml.json

The script performs two steps:
1) Topology sanity checks (inputs/outputs/intermediates referenced by layers).
2) If `coremltools` is installed, build a minimal NeuralNetwork specification
   using elementwise add for supported ops and save a `.mlmodel` alongside
   the JSON file to confirm CoreML compilation works on macOS.
"""
import json
import sys
from pathlib import Path
import base64


def check_topology(data: dict) -> None:
    names = set()
    for section in ("input", "output", "intermediate"):
        for entry in data["description"].get(section, []):
            names.add(entry["name"])
    for weight in data.get("neuralNetwork", {}).get("weights", []):
        names.add(weight["name"])
    for layer in data["neuralNetwork"]["layers"]:
        for name in layer["input"]:
            if name not in names:
                raise RuntimeError(f"Layer {layer['name']} references missing input {name}")
        for name in layer["output"]:
            names.add(name)


def compile_coreml(data: dict, model_path: Path) -> None:
    try:
        import coremltools as ct  # type: ignore
        from coremltools.models import datatypes  # type: ignore
        from coremltools.models.neural_network import NeuralNetworkBuilder  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        print("coremltools not installed; skipping CoreML compile test.")
        return

    def coerce_dims(name: str, shape):
        dims = list(shape or [1])
        if len(dims) == 2:
            dims = [1] + dims  # CoreML NN expects 1D or 3D; lift 2D to 3D
            print(f"Adjusted CoreML shape for {name} to 3D: {dims}")
        elif len(dims) > 3:
            prod = 1
            for d in dims:
                prod *= d
            dims = [prod]
            print(f"Flattened CoreML shape for {name} to 1D: {dims}")
        return dims

    def coerce_shape(name: str, shape):
        return datatypes.Array(*coerce_dims(name, shape))

    inputs = [
        (entry["name"], coerce_shape(entry["name"], entry["type"]["shape"]))
        for entry in data["description"]["input"]
    ]
    outputs = [
        (entry["name"], coerce_shape(entry["name"], entry["type"]["shape"]))
        for entry in data["description"]["output"]
    ]
    builder = NeuralNetworkBuilder(inputs, outputs)

    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "uint32": np.uint32,
        "int8": np.int8,
        "uint8": np.uint8,
    }

    for weight in data.get("neuralNetwork", {}).get("weights", []):
        name = weight["name"]
        np_dtype = dtype_map.get(weight["type"]["dataType"])
        if np_dtype is None:
            raise RuntimeError(f"Unsupported CoreML weight dtype: {weight['type']['dataType']}")
        dims = coerce_dims(name, weight["type"].get("shape", []))
        arr = np.frombuffer(
            base64.b64decode(weight["rawData"]),
            dtype=np_dtype,
        )
        arr = arr.reshape(dims)
        builder.add_load_constant(
            name=f"const_{name}",
            output_name=name,
            constant_value=arr,
            shape=dims,
        )

    supported = {"add"}
    for layer in data["neuralNetwork"]["layers"]:
        op = layer["type"].lower()
        if op not in supported:
            raise RuntimeError(f"Unsupported op for CoreML test: {op}")
        builder.add_elementwise(
            name=layer["name"],
            input_names=layer["input"],
            output_name=layer["output"][0],
            mode="ADD",
        )

    mlmodel = ct.models.MLModel(builder.spec, compute_units=ct.ComputeUnit.CPU_ONLY)
    mlmodel.save(model_path)
    print(f"CoreML model compiled to {model_path}")

    feeds = {}
    for entry in data["description"]["input"]:
        name = entry["name"]
        np_dtype = dtype_map.get(entry["type"]["dataType"], np.float32)
        dims = coerce_dims(name, entry["type"].get("shape", []))
        feeds[name] = np.zeros(dims, dtype=np_dtype)
    try:
        try:
            outputs = mlmodel.predict(feeds, useCPUOnly=True)
        except TypeError:
            outputs = mlmodel.predict(feeds)
        print("CoreML predict succeeded:")
        for k, v in outputs.items():
            shape = getattr(v, "shape", None)
            dtype = getattr(v, "dtype", type(v))
            print(f"  {k}: shape={shape}, dtype={dtype}")
    except Exception as exc:  # noqa: B902
        print(f"CoreML predict failed: {exc}")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    data = json.loads(path.read_text())

    check_topology(data)
    print("CoreML JSON topology looks consistent.")

    mlmodel_path = path.with_suffix(".mlmodel")
    compile_coreml(data, mlmodel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
