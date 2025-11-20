# rustnn

This standalone crate mirrors Chromium's WebNN graph handling while adding
pluggable format converters (ONNX/CoreML) and helper tooling to visualize and
validate exported graphs.

The Rust validator matches Chromium's C++ flow:

- JSON files model the `GraphInfo` mojo structure (operands, operations,
  constants, tensor handles).
- `GraphValidator` replicates operand bookkeeping, data type checks, constant
  byte-length verification, and operation dependency tracking.
- `ContextProperties` exposes knobs that mirror `WebNNContextImpl::properties()`
  so tensor limits or supported IO data types can be adjusted.
- A converter registry emits ONNX/CoreML variants of the graph for downstream
  consumption.

## Layout

```
rustnn/
├── Cargo.toml
├── README.md
├── Makefile                # helper targets (viz/onnx/coreml/validate)
├── examples/
│   └── sample_graph.json   # tiny graph with a constant weight
├── scripts/
│   ├── validate_coreml.py  # builds/executes CoreML from exported JSON
│   └── validate_onnx.py    # rebuilds/runs ONNX from exported JSON
└── src/
    ├── converters/         # ONNX/CoreML converters + registry
    ├── error.rs            # GraphError mirrors Chromium paths
    ├── graph.rs            # DataType/Operand/Operation/GraphInfo structs
    ├── graphviz.rs         # DOT exporter
    ├── loader.rs           # JSON loader
    ├── main.rs             # CLI entrypoint
    └── validator.rs        # GraphValidator + ContextProperties
```

## Running the validator

```
cd rustnn
cargo run -- examples/sample_graph.json
```

The CLI prints the number of operands/operations, the input/output tensor
descriptors, and the dependency fan-out recorded while validating operations.
Use `--tensor-limit <bytes>` to experiment with the limit enforced in
`WebNNGraphBuilderImpl::ValidateGraphImpl`.

To export a Graphviz DOT view of the graph, pass `--export-dot <path>`:

```
cargo run -- examples/sample_graph.json --export-dot /tmp/graph.dot
dot -Tpng /tmp/graph.dot > /tmp/graph.png
```

Or with the bundled helper target (requires `dot` and macOS `open`):

```
make viz
```

## Converting graphs

A pluggable converter registry can emit other graph formats. ONNX JSON is the
first built-in converter:

```
cargo run -- examples/sample_graph.json --convert onnx --convert-output /tmp/graph.onnx.json
```

Omit `--convert-output` to print the converted graph to stdout. More converters
can be registered via `ConverterRegistry`.

CoreML export works the same way:

```
cargo run -- examples/sample_graph.json --convert coreml --convert-output /tmp/graph.coreml.json
```

To validate the CoreML JSON locally with `coremltools` in a virtualenv:

```
make coreml-validate-env
```

For ONNX, you can build and validate (optionally running inference via onnxruntime) with:

```
make onnx-validate-env
```

To run the whole pipeline (build, tests, converters, ONNX + CoreML validation with venv-managed deps):

```
make validate-all-env
```
