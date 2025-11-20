# rust-webnn-graph

This standalone crate mirrors the validation flow implemented in Chromium's
`WebNNGraphBuilderImpl` (`services/webnn/webnn_graph_builder_impl.cc:3037`).
The C++ builder reads a `mojom::GraphInfo`, checks operand metadata,
verifies constants (`webnn_graph_builder_impl.cc:3083`), ensures the input and
output operand lists match the operand table (`webnn_graph_builder_impl.cc:3220`)
and finally walks the operations in topological order to verify dependencies
(`webnn_graph_builder_impl.cc:3249`).

The Rust version focuses on those same invariants:

* JSON files model the `GraphInfo` mojo structure (operands, operations,
  constants, tensor handles).
* `GraphValidator` replicates the operand bookkeeping, data type checks,
  constant byte-length verification, and operation dependency tracking.
* `ContextProperties` exposes the knobs that the C++ builder reads from
  `WebNNContextImpl::properties()` so tensor limits or supported IO data types
  can be adjusted.

## Layout

```
rust-webnn-graph/
├── Cargo.toml
├── README.md
├── examples/
│   └── sample_graph.json  # tiny add graph that exercises parsing
└── src/
    ├── error.rs          # GraphError mirrors ReportBadMessage paths
    ├── graph.rs          # DataType/Operand/Operation/GraphInfo structs
    ├── loader.rs         # JSON loader
    ├── validator.rs      # GraphValidator + ContextProperties
    └── main.rs           # CLI entrypoint
```

## Running the validator

```
cd rust-webnn-graph
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
