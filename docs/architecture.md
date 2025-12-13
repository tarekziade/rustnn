# Architecture

## Core Components

```
┌─────────────────────────────────────────────────────────────┐
│ CLI (main.rs) / Library API (lib.rs) / Python API (PyO3)    │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┬──────────────┬─────────────────┐
    ▼                     ▼              ▼                 ▼
┌────────┐     ┌──────────────┐   ┌──────────┐    ┌──────────────┐
│Loader  │────▶│  Validator   │──▶│ Context  │───▶│  Backend     │
│(JSON)  │     │(graph.rs)    │   │(selects) │    │  Selection   │
└────────┘     └──────────────┘   └────┬─────┘    └──────┬───────┘
                                        │                 │
                                        ▼                 ▼
                                  ┌──────────┐    ┌──────────────┐
                                  │ Builder  │    │  Converter   │
                                  │(backend- │    │  (Runtime)   │
                                  │agnostic) │    │              │
                                  └────┬─────┘    └──────┬───────┘
                                       │                 │
                                       ▼                 ▼
                              ┌─────────────┐   ┌────────────────┐
                              │  MLGraph    │   │ ONNX / CoreML  │
                              │(immutable)  │   │   Execution    │
                              └─────────────┘   └────────────────┘
```

## Key Principles

### 1. Backend-Agnostic Graph Representation
- `builder.build()` creates an immutable, platform-independent `GraphInfo` structure
- Contains operands, operations, inputs, outputs, and constant data
- No backend-specific artifacts at this stage

### 2. Runtime Backend Selection (WebNN Spec-Compliant)

Following the [W3C WebNN Device Selection Explainer](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md):

- Backend selection happens at **context creation** via `accelerated` and `power_preference` hints
- `accelerated=False` → ONNX Runtime CPU
- `accelerated=True` + `power="high-performance"` → GPU preferred (ONNX or CoreML)
- `accelerated=True` + `power="low-power"` → NPU preferred (CoreML Neural Engine on Apple Silicon)
- Platform autonomously selects actual device based on availability and runtime conditions
- Selection logic in `PyMLContext::select_backend()`

### 3. MLTensor Management

Following the [W3C WebNN MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md):

- Explicit tensor management with descriptor flags (readable, writable, exportableToGPU)
- `destroy()` method for explicit resource cleanup
- `dispatch()` for async execution with MLTensor inputs/outputs
- Permission enforcement on read/write operations

### 4. Lazy Backend Conversion
- Backend-specific conversion happens during `compute()`, not `build()`
- `compute()` routes to appropriate backend method:
  - `compute_onnx()` for ONNX Runtime
  - `compute_coreml()` for CoreML
  - `compute_fallback()` when no backend available
- Same graph can be executed on different backends via different contexts

### 5. Rust-First Architecture
- All core functionality in pure Rust (validation, conversion, execution)
- Python bindings are thin wrappers exposing Rust functionality
- Rust library usable independently without Python
- Design principle: "Rust is the implementation, Python is the interface"

## File Organization

```
src/
├── lib.rs              # Public Rust API exports
├── main.rs             # CLI entry point
├── graph.rs            # Core data structures (backend-agnostic)
├── error.rs            # Error types
├── validator.rs        # Graph validation
├── loader.rs           # JSON loading
├── graphviz.rs         # DOT export
├── protos.rs           # Protobuf module setup
├── converters/
│   ├── mod.rs          # Registry and trait
│   ├── onnx.rs         # ONNX converter
│   └── coreml.rs       # CoreML converter
├── executors/
│   ├── mod.rs          # Conditional compilation
│   ├── onnx.rs         # ONNX runtime
│   └── coreml.rs       # CoreML runtime
└── python/             # Python bindings (PyO3)
    ├── mod.rs          # Python module definition
    ├── context.rs      # ML and MLContext classes (backend selection)
    ├── graph_builder.rs # MLGraphBuilder class
    ├── graph.rs        # MLGraph class
    ├── operand.rs      # MLOperand class
    └── tensor.rs       # MLTensor class

python/webnn/           # Python package
├── __init__.py         # Package exports (AsyncMLContext)
└── __init__.pyi        # Type stubs

tests/
├── test_python_api.py  # Python API tests (320+ tests)
├── test_wpt_conformance.py # WPT spec compliance tests
└── test_integration.py # Integration tests

examples/
├── python_simple.py          # Basic Python example
├── python_matmul.py          # Matrix multiplication
├── mobilenetv2_complete.py   # Complete pretrained MobileNetV2
├── text_generation_gpt.py    # Transformer with attention
└── train_text_model.py       # Model training script
```

## Design Patterns

### Registry Pattern (Converters)
- `ConverterRegistry` manages converters dynamically
- Trait objects: `Box<dyn GraphConverter + Send + Sync>`
- Extensible without modifying core code

### Builder Pattern (Graph Construction)
- `MLGraphBuilder` provides fluent API for graph construction
- Incremental construction of complex structures
- Used in ONNX and CoreML converters

### Validation Pipeline
- Immutable graph input
- Stateful validator with progressive checks
- Comprehensive artifacts returned for downstream use

### Conditional Compilation
- `#[cfg(target_os = "macos")]` for platform-specific code
- `#[cfg(feature = "...")]` for optional features
- Graceful degradation on unsupported platforms

## Technical Decisions

1. **WebNN Spec Compliance**: Follows W3C WebNN Device Selection and MLTensor explainers
2. **Protobuf for Interop**: Native format for ONNX and CoreML
3. **Compile-time Codegen**: Protobufs compiled at build time
4. **Feature Flags**: Optional runtimes to minimize dependencies
5. **Objective-C FFI**: Direct CoreML access on macOS
6. **Zero-copy where possible**: `Bytes` type for efficiency
7. **Registry Pattern**: Pluggable converters without core changes

## Platform Support

- **Validation & Conversion**: Cross-platform (Linux, macOS, Windows)
- **ONNX Execution**: Cross-platform with `onnx-runtime` feature (CPU/GPU)
- **CoreML Execution**: macOS only with `coreml-runtime` feature (GPU/Neural Engine)
- **Neural Engine**: macOS with Apple Silicon (via CoreML)
- **Python Bindings**: Cross-platform with `python` feature (Python 3.11+)

## Implementation Status

**85 WebNN operations fully implemented** across all backends:
- Shape Inference: 85/85 (100%)
- Python API: 85/85 (100%)
- ONNX Backend: 85/85 (100%)
- CoreML MLProgram: 85/85 (100%)

See [implementation-status.md](implementation-status.md) for complete details.
