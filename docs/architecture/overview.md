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
- `accelerated=False` → ORT/LiteRT CPU
- `accelerated=True` + `power="high-performance"` → GPU preferred (TRTX, CoreML, LiteRT, ORT)
- `accelerated=True` + `power="low-power"` → NPU preferred (CoreML Neural Engine, LiteRT NPU)
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

## Shape Inference

**Shape inference** is the process of automatically computing output tensor shapes of neural network operations based on their input shapes and operation parameters, without executing the operation.

### Why Shape Inference Matters

Shape inference enables:

1. **Early validation** - Catch shape mismatches at build time, not runtime
2. **Memory allocation** - Backend runtimes know output buffer sizes before execution
3. **Graph optimization** - Enables static analysis and optimization passes
4. **Self-describing graphs** - Graphs are fully annotated and backend-agnostic

### How It Works

Each WebNN operation has a shape inference function in `src/shape_inference.rs` that computes output shapes. Shape inference happens during **graph building**, before any backend selection or execution.

**Binary Operations (add, mul, div, etc.):**
- Use NumPy-style broadcasting rules
- Two dimensions are compatible if equal or one is 1
- Output dimension is the maximum of the two
```rust
// broadcast_shapes([3, 1, 5], [3, 4, 5]) → [3, 4, 5]
// The dimension 1 broadcasts to 4
```

**Matrix Multiplication:**
```rust
// Simple 2D: [M, K] @ [K, N] → [M, N]
infer_matmul_shape([2, 3], [3, 4]) → [2, 4]

// Batched: [batch, M, K] @ [batch, K, N] → [batch, M, N]
infer_matmul_shape([5, 2, 3], [5, 3, 4]) → [5, 2, 4]

// Validates inner dimensions match (K must equal)
infer_matmul_shape([2, 3], [4, 5]) → Error: 3 != 4
```

**Convolution (conv2d):**
- Takes input shape, filter shape, strides, padding, dilations
- Computes spatial output dimensions:
  ```
  output_h = floor((input_h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h + 1)
  output_w = floor((input_w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1)
  ```
- Validates channel compatibility and group constraints
- Handles multiple layouts: NCHW, NHWC (inputs) and OIHW, HWIO, OHWI, IHWO (filters)

**Reshape:**
```rust
// Validates element count is preserved
validate_reshape([2, 3, 4], [6, 4]) → OK (24 elements in both)
validate_reshape([2, 3, 4], [5, 5]) → Error (24 != 25 elements)
```

**Pooling Operations:**
- Similar to convolution but without filters
- Computes output spatial dimensions based on window size, strides, padding
- Handles both average and max pooling
- Global pooling reduces spatial dimensions to 1x1

### Integration with Graph Builder

Shape inference is called automatically during graph construction:

```python
# Python API example
x = builder.input("x", [2, 3], "float32")    # Shape: [2, 3]
y = builder.input("y", [3, 4], "float32")    # Shape: [3, 4]
z = builder.matmul(x, y)                     # Shape: [2, 4] (inferred)
output = builder.relu(z)                     # Shape: [2, 4] (preserved)
```

When you call `builder.matmul(x, y)`, the implementation:
1. Calls `infer_matmul_shape([2, 3], [3, 4])` from `src/shape_inference.rs`
2. Gets result `[2, 4]`
3. Creates operand descriptor with inferred shape
4. Stores operation in graph with validated inputs/outputs

This creates a fully-annotated, backend-agnostic graph that can be:
- Validated for correctness
- Visualized with Graphviz
- Converted to ONNX, CoreML, or other formats
- Executed on different backends without re-inference

### Implementation Status

All 85 WebNN operations have shape inference implemented (100% coverage). Each operation includes:
- Shape inference function in `src/shape_inference.rs`
- Comprehensive validation (dimension compatibility, parameter constraints)
- Unit tests covering typical cases and edge cases
- Error messages with context for debugging

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
│   ├── coreml_mlprogram.rs  # CoreML converter (MIL)
│   ├── cann.rs         # CANN converter (HiAI IR, placeholder)
│   └── litert.rs       # LiteRT converter (TFLite flatbuffer)
├── backends/
│   ├── mod.rs          # DisabledContext + module registry
│   ├── ort.rs          # ONNX Runtime backend
│   ├── trtx.rs         # TensorRT backend (CUDA)
│   ├── coreml.rs       # CoreML backend (macOS)
│   ├── cann.rs         # CANN backend (Ascend NPU, OHOS)
│   └── litert.rs       # LiteRT backend (TFLite)
├── executors/
│   ├── mod.rs          # Conditional compilation
│   ├── onnx.rs         # ONNX runtime
│   ├── coreml.rs       # CoreML runtime
│   └── coreml_shim.mm  # CoreML shim (ObjC++)
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
├── run_wpt_conformance.rs  # WPT conformance (libtest_mimic + MLGraphBuilder)
├── wpt_conformance/        # WPT harness modules
├── test_python_api.py      # Python API tests (pywebnn)
└── test_integration.py     # Integration tests

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
- **TRTX Execution**: Linux/Windows with `trtx-runtime` feature (NVIDIA GPU)
- **LiteRT Execution**: Cross-platform with `litert-runtime` feature (CPU/GPU/NPU)
- **ONNX Execution**: Cross-platform with `onnx-runtime` feature (CPU/GPU)
- **CoreML Execution**: macOS only with `coreml-runtime` feature (GPU/Neural Engine)
- **CANN Execution**: OHOS only with `cann-runtime` feature (Huawei Ascend NPU, Kirin)
- **Neural Engine**: macOS with Apple Silicon (via CoreML)
- **Python Bindings**: Cross-platform with `python` feature (Python 3.11+)

## Implementation Status

**85 WebNN operations fully implemented** across all backends:

- Shape Inference: 85/85 (100%)
- Python API: 85/85 (100%)
- ONNX Backend: 85/85 (100%)
- CoreML MLProgram: 85/85 (100%)
- LiteRT Backend: 56/85
- TRTX Backend: 85/85 (100%)

See [implementation-status.md](../development/implementation-status.md) for complete details.
