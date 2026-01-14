# rustnn (rustnn) - Project Guide

## Project Overview

**rustnn** is a cross-platform Rust crate that implements the W3C WebNN (Web Neural Network) specification, mirroring Chromium's graph handling while adding pluggable format converters and tooling for visualization, execution, and validation.

**Core Capabilities:**
- Validates WebNN graph descriptions from JSON files
- Converts WebNN graphs to ONNX (cross-platform) and CoreML (macOS) formats
- Executes models on various backends: TensorRT (NVIDIA GPU), ONNX Runtime (CPU/GPU), and CoreML (macOS: GPU/Neural Engine)
- Visualizes graph structures using Graphviz DOT format
- Provides CLI tool and Rust library API
- **Python bindings** are available in the separate [pywebnn](https://github.com/rustnn/pywebnn) package

## Architecture

### Core Components

```

 CLI (main.rs) / Library API (lib.rs)





Loader    Validator    Backend
(JSON)       (graph.rs)       Selection




                                   Converter
                                  (Runtime)




                                ONNX / CoreML
                                Execution

```

**Note:** Python bindings are now in [pywebnn](https://github.com/rustnn/pywebnn), which uses rustnn as its core library.

### Key Architectural Principles

**1. Backend-Agnostic Graph Representation (WebNN Spec-Compliant)**
- `builder.build()` creates an immutable `GraphInfo` structure
- Graph representation is **platform-independent** and **backend-agnostic**
- No backend-specific artifacts at graph build time
- Same graph can be executed on multiple backends

**2. Runtime Backend Selection (WebNN Device Selection Explainer)**
- Follows [W3C WebNN Device Selection Explainer](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md)
- Backend selection happens at **context creation** using hints, not compile-time
- `MLContext::new()` takes `accelerated` (bool) and `power_preference` (str) hints:
  - `accelerated=false` → `Backend::OnnxCpu` (CPU only)
  - `accelerated=true` + `power="low-power"` → NPU > GPU > CPU
  - `accelerated=true` + `power="high-performance"` → TensorRT > GPU > NPU > CPU
  - `accelerated=true` + `power="default"` → TensorRT > GPU > NPU > CPU
- Platform autonomously selects actual device based on availability
- Selection logic in `PyMLContext::select_backend()` (src/python/context.rs:473)
- Feature flags control availability, not selection
- Per explainer: "implementations have a better grasp of the system...control should be relinquished to them"

**3. Lazy Backend Conversion**
- Backend conversion happens during **`compute()`**, not `build()`
- `compute()` method routes to backend-specific execution:
  - `compute_trtx()` → Converts to ONNX protobuf, executes with TensorRT
  - `compute_onnx()` → Converts to ONNX protobuf, executes with ONNX Runtime
  - `compute_coreml()` → Converts to CoreML protobuf, executes with CoreML
  - `compute_fallback()` → Returns zeros when no backend available
- Conversion is transparent to the user

**4. Rust-First Architecture**
- All core logic implemented in pure Rust
- Python bindings are thin PyO3 wrappers
- Zero Python code in critical path (validation, conversion, execution)
- Rust library usable independently without Python

### Key Modules

#### **graph.rs** - Core Data Model
- `DataType`: Float32, Float16, Int32, Uint32, Int8, Uint8
- `OperandDescriptor`: Shape and type information
- `OperandKind`: Input, Constant, Output
- `Operand`: Graph nodes with descriptors and metadata
- `Operation`: Graph operations with inputs/outputs
- `ConstantData`: Weight/constant storage (base64 encoded)
- `GraphInfo`: Complete graph representation

**Key Convention:** Operands are referenced by their array index (u32) within the graph's operands list.

#### **validator.rs** - Validation Pipeline
- `ContextProperties`: Validation constraints and limits
- `GraphValidator`: Validates graph structure and dependencies
- `ValidationArtifacts`: Results including I/O descriptors and operation dependencies

**Validation Checks:**
1. Operand count limits
2. Tensor byte length limits
3. Valid input/output names
4. Constant data integrity
5. Operation dependency ordering
6. Operand usage consistency

#### **converters/** - Pluggable Format Conversion
- **Registry Pattern**: `ConverterRegistry` manages converters dynamically
- **Trait Interface**: `GraphConverter` defines conversion contract
- **Implementations**:
  - `OnnxConverter` → ONNX protobuf format
  - `CoremlMlProgramConverter` → CoreML MLProgram (MIL) protobuf format

#### **executors/** - Runtime Execution
- **Platform-specific**: Conditional compilation for macOS
- **TensorRT Runtime**: `run_trtx_with_inputs()` - NVIDIA GPU execution (Linux/Windows, with mock mode for development)
- **ONNX Runtime**: `run_onnx_with_inputs()` - executes with actual tensor I/O (cross-platform)
- **CoreML Runtime**: `run_coreml_zeroed_cached()` - macOS only via Objective-C FFI

#### **Backend Selection**
- Backend selection follows [W3C WebNN Device Selection Explainer](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md)
- Selection based on `accelerated` (bool) and `power_preference` (str) hints
- Platform autonomously selects optimal backend based on availability
- Supports: TensorRT (NVIDIA GPU), ONNX Runtime (CPU/GPU), CoreML (macOS)

**For Python API documentation**, see [pywebnn](https://github.com/rustnn/pywebnn)

#### **graphviz.rs** - Visualization
- Generates DOT format for graph visualization
- Color-coded nodes: inputs (green), outputs (blue), constants (yellow)

## Development Conventions

### Design Principles

**Rust-First WebNN Implementation:**
- The Rust code is a fully valid, standalone WebNN implementation
- All core functionality, validation, and graph operations exist in pure Rust
- The Rust library is independently usable without any Python dependency
- Python bindings are a convenience layer to enable easy integration with Python projects
- Python code should be minimal wrappers that expose Rust functionality
- This ensures the library can be used in pure Rust projects, CLI tools, and Python projects alike

### Code Style

1. **Naming:**
   - Files: `snake_case.rs`
   - Types: `PascalCase`
   - Functions: `snake_case`
   - Enums: PascalCase variants, snake_case JSON serialization

2. **Error Handling:**
   - All fallible operations return `Result<T, GraphError>`
   - Use `?` operator for error propagation
   - `thiserror` for error type derivation
   - Include contextual information in errors

3. **Serde Integration:**
   - `#[derive(Serialize, Deserialize)]` on all data types
   - `#[serde(rename_all = "snake_case")]` for JSON compatibility
   - `serde_with` for base64 encoding of binary data
   - Optional fields use `Option<T>`

4. **Testing:**
   - Unit tests in `#[cfg(test)]` modules at end of files
   - Use realistic data structures matching actual usage
   - Test examples exist in `graphviz.rs` and `converters/mod.rs`

5. **Formatting:**
   - No emojis in code, documentation, commit messages, or any project files
   - Use plain text markers: [OK], [WARNING], [INFO], [TODO], etc.
   - Keep all text professional and readable in all terminals and editors
   - Prioritize clarity and accessibility over visual decoration

### Architecture Patterns

1. **Registry Pattern** (converters):
   - Trait objects: `Box<dyn GraphConverter + Send + Sync>`
   - Dynamic registration and lookup
   - Extensible without modifying core code

2. **Builder Pattern** (protobuf construction):
   - Incremental construction of complex structures
   - Used in ONNX and CoreML converters

3. **Validation Pipeline**:
   - Immutable graph input
   - Stateful validator with progressive checks
   - Comprehensive artifacts returned for downstream use

4. **Conditional Compilation**:
   - `#[cfg(target_os = "macos")]` for platform-specific code
   - `#[cfg(feature = "...")]` for optional features
   - Graceful degradation on unsupported platforms

5. **Explicit Dependencies**:
   - No singletons or global state
   - Pass dependencies via function parameters
   - Clear data flow through the system

### WebNN Specification Reference

The project uses the `search-bikeshed` tool for efficient browsing and searching of the W3C WebNN specification:

**Installation:**
```bash
# Install search-bikeshed (if not already installed)
pip install search-bikeshed
```

**Initial Setup:**
```bash
# Index the WebNN specification (run once per session or when spec updates)
search-bs index https://github.com/webmachinelearning/webnn/blob/main/index.bs --name webnn
```

**Common Usage:**

1. **Basic keyword search:**
   ```bash
   search-bs search --name webnn "MLTensor"
   ```

2. **Phrase search:**
   ```bash
   search-bs search --name webnn "graph builder"
   ```

3. **Search with context lines:**
   ```bash
   # Show 3 lines before and after each match
   search-bs search --name webnn "MLContext" --around 3
   ```

4. **JSON output for scripting:**
   ```bash
   search-bs search --name webnn "MLContext" --json
   ```

5. **Show URLs in results:**
   ```bash
   search-bs search --name webnn "operator" --show-url --max-results 10
   ```

6. **Get specific line ranges:**
   ```bash
   # Get 40 lines starting from line 1234
   search-bs get --name webnn --line 1234 --count 40
   ```

7. **JSON output for line ranges:**
   ```bash
   search-bs get --name webnn --line 1234 --count 40 --json
   ```

**When to Use:**
- Verifying WebNN API signatures and behavior
- Understanding operation semantics and constraints
- Checking data type support for operations
- Finding spec language for documentation
- Resolving ambiguities in implementation

**Advantages over web browsing:**
- Fast local search without network latency
- Context-aware results with surrounding lines
- Scriptable output formats (JSON)
- Works offline after initial indexing
- Integrated into development workflow

### File Organization

```
src/
 lib.rs              # Public API exports
 main.rs             # CLI entry point
 graph.rs            # Core data structures
 error.rs            # Error types
 validator.rs        # Graph validation
 loader.rs           # JSON loading
 graphviz.rs         # DOT export
 protos.rs           # Protobuf module setup
 converters/
    mod.rs              # Registry and trait
    onnx.rs             # ONNX converter
    coreml_mlprogram.rs # CoreML MLProgram (MIL) converter
 executors/
    mod.rs          # Conditional compilation
    trtx.rs         # TensorRT runtime
    onnx.rs         # ONNX runtime
    coreml.rs       # CoreML runtime

examples/
 sample_graph.json   # Sample WebNN graph
 # Python examples in pywebnn repository

tests/
 # Python tests in pywebnn repository
```

**For Python bindings**, see the [pywebnn repository](https://github.com/rustnn/pywebnn).

## Adding New Features

### Adding a New Converter

1. **Create converter file** in `src/converters/your_format.rs`
2. **Implement the trait:**
   ```rust
   pub struct YourFormatConverter;

   impl GraphConverter for YourFormatConverter {
       fn name(&self) -> &str { "your-format" }
       fn convert(&self, graph_info: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
           // Implementation
       }
   }
   ```
3. **Register in** `converters/mod.rs` or `main.rs`:
   ```rust
   registry.register(Box::new(YourFormatConverter));
   ```
4. **Add dependencies** to `Cargo.toml` if needed
5. **Add tests** in your converter file

### Adding a New Executor

1. **Create executor file** in `src/executors/your_runtime.rs`
2. **Add feature gate** in `Cargo.toml`:
   ```toml
   [features]
   your-runtime = ["dep:your-runtime-crate"]
   ```
3. **Implement execution function:**
   ```rust
   #[cfg(feature = "your-runtime")]
   pub fn run_your_runtime(model_data: &[u8]) -> Result<(), GraphError> {
       // Implementation
   }
   ```
4. **Add conditional compilation** in `executors/mod.rs`
5. **Wire up in CLI** (`main.rs`) if needed

### Adding New WebNN Operations (Standard Workflow)

**IMPORTANT: Before implementing any new operation, always check the Chromium reference implementation first:**
- **Chromium WebNN Source**: https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/
- **ONNX Runtime Backend**: https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/ort/graph_builder_ort.cc
- **CoreML Backend**: https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/coreml/graph_builder_coreml.mm
- **Why**: Chromium is the reference implementation of W3C WebNN spec. Their implementation shows:
  - Correct WebNN API signatures and behavior
  - How to handle type conversions (e.g., Cast nodes for ONNX bool types)
  - Edge cases and validation requirements
  - Backend-specific workarounds and best practices
- **How to use**: Search for the operation name in `graph_builder_ort.cc` or `graph_builder_coreml.mm` to see their implementation approach

**Complete implementation checklist for adding operations:**

1. **Shape Inference** (`src/shape_inference.rs`):
   - Add `infer_<operation>_shape()` function
   - Add Rust unit tests for shape inference
   - Handle all parameter variations (layouts, axes, etc.)

2. **Python API** (`src/python/graph_builder.rs`):
   - Add method following WebNN spec signature
   - Use PyO3 `#[pyo3(signature = (...))]` for optional parameters
   - Create `OperandDescriptor` with `pending_permutation: Vec::new()`
   - Call shape inference function
   - Store operation with all parameters in `attributes` JSON

3. **ONNX Converter** (`src/converters/onnx.rs`):
   - Add operation name mapping in `onnx_op_type()`
   - Add attribute handling if needed (see `create_conv2d_attributes()` example)

4. **CoreML Converter** (`src/converters/coreml_mlprogram.rs`):
   - Add MIL operation name mapping in `get_mil_op_type()`
   - Add operation input handling in `create_operation_inputs()` if needed
   - CoreML MLProgram uses MIL operations (more flexible than old NeuralNetwork format)

5. **Tests** (`tests/test_python_api.py`):
   - Add 3-5 tests covering:
     - Basic usage
     - Optional parameters (scale, bias, etc.)
     - Different layouts/shapes
     - Edge cases and validation
   - Run: `python -m pytest tests/test_python_api.py -v`

6. **WPT Conformance Tests** (`tests/wpt_data/conformance/<operation>.json`):
   - **IMPORTANT**: Add WPT test data for spec compliance validation
   - Reference: [WPT WebNN Tests](https://github.com/web-platform-tests/wpt/tree/master/webnn/conformance_tests)
   - Find the operation's test file in WPT repo (e.g., `relu.https.any.js`)
   - Extract 3-5 test cases covering:
     - Different tensor shapes (1D, 2D, 3D, 4D)
     - Different data types (float32 primarily)
     - Edge cases and boundary conditions
   - Create JSON file following format in existing test files
   - Include proper tolerance specifications (ULP or ATOL)
   - Verify tests: `pytest tests/test_wpt_conformance.py -k "<operation>" -v`
   - See: `docs/wpt-test-guide.md` for detailed instructions

7. **Documentation** (`docs/api-reference.md`):
   - Add operation to appropriate section
   - Include parameters, shape inference, formula
   - Add 2-3 practical examples
   - Show common use cases

8. **Update TODO.txt**:
   - Mark operation as done with implementation summary

9. **Rebuild Python module**:
   ```bash
   maturin develop --features python
   ```

10. **Run all tests before committing**:
   ```bash
   cargo test --lib        # Rust tests
   pytest tests/ -v        # Python + WPT tests
   cargo fmt              # Format code
   ```

**Example PR titles:**
- "Add batch_normalization operation"
- "Add element-wise operations (abs, exp, log)"
- "Add reduction operations (reduceSum, reduceMean)"

### Adding Protobuf Definitions

1. **Add .proto files** to `protos/your_format/`
2. **Update** `build.rs` to compile them:
   ```rust
   prost_build::compile_protos(&["protos/your_format/schema.proto"], &["protos/"])?;
   ```
3. **Include generated code** in `src/protos.rs`:
   ```rust
   pub mod your_format {
       include!(concat!(env!("OUT_DIR"), "/your.format.namespace.rs"));
   }
   ```

## Development

**IMPORTANT: Always use Make commands instead of direct cargo/maturin/pytest commands.**

The Makefile provides consistent build targets with proper feature flags, environment setup, and dependency management. Using Make ensures builds are reproducible and properly configured.

For detailed development instructions, build commands, and troubleshooting, see **[docs/development.md](docs/development.md)**.

Common Make targets:
```bash
make build              # Build Rust library with proper features
make python-dev         # Install Python package in development mode
make test               # Run Rust tests
make python-test        # Run all Python tests (API + WPT conformance)
make python-test-wpt    # Run WPT conformance tests only
make fmt                # Format Rust code
make help               # Show all available targets
```

Direct cargo commands (AVOID - use Make instead):
```bash
cargo build --release                      # DON'T USE - use `make build` instead
maturin develop --features python          # DON'T USE - use `make python-dev` instead
cargo test && python -m pytest tests/      # DON'T USE - use `make test && make python-test` instead
```

## Dependencies

### Core Dependencies
- **clap 4.5** - CLI argument parsing
- **serde 1.0** + **serde_json 1.0** - JSON serialization
- **serde_with 3.8** - Base64 encoding
- **thiserror 1.0** - Error derivation
- **prost 0.12** + **prost-types 0.12** - Protobuf runtime
- **bytes 1.6** - Byte buffer utilities
- **bytemuck 1.15** - Type casting

### Optional Runtime Dependencies
- **trtx 0.2.0** - TensorRT execution (NVIDIA GPU, with mock mode for development)
- **objc 0.2** - Objective-C FFI for CoreML (macOS)
- **ort 2.0.0-rc.10** - ONNX execution (successor to onnxruntime-rs)
- **pyo3 0.22** - Python bindings (optional, with `python` feature)

### Build Dependencies
- **prost-build 0.12** - Protobuf code generation
- **maturin** - Python package build system (for Python bindings)

## Platform Support

- **Validation & Conversion**: Cross-platform (Linux, macOS, Windows)
- **TensorRT Execution**: Linux/Windows with NVIDIA GPU and `trtx-runtime` feature (mock mode via `trtx-runtime-mock` for development on any platform)
- **ONNX Execution**: Cross-platform with `onnx-runtime` feature
- **CoreML Execution**: macOS only with `coreml-runtime` feature
- **Neural Engine**: macOS with Apple Silicon (via CoreML)
- **Python Bindings**: Cross-platform with `python` feature (Python 3.8+)

## Key Technical Decisions

1. **WebNN Device Selection Explainer**: Follows [W3C WebNN Device Selection spec](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md) for platform-autonomous device selection using hints
2. **WebNN MLTensor Explainer**: Follows [W3C WebNN MLTensor spec](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md) for explicit tensor management with descriptor flags (readable, writable, exportableToGPU), destroy() for resource cleanup, and dispatch() for async execution
3. **Protobuf for interop**: Native format for ONNX and CoreML
4. **Compile-time codegen**: Protobufs compiled at build time
5. **Feature flags**: Optional runtimes to minimize dependencies
6. **Objective-C FFI**: Direct CoreML access on macOS
7. **Zero-copy where possible**: `Bytes` type for efficiency
8. **Registry pattern**: Pluggable converters without core changes

## Future Extension Points

- **More converters**: TensorFlow Lite, TensorRT, OpenVINO
- **More executors**: Additional backend runtimes
- **Operation typing**: Strongly-typed operation variants
- **Graph optimization**: Pre-conversion graph transformations
- **Benchmarking**: Performance measurement tools
- **Graph diff**: Compare graphs for equivalence

## Python Integration

Python bindings for rustnn are available in the separate **[pywebnn](https://github.com/rustnn/pywebnn)** package.

The pywebnn package provides:
- Full W3C WebNN API implementation
- PyO3-based bindings to rustnn's core functionality
- ONNX Runtime and CoreML backend support
- NumPy array integration
- Comprehensive test suite and examples

**Installation:**
```bash
pip install pywebnn
```

**Usage:**
```python
import webnn
ml = webnn.ML()
context = ml.create_context(accelerated=False)
builder = context.create_graph_builder()
# ... build and execute graphs
```

For complete Python API documentation, examples, and development instructions, see the [pywebnn repository](https://github.com/rustnn/pywebnn).

## Claude Code - Approved Permissions

The following operations have been approved for Claude Code to execute without requiring additional user confirmation:

### Build & Development
- `cargo *` - All Cargo commands (check, build, fmt, test, clean, clippy, doc, etc.)
- `pip *` - All pip commands (install, uninstall, list, freeze, etc.)
- `maturin *` - All maturin commands (develop, build, publish, etc.)
- `make *` - All Makefile targets approved

### Python Execution & Testing
- `python` - Run Python scripts
- `python3.12` - Run Python 3.12 interpreter specifically
- `.venv/bin/python` - Run Python from virtual environment
- `.venv-test/bin/python` - Run Python from test virtual environment
- `python3.12 -m venv` - Create Python virtual environments
- `source` - Activate virtual environments (e.g., `source .venv-test/bin/activate`)
- `pytest` - Run Python test suite

### Documentation
- `mkdocs build` - Build documentation site

### Specification Tools
- `search-bs *` - All search-bikeshed commands (index, search, get)
  - Used for browsing and searching W3C WebNN specification
  - Provides fast local access to spec content without web browsing

### File Operations
- `find` - Search for files
- `cat` - Read file contents

### Git Operations
- `git *` - All git commands approved (status, add, commit, push, pull, checkout, branch, tag, log, diff, show, reset, rebase, merge, etc.)
  - Note: Destructive operations (force push to main, hard reset) should still be used cautiously

### GitHub Operations
- `gh run list` - List GitHub Actions workflow runs
- `gh run view` - View details of GitHub Actions runs

### Web Resources
- `WebFetch(domain:www.w3.org)` - Fetch W3C WebNN specifications

These permissions enable Claude Code to efficiently assist with development, testing, documentation, version control, and CI/CD monitoring tasks without interrupting the workflow.

## Testing Validation Requirement

**CRITICAL: All code changes MUST be validated by running tests before committing.**

Before creating any git commit:

1. **Run Rust Tests:**
   ```bash
   cargo test --lib
   ```
   - All tests must pass
   - No new warnings should be introduced

2. **Run Python Tests (if Python code changed):**
   ```bash
   make python-test
   ```
   - All tests must pass or be explicitly skipped (when dependencies unavailable)
   - Skipped tests are acceptable if the skip reason is valid (e.g., ONNX runtime not available)

3. **Format Rust Code (if Rust code changed):**
   ```bash
   cargo fmt
   ```
   - MUST be run after any Rust code changes
   - CI will fail if code is not formatted
   - This ensures consistent code style across the project

4. **Fix Any Failures:**
   - Never commit code with failing tests
   - If tests fail, fix the code or update the tests
   - Document any intentional test skips with clear skip conditions

**Rationale:** Running tests catches regressions early and ensures code quality. Tests are the safety net that allows confident refactoring and feature additions.

**Example workflow:**
```bash
# Make changes to code
vim src/python/context.rs

# Run tests to validate
cargo test --lib
make python-test

# If tests pass, commit
git add src/python/context.rs
git commit -m "Update context implementation"
```

## Implemented Operations (as of 2025-12-20)

### Implementation Summary

**WebNN Specification Coverage:**
- **Total Operations in Spec:** 105
- **Fully Implemented:** 88 (84%)
- **Not Yet Implemented:** 13 (12%)
- **Intentionally Deferred:** 4 (4%) - RNN operations (gru, gruCell, lstm, lstmCell)

**Not Yet Implemented Operations:**
- `cumulativeSum` - Element-wise cumulative sum along axis
- `gatherElements` - Gather elements using index tensor
- `gatherND` - Gather N-dimensional slices
- `isInfinite` - Check for infinite values
- `isNaN` - Check for NaN values
- `l2Pool2d` - L2 pooling (L2 norm within window)
- `linear` - Linear transformation (alpha*x + beta)
- `max` - Element-wise maximum of two tensors
- `min` - Element-wise minimum of two tensors
- `notEqual` - Element-wise inequality comparison
- `resample2d` - Resize/resample 2D tensor
- `reverse` - Reverse elements along axes
- `roundEven` - Round to nearest even integer

### Fully Implemented Operations by Category

**Binary Element-wise Operations (11):**
- `add`, `sub`, `mul`, `div`, `pow`, `matmul`
- `equal`, `greater`, `greaterOrEqual`, `lesser`, `lesserOrEqual`
- Full NumPy-style broadcasting support

**Unary Element-wise Operations (28):**
- **Arithmetic:** `abs`, `ceil`, `floor`, `neg`, `reciprocal`, `sign`, `sqrt`
- **Trigonometric:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- **Hyperbolic:** `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- **Exponential/Log:** `exp`, `log`, `erf`
- **Rounding:** `round`

**Activation Functions (11):**
- `relu`, `sigmoid`, `tanh`, `softmax`, `softplus`, `softsign`
- `elu`, `leakyRelu`, `prelu`, `gelu`
- `hardSigmoid`, `hardSwish`

**Convolution Operations (2):**
- `conv2d` - 2D convolution with strides, dilations, padding, groups
- `convTranspose2d` - Transposed convolution with output padding/sizes
- Supports NCHW and NHWC layouts
- Depthwise convolution via groups parameter

**Pooling Operations (4):**
- `averagePool2d`, `maxPool2d` - 2D pooling with window, stride, dilation, padding
- `global_average_pool`, `global_max_pool` - Global pooling (reduces spatial to 1x1)
- Supports NCHW and NHWC layouts

**Normalization Operations (3):**
- `batchNormalization` - Batch norm with mean, variance, scale, bias, epsilon
- `instanceNormalization` - Instance norm with scale, bias, epsilon
- `layerNormalization` - Layer norm with scale, bias, epsilon, axes (for transformers)

**Reduction Operations (10):**
- `reduceSum`, `reduceMean`, `reduceMax`, `reduceMin`, `reduceProduct`
- `reduceL1`, `reduceL2`, `reduceLogSum`, `reduceLogSumExp`, `reduceSumSquare`
- All support axes parameter and keepDimensions option

**Shape Manipulation (9):**
- `reshape`, `transpose`, `expand`, `squeeze`, `unsqueeze`
- `concat`, `split`, `slice`, `tile`

**Indexing/Gathering (4):**
- `gather` - Gather elements along axis
- `scatterElements` - Scatter elements using indices
- `scatterND` - Scatter N-dimensional updates
- `where` - Select elements based on condition

**Matrix Operations (2):**
- `matmul` - Matrix multiplication with batched support
- `gemm` - General matrix multiplication (C = alpha*A*B + beta*C)

**Quantization (2):**
- `quantizeLinear` - Quantize float to integer
- `dequantizeLinear` - Dequantize integer to float

**Other Operations (2):**
- `cast` - Type conversion between data types
- `clamp` - Clamp values to range [min, max]
- `identity` - Identity operation
- `pad` - Pad tensor with constant/edge/reflection modes
- `argMax`, `argMin` - Find indices of max/min values
- `logical_and`, `logical_or`, `logical_xor`, `logical_not` - Logical operations
- `triangular` - Extract triangular part of matrices

**Test Coverage:**
- 1350+ ONNX tests passing (100% of supported operations)
- WPT conformance tests for 44 operations
- Python API tests covering all 88 implemented operations

For complete implementation status including WPT test coverage, see [docs/development/implementation-status.md](docs/development/implementation-status.md).

## Documentation

**User Documentation:**
- **[Getting Started](docs/user-guide/getting-started.md)** - Installation and first steps
- **[API Reference](docs/user-guide/api-reference.md)** - Complete Python API documentation
- **[Examples](docs/user-guide/examples.md)** - Working code samples
- **[Advanced Topics](docs/user-guide/advanced.md)** - Advanced usage patterns

**Architecture & Design:**
- **[Architecture Overview](docs/architecture/overview.md)** - Core design principles and components
- **[Chromium Comparison](docs/architecture/chromium-comparison.md)** - How we compare to Chromium's WebNN

**Development:**
- **[Development Setup](docs/development/setup.md)** - Build and test environment
- **[Implementation Status](docs/development/implementation-status.md)** - Comprehensive status across all backends with WPT test integration

**Testing:**
- **[WPT Test Guide](docs/testing/wpt-test-guide.md)** - W3C WebNN conformance testing
- **[Performance Benchmarks](docs/testing/performance-benchmarks.md)** - Performance tracking

**Integration Guides:**
- **[TensorRT Integration](docs/integration/tensorrt.md)** - NVIDIA GPU backend (Linux/Windows)
- **[Windows TensorRT Setup](docs/integration/windows-tensorrt-setup.md)** - Windows-specific TensorRT setup
- **[GGML Integration](docs/integration/ggml.md)** - GGML backend (future)

**Reference:**
- **[WebNN Spec Reference](docs/reference/webnn-spec.md)** - W3C spec excerpts for offline use
- **[IPC Design](docs/reference/ipc-design.md)** - Inter-process communication design

**Other Resources:**
- **README.md**: Crisp project overview and quickstart
- **examples/**: Sample WebNN graph JSON files and Python examples
- **tests/test_python_api.py**: Python API test suite (320+ tests passing)
- **TODO.txt**: Implementation roadmap and completed features
- **Makefile**: Common build and validation targets

---

*This AGENTS.md evolves with the project. Update it as new patterns emerge or architecture changes.*
