# rust-webnn-graph (rustnn) - Project Guide

## Project Overview

**rustnn** is a standalone Rust crate that mirrors Chromium's WebNN (Web Neural Network) graph handling while adding pluggable format converters and helper tooling to visualize, execute, and validate exported graphs on macOS.

**Core Capabilities:**
- Validates WebNN graph descriptions from JSON files
- Converts WebNN graphs to ONNX and CoreML formats
- Executes converted models on various compute units (CPU, GPU, Neural Engine)
- Visualizes graph structures using Graphviz DOT format
- Provides both a CLI tool and library API
- **Python bindings** via PyO3/maturin implementing the W3C WebNN API specification

## Architecture

### Core Components

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
  - `accelerated=true` + `power="high-performance"` → GPU > NPU > CPU
  - `accelerated=true` + `power="default"` → GPU > NPU > CPU
- Platform autonomously selects actual device based on availability
- Selection logic in `PyMLContext::select_backend()` (src/python/context.rs:473)
- Feature flags control availability, not selection
- Per explainer: "implementations have a better grasp of the system...control should be relinquished to them"

**3. Lazy Backend Conversion**
- Backend conversion happens during **`compute()`**, not `build()`
- `compute()` method routes to backend-specific execution:
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
  - `CoremlConverter` → CoreML protobuf format

#### **executors/** - Runtime Execution
- **Platform-specific**: Conditional compilation for macOS
- **ONNX Runtime**: `run_onnx_with_inputs()` - executes with actual tensor I/O (cross-platform)
- **CoreML Runtime**: `run_coreml_zeroed_cached()` - macOS only via Objective-C FFI

#### **python/context.rs** - Backend Selection & Execution
- **Backend Enum**: Tracks selected backend (OnnxCpu, OnnxGpu, CoreML, None)
- **Context Creation**: `PyMLContext::new()` implements [WebNN Device Selection Explainer](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md)
  - Takes `accelerated` (bool) and `power_preference` (str) hints
  - Returns `(Backend, accelerated_available)` tuple
  - `accelerated` property indicates actual platform capability
- **Backend Selection**: `select_backend()` maps hints to available backend using platform autonomy
  - No explicit device types - uses hints and availability
  - Platform decides actual device allocation
- **Compute Routing**: `compute()` routes to appropriate backend method
  - `compute_onnx()` - ONNX Runtime execution (feature-gated)
  - `compute_coreml()` - CoreML execution (feature-gated)
  - `compute_fallback()` - Fallback when no backend available
- **Tensor Management**: Implements [WebNN MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md)
  - `create_tensor()` - Creates MLTensor with descriptor flags
  - `read_tensor()` / `write_tensor()` - Synchronous I/O with permission checks
  - `dispatch()` - Async execution with MLTensor inputs/outputs

#### **python/tensor.rs** - MLTensor Implementation
- **MLTensorDescriptor**: Descriptor with usage flags following [WebNN MLTensor Explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md)
  - `readable` - Can read tensor data back to CPU
  - `writable` - Can write tensor data from CPU
  - `exportable_to_gpu` - Can export as GPU texture (future use)
- **PyMLTensor**: Opaque typed tensor with explicit resource management
  - Properties: `shape`, `data_type`, `size`, `readable`, `writable`, `exportable_to_gpu`
  - Methods: `destroy()` for explicit cleanup
  - Permission enforcement: read/write operations check descriptor flags
  - Lifecycle tracking: prevents use after destroy()

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

### File Organization

```
src/
├── lib.rs              # Public API exports
├── main.rs             # CLI entry point
├── graph.rs            # Core data structures
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
    ├── context.rs      # ML and MLContext classes
    ├── graph_builder.rs # MLGraphBuilder class
    ├── graph.rs        # MLGraph class
    └── operand.rs      # MLOperand class

python/webnn/           # Python package
├── __init__.py         # Package exports
└── __init__.pyi        # Type stubs

tests/
└── test_python_api.py  # Python API tests

examples/
├── python_simple.py    # Basic Python example
└── python_matmul.py    # Matrix multiplication example
```

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

### Adding New Graph Operations

Currently, operations are validated but not typed. To add operation-specific validation:

1. **Extend** `Operation` struct in `graph.rs` if needed
2. **Add validation logic** in `validator.rs`
3. **Update converters** to handle the new operation
4. **Add test cases** with example graphs

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

## Common Tasks

### Building the Project
```bash
make build                    # Debug build
cargo build --release         # Release build
cargo build --features coreml-runtime,onnx-runtime  # All features
```

### Running Validation
```bash
cargo run -- validate examples/sample_graph.json
```

### Converting Graphs
```bash
# To ONNX
cargo run -- convert examples/sample_graph.json onnx -o output.onnx

# To CoreML
cargo run -- convert examples/sample_graph.json coreml -o output.mlmodel
```

### Executing Models
```bash
# ONNX (requires onnx-runtime feature)
cargo run --features onnx-runtime -- run-onnx model.onnx

# CoreML (macOS only, requires coreml-runtime feature)
cargo run --features coreml-runtime -- run-coreml model.mlmodel --device gpu
```

### Visualization
```bash
cargo run -- visualize examples/sample_graph.json -o graph.dot
dot -Tpng graph.dot -o graph.png
```

### Running Tests
```bash
cargo test                    # All tests
cargo test --lib              # Library tests only
cargo test converters         # Specific module
```

### Clean Build Artifacts
```bash
make clean
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
- **objc 0.2** - Objective-C FFI for CoreML (macOS)
- **onnxruntime 0.0.14** - ONNX execution
- **pyo3 0.22** - Python bindings (optional, with `python` feature)

### Build Dependencies
- **prost-build 0.12** - Protobuf code generation
- **maturin** - Python package build system (for Python bindings)

## Platform Support

- **Validation & Conversion**: Cross-platform (Linux, macOS, Windows)
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

### Python Bindings (WebNN API)

The project includes full Python bindings implementing the W3C WebNN API specification.

**Installation:**
```bash
# Development mode
maturin develop --features python

# Or build a wheel
maturin build --release --features python
pip install target/wheels/webnn-*.whl
```

**Quick Example:**
```python
import webnn
import numpy as np

# Create context and builder
ml = webnn.ML()
context = ml.create_context(device_type="cpu")
builder = context.create_graph_builder()

# Build graph: output = relu(x + y)
x = builder.input("x", [2, 3], "float32")
y = builder.input("y", [2, 3], "float32")
z = builder.add(x, y)
output = builder.relu(z)

# Compile and execute
graph = builder.build({"output": output})
context.convert_to_onnx(graph, "model.onnx")
```

**API Classes:**
- `webnn.ML` - Entry point
- `webnn.MLContext` - Execution context
- `webnn.MLGraphBuilder` - Graph construction
- `webnn.MLGraph` - Compiled graph
- `webnn.MLOperand` - Tensor operands

See **README.md** (Python API Reference section) for complete documentation and examples.

### Python-Rust Integration Architecture

**Important:** The Python API is a thin wrapper around the Rust implementation. All core functionality is implemented in Rust:

**Execution Flow:**
```
Python compute() → Rust OnnxConverter → Rust run_onnx_with_inputs() → Results to Python
```

**Key Integration Points:**
- `context.rs:79` - Uses Rust `OnnxConverter.convert()` to convert graphs
- `context.rs:129` - Calls Rust `run_onnx_with_inputs()` executor
- Data conversion: NumPy arrays → Rust Vec<f32> → ONNX Runtime → Rust Vec<f32> → NumPy arrays

**Benefits:**
- Python provides W3C WebNN API interface
- Rust provides performance-critical validation, conversion, and execution
- Zero-copy operations where possible for efficiency
- Native ONNX Runtime integration through Rust bindings

### ONNX to WebNN Converter Script

The `scripts/convert_onnx_to_webnn.py` script converts ONNX models to WebNN JSON format:
- Uses `huningxin/onnx2webnn` package
- Includes preprocessing and optimization
- Handles operator normalization

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

## Git Commit Attribution Policy

**CRITICAL: All commits must use ONLY the user's identity - NEVER add Claude attribution.**

When creating git commits:
- ❌ **NEVER** add "Co-Authored-By: Claude <noreply@anthropic.com>"
- ❌ **NEVER** add "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
- ❌ **NEVER** add any other Claude attribution or signature
- ✅ **ALWAYS** use ONLY the user's git identity from `git config user.name` and `git config user.email`
- ✅ **NEVER** modify git config - use the existing user configuration

**Rationale:** The user's commits should reflect their own work and identity. Claude Code is a tool that assists the user, but the commits belong to the user, not to Claude.

**Example of correct commit:**
```
git commit -m "Add feature X

- Implementation details
- Test coverage
- Documentation updates"
```

**Example of INCORRECT commit (DO NOT DO THIS):**
```
git commit -m "Add feature X

- Implementation details

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Resources

- **README.md**: Complete documentation including Python API, Rust CLI, and architecture
- **examples/**: Sample WebNN graph JSON files and Python examples
- **tests/test_python_api.py**: Python API test suite (45 tests)
- **Makefile**: Common build and validation targets
- **pyproject.toml**: Python package configuration
- **LICENSE**: Apache 2.0 license

---

*This CLAUDE.md evolves with the project. Update it as new patterns emerge or architecture changes.*
