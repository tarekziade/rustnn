# GGML Integration Guide

**Date:** December 8, 2024
**Purpose:** Guide for adding GGML converter and executor to rustnn

---

## [TARGET] Overview

This document outlines the integration of [GGML (GPT-Generated Model Language)](https://github.com/ggml-org/ggml) as a third execution backend for rustnn, alongside ONNX Runtime and CoreML.

**Why GGML?**
- **CPU-optimized inference**: Excellent performance on CPUs without GPU
- **Quantization support**: 4-bit, 8-bit quantized models for reduced memory usage
- **LLM-focused**: Widely used for large language model inference (llama.cpp, whisper.cpp)
- **Cross-platform**: Linux, macOS, Windows with various backends (CPU, CUDA, Metal, Vulkan)
- **Lightweight**: Minimal dependencies, no runtime memory allocation

---

##  GGML Background

### What is GGML?

GGML is a C tensor library for machine learning created by Georgi Gerganov. It's designed for:
- Fast and portable tensor operations
- Efficient LLM inference on consumer hardware
- Quantization to reduce model size (JPEG-like compression for tensors)
- Multiple hardware backends (CPU, CUDA, Metal, Vulkan)

**Key Resources:**
- [GGML GitHub](https://github.com/ggml-org/ggml)
- [Introduction to GGML](https://huggingface.co/blog/introduction-to-ggml)
- [GGML Glossary](https://klu.ai/glossary/ggml)

### GGML Architecture

**Core Concepts:**
1. **`ggml_context`**: Container holding tensors, graphs, and optionally data
2. **`ggml_cgraph`**: Computational graph representing order of operations
3. **`ggml_backend`**: Interface for executing graphs (CPU, CUDA, Metal, etc.)
4. **`ggml_tensor`**: Tensor data structure with shape and type
5. **Deferred execution**: Operations build a graph; computation happens at `graph_compute()`

**Workflow:**
```rust
// 1. Create context
let ctx = ggml_init(...);

// 2. Define tensors and operations
let a = ggml_new_tensor_2d(ctx, type, rows, cols);
let b = ggml_new_tensor_2d(ctx, type, rows, cols);
let result = ggml_add(ctx, a, b);

// 3. Build computation graph
let gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, result);

// 4. Execute
ggml_backend_graph_compute(backend, gf);
```

### Rust Bindings

**Available Crates:**
- **[ggml](https://crates.io/crates/ggml)** (v0.1.1): Semi-idiomatic Rust bindings (minimal maintenance)
- **[rusty-ggml](https://crates.io/crates/rusty-ggml)** (v0.0.8): Idiomatic Rust approach (pre-alpha)
- **[ggml-sys](https://crates.io/crates/ggml-sys)**: Raw C bindings

**Recommendation:** Use `ggml` crate (most stable, used by llm library)

---

##  Integration Architecture

### Following rustnn Patterns

rustnn uses a **converter + executor** pattern:

```
WebNN GraphInfo → Converter → Backend Format → Executor → Results
```

**Existing Backends:**
1. **ONNX Runtime**: Cross-platform, protobuf → ONNX Runtime execution
2. **CoreML**: macOS-only, protobuf → CoreML execution

**New GGML Backend:**
3. **GGML**: Cross-platform, computation graph → GGML execution

### File Structure

```
src/
 converters/
    mod.rs              # Add GgmlConverter registration
    onnx.rs
    coreml_mlprogram.rs
    ggml.rs             # NEW: GGML converter
 executors/
    mod.rs              # Add #[cfg(feature = "ggml-runtime")]
    onnx.rs
    coreml.rs
    ggml.rs             # NEW: GGML executor
 python/
     context.rs          # Add Backend::Ggml variant
```

---

##  Implementation Plan

### Phase 1: Converter (GraphInfo → GGML)

**File:** `src/converters/ggml.rs`

**Implementation:**
```rust
use crate::converters::{ConvertedGraph, GraphConverter};
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, Operation, OperandKind};

#[derive(Default)]
pub struct GgmlConverter;

impl GraphConverter for GgmlConverter {
    fn format(&self) -> &'static str {
        "ggml"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        // Convert WebNN graph to GGML computation graph
        // Return serialized graph (or placeholder for in-memory graph)

        // Strategy: Since GGML graphs are typically built in-memory,
        // we may serialize graph structure as JSON/GGUF format or
        // keep graph construction in executor

        Ok(ConvertedGraph {
            format: "ggml",
            content_type: "application/octet-stream",
            data: vec![], // Placeholder or GGUF format
        })
    }
}
```

**Key Challenges:**
1. **In-memory vs serialized**: GGML graphs are typically built in-memory, not serialized
2. **Format choice**:
   - Option A: Serialize as JSON (graph structure only)
   - Option B: Use GGUF format (weights + structure)
   - Option C: Pass GraphInfo directly to executor (no conversion)
3. **Quantization**: GGML supports quantized tensors; how to handle quantization metadata?

**Recommended Approach:** Pass GraphInfo directly to executor (Option C) since GGML graphs must be built in-memory with a context. Converter returns a lightweight "marker" indicating GGML format.

### Phase 2: Executor (GGML Execution)

**File:** `src/executors/ggml.rs`

**Feature Gate:** `#[cfg(feature = "ggml-runtime")]`

**Implementation:**
```rust
#![cfg(feature = "ggml-runtime")]

use crate::error::GraphError;
use crate::graph::{GraphInfo, OperandDescriptor};
use std::collections::HashMap;

pub struct GgmlOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn run_ggml_with_inputs(
    graph: &GraphInfo,
    inputs: HashMap<String, GgmlInput>,
) -> Result<Vec<GgmlOutput>, GraphError> {
    // 1. Create GGML context
    let ctx = ggml::Context::init(...);

    // 2. Build GGML computation graph from GraphInfo
    let tensors = build_tensors(&ctx, graph, inputs)?;
    let cgraph = build_computation_graph(&ctx, graph, &tensors)?;

    // 3. Execute graph
    let backend = ggml::Backend::cpu_default();
    backend.graph_compute(&cgraph)?;

    // 4. Extract results
    extract_outputs(graph, &tensors)
}

pub struct GgmlInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}
```

**Key Challenges:**
1. **Tensor creation**: Map WebNN operands to GGML tensors
2. **Operation mapping**: Translate WebNN ops to GGML ops
3. **Data flow**: Connect operations in correct order
4. **Memory management**: GGML context owns all tensors
5. **Backend selection**: CPU, CUDA, Metal based on device hints

### Phase 3: Feature Flag & Dependencies

**File:** `Cargo.toml`

**Changes:**
```toml
[features]
default = []
coreml-runtime = ["objc"]
onnx-runtime = ["onnxruntime"]
ggml-runtime = ["ggml"]  # NEW
python = ["pyo3"]

[dependencies]
# ... existing dependencies ...
ggml = { version = "0.1", optional = true }  # NEW
```

### Phase 4: Registration

**File:** `src/converters/mod.rs`

**Changes:**
```rust
mod coreml_mlprogram;
mod onnx;
mod ggml;  // NEW

pub use coreml_mlprogram::CoremlMlProgramConverter;
pub use onnx::OnnxConverter;
pub use ggml::GgmlConverter;  // NEW

impl ConverterRegistry {
    pub fn with_defaults() -> Self {
        let mut registry = Self {
            converters: HashMap::new(),
        };
        registry.register(Box::new(OnnxConverter::default()));
        registry.register(Box::new(CoremlMlProgramConverter::default()));
        registry.register(Box::new(GgmlConverter::default()));  // NEW
        registry
    }
}
```

**File:** `src/executors/mod.rs`

**Changes:**
```rust
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub mod coreml;
#[cfg(feature = "onnx-runtime")]
pub mod onnx;
#[cfg(feature = "ggml-runtime")]  // NEW
pub mod ggml;
```

### Phase 5: Python API Integration

**File:** `src/python/context.rs`

**Changes:**
```rust
#[derive(Debug, Clone)]
enum Backend {
    OnnxCpu,
    OnnxGpu,
    CoreML,
    Ggml,  // NEW
    None,
}

impl PyMLContext {
    fn select_backend(accelerated: bool, power: &str) -> (Backend, bool) {
        // Add GGML selection logic
        // GGML is CPU-optimized, so prefer when accelerated=false
        if !accelerated {
            #[cfg(feature = "ggml-runtime")]
            return (Backend::Ggml, false);
        }

        // Existing logic for ONNX/CoreML...
    }

    fn compute_ggml(
        &self,
        graph: &PyMLGraph,
        inputs: HashMap<String, Py<PyArray<f32, Dim<IxDyn>>>>,
    ) -> Result<HashMap<String, Py<PyArray<f32, Dim<IxDyn>>>>, GraphError> {
        #[cfg(feature = "ggml-runtime")]
        {
            use crate::executors::ggml::{run_ggml_with_inputs, GgmlInput};

            // Convert inputs to GgmlInput
            let ggml_inputs = convert_numpy_to_ggml(inputs)?;

            // Execute
            let outputs = run_ggml_with_inputs(&graph.graph, ggml_inputs)?;

            // Convert outputs back to NumPy
            convert_ggml_to_numpy(outputs)
        }
        #[cfg(not(feature = "ggml-runtime"))]
        Err(GraphError::BackendUnavailable {
            backend: "GGML".to_string(),
        })
    }
}
```

---

## [STATS] WebNN to GGML Operation Mapping

### Supported Operations

| WebNN Operation | GGML Equivalent | Notes |
|----------------|-----------------|-------|
| `add` | `ggml_add` | Element-wise addition |
| `mul` | `ggml_mul` | Element-wise multiplication |
| `matmul` | `ggml_mul_mat` | Matrix multiplication |
| `relu` | `ggml_relu` | ReLU activation |
| `gelu` | `ggml_gelu` | GELU activation |
| `sigmoid` | Custom | Needs implementation via composition |
| `tanh` | Custom | Needs implementation via composition |
| `softmax` | Custom | Needs implementation via `ggml_soft_max` |
| `transpose` | `ggml_transpose` | Tensor transpose |
| `reshape` | `ggml_reshape_*` | Shape transformation |
| `conv2d` | `ggml_conv_1d_*` / `ggml_conv_2d` | Convolution (limited) |
| `abs` | `ggml_abs` | Absolute value |
| `neg` | `ggml_neg` | Negation |
| `div` | Custom | Via `ggml_div` |

### Operations Requiring Composition

Some WebNN operations require composing multiple GGML operations:

**Sigmoid:**
```rust
// sigmoid(x) = 1 / (1 + exp(-x))
let neg_x = ggml_neg(ctx, x);
let exp_neg_x = ggml_exp(ctx, neg_x);
let one_plus_exp = ggml_add1(ctx, exp_neg_x, 1.0);
let sigmoid = ggml_div(ctx, ggml_new_f32(ctx, 1.0), one_plus_exp);
```

**Tanh:**
```rust
// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// Or use built-in if available
```

### Operations Not Supported

GGML has limited support for some operations:
- **Pooling**: No direct max_pool2d/avg_pool2d (may need custom implementation)
- **Normalization**: Limited batch_norm/layer_norm support
- **Advanced activations**: prelu, elu, leakyRelu require composition
- **Quantization ops**: Limited to GGML's internal quantization

---

##  Challenges & Solutions

### Challenge 1: In-Memory Graph Construction

**Problem:** GGML graphs must be built in-memory with a context. Cannot serialize to bytes like ONNX/CoreML.

**Solution:**
- Converter returns lightweight marker (empty Vec<u8> or JSON metadata)
- Executor receives original GraphInfo and builds GGML graph on-demand
- Pass GraphInfo directly to `run_ggml_with_inputs()` instead of serialized bytes

### Challenge 2: Limited Operation Coverage

**Problem:** GGML has fewer operations than ONNX (focused on LLMs, not general CV/NLP).

**Solution:**
- Implement missing operations via composition (e.g., sigmoid from exp/div)
- Return clear error for unsupported operations
- Document operation coverage in implementation-status.md
- Focus on LLM-relevant operations initially

### Challenge 3: Quantization Integration

**Problem:** GGML's strength is quantization, but WebNN spec doesn't specify quantization formats.

**Solution:**
- Initially support float32 only (matching ONNX/CoreML)
- Future: Add GGML-specific quantization hints via device selection
- Could use `power_preference="low-power"` as hint to enable quantization

### Challenge 4: Backend Selection

**Problem:** GGML supports multiple backends (CPU, CUDA, Metal). How to select?

**Solution:**
- Follow WebNN Device Selection Explainer pattern
- `accelerated=false` → GGML CPU (best use case)
- `accelerated=true` → GGML CUDA/Metal if available
- Query available backends at runtime: `ggml::Backend::available()`

### Challenge 5: Rust Bindings Maturity

**Problem:** GGML Rust bindings are in minimal maintenance mode (v0.1.1).

**Solution:**
- Use stable `ggml` crate (0.1.1) with limited but working API
- Consider `rusty-ggml` if need more idiomatic Rust (pre-alpha risk)
- Contribute improvements back to ggml-rs if needed
- Fallback: Use `ggml-sys` (raw bindings) if safe wrapper insufficient

---

## [TARGET] Implementation Roadmap

### Phase 1: Proof of Concept (1-2 days)
- [ ] Add `ggml` dependency with feature flag
- [ ] Create minimal executor for basic operations (add, mul, matmul)
- [ ] Test with simple WebNN graph (2 inputs + add + output)
- [ ] Validate tensor I/O works correctly

### Phase 2: Core Operations (3-5 days)
- [ ] Implement operation mapping for 20 core operations
- [ ] Add tensor shape inference for GGML tensors
- [ ] Implement computation graph building from GraphInfo
- [ ] Add comprehensive unit tests

### Phase 3: Python Integration (2-3 days)
- [ ] Add Backend::Ggml to context selection
- [ ] Implement `compute_ggml()` method
- [ ] Add device selection logic (GGML for CPU-only)
- [ ] Test with Python API examples

### Phase 4: Documentation & Examples (1-2 days)
- [ ] Update docs/implementation-status.md with GGML coverage
- [ ] Update docs/architecture.md with GGML backend
- [ ] Create example: `examples/ggml_inference.py`
- [ ] Update README.md with GGML backend section

### Phase 5: Advanced Features (Future)
- [ ] Quantization support (Q4, Q8)
- [ ] Multiple backend selection (CPU/CUDA/Metal)
- [ ] Performance benchmarks vs ONNX
- [ ] LLM-specific optimizations

**Total Estimated Time:** 7-12 days for phases 1-4

---

##  Testing Strategy

### Unit Tests (Rust)

**File:** `src/converters/ggml.rs`
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn converts_simple_graph() {
        let graph = create_add_graph();
        let converter = GgmlConverter::default();
        let result = converter.convert(&graph);
        assert!(result.is_ok());
    }
}
```

**File:** `src/executors/ggml.rs`
```rust
#[cfg(all(test, feature = "ggml-runtime"))]
mod tests {
    #[test]
    fn executes_add_operation() {
        let graph = create_add_graph();
        let inputs = create_test_inputs();
        let outputs = run_ggml_with_inputs(&graph, inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        // Verify output values
    }
}
```

### Python Tests

**File:** `tests/test_ggml_backend.py`
```python
import pytest
import webnn
import numpy as np

@pytest.mark.skipif(not has_ggml_runtime(), reason="GGML runtime not available")
def test_ggml_add():
    ml = webnn.ML()
    context = ml.create_context(accelerated=False)  # Should select GGML
    builder = context.create_graph_builder()

    x = builder.input("x", [2, 3], "float32")
    y = builder.input("y", [2, 3], "float32")
    z = builder.add(x, y)

    graph = builder.build({"output": z})

    inputs = {
        "x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "y": np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
    }

    outputs = context.compute(graph, inputs)
    expected = np.array([[2, 3, 4], [6, 7, 8]], dtype=np.float32)
    np.testing.assert_allclose(outputs["output"], expected)
```

### Makefile Targets

```makefile
# Add to Makefile
.PHONY: ggml-dev
ggml-dev:
	maturin develop --features python,ggml-runtime

.PHONY: test-ggml
test-ggml:
	cargo test --features ggml-runtime
	pytest tests/test_ggml_backend.py -v
```

---

##  References

### GGML Resources
- [GGML GitHub Repository](https://github.com/ggml-org/ggml)
- [Introduction to GGML (HuggingFace)](https://huggingface.co/blog/introduction-to-ggml)
- [GGML Glossary](https://klu.ai/glossary/ggml)
- [Experimenting with GGML Tutorial](https://omkar.xyz/intro-ggml/)

### Rust Bindings
- [ggml crate (crates.io)](https://crates.io/crates/ggml)
- [ggml docs.rs](https://docs.rs/ggml/latest/ggml/)
- [rusty-ggml (GitHub)](https://github.com/KerfuffleV2/rusty-ggml)
- [ggml-sys (raw bindings)](https://crates.io/crates/ggml-sys)

### WebNN Spec
- [W3C WebNN API Specification](https://www.w3.org/TR/webnn/)
- [WebNN Device Selection Explainer](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md)

### Existing Implementations
- [llama.cpp (uses GGML)](https://github.com/ggml-org/llama.cpp)
- [whisper.cpp (uses GGML)](https://github.com/ggml-org/whisper.cpp)

---

##  Summary

**GGML Integration Value:**
- [OK] **CPU-optimized inference** for environments without GPU
- [OK] **Quantization support** for memory-constrained devices
- [OK] **Cross-platform** (Linux, macOS, Windows)
- [OK] **LLM-focused** operations and optimizations
- [OK] **Lightweight** with minimal dependencies

**Key Design Decisions:**
1. Pass GraphInfo directly to executor (no serialization)
2. Focus on LLM-relevant operations initially
3. Use `accelerated=false` hint to select GGML backend
4. Start with float32, add quantization later
5. Use stable `ggml` crate (v0.1.1)

**Next Steps:**
1. Create proof of concept with basic operations
2. Validate tensor I/O and graph execution
3. Expand operation coverage incrementally
4. Integrate with Python API
5. Document and test thoroughly

---

**Status:** Planning document (not yet implemented)
