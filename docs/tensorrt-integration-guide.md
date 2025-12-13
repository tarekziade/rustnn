# TensorRT Integration Guide

**Date:** December 8, 2024
**Purpose:** Guide for adding NVIDIA TensorRT converter and executor to rustnn

---

## [TARGET] Overview

This document outlines the integration of [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) as a fourth execution backend for rustnn, optimized for NVIDIA GPU inference alongside ONNX Runtime, CoreML, and GGML.

**Why TensorRT?**
- **GPU-optimized inference**: Best-in-class performance on NVIDIA GPUs (RTX, A100, H100)
- **Advanced quantization**: FP16, INT8, INT4, FP8, FP4 for maximum throughput
- **JIT optimization**: Just-In-Time compilation for specific GPU architectures
- **Production-ready**: Widely deployed in NVIDIA-accelerated inference (Triton, TensorRT-LLM)
- **ONNX-native**: Primary import via ONNX format (perfect match for rustnn)

**TensorRT for RTX (New in 2025):**
- Lightweight library (<200 MB) optimized for Windows 11 + NVIDIA RTX GPUs
- 50%+ performance improvement vs baseline DirectML
- JIT compilation in <30 seconds
- Supports Turing through Blackwell GPU generations

---

##  TensorRT Background

### What is TensorRT?

TensorRT is NVIDIA's high-performance deep learning inference SDK. It optimizes trained models through:
- **Layer fusion**: Combines operations to reduce kernel launches
- **Precision calibration**: INT8/FP16 quantization with minimal accuracy loss
- **Kernel auto-tuning**: Selects fastest implementation for target GPU
- **Dynamic tensor memory**: Minimizes memory footprint

**Key Resources:**
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
- [TensorRT SDK](https://developer.nvidia.com/tensorrt)
- [TensorRT for RTX (Windows 11)](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html)
- [ONNX-TensorRT GitHub](https://github.com/onnx/onnx-tensorrt)

### TensorRT Architecture

**Core Workflow:**
```
ONNX Model → TensorRT Builder → Optimized Engine → Inference Runtime
```

**Key Concepts:**
1. **Builder (`IBuilder`)**: Configures optimization settings (precision, batch size, workspace)
2. **Network (`INetworkDefinition`)**: Graph of layers and tensors
3. **Engine (`ICudaEngine`)**: Optimized executable for specific GPU + precision
4. **Context (`IExecutionContext`)**: Runtime state for executing inference
5. **Parser (`IParser`)**: Imports ONNX models into TensorRT network

**Optimization Pipeline:**
```rust
// 1. Create builder and network
let builder = create_infer_builder();
let network = builder.create_network();

// 2. Parse ONNX model
let parser = create_onnx_parser(network);
parser.parse_from_file("model.onnx");

// 3. Build optimized engine
let config = builder.create_builder_config();
config.set_flag(BuilderFlag::FP16);  // Enable FP16
let engine = builder.build_engine(network, config);

// 4. Execute inference
let context = engine.create_execution_context();
context.execute_v2(&bindings);  // Run inference
```

### Supported Operations

**300+ ONNX Operators** (opset 9-20) including:

**Binary Operations:**
- Add, Sub, Mul, Div, MatMul, Pow
- Broadcasting support

**Activations:**
- Relu, Sigmoid, Tanh, Softmax, Gelu, Elu, LeakyRelu, PRelu, Selu, HardSigmoid, HardSwish, Softplus, Softsign

**Convolution & Pooling:**
- Conv, ConvTranspose (2D and 3D)
- MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool
- LpPool (with restrictions)

**Normalization:**
- BatchNormalization, InstanceNormalization, LayerNormalization, GroupNormalization, LRN

**Reduction:**
- ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd
- ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceSumSquare

**Tensor Manipulation:**
- Reshape, Transpose, Concat, Split, Slice, Gather, Scatter, Squeeze, Unsqueeze, Expand, Pad, Tile

**Comparison & Logic:**
- Equal, Greater, GreaterOrEqual, Less, LessOrEqual
- And, Or, Xor, Not

**Math Functions:**
- Abs, Neg, Ceil, Floor, Round, Sqrt, Exp, Log, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Erf, Sign, Reciprocal

**Advanced:**
- LSTM, GRU (with restrictions)
- Attention mechanisms
- Einsum
- TopK, ArgMax, ArgMin
- Cast, Clip, Where

**Quantization:**
- QuantizeLinear, DequantizeLinear

**Data Types:**
DOUBLE, FLOAT32, FLOAT16, BFLOAT16, INT32, INT64, FP8, INT8, INT4, UINT8, BOOL

**Important Limitations:**
- DOUBLE cast to FLOAT32 (with clamping)
- UINT8 only for input/output tensors
- INT8/INT4/FP8 require quantization from FP32/FP16
- Some ops restricted to 2D/3D (e.g., pooling)

---

##  Integration Architecture

### Following rustnn Patterns

rustnn uses a **converter + executor** pattern:

```
WebNN GraphInfo → Converter → ONNX → TensorRT Engine → Executor → Results
```

**Existing Backends:**
1. **ONNX Runtime**: Cross-platform, protobuf → ONNX Runtime execution
2. **CoreML**: macOS-only, protobuf → CoreML execution
3. **GGML**: CPU-optimized, in-memory graph → GGML execution

**New TensorRT Backend:**
4. **TensorRT**: NVIDIA GPU, ONNX → TensorRT Engine → GPU execution

**Key Advantage:** We already have ONNX converter! TensorRT can consume ONNX directly.

### File Structure

```
src/
 converters/
    mod.rs              # Already has OnnxConverter (reuse!)
    onnx.rs
    coreml_mlprogram.rs
    ggml.rs
    tensorrt.rs         # NEW: TensorRT-specific converter (optional)
 executors/
    mod.rs              # Add #[cfg(feature = "tensorrt-runtime")]
    onnx.rs
    coreml.rs
    ggml.rs
    tensorrt.rs         # NEW: TensorRT executor
 python/
     context.rs          # Add Backend::TensorRT variant
```

---

##  Implementation Plan

### Phase 1: Executor (ONNX → TensorRT Engine)

**File:** `src/executors/tensorrt.rs`

**Feature Gate:** `#[cfg(feature = "tensorrt-runtime")]`

**Strategy:** Reuse existing ONNX converter, build TensorRT engine from ONNX bytes

**Implementation:**
```rust
#![cfg(feature = "tensorrt-runtime")]

use crate::error::GraphError;
use crate::graph::{GraphInfo, OperandDescriptor};
use std::collections::HashMap;

pub struct TensorRTOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct TensorRTInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Execute TensorRT inference from ONNX model bytes
pub fn run_tensorrt_with_inputs(
    onnx_model: &[u8],
    inputs: HashMap<String, TensorRTInput>,
    precision: TensorRTPrecision,
) -> Result<Vec<TensorRTOutput>, GraphError> {
    // 1. Create TensorRT builder
    let logger = create_logger();
    let builder = create_infer_builder(&logger)?;

    // 2. Parse ONNX model
    let network_flags = 1u32 << NetworkDefinitionCreationFlag::ExplicitBatchDimensions as u32;
    let network = builder.create_network_v2(network_flags)?;

    let parser = create_onnx_parser(&network, &logger)?;
    parser.parse(onnx_model)?;

    // 3. Configure builder
    let config = builder.create_builder_config()?;
    config.set_memory_pool_limit(MemoryPoolType::Workspace, 1 << 30)?; // 1GB

    // Set precision mode
    match precision {
        TensorRTPrecision::FP32 => {},
        TensorRTPrecision::FP16 => config.set_flag(BuilderFlag::FP16)?,
        TensorRTPrecision::INT8 => config.set_flag(BuilderFlag::INT8)?,
    }

    // 4. Build engine
    let engine = builder.build_serialized_network(&network, &config)?;
    let runtime = create_infer_runtime(&logger)?;
    let engine = runtime.deserialize_cuda_engine(&engine)?;

    // 5. Create execution context
    let context = engine.create_execution_context()?;

    // 6. Allocate GPU buffers and copy inputs
    let bindings = allocate_and_copy_inputs(&engine, inputs)?;

    // 7. Execute inference
    context.execute_v2(&bindings)?;

    // 8. Copy outputs back to CPU
    let outputs = copy_outputs_from_gpu(&engine, &bindings)?;

    Ok(outputs)
}

#[derive(Debug, Clone, Copy)]
pub enum TensorRTPrecision {
    FP32,
    FP16,
    INT8,
}
```

**Key Challenges:**
1. **Rust bindings**: Use `tensorrt-rs` or `easy-tensorrt-sys` (FFI to C++ API)
2. **GPU memory management**: Allocate CUDA buffers for inputs/outputs
3. **Engine caching**: Serialized engines can be cached for faster startup
4. **Precision selection**: FP32/FP16/INT8 based on device hints
5. **Batch size**: Dynamic batch support vs fixed batch

### Phase 2: Feature Flag & Dependencies

**File:** `Cargo.toml`

**Changes:**
```toml
[features]
default = []
coreml-runtime = ["objc"]
onnx-runtime = ["onnxruntime"]
ggml-runtime = ["ggml"]
tensorrt-runtime = ["tensorrt-rs", "cuda-runtime"]  # NEW

[dependencies]
# ... existing dependencies ...
tensorrt-rs = { version = "0.8", optional = true }  # NEW
cuda-runtime = { version = "0.7", optional = true }  # NEW
# Alternative: easy-tensorrt-sys for more recent bindings
```

**Rust Bindings Options:**

| Crate | Status | Notes |
|-------|--------|-------|
| `tensorrt-rs` | Older (2020) | Supports TensorRT 5-7, may need fork |
| `easy-tensorrt-sys` | Newer fork | Uses `cudarc` instead of old `cuda-rs` |
| Custom FFI | Most control | Bindgen to TensorRT C++ API |

**Recommendation:** Start with `easy-tensorrt-sys` or custom FFI for TensorRT 10.x support

### Phase 3: Registration

**File:** `src/executors/mod.rs`

**Changes:**
```rust
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub mod coreml;
#[cfg(feature = "onnx-runtime")]
pub mod onnx;
#[cfg(feature = "ggml-runtime")]
pub mod ggml;
#[cfg(feature = "tensorrt-runtime")]  // NEW
pub mod tensorrt;
```

**File:** `src/converters/mod.rs`

**No changes needed!** Reuse existing `OnnxConverter` to generate ONNX bytes, then TensorRT executor parses ONNX directly.

### Phase 4: Python API Integration

**File:** `src/python/context.rs`

**Changes:**
```rust
#[derive(Debug, Clone)]
enum Backend {
    OnnxCpu,
    OnnxGpu,
    CoreML,
    Ggml,
    TensorRT,  // NEW
    None,
}

impl PyMLContext {
    fn select_backend(accelerated: bool, power: &str) -> (Backend, bool) {
        // TensorRT selection logic
        if accelerated {
            #[cfg(feature = "tensorrt-runtime")]
            if is_nvidia_gpu_available() {
                // Prefer TensorRT on NVIDIA GPUs for high-performance
                if power == "high-performance" {
                    return (Backend::TensorRT, true);
                }
            }
        }

        // Existing logic for ONNX/CoreML/GGML...
    }

    fn compute_tensorrt(
        &self,
        graph: &PyMLGraph,
        inputs: HashMap<String, Py<PyArray<f32, Dim<IxDyn>>>>,
    ) -> Result<HashMap<String, Py<PyArray<f32, Dim<IxDyn>>>>, GraphError> {
        #[cfg(feature = "tensorrt-runtime")]
        {
            use crate::converters::OnnxConverter;  // Reuse ONNX converter!
            use crate::executors::tensorrt::{run_tensorrt_with_inputs, TensorRTInput, TensorRTPrecision};

            // 1. Convert GraphInfo to ONNX
            let converter = OnnxConverter::default();
            let converted = converter.convert(&graph.graph)?;

            // 2. Convert inputs to TensorRTInput
            let trt_inputs = convert_numpy_to_tensorrt(inputs)?;

            // 3. Execute with TensorRT
            let precision = TensorRTPrecision::FP16;  // Could be configurable
            let outputs = run_tensorrt_with_inputs(&converted.data, trt_inputs, precision)?;

            // 4. Convert outputs back to NumPy
            convert_tensorrt_to_numpy(outputs)
        }
        #[cfg(not(feature = "tensorrt-runtime"))]
        Err(GraphError::BackendUnavailable {
            backend: "TensorRT".to_string(),
        })
    }
}

#[cfg(feature = "tensorrt-runtime")]
fn is_nvidia_gpu_available() -> bool {
    // Check for CUDA-capable NVIDIA GPU
    // Could use cuda-runtime or parse nvidia-smi
    std::process::Command::new("nvidia-smi")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
```

### Phase 5: Engine Caching (Performance Optimization)

**Problem:** TensorRT engine building can take 10-60 seconds on first run.

**Solution:** Cache serialized engines to disk, keyed by model hash + GPU architecture.

**Implementation:**
```rust
use std::path::PathBuf;
use std::fs;
use sha2::{Sha256, Digest};

fn get_engine_cache_path(onnx_model: &[u8], gpu_arch: &str, precision: TensorRTPrecision) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(onnx_model);
    hasher.update(gpu_arch.as_bytes());
    hasher.update(format!("{:?}", precision).as_bytes());
    let hash = format!("{:x}", hasher.finalize());

    PathBuf::from(format!(".tensorrt_cache/engine_{}.trt", hash))
}

pub fn run_tensorrt_with_caching(
    onnx_model: &[u8],
    inputs: HashMap<String, TensorRTInput>,
    precision: TensorRTPrecision,
) -> Result<Vec<TensorRTOutput>, GraphError> {
    let gpu_arch = get_gpu_architecture()?;  // e.g., "sm_89" for RTX 4090
    let cache_path = get_engine_cache_path(onnx_model, &gpu_arch, precision);

    let engine = if cache_path.exists() {
        // Load cached engine
        let serialized = fs::read(&cache_path)?;
        let runtime = create_infer_runtime(&logger)?;
        runtime.deserialize_cuda_engine(&serialized)?
    } else {
        // Build new engine
        let engine = build_engine(onnx_model, precision)?;

        // Cache for future use
        let serialized = engine.serialize()?;
        fs::create_dir_all(cache_path.parent().unwrap())?;
        fs::write(&cache_path, serialized)?;

        engine
    };

    // Execute with cached/new engine
    execute_engine(engine, inputs)
}
```

---

## [STATS] Operation Coverage Analysis

### WebNN Operations → TensorRT Support

| WebNN Operation | TensorRT Support | Notes |
|----------------|------------------|-------|
| **Binary Ops** | | |
| `add`, `sub`, `mul`, `div` | [OK] Full | Via Add, Sub, Mul, Div |
| `matmul` | [OK] Full | Via MatMul |
| `pow` | [OK] Full | Via Pow |
| **Activations** | | |
| `relu`, `sigmoid`, `tanh`, `softmax` | [OK] Full | Native support |
| `gelu`, `elu`, `leakyRelu`, `prelu` | [OK] Full | Native support |
| `hardSigmoid`, `hardSwish`, `softplus`, `softsign` | [OK] Full | Native support |
| **Convolution** | | |
| `conv2d`, `convTranspose2d` | [OK] Full | 2D and 3D supported |
| **Pooling** | | |
| `averagePool2d`, `maxPool2d` | [OK] Full | 2D/3D, indices unsupported for MaxPool |
| `globalAveragePool`, `globalMaxPool` | [OK] Full | Native support |
| **Normalization** | | |
| `batchNormalization` | [OK] Full | Native support |
| `instanceNormalization` | [OK] Full | Native support |
| `layerNormalization` | [OK] Full | Native support |
| **Reduction** | | |
| All `reduce*` operations | [OK] Full | 10 reduction ops supported |
| **Tensor Ops** | | |
| `reshape`, `transpose`, `concat`, `split` | [OK] Full | Native support |
| `slice`, `gather`, `scatter`, `pad`, `tile` | [OK] Full | Native support |
| `squeeze`, `unsqueeze`, `expand` | [OK] Full | Native support |
| **Logic** | | |
| All comparison and logical ops | [OK] Full | 9 ops supported |
| **Math** | | |
| All element-wise math | [OK] Full | 23 ops supported |
| **Quantization** | | |
| `quantizeLinear`, `dequantizeLinear` | [OK] Full | Native support |
| **Advanced** | | |
| `argMax`, `argMin` | [OK] Full | Via ArgMax, ArgMin |
| `cast`, `clamp`, `where` | [OK] Full | Via Cast, Clip, Where |
| `gemm` | [OK] Full | Via Gemm |

**Coverage:** ~95%+ of WebNN spec (TensorRT has 300+ ONNX ops, WebNN has 85-95 ops)

**Not Supported:**
- Some RNN/LSTM restrictions (bidirectional requires matching activations)
- MaxPool indices output
- Certain dilation/padding combinations
- DOUBLE precision (cast to FLOAT32)

---

##  Challenges & Solutions

### Challenge 1: Rust Bindings Maturity

**Problem:** Existing Rust bindings (`tensorrt-rs`) are outdated (TensorRT 5-7, last update 2020).

**Solutions:**
1. **Use `easy-tensorrt-sys`**: Newer fork with better CUDA integration via `cudarc`
2. **Create custom FFI**: Use `bindgen` to generate fresh bindings for TensorRT 10.x
3. **Fork and update `tensorrt-rs`**: Modernize existing crate for TensorRT 10.x
4. **Wait for official bindings**: NVIDIA may release official Rust support (unlikely short-term)

**Recommendation:** Create custom FFI bindings for TensorRT 10.x C++ API using `bindgen`. Focus on core interfaces: IBuilder, INetworkDefinition, IExecutionContext, IParser.

### Challenge 2: CUDA Dependency

**Problem:** TensorRT requires CUDA toolkit and NVIDIA GPU runtime.

**Solutions:**
- **Feature flag**: Only enable with `tensorrt-runtime` feature
- **Runtime detection**: Check for NVIDIA GPU before selecting backend
- **Clear errors**: Provide helpful error if CUDA unavailable
- **Documentation**: Document CUDA installation requirements

### Challenge 3: Engine Build Time

**Problem:** Building TensorRT engine can take 10-60 seconds on first run.

**Solutions:**
- **Engine caching**: Serialize engines to disk, key by model hash + GPU arch
- **Ahead-of-time compilation**: Pre-build engines for target GPUs
- **JIT progress**: Show progress during engine building
- **TensorRT for RTX**: JIT compilation in <30 seconds (Windows 11)

### Challenge 4: Precision Selection

**Problem:** TensorRT supports FP32, FP16, INT8, FP8, FP4. How to select?

**Solutions:**
- Follow WebNN device hints:
  - `power="high-performance"` → FP16 (2x faster than FP32)
  - `power="default"` → FP16
  - `power="low-power"` → INT8 (requires calibration)
- Add optional precision parameter to `compute()`
- Auto-detect GPU capability (e.g., FP8 only on Ada/Hopper)

### Challenge 5: Platform Support

**Problem:** TensorRT is NVIDIA GPU-only (Linux, Windows). No macOS/AMD support.

**Solutions:**
- **Runtime detection**: Check for NVIDIA GPU at context creation
- **Graceful fallback**: Fall back to ONNX Runtime if TensorRT unavailable
- **Clear documentation**: Document platform requirements
- **Windows focus**: Leverage TensorRT for RTX (Windows 11 + RTX GPUs)

### Challenge 6: Dynamic Shapes

**Problem:** TensorRT engines can have fixed or dynamic input shapes.

**Solutions:**
- **Use explicit batch**: Set `ExplicitBatchDimensions` flag
- **Optimization profiles**: Define min/opt/max shapes for dynamic inputs
- **Runtime binding**: Bind shapes at execution time
- **Future work**: Add dynamic shape support incrementally

---

## [TARGET] Implementation Roadmap

### Phase 1: Proof of Concept (2-3 days)
- [ ] Research TensorRT C++ API and identify core interfaces needed
- [ ] Create minimal FFI bindings using `bindgen` for TensorRT 10.x
- [ ] Implement basic executor for ONNX → TensorRT → inference
- [ ] Test with simple operation (add, matmul) on NVIDIA GPU
- [ ] Validate FP32 precision works correctly

### Phase 2: Core Functionality (5-7 days)
- [ ] Expand FFI bindings for full IBuilder/INetworkDefinition API
- [ ] Implement ONNX parser integration
- [ ] Add FP16/INT8 precision support
- [ ] Implement GPU memory management (CUDA buffers)
- [ ] Add error handling and validation
- [ ] Test with 20+ WebNN operations

### Phase 3: Performance Optimization (3-5 days)
- [ ] Implement engine caching to disk
- [ ] Add engine serialization/deserialization
- [ ] Optimize memory allocation/deallocation
- [ ] Add batch size optimization
- [ ] Profile and benchmark vs ONNX Runtime

### Phase 4: Python Integration (2-3 days)
- [ ] Add Backend::TensorRT to context selection
- [ ] Implement `compute_tensorrt()` method
- [ ] Add NVIDIA GPU detection
- [ ] Add device selection logic (prefer TensorRT on NVIDIA)
- [ ] Test with Python API examples

### Phase 5: Documentation & Testing (2-3 days)
- [ ] Update docs/implementation-status.md with TensorRT coverage
- [ ] Update docs/architecture.md with TensorRT backend
- [ ] Create example: `examples/tensorrt_inference.py`
- [ ] Add comprehensive unit tests (Rust + Python)
- [ ] Document CUDA installation requirements
- [ ] Update README.md with TensorRT backend section

### Phase 6: Advanced Features (Future)
- [ ] TensorRT for RTX support (Windows 11)
- [ ] INT8 calibration for quantization
- [ ] Dynamic shape support
- [ ] Multi-stream execution
- [ ] DLA (Deep Learning Accelerator) support
- [ ] TensorRT-LLM integration for transformer models

**Total Estimated Time:** 14-21 days for phases 1-5

---

##  Testing Strategy

### Unit Tests (Rust)

**File:** `src/executors/tensorrt.rs`
```rust
#[cfg(all(test, feature = "tensorrt-runtime"))]
mod tests {
    use super::*;

    #[test]
    fn builds_engine_from_onnx() {
        let onnx_model = create_simple_add_onnx();
        let logger = create_logger();
        let builder = create_infer_builder(&logger).unwrap();
        assert!(builder.is_valid());
    }

    #[test]
    fn executes_add_operation() {
        if !is_nvidia_gpu_available() {
            eprintln!("Skipping test: No NVIDIA GPU available");
            return;
        }

        let onnx_model = create_simple_add_onnx();
        let inputs = create_test_inputs();
        let outputs = run_tensorrt_with_inputs(&onnx_model, inputs, TensorRTPrecision::FP32).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, vec![2, 3]);
        // Verify output values
    }

    #[test]
    fn fp16_precision_works() {
        // Test FP16 execution
    }

    #[test]
    fn engine_caching_works() {
        // Test cache hit/miss
    }
}
```

### Python Tests

**File:** `tests/test_tensorrt_backend.py`
```python
import pytest
import webnn
import numpy as np
import subprocess

def has_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def has_tensorrt_runtime():
    """Check if TensorRT runtime is available"""
    try:
        import webnn._rustnn as rustnn
        return hasattr(rustnn, 'tensorrt_available')
    except:
        return False

@pytest.mark.skipif(not has_nvidia_gpu(), reason="No NVIDIA GPU available")
@pytest.mark.skipif(not has_tensorrt_runtime(), reason="TensorRT runtime not available")
def test_tensorrt_add():
    ml = webnn.ML()
    context = ml.create_context(accelerated=True, power_preference="high-performance")

    # Should select TensorRT on NVIDIA GPU
    assert context.backend == "tensorrt"

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

@pytest.mark.skipif(not has_nvidia_gpu(), reason="No NVIDIA GPU available")
def test_tensorrt_fp16_precision():
    # Test FP16 execution
    pass

@pytest.mark.skipif(not has_nvidia_gpu(), reason="No NVIDIA GPU available")
def test_tensorrt_mobilenet():
    # Test full MobileNetV2 model on TensorRT
    pass
```

### Performance Benchmarks

**File:** `benchmarks/tensorrt_vs_onnx.py`
```python
import time
import webnn
import numpy as np

def benchmark_backend(backend_name, accelerated, power_preference):
    ml = webnn.ML()
    context = ml.create_context(accelerated=accelerated, power_preference=power_preference)

    # Build MobileNetV2 graph
    graph = build_mobilenetv2(context)

    # Warmup
    for _ in range(5):
        context.compute(graph, inputs)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        outputs = context.compute(graph, inputs)
        times.append(time.perf_counter() - start)

    return {
        "backend": backend_name,
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
    }

# Compare backends
onnx_gpu = benchmark_backend("ONNX GPU", True, "high-performance")
tensorrt = benchmark_backend("TensorRT", True, "high-performance")

print(f"ONNX GPU: {onnx_gpu['mean_ms']:.2f}ms ± {onnx_gpu['std_ms']:.2f}ms")
print(f"TensorRT: {tensorrt['mean_ms']:.2f}ms ± {tensorrt['std_ms']:.2f}ms")
print(f"Speedup: {onnx_gpu['mean_ms'] / tensorrt['mean_ms']:.2f}x")
```

### Makefile Targets

```makefile
# Add to Makefile
.PHONY: tensorrt-dev
tensorrt-dev:
	maturin develop --features python,tensorrt-runtime

.PHONY: test-tensorrt
test-tensorrt:
	cargo test --features tensorrt-runtime
	pytest tests/test_tensorrt_backend.py -v

.PHONY: benchmark-tensorrt
benchmark-tensorrt:
	python benchmarks/tensorrt_vs_onnx.py
```

---

##  References

### TensorRT Resources
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
- [TensorRT SDK](https://developer.nvidia.com/tensorrt)
- [TensorRT Architecture Overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html)
- [TensorRT for RTX (Windows 11)](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html)
- [TensorRT for RTX Announcement](https://developer.nvidia.com/blog/nvidia-tensorrt-for-rtx-introduces-an-optimized-inference-ai-library-on-windows/)
- [Run High-Performance AI with TensorRT for RTX](https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/)

### ONNX-TensorRT
- [ONNX-TensorRT GitHub](https://github.com/onnx/onnx-tensorrt)
- [Supported ONNX Operators](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
- [TensorRT Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)

### Rust Bindings
- [tensorrt-rs (GitHub)](https://github.com/mstallmo/tensorrt-rs)
- [tensorrt-rs (crates.io)](https://crates.io/crates/tensorrt-rs)
- [easy-tensorrt-sys (crates.io)](https://crates.io/crates/easy-tensorrt-sys)
- [TensorRT-sys](https://lib.rs/crates/tensorrt-sys)

### WebNN Spec
- [W3C WebNN API Specification](https://www.w3.org/TR/webnn/)
- [WebNN Device Selection Explainer](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md)

### Related Projects
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Torch-TensorRT](https://github.com/pytorch/TensorRT)

---

##  Summary

**TensorRT Integration Value:**
- [OK] **Best GPU performance** on NVIDIA hardware (RTX, A100, H100)
- [OK] **Advanced quantization** (FP16, INT8, FP8, FP4)
- [OK] **Production-ready** (widely deployed in NVIDIA ecosystem)
- [OK] **ONNX-native** (reuse existing ONNX converter)
- [OK] **95%+ operation coverage** (300+ ONNX ops)
- [OK] **TensorRT for RTX** (optimized for Windows 11 + RTX GPUs)

**Key Design Decisions:**
1. **Reuse ONNX converter** (no new converter needed!)
2. **Custom FFI bindings** for TensorRT 10.x C++ API
3. **Engine caching** to avoid rebuild overhead
4. **FP16 default** for 2x speedup over FP32
5. **Prefer TensorRT** on NVIDIA GPUs with `accelerated=True` + `power="high-performance"`
6. **Graceful fallback** to ONNX Runtime if TensorRT unavailable

**Platform Support:**
- **Primary**: Linux + NVIDIA GPU (CUDA)
- **Secondary**: Windows 11 + NVIDIA RTX GPU (TensorRT for RTX)
- **Not supported**: macOS (no NVIDIA GPU), AMD GPUs

**Next Steps:**
1. Create FFI bindings for TensorRT 10.x
2. Implement basic executor with FP32 support
3. Add FP16/INT8 precision modes
4. Implement engine caching
5. Integrate with Python API
6. Benchmark vs ONNX Runtime GPU

---

**Status:** Planning document (not yet implemented)

**Estimated Effort:** 14-21 days for full integration with caching and FP16/INT8 support
