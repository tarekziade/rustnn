# Release Notes - v0.2.0

**Release Date:** December 8, 2024

This is a major feature release that transforms rustnn from a graph validation and conversion tool into a **fully functional WebNN implementation** with real execution capabilities, 85 implemented operations, and production-ready examples.

---

## [SUCCESS] Highlights

### 85 WebNN Operations Fully Implemented (89% Spec Coverage)
Complete implementation of 85 WebNN operations across all backends:
- **Shape Inference**: 85/85 operations (100%)
- **Python API**: 85/85 operations (100%)
- **ONNX Backend**: 85/85 operations (100%)
- **CoreML MLProgram**: 85/85 operations (100%)

### Real Execution Engine
- **ONNX Runtime Integration**: Execute graphs with actual tensor I/O on CPU and GPU
- **CoreML Execution**: Native macOS GPU and Neural Engine support via MLProgram backend
- **NumPy Integration**: Seamless input/output with NumPy arrays
- **Multi-Backend Support**: Run the same graph on CPU, GPU, or Neural Engine

### Production Examples
- **MobileNetV2**: Complete 106-layer pretrained model achieving 99.60% accuracy
- **Text Generation**: Transformer architecture with attention mechanism
- **Model Training**: Train models with gradient descent and save weights

### W3C WebNN Spec Compliance
- Implements W3C WebNN Device Selection Explainer
- Implements W3C WebNN MLTensor Explainer
- Includes W3C Web Platform Tests (WPT) for conformance validation

---

##  New Features

### Operations (85 Total)

**Binary Operations (6):**
- `add`, `sub`, `mul`, `div`, `matmul`, `pow`

**Activation Functions (11):**
- Standard: `relu`, `sigmoid`, `tanh`, `softmax`
- Specialized: `prelu`, `elu`, `leakyRelu`, `hardSigmoid`, `hardSwish`, `softplus`, `softsign`

**Element-wise Math (23):**
- Basic: `abs`, `ceil`, `floor`, `round`, `neg`, `sign`, `identity`, `reciprocal`
- Exponential: `exp`, `log`, `sqrt`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
- Special: `erf`

**Convolution & Pooling (6):**
- `conv2d`, `conv_transpose2d` (with full parameter support: strides, dilations, padding, groups)
- `average_pool2d`, `max_pool2d`, `global_average_pool`, `global_max_pool`

**Normalization (3):**
- `batch_normalization`, `instance_normalization`, `layer_normalization`

**Reduction (10):**
- `reduce_sum`, `reduce_mean`, `reduce_max`, `reduce_min`, `reduce_product`
- `reduce_l1`, `reduce_l2`, `reduce_log_sum`, `reduce_log_sum_exp`, `reduce_sum_square`

**Logic Operations (9):**
- Comparison: `equal`, `greater`, `greater_or_equal`, `lesser`, `lesser_or_equal`
- Logical: `logical_not`, `logical_and`, `logical_or`, `logical_xor`

**Tensor Manipulation (8):**
- `transpose`, `concat`, `slice`, `expand`, `gather`, `split`, `where`, `pad`

**Advanced Operations (9):**
- Shape: `reshape`, `squeeze`, `unsqueeze`
- Selection: `argMax`, `argMin`
- Activation: `gelu`
- Type: `cast`
- Advanced: `gemm`, `clamp`

**Additional Features (4):**
- `scatterElements`, `scatterND`, `tile`, `triangular`

**Quantization (2):**
- `quantize_linear`, `dequantize_linear`

### Execution & Runtime

**Async Execution:**
- `AsyncMLContext` class for non-blocking operations
- Python asyncio integration
- `dispatch()` method for async graph execution

**MLTensor Management:**
- Explicit tensor creation with `create_tensor()`
- Descriptor flags: `readable`, `writable`, `exportable_to_gpu`
- Synchronous I/O: `read_tensor()`, `write_tensor()`
- Explicit cleanup: `destroy()`

**Backend Selection:**
- Runtime device selection via `accelerated` and `power_preference` hints
- Platform-autonomous device allocation
- Query actual capability with `context.accelerated` property
- Supports: ONNX CPU, ONNX GPU, CoreML (GPU/Neural Engine)

### Examples & Documentation

**Production Examples:**
- `mobilenetv2_complete.py` - Complete 106-layer pretrained MobileNetV2
- `text_generation_gpt.py` - Transformer with attention mechanism
- `text_generation_enhanced.py` - Enhanced version with KV caching
- `train_text_model.py` - Model training with gradient descent
- `image_classification.py` - Complete classification pipeline

**Documentation:**
- Complete API reference with execution examples
- Getting started guide with `compute()` examples
- Advanced usage patterns
- Architecture documentation
- Development guide with Makefile workflow
- Operator implementation status tracking

### Developer Experience

**Makefile Targets:**
- `make python-dev` - Install package with ONNX Runtime
- `make python-test` - Run all tests including WPT conformance
- `make mobilenet-demo` - Run MobileNetV2 on all backends
- `make text-gen-demo` - Run text generation demo
- `make text-gen-train` - Train text generation model
- `make docs-serve` - Serve documentation with live reload

**Pre-commit Hooks:**
- Automatic `cargo fmt` formatting before commits
- Install with `./scripts/install-git-hooks.sh`

**WPT Conformance:**
- Integration with W3C Web Platform Tests
- Conformance test data for all operations
- Automated testing with `make python-test-wpt`

---

##  Major Changes

### CoreML Backend Migration
- **Complete migration from NeuralNetwork to MLProgram format**
- Modern MIL (Model Intermediate Language) operations
- Full parameter support for all operations
- End-to-end execution on macOS GPU and Neural Engine
- Requires iOS 18+ / macOS 15+ (CoreML spec version 9)

### Project Rename
- **rust-webnn-graph → rustnn**
- New PyPI package: `pywebnn`
- Updated GitHub repository and documentation URLs

### Shape Inference & Broadcasting
- Automatic shape computation for all operations
- NumPy-style broadcasting support
- Validation at graph build time

### Python API Enhancements
- Real execution with `compute()` method
- Returns actual computed NumPy arrays
- Multi-backend support with runtime selection
- Async execution support

---

##  Bug Fixes

- Fixed ONNX model format compatibility with older ONNX Runtime versions
- Fixed logic operations Cast node type conversion for onnxruntime-rs v0.0.14
- Fixed CoreML constant handling and complete gemm support
- Fixed Python examples to include proper environment setup
- Fixed all unsafe_op_in_unsafe_fn warnings (271 → 0)
- Fixed Rust compiler warnings (278 → 0)
- Fixed CI to build Python package with ONNX runtime support
- Fixed async dispatch tests and tensor workflow tests

---

## [STATS] Statistics

- **97 commits** since v0.1.0
- **85 operations** implemented (89% of WebNN spec)
- **320+ tests** (Python API + WPT conformance + Rust)
- **0 compiler warnings** (clean codebase)
- **3 execution backends** (ONNX CPU, ONNX GPU, CoreML)

---

##  Breaking Changes

### API Changes
- `device_type` parameter replaced with `accelerated` (bool) and `power_preference` (str)
  ```python
  # Old (v0.1.0)
  context = ml.create_context(device_type="cpu")

  # New (v0.2.0)
  context = ml.create_context(accelerated=False)
  # Or: context = ml.create_context(accelerated=True, power_preference="high-performance")
  ```

### Project Name
- Package name changed: `rust-webnn-graph` → `rustnn`
- Python package on PyPI: `pywebnn`
- Repository URL unchanged (GitHub handles redirect)

### Requirements
- Python 3.11+ (was 3.8+ in v0.1.0)
- CoreML requires iOS 18+ / macOS 15+ for MLProgram backend

---

## [PACKAGE] Installation

### From PyPI
```bash
pip install pywebnn
```

### From Source
```bash
git clone https://github.com/tarekziade/rustnn.git
cd rustnn
make python-dev
```

---

## [TARGET] What's Next (v0.3.0)

- Additional specialized activations (~6 operations)
- Graph optimization passes
- Multi-platform wheel building (manylinux, Windows)
- Performance benchmarks
- Additional production examples

---

##  Acknowledgments

- W3C WebNN Community Group for the specification
- Chromium WebNN implementation for reference
- ONNX and CoreML teams for runtime support
- PyO3 project for excellent Python-Rust bindings

---

**Full Changelog**: https://github.com/tarekziade/rustnn/compare/v0.1.0...v0.2.0
