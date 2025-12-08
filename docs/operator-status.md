# WebNN Operator Implementation Status

This document tracks the implementation status of all WebNN operators across different backends.

**Legend:**
- ✅ = Fully implemented
- ⏸️ = Partially implemented (shape inference only, or missing parameters)
- ❌ = Not implemented

**Last Updated:** 2025-12-08

---

## Binary Operations

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `add` | ✅ | ✅ | ✅ | ✅ |
| `sub` | ✅ | ✅ | ✅ | ✅ |
| `mul` | ✅ | ✅ | ✅ | ✅ |
| `div` | ✅ | ✅ | ✅ | ✅ |
| `matmul` | ✅ | ✅ | ✅ | ✅ |
| `pow` | ✅ | ✅ | ✅ | ✅ |

## Activation Functions

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `relu` | ✅ | ✅ | ✅ | ✅ |
| `sigmoid` | ✅ | ✅ | ✅ | ✅ |
| `tanh` | ✅ | ✅ | ✅ | ✅ |
| `softmax` | ✅ | ✅ | ✅ | ✅ |

## Element-wise Math

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `abs` | ✅ | ✅ | ✅ | ✅ |
| `ceil` | ✅ | ✅ | ✅ | ✅ |
| `floor` | ✅ | ✅ | ✅ | ✅ |
| `round` | ✅ | ✅ | ✅ | ✅ |
| `neg` | ✅ | ✅ | ✅ | ✅ |
| `sign` | ✅ | ✅ | ✅ | ✅ |
| `exp` | ✅ | ✅ | ✅ | ✅ |
| `log` | ✅ | ✅ | ✅ | ✅ |
| `sqrt` | ✅ | ✅ | ✅ | ✅ |
| `reciprocal` | ✅ | ✅ | ✅ | ✅ |
| `identity` | ✅ | ✅ | ✅ | ✅ |

## Trigonometric

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `sin` | ✅ | ✅ | ✅ | ✅ |
| `cos` | ✅ | ✅ | ✅ | ✅ |
| `tan` | ✅ | ✅ | ✅ | ✅ |
| `asin` | ✅ | ✅ | ✅ | ✅ |
| `acos` | ✅ | ✅ | ✅ | ✅ |
| `atan` | ✅ | ✅ | ✅ | ✅ |

## Hyperbolic

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `sinh` | ✅ | ✅ | ✅ | ✅ |
| `cosh` | ✅ | ✅ | ✅ | ✅ |
| `asinh` | ✅ | ✅ | ✅ | ✅ |
| `acosh` | ✅ | ✅ | ✅ | ✅ |
| `atanh` | ✅ | ✅ | ✅ | ✅ |

## Special Functions

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `erf` | ✅ | ✅ | ✅ | ✅ |

## Logic Operations

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `equal` | ✅ | ✅ | ✅ | ✅ |
| `greater` | ✅ | ✅ | ✅ | ✅ |
| `greater_or_equal` | ✅ | ✅ | ✅ | ✅ |
| `lesser` | ✅ | ✅ | ✅ | ✅ |
| `lesser_or_equal` | ✅ | ✅ | ✅ | ✅ |
| `logical_not` | ✅ | ✅ | ✅ | ✅ |
| `logical_and` | ✅ | ✅ | ✅ | ✅ |
| `logical_or` | ✅ | ✅ | ✅ | ✅ |
| `logical_xor` | ✅ | ✅ | ✅ | ✅ |

## Convolution

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `conv2d` | ✅ | ✅ | ✅ | ✅ |
| `conv_transpose2d` | ✅ | ✅ | ✅ | ✅ |

## Pooling

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `average_pool2d` | ✅ | ✅ | ✅ | ✅ |
| `max_pool2d` | ✅ | ✅ | ✅ | ✅ |
| `global_average_pool` | ✅ | ✅ | ✅ | ✅ |
| `global_max_pool` | ✅ | ✅ | ✅ | ✅ |

## Normalization

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `batch_normalization` | ✅ | ✅ | ✅ | ✅ |
| `instance_normalization` | ✅ | ✅ | ✅ | ✅ |
| `layer_normalization` | ✅ | ✅ | ✅ | ✅ |

## Reduction

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `reduce_sum` | ✅ | ✅ | ✅ | ✅ |
| `reduce_mean` | ✅ | ✅ | ✅ | ✅ |
| `reduce_max` | ✅ | ✅ | ✅ | ✅ |
| `reduce_min` | ✅ | ✅ | ✅ | ✅ |
| `reduce_product` | ✅ | ✅ | ✅ | ✅ |
| `reduce_l1` | ✅ | ✅ | ✅ | ✅ |
| `reduce_l2` | ✅ | ✅ | ✅ | ✅ |
| `reduce_log_sum` | ✅ | ✅ | ✅ | ✅ |
| `reduce_log_sum_exp` | ✅ | ✅ | ✅ | ✅ |
| `reduce_sum_square` | ✅ | ✅ | ✅ | ✅ |

## Quantization

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `dequantize_linear` | ✅ | ✅ | ✅ | ✅ |
| `quantize_linear` | ✅ | ✅ | ✅ | ✅ |

## Shape Operations

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `reshape` | ✅ | ✅ | ✅ | ✅ |

## Tensor Manipulation

| Operation | Shape Inference | Python API | ONNX | CoreML MLProgram |
|-----------|----------------|------------|------|------------------|
| `transpose` | ✅ | ✅ | ✅ | ✅ |
| `concat` | ✅ | ✅ | ✅ | ✅ |
| `slice` | ✅ | ✅ | ✅ | ✅ |
| `expand` | ✅ | ✅ | ✅ | ✅ |
| `gather` | ✅ | ✅ | ✅ | ✅ |
| `split` | ✅ | ✅ | ✅ | ✅ |
| `where` | ✅ | ✅ | ✅ | ✅ |
| `pad` | ✅ | ✅ | ✅ | ✅ |

---

## Summary Statistics

```
WebNN Spec (CR Draft Dec 2025): ~95 total operations
Core Operations Implemented:     68/68 (100%) ✅
Deferred Operations:              4 (RNN: lstm, lstmCell, gru, gruCell)
Remaining Operations:             ~23 (advanced tensor ops, additional activations)

Implementation Status:
Shape Inference:                  68/68 (100%)
Python API:                       68/68 (100%)
ONNX Backend:                     68/68 (100%)
CoreML MLProgram:                 68/68 (100%) ✅
```

**🎉 CORE OPERATIONS FULLY IMPLEMENTED! 🎉**

### Implementation Status

All 68 core WebNN operations are now fully implemented across all backends:
- ✅ **Shape Inference**: Complete type and shape validation for all operations
- ✅ **Python API**: W3C WebNN spec-compliant Python bindings
- ✅ **ONNX Backend**: Cross-platform execution with full parameter support
- ✅ **CoreML MLProgram**: macOS GPU/Neural Engine execution with full parameter support

**Recent Additions:**
- **Tensor Manipulation Operations (8 operations):** `transpose`, `concat`, `slice`, `expand`, `gather`, `split`, `where`, `pad`
  - Full implementation across all backends (shape inference, Python API, ONNX, CoreML)
  - 46 comprehensive Python tests covering various scenarios
  - Essential for Transformers, CNNs, and modern ML architectures
- Added full parameter support (strides, dilations, pads, groups, epsilon, etc.) for:
  - Convolution operations: `conv2d`, `conv_transpose2d`
  - Pooling operations: `average_pool2d`, `max_pool2d`
  - Normalization operations: `batch_normalization`, `instance_normalization`, `layer_normalization`

---

## Deferred Operations

The following operations are defined in the WebNN specification but are **intentionally deferred** for later implementation:

### Recurrent Neural Networks (4 operations)

| Operation | Status | Rationale |
|-----------|--------|-----------|
| `lstm` | ⏭️ Deferred | Complex composite operation; spec under review; Transformers more common |
| `lstmCell` | ⏭️ Deferred | Complex composite operation; lower priority than simpler ops |
| `gru` | ⏭️ Deferred | Complex composite operation; spec under review; Transformers more common |
| `gruCell` | ⏭️ Deferred | Complex composite operation; lower priority than simpler ops |

**Deferral Rationale:**
- **Complexity**: Each operation requires 10-15 parameters with complex shape inference (~2000-3000 LOC total)
- **Spec Evolution**: Active [W3C discussion](https://github.com/webmachinelearning/webnn/issues/453) about removing these in favor of lower-level primitives
- **Modern ML Trends**: LSTM/GRU largely obsoleted by Transformer architectures
- **Priority**: Simpler, more widely-used operations should be implemented first
- **Test Coverage**: WPT tests exist but can be added when/if implementation is prioritized

### Priority Operations for Next Implementation

Based on modern ML architecture requirements, the following operations should be prioritized:

**High Priority (Advanced architectures):**
- `gelu` - GELU activation (Transformers)
- `squeeze` / `unsqueeze` - Dimension manipulation
- `argMax` / `argMin` - Find indices of extreme values
- `cast` - Type conversion

**Medium Priority (Additional features):**
- `softmax` parameters - Add axis parameter
- `scatter` - Scatter updates
- `tile` - Repeat tensor
- `triangular` - Extract triangular part

**Lower Priority (Specialized activations):**
- `prelu`, `elu`, `leakyRelu` - Additional activations
- `hardSigmoid`, `hardSwish`, `softplus`, `softsign` - Specialized activations

---

## Notes

### ONNX Backend
The ONNX converter has a default fallback mechanism that capitalizes the first letter of any operation name. This means it automatically supports all WebNN operations without requiring explicit mappings.

**Example:**
```rust
// Default: capitalize first letter
"round" → "Round"
"asin" → "Asin"
"globalAveragePool" → "GlobalAveragePool"
```

### CoreML MLProgram Backend
The CoreML MLProgram converter uses explicit operation mappings to MIL (Model Intermediate Language) operations. Operations not explicitly mapped will fail during conversion with an error.

**Implementation Location:** `src/converters/coreml_mlprogram.rs`

### Implementation Priority

**Phase 1 - Simple Operations (Quick Wins):**
1. Global pooling: `global_average_pool`, `global_max_pool`
2. Element-wise basic: `round`, `neg`, `identity`
3. Binary: `pow`

**Phase 2 - Transcendental Functions:**
4. Trigonometric: `asin`, `acos`, `atan`
5. Hyperbolic: `sinh`, `cosh`, `asinh`, `acosh`, `atanh`

**Phase 3 - Parameter Handling:**
6. Complete parameter handling for conv/pool/norm operations (requires MIL Value creation)

### MIL Operation Names

CoreML MIL operation names for missing operations:
- `global_average_pool` → `"reduce_mean"` (with axes parameter)
- `global_max_pool` → `"reduce_max"` (with axes parameter)
- `round` → `"round"`
- `neg` → `"mul"` (multiply by -1) or `"neg"` if available
- `identity` → `"identity"`
- `pow` → `"pow"`
- `asin` → `"asin"`
- `acos` → `"acos"`
- `atan` → `"atan"`
- `sinh` → `"sinh"`
- `cosh` → `"cosh"`
- `asinh` → `"asinh"`
- `acosh` → `"acosh"`
- `atanh` → `"atanh"`
