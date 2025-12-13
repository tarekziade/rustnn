# WebNN Implementation Status & Testing Strategy

**Last Updated:** 2025-12-13

## Executive Summary

rustnn implements 85 of ~95 WebNN operations (89% coverage) with full backend support across ONNX Runtime, CoreML MLProgram, and TensorRT. WPT test infrastructure exists but test data conversion is incomplete.

**Current Status:**
- [OK] 85 operations fully implemented (Shape Inference + Python API + ONNX + CoreML)
- [OK] WPT test infrastructure in place
- [WARNING] WPT test data files empty - conversion from JavaScript incomplete
- [WARNING] 260 Python API tests exist but skipped (runtime dependencies)

---

## Implementation Status by Operation Category

**Legend:**
- [OK] = Fully implemented across all backends
- [PAUSE] = Partially implemented
-  = Not implemented
- ⏭ = Intentionally deferred

### Binary Operations (6/6 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `add` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `sub` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `mul` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `div` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `matmul` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `pow` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |

### Activation Functions (4/4 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `relu` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `sigmoid` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `tanh` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `softmax` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |

### Specialized Activations (7/7 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `prelu` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `elu` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `leakyRelu` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `hardSigmoid` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `hardSwish` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `softplus` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `softsign` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Element-wise Math (11/11 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `abs` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `ceil` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `floor` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `round` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `neg` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `sign` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `exp` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `log` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `sqrt` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reciprocal` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `identity` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Trigonometric (6/6 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `sin` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `cos` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `tan` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `asin` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `acos` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `atan` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Hyperbolic (5/5 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `sinh` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `cosh` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `asinh` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `acosh` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `atanh` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Special Functions (1/1 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `erf` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Logic Operations (9/9 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `equal` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `greater` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `greater_or_equal` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `lesser` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `lesser_or_equal` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `logical_not` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `logical_and` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `logical_or` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `logical_xor` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Convolution (2/2 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `conv2d` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `conv_transpose2d` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |

### Pooling (4/4 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `average_pool2d` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `max_pool2d` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `global_average_pool` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `global_max_pool` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Normalization (3/3 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `batch_normalization` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `instance_normalization` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `layer_normalization` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |

### Reduction (10/10 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `reduce_sum` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reduce_mean` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reduce_max` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reduce_min` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reduce_product` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reduce_l1` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reduce_l2` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reduce_log_sum` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reduce_log_sum_exp` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `reduce_sum_square` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |

### Quantization (2/2 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `dequantize_linear` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `quantize_linear` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Shape Operations (1/1 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `reshape` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |

### Tensor Manipulation (8/8 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `transpose` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `concat` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `slice` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `expand` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `gather` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `split` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |
| `where` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `pad` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Advanced Architecture Operations (6/6 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `gelu` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `squeeze` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `unsqueeze` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `argMax` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `argMin` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `cast` | [OK] | [OK] | [OK] | [OK] | [WARNING] Empty |

### Additional Features (4/4 - 100%)

| Operation | Shape Inference | Python API | ONNX | CoreML | WPT Data |
|-----------|-----------------|------------|------|--------|----------|
| `scatterElements` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `scatterND` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `tile` | [OK] | [OK] | [OK] | [OK] | Status unknown |
| `triangular` | [OK] | [OK] | [OK] | [OK] | Status unknown |

### Deferred Operations (4 operations)

| Operation | Status | Rationale |
|-----------|--------|-----------|
| `lstm` | ⏭ Deferred | Complex RNN operation; Transformers more common; spec under review |
| `lstmCell` | ⏭ Deferred | Complex RNN operation; lower priority |
| `gru` | ⏭ Deferred | Complex RNN operation; Transformers more common |
| `gruCell` | ⏭ Deferred | Complex RNN operation; lower priority |

**Deferral Rationale:** Each operation requires 10-15 parameters with complex shape inference (~2000-3000 LOC total). Active [W3C discussion](https://github.com/webmachinelearning/webnn/issues/453) about removing these in favor of lower-level primitives. Modern ML trends favor Transformer architectures over LSTM/GRU.

---

## Summary Statistics

```
WebNN Spec (CR Draft Dec 2025):  ~95 total operations
Core Operations:                  68/68 (100%) [OK]
Specialized Activations:           7/7  (100%) [OK]
Advanced Architecture Ops:         6/6  (100%) [OK]
Additional Features:               4/4  (100%) [OK]
Total Implemented:                85/95 (89%) [OK]
Deferred Operations:               4     (RNN operations)
Remaining Operations:             ~6     (specialized activations)

Backend Coverage:
Shape Inference:                  85/85 (100%) [OK]
Python API:                       85/85 (100%) [OK]
ONNX Backend:                     85/85 (100%) [OK]
CoreML MLProgram:                 85/85 (100%) [OK]

Test Coverage:
Python API Tests:                 260 tests (all skipped - runtime deps)
WPT Test Infrastructure:          [OK] Complete
WPT Test Data Files:              54 conformance + 17 validation
WPT Test Data Populated:          [WARNING] 0/71 (test data empty)
```

---

## WPT Integration Status

### What Exists

[OK] **Infrastructure:**
- `tests/wpt_data/` directory with conformance/ and validation/ subdirectories
- `tests/test_wpt_conformance.py` - Test runner framework
- `tests/wpt_utils.py` - ULP distance calculation, tolerance checking
- `scripts/convert_wpt_tests.py` - Test data converter
- `scripts/update_wpt_tests.sh` - Update automation script

[OK] **Test Data Files:**
- 54 conformance test JSON files created
- 17 validation test JSON files created
- Files include metadata: operation name, WPT version, commit SHA, source file

[WARNING] **Critical Gap:**
- All test data files have empty "tests": [] arrays
- Test data conversion from WPT JavaScript to JSON incomplete
- 0 actual test cases available despite infrastructure

### Why WPT Tests Are Skipped

Current test run shows: `54 skipped in 0.02s`

**Root Cause:** Test data files exist but contain no test cases:
```json
{
  "operation": "relu",
  "wpt_version": "2025-12-08",
  "wpt_commit": "25b26d4cfd4e1ca1b288849c03fa58ee7b049ef1",
  "source_file": "webnn/conformance_tests/relu.https.any.js",
  "tests": []  # <-- Empty!
}
```

---

## Next Steps (Prioritized)

### Priority 1: Complete WPT Test Data Conversion (HIGH IMPACT)

**Goal:** Populate WPT test data files with actual test cases from upstream WPT repository

**Current Blocker:** `scripts/convert_wpt_tests.py` exists but hasn't successfully converted test cases from JavaScript to JSON format.

**Action Items:**
1. **Fix test data converter** (`scripts/convert_wpt_tests.py`)
   - Debug why conversion produces empty test arrays
   - Verify JavaScript parsing logic handles WPT test format
   - Test with 2-3 simple operations first (relu, add, reshape)
   - Expected output: JSON files with populated "tests" arrays

2. **Convert high-priority operations** (Tier 1 - 29 ops)
   - Binary: add, sub, mul, div, matmul, pow
   - Activations: relu, sigmoid, tanh, softmax
   - Reductions: all 10 reduce operations
   - Pooling: averagePool2d, maxPool2d, globalAveragePool, globalMaxPool
   - Convolution: conv2d, convTranspose2d
   - Normalization: batchNormalization, instanceNormalization, layerNormalization
   - Shape: reshape

3. **Verify test execution**
   ```bash
   pytest tests/test_wpt_conformance.py -k "relu" -v
   ```
   - Should see actual test cases running (not skipped)
   - Fix any tolerance or execution issues

**Expected Outcome:**
- 100+ WPT conformance tests passing
- Numerical correctness validated against official W3C test suite
- CI pipeline running WPT tests on every PR

**Estimated Effort:** 1-2 days

---

### Priority 2: Enable Python API Tests (MEDIUM IMPACT)

**Goal:** Diagnose why 260 Python API tests are skipped and enable execution

**Current Issue:** All Python API tests skipped, likely due to missing ONNX Runtime or other dependencies.

**Action Items:**
1. **Investigate skip conditions**
   ```bash
   pytest tests/test_python_api.py -v --collect-only
   ```
   - Identify why tests are marked as skipped
   - Check for missing pytest markers (e.g., `pytest.mark.asyncio` warning)

2. **Fix runtime dependencies**
   - Ensure ONNX Runtime properly installed: `pip install onnxruntime`
   - Verify `webnn` Python module built: `maturin develop --features python`
   - Check for feature flags or environment variables required

3. **Run tests and document results**
   ```bash
   pytest tests/test_python_api.py -v
   cargo test --lib
   ```

**Expected Outcome:**
- Python API tests passing (or failing with actionable errors)
- Clear documentation of which tests require specific backends
- Skipped tests only for unavailable backends (TensorRT on macOS, CoreML on Linux)

**Estimated Effort:** 4-6 hours

---

### Priority 3: Document Remaining Operations (LOW IMPACT)

**Goal:** Complete WebNN specification coverage analysis

**Action Items:**
1. **Identify remaining ~6 operations** from WebNN spec not yet implemented
2. **Assess priority** based on:
   - Usage in popular models (BERT, ResNet, etc.)
   - Complexity of implementation
   - Backend support availability
3. **Update TODO.txt** with findings

**Expected Outcome:**
- Clear roadmap for reaching 95/95 (100%) operation coverage
- Priority ranking for next implementation phase

**Estimated Effort:** 2-3 hours

---

### Priority 4: CI/CD Integration (MEDIUM IMPACT)

**Goal:** Automate WPT tests in continuous integration pipeline

**Prerequisites:** Priority 1 must be complete (WPT test data populated)

**Action Items:**
1. **Add WPT tests to CI workflow** (`.github/workflows/`)
   - Run on every PR
   - Generate coverage report
   - Fail build on test failures
2. **Create test matrix**
   - Test on multiple platforms (Linux, macOS, Windows)
   - Test with different backends (ONNX CPU, ONNX GPU, CoreML)
3. **Add status badges** to README.md

**Expected Outcome:**
- Automated validation of every code change
- Visible test status for contributors
- Regression prevention

**Estimated Effort:** 4-6 hours (after Priority 1 complete)

---

## Testing Strategy Details

### WPT Test Structure

**Conformance Tests** (`tests/wpt_data/conformance/`)
- Validate numerical correctness of operations
- Use ULP (Units in Last Place) or ATOL (absolute tolerance) based checking
- Test multiple input shapes, data types, and parameter combinations

**Validation Tests** (`tests/wpt_data/validation/`)
- Validate parameter constraints and error handling
- Test invalid inputs produce correct error messages
- Test boundary conditions

### Tolerance Checking

The `wpt_utils.py` module implements WPT-compatible precision validation:

```python
def ulp_distance(a: float, b: float, dtype: str) -> int:
    """Calculate ULP distance between two floating-point values"""
    # Handles float32 and float16
    # Returns number of representable values between a and b
```

**Per-Operation Tolerances:**
- `relu`: 0 ULP (exact)
- `sigmoid`: 34 ULP (float32), 3 ULP (float16)
- `tanh`: 44 ULP (float32), 4 ULP (float16)
- `reduce_*`: Varies based on input size (accumulation error)

### Running Tests

```bash
# Run all WPT conformance tests (when data populated)
pytest tests/test_wpt_conformance.py -v

# Run tests for specific operation
pytest tests/test_wpt_conformance.py -k "reduce_sum" -v

# Run with coverage report
pytest tests/test_wpt_conformance.py --cov=webnn --cov-report=html

# Run Python API tests (when runtime available)
pytest tests/test_python_api.py -v

# Run all tests
make python-test
```

---

## References

- **W3C WebNN Specification:** https://www.w3.org/TR/webnn/
- **WPT WebNN Tests:** https://github.com/web-platform-tests/wpt/tree/master/webnn
- **Local WebNN Spec Reference:** `docs/webnn-spec-reference.md`
- **API Reference:** `docs/api-reference.md`
- **Development Guide:** `docs/development.md`

---

## Revision History

- **2025-12-13:** Merged operator-status.md and wpt-integration-plan.md; identified WPT test data gap as critical blocker
- **2025-12-08:** 85 operations fully implemented; CoreML end-to-end execution verified
- **2025-12-07:** WPT test infrastructure created; test data files initialized (empty)

---

**Document Status:** Living Document - Update after major implementation milestones
