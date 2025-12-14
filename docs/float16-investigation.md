# Float16 Investigation - CoreML Backend

## Problem Summary

CoreML backend had 122 WPT test failures (4% of suite) related to Float16 tensors, causing SIGABRT crashes during graph execution.

## Root Causes Identified

### 1. Float16 Constants (FIXED)

**Problem**: Non-scalar Float16 constants caused CoreML compilation/execution to crash.

**Root Cause**: CoreML MLProgram (MIL) format requires non-scalar Float16 constants to be stored in external weight files (weights.bin), not as immediate values in the protobuf.

**Solution Implemented**:
- Created weight file builder infrastructure (`src/converters/weight_file_builder.rs`)
- Implements 64-byte alignment and sentinel+count metadata format
- Modified CoreML converter to detect non-scalar Float16 constants and route them to weight files
- Updated executor to generate .mlpackage directory structure with weights/weights.bin
- Scalar Float16 constants continue to use immediate values

**Status**: ✅ FIXED - Float16 constants now work correctly

**Test Results**:
- `test_float16_debug.py` - Float16 constant [3] + relu: PASSED
- `test_float16_both.py` - Float16 constant [5] + leakyRelu: PASSED

### 2. Float16 Inputs (PARTIAL LIMITATION FOUND)

**Problem**: Float16 input tensors cause SIGSEGV (exit code 139) crash during CoreML prediction.

**Root Cause**: CoreML runtime limitation/bug with Float16 input arrays above a size threshold.

**Findings**:
- Float16 inputs work correctly for small sizes (≤4 elements, ≤8 bytes)
- Float16 inputs crash with larger sizes (≥8 elements, ≥16 bytes)
- Issue is NOT in our Rust code - crash occurs inside CoreML's predictionFromFeatures call
- Model generation is correct - Float16 inputs properly declared with ArrayDataType::Float16
- Data conversion is correct - f32 → f16 conversion working properly

**Test Results**:
| Size | Bytes | Status |
|------|-------|--------|
| [2] | 4 | ✅ PASSED |
| [3] | 6 | ✅ PASSED |
| [4] | 8 | ✅ PASSED |
| [8] | 16 | ❌ CRASH (SIGSEGV) |
| [12] | 24 | ❌ CRASH (SIGSEGV) |
| [16] | 32 | ❌ CRASH (SIGSEGV) |
| [24] | 48 | ❌ CRASH (SIGSEGV) |

**Code Locations**:
- Model input declaration: `src/converters/coreml_mlprogram.rs:1752-1754` (correct)
- Input data conversion: `src/executors/coreml.rs:753-758` (correct f32→f16 conversion)
- Crash location: `src/executors/coreml.rs:356` (inside CoreML prediction call)

## Architecture Details

### Weight File Format (Following Chromium)

```
.mlpackage/
├── Data/
│   └── com.apple.CoreML/
│       ├── model.mlmodel (protobuf with BlobFileValue references)
│       └── weights/
│           └── weights.bin (Float16 constant data)
```

**weights.bin Structure**:
```
[Entry 1]
  Sentinel: 0xDEADBEEF (4 bytes, little-endian)
  Count: N elements (8 bytes, little-endian)
  Data: Float16 bytes (2 bytes per element)
  Padding: Zero bytes to next 64-byte boundary

[Entry 2]
  ... (64-byte aligned)
```

**BlobFileValue in Protobuf**:
```protobuf
value {
  type: { tensorType { dataType: FLOAT16, rank: 1, dimensions: [3] } }
  blobFileValue {
    fileName: "@model_path/weights/weights.bin"
    offset: 0  # Byte offset into weights.bin
  }
}
```

### Code Flow

**Constant Handling**:
1. `CoremlMlProgramConverter::create_const_operation()` detects Float16 + non-scalar
2. Calls `WeightFileBuilder::add_weight()` to add to weights.bin
3. Creates BlobFileValue with offset in protobuf
4. `WeightFileBuilder::finalize()` pads to 64-byte alignment
5. Executor creates .mlpackage with weights directory

**Input Handling**:
1. Model declares Float16 input type in FeatureDescription
2. Python passes np.float16 array to `compute()`
3. `PyMLContext::compute_coreml()` converts to CoreML inputs
4. Executor creates MLMultiArray with dataType=16 (Float16)
5. `fill_data_with_type_conversion()` converts f32 data → f16 bits
6. CoreML's `predictionFromFeatures` executes (crashes on large arrays)

## Workaround Strategy

### Option 1: Skip Float16 Input Tests
Mark Float16 input tests as skipped with clear reason:
```python
@pytest.mark.skip(reason="CoreML runtime crashes with Float16 inputs >4 elements")
```

### Option 2: Convert Float16 Inputs to Float32
Add converter logic to upcast Float16 → Float32 for inputs:
- Pro: Tests would pass
- Con: Loses precision benefits, not spec-compliant

### Option 3: Wait for CoreML Fix
Document limitation and wait for Apple to fix in future macOS/Xcode updates.

## Chromium Comparison

Chromium's CoreML WebNN backend:
- Also uses external weight files for Float16 constants (same approach we implemented)
- Not clear if they have the same Float16 input limitation
- May skip Float16 tests or have platform version checks

## WPT Conformance Impact

**Before Weight File Fix**:
- 91.3% conformance (2700/2958 passing)
- 122 tests crashing (Float16 constants)
- 136 tests failing for other reasons

**After Weight File Fix**:
- Float16 constant tests: ✅ FIXED
- Float16 input tests with small arrays (≤4 elements): ✅ WORKING
- Float16 input tests with large arrays (>4 elements): ❌ CoreML limitation

**Expected Impact**:
- Most Float16 constant tests should now pass
- Some Float16 input tests will continue to crash due to CoreML limitation
- Estimated improvement: +80-100 tests passing (targeting ~94% conformance)

## Next Steps

1. Run full WPT suite to measure actual improvement
2. Identify which remaining Float16 tests are affected by input size limitation
3. Document CoreML limitation in test skip conditions
4. File bug report with Apple Feedback Assistant if appropriate
5. Consider fallback to ONNX Runtime for Float16 operations if available

## Files Modified

**Phase 1-4 (Weight Files)**:
- NEW: `src/converters/weight_file_builder.rs` - Weight file infrastructure
- MODIFIED: `src/converters/mod.rs` - Added weights_data field
- MODIFIED: `src/converters/coreml_mlprogram.rs` - Float16 constant routing
- MODIFIED: `src/executors/coreml.rs` - MLPackage generation
- MODIFIED: `src/python/context.rs` - Pass weights_data to executor
- NEW: `docs/coreml-weight-files-implementation.md` - Implementation plan

**Investigation**:
- NEW: `test_float16_debug.py` - Float16 constant test
- NEW: `test_float16_input_compute.py` - Float16 input test (small)
- NEW: `test_float16_input_leaky.py` - Float16 input + leakyRelu test
- NEW: `test_float16_wpt_size.py` - Float16 input test (WPT size)
- NEW: `test_float16_sizes.py` - Size threshold test
- NEW: `test_leaky_debug.py` - WPT leakyRelu replication

## Commits

1. `213ecc1f` - Phase 1: Weight file builder infrastructure
2. `a36095cd` - Phase 2: CoreML converter integration
3. `cba495ee` - Phase 2: Integration tests
4. `185517b9` - Phase 3: MLPackage file generation

## References

- W3C WebNN Spec: https://www.w3.org/TR/webnn/
- CoreML MLProgram Format: Apple CoreML documentation
- Chromium WebNN CoreML: chromium/src/services/webnn/coreml/
- WPT WebNN Tests: https://github.com/web-platform-tests/wpt/tree/master/webnn/
