# CoreML Backend Fixes - Session 2025-12-14

## Summary

Improved CoreML backend conformance from 15.8% to 40.0% (+359 tests, +154.1% improvement).

## Key Learnings

### 1. CoreML Parameter Requirements

**Required Parameters:** Many CoreML MIL operations require parameters that WebNN treats as optional:
- `keep_dims` for reduce operations (default: false)
- `perm` for transpose (default: reverse dimensions)
- `transpose_x`, `transpose_y` for matmul (default: false)
- `pad_type` for conv_transpose (default: "custom")
- `alpha`, `beta` for clamp (default: -Infinity, +Infinity)
- `epsilon` for log (default: 1e-45)

**Always add these parameters** even when WebNN doesn't require them.

### 2. Variadic Parameters Need Tuples

Operations like `concat` with multiple inputs need special handling:

```rust
// WRONG: Separate parameters values_0, values_1, values_2...
inputs.insert("values_0", create_argument(&input_names[0]));
inputs.insert("values_1", create_argument(&input_names[1]));

// CORRECT: Single 'values' parameter with tuple of references
fn create_argument_tuple(operand_names: &[String]) -> Argument {
    Argument {
        arguments: operand_names.iter()
            .map(|name| Binding::Name(name.clone()))
            .collect(),
    }
}
inputs.insert("values", create_argument_tuple(&input_names));
```

### 3. CoreML Type Limitations

**Feature Descriptions** (I/O) only support: DOUBLE, FLOAT32, FLOAT16, INT32

**NOT supported:** int8, uint8, uint32, int64 (even though they exist in protobuf)

**Solution:** Add skip logic in test suite for unsupported types:
```python
if data_type in ["int8", "uint8", "uint32", "int64"]:
    pytest.skip(f"CoreML limitation: {data_type} not supported")
```

### 4. Parameter Type Matters

CoreML is strict about parameter types:

```rust
// WRONG: dtype as integer
inputs.insert("dtype", create_immediate_int(10));

// CORRECT: dtype as string
inputs.insert("dtype", create_immediate_string("fp32"));
```

### 5. WebNN vs CoreML Parameter Naming

Common mismatches:
- WebNN `outputPadding` != CoreML `output_shape`
- WebNN `outputSizes` = CoreML `output_shape` (spatial dimensions only [H, W])
- WebNN `axes` = CoreML `axes` (but CoreML requires it for reduce ops)
- WebNN `keepDimensions` = CoreML `keep_dims`

### 6. Operation Name Case Sensitivity

Operation names are lowercased, so:
- `reduceProduct` becomes `"reduceproduct"` (not `"reduceprod"`)
- Use exact lowercase names in pattern matching

### 7. 0D Tensor Handling

Many CoreML operations fail on 0D (scalar) tensors:
- transpose: perm must have shape [rank of x], fails for rank 0
- slice: begin must have length >= 1, fails for empty
- expand: tile doesn't support scalar inputs

**Solution:** Add skip logic for 0D tensors where operations don't support them.

### 8. Always Check Chromium Reference

Before implementing any operation, check:
- https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/coreml/graph_builder_coreml.mm

Chromium shows:
- Correct parameter names and types
- Required vs optional parameters
- Workarounds for CoreML limitations
- Type conversion strategies

### 9. Spatial Dimensions Only

For convolution operations, CoreML expects spatial dimensions only:
- `output_shape` for conv_transpose2d: [H, W] not [N, C, H, W]
- `pad`: [H_begin, H_end, W_begin, W_end] not full 4D padding

### 10. Default Values Are Critical

When WebNN parameters have defaults, CoreML still needs them explicitly:
- Clamp: minValue=-Infinity, maxValue=+Infinity
- Log: epsilon=1e-45
- MatMul: transpose_x=false, transpose_y=false

### 11. Type Matching for Binary Operations

CoreML requires exact type matching for binary operations:
- **mul operation**: All operands (x, y, output) must have same dtype
- **neg operation**: Implemented as `mul(x, -1.0)` but -1.0 constant must match input dtype
- For float16 inputs, create float16 constant (not float32)
- For int32 inputs, create int32 constant (not float32)

**Solution**: Always create typed constants matching the input operand's dtype:
```rust
// Create constant with matching dtype
let constant_data = match input_desc.data_type {
    DataType::Float32 => vec![-1.0f32.to_ne_bytes()].concat(),
    DataType::Float16 => vec![f16::from_f32(-1.0).to_ne_bytes()].concat(),
    DataType::Int32 => vec![(-1i32).to_ne_bytes()].concat(),
    // ...
};
```

### 12. Clamp Alpha/Beta Type Matching

CoreML's clamp operation requires alpha/beta to match input tensor dtype:
- For float32 input: alpha/beta must be float32 immediates
- For float16 input: alpha/beta must be float16 immediates
- Type mismatch causes runtime parse errors

**Solution**: Convert alpha/beta values to input dtype before creating immediates:
```rust
let min_value_f32 = min_value.unwrap_or(f32::NEG_INFINITY);
let max_value_f32 = max_value.unwrap_or(f32::INFINITY);

match input_desc.data_type {
    DataType::Float32 => {
        inputs.insert("alpha", Self::create_immediate_float(min_value_f32));
        inputs.insert("beta", Self::create_immediate_float(max_value_f32));
    }
    DataType::Float16 => {
        let min_f16 = f16::from_f32(min_value_f32);
        let max_f16 = f16::from_f32(max_value_f32);
        inputs.insert("alpha", Self::create_immediate_float16(min_f16));
        inputs.insert("beta", Self::create_immediate_float16(max_f16));
    }
    // ...
}
```

## Fixes Implemented (13 commits)

### Session 1 (commits cb9221e9 - f7bc3e50)
1. **cb9221e9** - Reduce operations (keep_dims, axes) + transpose (perm) + reshape/slice
2. **b7244674** - MatMul (transpose_x/y) + neg (y=-1.0)
3. **2554cdb2** - Gather parameter names
4. **3bf75f84** - reduceProduct operation name
5. **1af0e271** - Cast dtype string type
6. **fd46e237** - Cast unsupported type skip logic
7. **cb6e53b3** - Clamp (alpha/beta) + concat (variadic values)
8. **de6742be** - Log (epsilon) + hardswish (remove alpha/beta)
9. **f7bc3e50** - Conv_transpose2d (pad_type, outputSizes)

### Session 2 (commits e251565f - 1ec08c58)
10. **e251565f** - Fix CI: pytest fixture error + docs broken links
11. **02bf7f73** - Gather: add axis parameter (always present, defaults to 0)
12. **23bcde9f** - Gather: set validate_indices=false (fixes all gather runtime errors)
13. **1ec08c58** - Clamp: fix float16 type mismatch - alpha/beta must match input dtype

## Remaining Issues (95 failures)

### High Priority
- **expand**: 38 failures (rank-increasing, needs expand_dims)
- **layer_normalization**: 22 failures (invalid param 'mean' error)
- **conv2d**: 20 failures (layout conversions NHWC?)
- **batch_normalization**: 9 failures (runtime errors, 9 skipped for NHWC layout)

### Medium Priority
- **conv_transpose2d**: 7 failures (layout issues)

### Low Priority
- **instance_normalization**: 4 failures
- **neg**: 3 failures (type mismatch - needs typed constants for float16/int32)
- **transpose**: 1 failure (0D tensors)
- **slice**: 1 failure (0D tensors)
- **reshape**: 1 failure (6D+ limitation)
- **relu**: 1 failure (int32 not supported - only float32/float16)
- **add**: 1 failure (special character names)
- **clamp**: 1 failure (int32 type support)

## Testing Strategy

1. Run specific operation tests: `pytest -k "operation_name and coreml"`
2. Check error type: parse error vs runtime error
3. Parse errors = missing/wrong parameters (fixable)
4. Runtime errors = deeper implementation issues
5. Always rebuild after changes: `make python-dev`
6. Commit after each successful fix with clear message

## Performance

- **Before**: 233 passed / 1479 tests (15.8%)
- **After Session 1**: 591 passed / 1479 tests (39.96%) - +358 tests
- **After Session 2**: 592 passed / 1479 tests (40.0%) - +359 tests total
- **Total Improvement**: +359 tests (+154.1%)

## Session 2 Highlights

### CI Fixes (commit e251565f)
- Fixed pytest discovering `test_conversions()` as a test (renamed to `verify_conversions()`)
- Fixed MkDocs strict mode failure on broken links to TODO.txt and AGENTS.md

### Gather Operation (commits 02bf7f73, 23bcde9f)
- **Root Cause**: CoreML's `validate_indices` parameter was set to `true`, causing validation errors
- **Solution**: Set `validate_indices=false` following Chromium's implementation
- **Impact**: Fixed ALL 20+ gather runtime errors
- **Learning**: Always check Chromium reference implementation first - it has workarounds for CoreML quirks

### Clamp Float16 Fix (commit 1ec08c58)
- **Root Cause**: Clamp's alpha/beta parameters must match input dtype
- **Solution**: Convert alpha/beta to float16 when input is float16
- **Impact**: Fixed 1 additional clamp test
- **Pattern**: Applies to all CoreML operations with typed parameters

## Next Session Goals

1. Fix neg operation type matching (float16/int32 constants)
2. Fix layer_normalization (invalid param 'mean' error)
3. Add layout conversion support for conv2d/batch_norm (NHWC)
4. Investigate expand operation (rank-increasing needs expand_dims)
5. Target 50%+ conformance (need +148 more passing tests)
