# WPT WebNN Test Integration Plan

**Status:** Design Document
**Created:** 2025-12-07
**Author:** Claude Code (via analysis of WPT test suite)

## Executive Summary

This document outlines a strategy for integrating the W3C Web Platform Tests (WPT) for WebNN with the rustnn implementation. The goal is to leverage the comprehensive WPT test suite (110+ conformance tests, 64+ validation tests) to validate our implementation's correctness and spec compliance.

## Background

### What are WPT WebNN Tests?

The [Web Platform Tests for WebNN](https://github.com/web-platform-tests/wpt/tree/master/webnn) are the official conformance tests for the W3C WebNN specification. They consist of:

1. **Conformance Tests** (`conformance_tests/`): 110+ tests validating that operations produce mathematically correct results
2. **Validation Tests** (`validation_tests/`): 64+ tests ensuring proper error handling and parameter validation
3. **IDL Tests**: Web IDL interface validation tests
4. **Test Resources**: Shared utilities and test data

### WPT Test Structure

**Conformance Test Example:**
```javascript
{
  "name": "relu float32 2D tensor",
  "graph": {
    "inputs": {
      "reluInput": {
        "data": [1, -2, 3, -4, 5, -6],
        "descriptor": {"dimensions": [2, 3], "dataType": "float32"}
      }
    },
    "operators": [{
      "name": "relu",
      "arguments": [{"input": "reluInput"}],
      "outputs": "reluOutput"
    }],
    "expectedOutputs": {
      "reluOutput": {
        "data": [1, 0, 3, 0, 5, 0],
        "descriptor": {"dimensions": [2, 3], "dataType": "float32"}
      }
    }
  }
}
```

**Key Components:**
- **Test data format**: JSON-like structures with inputs, operators, and expected outputs
- **Precision tolerances**: ULP (Units in Last Place) or ATOL (absolute tolerance) based
- **Multi-context**: Tests run on CPU, GPU, NPU variants
- **Data types**: float32, float16, int8, int32, int64, uint8, uint32, uint64, int4, uint4

## Current State Analysis

### What We Have
[OK] **Python API**: Full WebNN-compliant API with all reduction operations
[OK] **Graph Builder**: Backend-agnostic graph construction
[OK] **ONNX Backend**: Cross-platform execution via ONNX Runtime
[OK] **CoreML Backend**: macOS-optimized execution
[OK] **Basic Tests**: 109 Python tests (18 for reduction operations)

### What We're Missing
 **Comprehensive test coverage**: Only ~10% of WPT test cases covered
 **Precision validation**: No ULP-based tolerance checking
 **Data type coverage**: Missing float16, int64, int4/uint4 support
 **Validation tests**: No systematic parameter validation testing
 **Test automation**: Manual test writing vs. data-driven approach

## Integration Strategy

### Approach: Python Test Adapter

We propose creating a **Python-based test adapter** that:
1. Loads WPT test data (converted from JavaScript to JSON)
2. Executes tests against our Python WebNN API
3. Validates results using WPT-compatible tolerance checking
4. Reports results in pytest format

### Why Python (not JavaScript)?

**Advantages:**
- [OK] Our Python API already implements WebNN spec
- [OK] Can reuse existing pytest infrastructure
- [OK] Direct access to NumPy for numerical validation
- [OK] Easier to integrate with CI/CD
- [OK] No need for JavaScript runtime

**Trade-offs:**
- [WARNING] Need to convert JS test data to Python/JSON format
- [WARNING] Some WPT utilities need reimplementation (tolerance checking, type conversion)

## Implementation Plan

### Phase 1: Test Infrastructure (Week 1)

**Goal:** Build the foundation for running WPT-style tests in Python

**Tasks:**

1. **Create test data converter** (`scripts/convert_wpt_tests.py`)
   - Parse JavaScript test files from WPT repo
   - Extract test case data structures
   - Convert to JSON format (one file per operation)
   - Store in `tests/wpt_data/conformance/` and `tests/wpt_data/validation/`

2. **Implement tolerance checking** (`tests/wpt_utils.py`)
   - Port ULP distance calculation from WPT `utils.js`
   - Implement ATOL checking
   - Create precision tolerance lookup tables
   - Add tolerance accumulation for multi-operator graphs

3. **Build test loader** (`tests/test_wpt_conformance.py`)
   - Load JSON test data
   - Parameterize pytest tests from data
   - Map test data to WebNN API calls
   - Execute and validate results

**Deliverables:**
- [ ] `scripts/convert_wpt_tests.py` - Converter script
- [ ] `tests/wpt_utils.py` - WPT-compatible utilities
- [ ] `tests/wpt_data/` - Test data directory structure
- [ ] `tests/test_wpt_conformance.py` - Test runner (initial version)

### Phase 2: Conformance Tests (Week 2-3)

**Goal:** Run all WPT conformance tests for implemented operations

**Priority Operations (in order):**

**Tier 1 - Already Implemented:**
1. [OK] Binary ops: add, sub, mul, div, matmul (5 ops)
2. [OK] Activations: relu, sigmoid, tanh, softmax (4 ops)
3. [OK] Reductions: reduceSum, reduceMean, reduceMax, reduceMin, reduceProduct, reduceL1, reduceL2, reduceLogSum, reduceLogSumExp, reduceSumSquare (10 ops)
4. [OK] Pooling: averagePool2d, maxPool2d, globalAveragePool, globalMaxPool (4 ops)
5. [OK] Convolution: conv2d, convTranspose2d (2 ops)
6. [OK] Normalization: batchNormalization, instanceNormalization, layerNormalization (3 ops)
7. [OK] Shape: reshape (1 op)

**Total Tier 1:** 29 operations (estimated ~35-40 WPT test files)

**Tier 2 - High Priority Missing Ops:**
1. Element-wise: abs, ceil, floor, exp, log, sqrt, neg, reciprocal (8 ops)
2. Shape: transpose, concat, split, slice, expand (5 ops)
3. Logical: equal, greater, lesser, logical_not (4 ops)

**Tier 3 - Future:**
- Recurrent: lstm, gru, lstmCell, gruCell
- Advanced: gemm, where, cast, clamp, gather, scatter
- Quantization: quantizeLinear, dequantizeLinear

**Tasks:**

1. **Convert Tier 1 test data** (Priority: Reductions first)
   ```bash
   python scripts/convert_wpt_tests.py \
     --wpt-repo ~/wpt \
     --operations reduce_sum,reduce_mean,reduce_max,reduce_min \
     --output tests/wpt_data/conformance/
   ```

2. **Implement data type support**
   - Add float16 support (map to float32 for computation, track separately)
   - Add int64/uint64 support (use Python int, convert to/from NumPy)
   - Document unsupported types (int4/uint4 - defer to future)

3. **Run and validate conformance tests**
   ```bash
   pytest tests/test_wpt_conformance.py -k "reduce" -v
   ```

4. **Fix failures**
   - Investigate numerical precision issues
   - Adjust tolerance settings if needed
   - Fix implementation bugs

5. **Add CI integration**
   - Add GitHub Actions workflow
   - Run WPT tests on every PR
   - Generate coverage reports

**Success Metrics:**
- [ ] 100% of Tier 1 tests passing (within WPT tolerances)
- [ ] Coverage report showing tested operations
- [ ] CI pipeline green

### Phase 3: Validation Tests (Week 4)

**Goal:** Ensure proper error handling and parameter validation

**Tasks:**

1. **Convert validation test data**
   - Extract validation tests from WPT
   - Focus on operations we've implemented
   - Convert error-checking patterns to pytest assertions

2. **Implement validation test runner** (`tests/test_wpt_validation.py`)
   - Test parameter constraints (shape, type, range)
   - Test error messages and exception types
   - Test cross-builder validation
   - Test invalid input combinations

3. **Enhance error handling**
   - Improve error messages to match WPT expectations
   - Add missing validation checks
   - Document validation behavior

**Example Validation Test:**
```python
def test_reduce_sum_invalid_axes():
    """Test that reduceSum rejects out-of-bounds axes"""
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3, 4], "float32")

    # Axis 5 is out of bounds for rank-3 tensor
    with pytest.raises(ValueError, match="out of bounds"):
        output = builder.reduce_sum(x, axes=[5])
```

**Success Metrics:**
- [ ] All validation tests passing for implemented ops
- [ ] Consistent error messages with WPT expectations
- [ ] Full parameter validation coverage

### Phase 4: Continuous Integration (Week 5)

**Goal:** Automate WPT test execution and reporting

**Tasks:**

1. **Set up test automation**
   - Create `make wpt-test` target
   - Add to CI pipeline
   - Configure test matrix (backends, data types)

2. **Implement test filtering**
   - Skip tests for unimplemented operations
   - Mark known failures with xfail
   - Tag tests by operation category

3. **Create test reports**
   - Generate HTML coverage report
   - Show pass/fail/skip breakdown by operation
   - Track test status over time

4. **Document test usage**
   - Update README with WPT test instructions
   - Document how to add new test data
   - Explain tolerance tuning process

**Success Metrics:**
- [ ] WPT tests run automatically on every PR
- [ ] Test coverage visible in CI
- [ ] Clear documentation for contributors

## Technical Design

### Directory Structure

```
rustnn/
 tests/
    wpt_data/                    # WPT test data (converted)
       conformance/
          relu.json
          reduce_sum.json
          ...
       validation/
           relu.json
           ...
    wpt_utils.py                 # WPT-compatible utilities
    test_wpt_conformance.py      # Conformance test runner
    test_wpt_validation.py       # Validation test runner
    conftest.py                  # Pytest fixtures
 scripts/
    convert_wpt_tests.py         # WPT test converter
    update_wpt_tests.sh          # Auto-update script
 docs/
     wpt-integration-plan.md      # This document
     wpt-test-guide.md            # User guide (TBD)
```

### Test Data Format

**JSON Test Case Structure:**
```json
{
  "name": "reduce_sum float32 2D tensor with axes=[1]",
  "inputs": {
    "input": {
      "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      "shape": [2, 3],
      "dataType": "float32"
    }
  },
  "operators": [
    {
      "name": "reduce_sum",
      "arguments": {
        "input": "input",
        "axes": [1],
        "keepDimensions": false
      },
      "output": "output"
    }
  ],
  "expectedOutputs": {
    "output": {
      "data": [6.0, 15.0],
      "shape": [2],
      "dataType": "float32"
    }
  },
  "tolerance": {
    "type": "ULP",
    "value": 0
  }
}
```

### Tolerance Checking Implementation

**ULP Distance Function:**
```python
def ulp_distance(a: float, b: float, dtype: str) -> int:
    """Calculate ULP distance between two floating-point values"""
    if dtype == "float32":
        # Convert to int32 bit representation
        a_bits = struct.unpack('!i', struct.pack('!f', a))[0]
        b_bits = struct.unpack('!i', struct.pack('!f', b))[0]
        return abs(a_bits - b_bits)
    elif dtype == "float16":
        # Use numpy float16
        a_half = np.float16(a)
        b_half = np.float16(b)
        a_bits = a_half.view(np.uint16)
        b_bits = b_half.view(np.uint16)
        return int(abs(int(a_bits) - int(b_bits)))
    else:
        raise ValueError(f"ULP not supported for {dtype}")
```

**Precision Tolerance Lookup:**
```python
OPERATION_TOLERANCES = {
    "relu": {"ULP": 0},
    "sigmoid": {"ULP": {"float32": 34, "float16": 3}},
    "tanh": {"ULP": {"float32": 44, "float16": 4}},
    "reduce_sum": lambda test: {"ULP": 0},  # Varies by input size
    "reduce_mean": lambda test: {
        "ULP": compute_reduce_tolerance(test, "mean")
    },
    # ... more operations
}
```

### Test Parameterization

**Pytest Parameterization:**
```python
@pytest.fixture
def wpt_test_loader():
    """Load WPT test data files"""
    def load(operation: str, category: str = "conformance"):
        path = f"tests/wpt_data/{category}/{operation}.json"
        with open(path) as f:
            return json.load(f)
    return load

def load_conformance_tests(operation: str):
    """Generate pytest parameters from WPT test data"""
    test_data = load_wpt_test_data(f"conformance/{operation}.json")
    return [
        pytest.param(test, id=test["name"])
        for test in test_data["tests"]
    ]

@pytest.mark.parametrize(
    "test_case",
    load_conformance_tests("reduce_sum")
)
def test_reduce_sum_conformance(context, test_case):
    """Run WPT conformance test for reduce_sum"""
    result = execute_wpt_test(context, test_case)
    validate_wpt_result(result, test_case, tolerance=get_tolerance(test_case))
```

## Migration Path

### Short-term (Immediate)

**Focus:** Get 10 reduction operations fully tested with WPT conformance tests

1. Convert WPT test data for all 10 reduction operations
2. Implement basic tolerance checking (ULP + ATOL)
3. Run tests, fix any failures
4. Document results and learnings

**Effort:** 2-3 days
**Value:** High - validates our recent reduction implementation

### Medium-term (1-2 weeks)

**Focus:** Expand coverage to all implemented operations (29 ops)

1. Convert test data for remaining Tier 1 operations
2. Add float16 support where needed
3. Implement validation tests for parameter checking
4. Integrate into CI pipeline

**Effort:** 1-2 weeks
**Value:** High - comprehensive validation of current implementation

### Long-term (1-2 months)

**Focus:** Full WPT compliance

1. Implement missing Tier 2 operations
2. Add all conformance and validation tests
3. Track WPT upstream changes
4. Contribute fixes back to WPT if needed

**Effort:** 1-2 months
**Value:** Medium-High - full spec compliance, industry standard testing

## Risks and Mitigation

### Risk 1: Test Data Conversion Complexity
**Risk:** WPT tests use JavaScript; conversion may be error-prone
**Mitigation:** Start with simple operations (reductions), validate manually, automate gradually

### Risk 2: Precision Tolerance Mismatches
**Risk:** Our backends may have different numerical characteristics than WPT expects
**Mitigation:** Make tolerances configurable, document backend-specific tolerances

### Risk 3: Unsupported Data Types
**Risk:** ONNX Runtime may not support all WPT data types (float16, int4, uint4)
**Mitigation:** Clearly document unsupported types, skip those tests gracefully

### Risk 4: Test Maintenance Burden
**Risk:** WPT tests update frequently, keeping in sync is effort
**Mitigation:** Automate test data updates, pin to specific WPT version initially

### Risk 5: Performance Impact
**Risk:** 110+ conformance tests may slow down CI significantly
**Mitigation:** Run subset on PR (smoke tests), full suite nightly or on-demand

## Success Criteria

### Milestone 1: Reduction Operations (Week 1)
- [ ] 10 reduction operations have WPT conformance tests
- [ ] All tests passing within specified tolerances
- [ ] Test infrastructure proven and documented

### Milestone 2: Tier 1 Coverage (Week 3)
- [ ] 29 implemented operations have WPT conformance tests
- [ ] 80%+ pass rate (some tolerance tuning expected)
- [ ] CI integration complete

### Milestone 3: Validation Coverage (Week 4)
- [ ] Parameter validation tests for all Tier 1 operations
- [ ] Consistent error handling across API
- [ ] Test coverage report generated

### Milestone 4: Production Ready (Week 5)
- [ ] 90%+ WPT conformance test pass rate
- [ ] Full documentation and contributor guide
- [ ] Automated test updates from WPT upstream

## Open Questions

1. **Q:** Should we fork WPT or reference it as a submodule?
   **A:** TBD - Recommend submodule for official tests, convert to JSON as needed

2. **Q:** Do we need to support all WPT data types immediately?
   **A:** No - Start with float32/int32, add float16 next, defer int4/uint4

3. **Q:** Should tolerance settings be configurable per backend?
   **A:** Yes - Different backends (ONNX CPU vs CoreML) may need different tolerances

4. **Q:** How to handle flaky tests?
   **A:** Mark with pytest.mark.flaky, investigate root cause, consider backend-specific skips

5. **Q:** Should we contribute test results back to WPT?
   **A:** Future consideration - Once stable, could report implementation status

## References

- **WPT WebNN Tests:** https://github.com/web-platform-tests/wpt/tree/master/webnn
- **WebNN Spec:** https://www.w3.org/TR/webnn/
- **WPT Contributing Guide:** https://web-platform-tests.org/writing-tests/
- **Local WebNN Spec Reference:** `docs/webnn-spec-reference.md`

## Appendices

### Appendix A: WPT Test Coverage Matrix

| Operation | WPT Tests | Implemented | Priority |
|-----------|-----------|-------------|----------|
| relu | [OK] | [OK] | Tier 1 |
| sigmoid | [OK] | [OK] | Tier 1 |
| tanh | [OK] | [OK] | Tier 1 |
| softmax | [OK] | [OK] | Tier 1 |
| add | [OK] | [OK] | Tier 1 |
| sub | [OK] | [OK] | Tier 1 |
| mul | [OK] | [OK] | Tier 1 |
| div | [OK] | [OK] | Tier 1 |
| matmul | [OK] | [OK] | Tier 1 |
| reduceSum | [OK] | [OK] | Tier 1 |
| reduceMean | [OK] | [OK] | Tier 1 |
| reduceMax | [OK] | [OK] | Tier 1 |
| reduceMin | [OK] | [OK] | Tier 1 |
| reduceProduct | [OK] | [OK] | Tier 1 |
| reduceL1 | [OK] | [OK] | Tier 1 |
| reduceL2 | [OK] | [OK] | Tier 1 |
| reduceLogSum | [OK] | [OK] | Tier 1 |
| reduceLogSumExp | [OK] | [OK] | Tier 1 |
| reduceSumSquare | [OK] | [OK] | Tier 1 |
| averagePool2d | [OK] | [OK] | Tier 1 |
| maxPool2d | [OK] | [OK] | Tier 1 |
| globalAveragePool | [OK] | [OK] | Tier 1 |
| globalMaxPool | [OK] | [OK] | Tier 1 |
| conv2d | [OK] | [OK] | Tier 1 |
| convTranspose2d | [OK] | [OK] | Tier 1 |
| batchNormalization | [OK] | [OK] | Tier 1 |
| instanceNormalization | [OK] | [OK] | Tier 1 |
| layerNormalization | [OK] | [OK] | Tier 1 |
| reshape | [OK] | [OK] | Tier 1 |
| abs | [OK] |  | Tier 2 |
| exp | [OK] |  | Tier 2 |
| log | [OK] |  | Tier 2 |
| sqrt | [OK] |  | Tier 2 |
| transpose | [OK] |  | Tier 2 |
| concat | [OK] |  | Tier 2 |
| split | [OK] |  | Tier 2 |
| slice | [OK] |  | Tier 2 |
| ... | ... | ... | ... |

### Appendix B: Example Test Converter Pseudocode

```python
#!/usr/bin/env python3
"""Convert WPT WebNN tests from JavaScript to JSON format"""

import re
import json
from pathlib import Path

def parse_js_test_file(js_content: str) -> dict:
    """Parse JavaScript test array into Python dict"""
    # Extract test array using regex
    match = re.search(r'const \w+Tests = (\[.*?\]);', js_content, re.DOTALL)
    if not match:
        raise ValueError("No test array found")

    # Use ast/esprima to parse JavaScript safely
    # Or use simple regex for basic cases
    test_array_str = match.group(1)

    # Convert to JSON-compatible format
    tests = parse_test_array(test_array_str)
    return {"tests": tests}

def convert_wpt_operation(wpt_dir: Path, operation: str, output_dir: Path):
    """Convert a single operation's tests"""
    js_file = wpt_dir / "conformance_tests" / f"{operation}.https.any.js"

    with open(js_file) as f:
        js_content = f.read()

    test_data = parse_js_test_file(js_content)

    output_file = output_dir / f"{operation}.json"
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"Converted {operation}: {len(test_data['tests'])} tests")
```

---

**Document Status:** Draft v1.0
**Next Review:** After Phase 1 completion
**Feedback:** Submit issues or PRs to rustnn repository
