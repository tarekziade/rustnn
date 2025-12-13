# WPT WebNN Test Guide

This guide explains how to use the W3C Web Platform Tests (WPT) for WebNN with the rustnn implementation.

## Overview

The WPT integration provides:
- **Conformance Tests**: Validate that operations produce mathematically correct results
- **Validation Tests**: Ensure proper error handling and parameter validation
- **Automatic Test Generation**: Convert official WPT tests to run against our implementation
- **Precision Checking**: ULP-based and ATOL-based tolerance validation
- **Easy Updates**: Simple scripts to sync with upstream WPT changes

## Quick Start

### Running WPT Tests

```bash
# Run all WPT conformance tests
pytest tests/test_wpt_conformance.py -v

# Run tests for specific operation
pytest tests/test_wpt_conformance.py -k "reduce_sum" -v

# Run with detailed output
pytest tests/test_wpt_conformance.py -vv --tb=short

# Run only WPT-marked tests
pytest -m wpt -v
```

### Current Status

The WPT test infrastructure is fully implemented and ready to use. Currently:

[OK] Test infrastructure complete
[OK] Tolerance checking (ULP and ATOL)
[OK] Test data loader and runner
[OK] Sample test data for reduce_sum
⏳ Full test data population (requires manual conversion or WPT sync)
⏳ Graph execution (compute) implementation

Tests currently skip with message: "Graph execution (compute) not yet implemented - graph build validated"

## Architecture

### Directory Structure

```
rustnn/
 tests/
    wpt_data/              # WPT test data (JSON format)
       conformance/       # Correctness tests
          reduce_sum.json  # Sample test data
       validation/        # Parameter validation tests
    wpt_utils.py           # WPT utilities (tolerance checking)
    test_wpt_conformance.py  # Conformance test runner
    conftest.py            # Shared pytest fixtures
    test_python_api.py     # Regular API tests
 scripts/
    convert_wpt_tests.py   # Convert JS tests to JSON
    update_wpt_tests.sh    # Auto-update script
 docs/
     implementation-status.md # Implementation status & testing strategy
     wpt-test-guide.md        # This guide
```

### Components

#### 1. Test Data (`tests/wpt_data/`)

Test data is stored in JSON format, one file per operation:

```json
{
  "operation": "reduce_sum",
  "wpt_version": "2025-12-07",
  "wpt_commit": "abc123...",
  "source_file": "webnn/conformance_tests/reduce.https.any.js",
  "tests": [
    {
      "name": "reduce_sum float32 2D tensor axis 1",
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
  ]
}
```

#### 2. Test Utilities (`tests/wpt_utils.py`)

Provides WPT-compatible utilities:

- **`ulp_distance(a, b, dtype)`**: Calculate ULP distance between values
- **`check_ulp_tolerance(actual, expected, tolerance, dtype)`**: Validate with ULP tolerance
- **`check_atol_tolerance(actual, expected, tolerance)`**: Validate with absolute tolerance
- **`get_operation_tolerance(operation, test_case)`**: Get tolerance spec for operation
- **`validate_result(actual, expected, tolerance, dtype)`**: Main validation function
- **`load_wpt_test_data(operation, category)`**: Load test data from JSON
- **`format_test_failure(test_name, failures)`**: Format failure messages

#### 3. Test Runner (`tests/test_wpt_conformance.py`)

Pytest-based test runner that:

1. Discovers all operations with test data
2. Loads test cases for each operation
3. Dynamically generates parameterized tests
4. Executes tests against WebNN API
5. Validates results with WPT tolerance specs

#### 4. Converter Script (`scripts/convert_wpt_tests.py`)

Converts WPT JavaScript tests to JSON format:

```bash
# Convert single operation
python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --operation reduce_sum

# Convert multiple operations
python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --operations reduce_sum,relu,add

# List available operations
python scripts/convert_wpt_tests.py --wpt-repo ~/wpt --list-operations
```

#### 5. Update Script (`scripts/update_wpt_tests.sh`)

Automates WPT repository management and test conversion:

```bash
# Update all operations
./scripts/update_wpt_tests.sh

# Update specific operations
./scripts/update_wpt_tests.sh --operations reduce_sum,relu,add

# Force fresh clone of WPT repo
./scripts/update_wpt_tests.sh --force-clone
```

## Tolerance Checking

### ULP (Units in Last Place)

ULP distance measures how many representable floating-point values exist between two numbers. This is more robust than absolute or relative tolerance for floating-point comparisons.

**Example tolerances:**
- Exact operations (relu, add): 0 ULP
- Approximate operations (sigmoid): 34 ULP (float32), 3 ULP (float16)
- Accumulated error (matmul): 100 ULP

### Absolute Tolerance (ATOL)

Absolute tolerance checks if |actual - expected| ≤ tolerance.

**When to use:**
- Integer operations
- Operations where ULP is not meaningful
- Custom precision requirements

### Default Tolerances

See `wpt_utils.py:get_operation_tolerance()` for full list:

```python
DEFAULT_TOLERANCES = {
    "relu": {"type": "ULP", "value": 0},
    "sigmoid": {"type": "ULP", "value": 34},
    "reduce_sum": {"type": "ULP", "value": 0},
    "matmul": {"type": "ULP", "value": 100},
    # ... more operations
}
```

Override tolerance per test case in JSON:

```json
{
  "tolerance": {
    "type": "ULP",
    "value": 50
  }
}
```

## Adding Test Data

### Method 1: Automatic Conversion (Preferred)

1. Clone WPT repository if not already available:
   ```bash
   git clone --depth 1 https://github.com/web-platform-tests/wpt.git ~/wpt
   ```

2. Run update script:
   ```bash
   ./scripts/update_wpt_tests.sh --operations reduce_sum,reduce_mean
   ```

3. Review generated JSON files in `tests/wpt_data/conformance/`

4. Manually populate test cases if converter couldn't parse JavaScript

### Method 2: Manual Creation

1. Create JSON file in `tests/wpt_data/conformance/`:
   ```bash
   touch tests/wpt_data/conformance/my_operation.json
   ```

2. Populate with test cases following the JSON schema (see example above)

3. Verify JSON is valid:
   ```bash
   python3 -m json.tool tests/wpt_data/conformance/my_operation.json
   ```

4. Run tests:
   ```bash
   pytest tests/test_wpt_conformance.py -k "my_operation" -v
   ```

### Method 3: Copy from WPT Source

1. Find the operation's test file in WPT:
   ```bash
   cd ~/wpt/webnn/conformance_tests
   ls -la | grep my_operation
   ```

2. Open the JavaScript file and manually extract test cases

3. Convert to JSON format matching our schema

4. Add metadata (wpt_version, wpt_commit, source_file)

## Workflow

### For Contributors

1. **Implement Operation**: Add new operation to rustnn
   ```rust
   // src/python/graph_builder.rs
   fn my_operation(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
       // implementation
   }
   ```

2. **Add WPT Test Data**: Get test data from WPT
   ```bash
   ./scripts/update_wpt_tests.sh --operations my_operation
   ```

3. **Run Tests**: Validate implementation
   ```bash
   pytest tests/test_wpt_conformance.py -k "my_operation" -v
   ```

4. **Fix Failures**: Debug and fix implementation or tolerance issues

5. **Commit**: Include both implementation and test data
   ```bash
   git add src/ tests/wpt_data/conformance/my_operation.json
   git commit -m "Add my_operation with WPT conformance tests"
   ```

### For Maintainers

**Regular Updates:**
```bash
# Weekly or monthly: sync with WPT upstream
./scripts/update_wpt_tests.sh

# Review changes
git diff tests/wpt_data/

# Run full test suite
pytest tests/test_wpt_conformance.py

# Commit updated test data
git add tests/wpt_data/
git commit -m "Update WPT test data from upstream"
```

**New Operation Support:**
1. Check WPT for tests: `./scripts/convert_wpt_tests.py --wpt-repo ~/wpt --list-operations`
2. Add operation to rustnn
3. Add test data: `./scripts/update_wpt_tests.sh --operations new_op`
4. Document in `docs/api-reference.md`

## Troubleshooting

### Test Discovery Issues

**Problem:** `pytest` doesn't find WPT tests

**Solution:**
```bash
# Verify test data exists
ls tests/wpt_data/conformance/

# Run with verbose collection
pytest tests/test_wpt_conformance.py --collect-only -v
```

### Tolerance Failures

**Problem:** Tests fail with ULP distance errors

**Solutions:**
1. **Check expected values:** Verify test data is correct
2. **Adjust tolerance:** Override in JSON or update `wpt_utils.py` defaults
3. **Backend differences:** Different backends may need different tolerances
4. **Implementation bug:** Fix the operation implementation

Example debugging:
```bash
# Run with detailed failure output
pytest tests/test_wpt_conformance.py -k "failing_test" -vv --tb=long
```

### Missing Test Data

**Problem:** `FileNotFoundError: WPT test data not found`

**Solution:**
```bash
# Generate test data for the operation
./scripts/update_wpt_tests.sh --operations <operation_name>

# Or create manually following the JSON schema
```

### JavaScript Parsing Errors

**Problem:** Converter can't parse WPT JavaScript tests

**Solution:**
- The converter provides a template - manually populate test cases
- Refer to the WPT JavaScript source file
- Follow the JSON schema in sample files
- Contribute improvements to the converter script

## Tips and Best Practices

### Testing Strategy

1. **Start Small**: Test with simple operations first (relu, add)
2. **Verify Manually**: Check a few test cases by hand
3. **Use Markers**: Tag tests with `@pytest.mark.wpt` for organization
4. **Parallel Tests**: Run tests in parallel with `pytest -n auto`
5. **Coverage**: Track which operations have WPT tests

### Performance

```bash
# Run subset for quick validation
pytest tests/test_wpt_conformance.py -k "reduce_sum" --maxfail=1

# Run in parallel
pytest tests/test_wpt_conformance.py -n 4

# Profile test execution
pytest tests/test_wpt_conformance.py --durations=10
```

### CI Integration

Add to `.github/workflows/tests.yml`:

```yaml
- name: Run WPT Conformance Tests
  run: |
    pytest tests/test_wpt_conformance.py -v --tb=short
  continue-on-error: true  # Until all operations implemented
```

## Future Enhancements

- [ ] Full JavaScript parser for automated conversion
- [ ] Validation test runner (`test_wpt_validation.py`)
- [ ] Coverage report generator
- [ ] Automatic WPT sync via GitHub Actions
- [ ] Backend-specific tolerance profiles
- [ ] Test result dashboard

## Resources

- **WPT WebNN Tests**: https://github.com/web-platform-tests/wpt/tree/master/webnn
- **WebNN Spec**: https://www.w3.org/TR/webnn/
- **Implementation Status & Testing Strategy**: `docs/implementation-status.md`
- **Local Spec Reference**: `docs/webnn-spec-reference.md`
- **Test Data README**: `tests/wpt_data/README.md`

## Getting Help

- **Issues**: Report problems at https://github.com/your-org/rustnn/issues
- **Questions**: Ask in discussions or issues
- **Contributing**: See `CONTRIBUTING.md` (if available)

---

**Last Updated:** 2025-12-07
**Status:** Infrastructure Complete, Test Population In Progress
