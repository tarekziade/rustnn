# Code Coverage Guide

This guide explains how to generate and analyze code coverage metrics for the rustnn project.

## Overview

rustnn uses [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) for code coverage analysis. This tool provides accurate coverage data using LLVM's source-based code coverage instrumentation.

## Current Coverage Metrics

As of the latest test run, the project has:

- **Total Line Coverage:** 47.25% (9,937 / 21,030 lines)
- **Total Function Coverage:** 42.26% (661 / 1,564 functions)
- **Total Region Coverage:** 47.38% (15,767 / 33,278 regions)

### Coverage by Module

High coverage modules (>80%):
- `weight_file_builder.rs`: 99.13% lines
- `graph.rs`: 97.14% lines
- `shape_inference.rs`: 86.84% lines
- `graphviz.rs`: 92.70% lines
- `validator.rs`: 72.02% lines

Modules needing attention (<30%):
- `coreml_mlprogram.rs`: 29.56% lines (converter)
- `onnx.rs`: 24.57% lines (converter)
- `webnn_json.rs`: 26.44% lines
- `coreml.rs`: 0% lines (platform-specific executor)
- `loader.rs`: 0% lines (needs tests)
- `error.rs`: 0% lines (error types, display-only)

## Prerequisites

The coverage tools are automatically installed when you first run a coverage command. However, you can install them manually:

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Install LLVM tools component (required, installed automatically on first run)
rustup component add llvm-tools-preview
```

## Usage

### Quick Start

Generate a text-based coverage report:
```bash
make coverage
```

Generate and open an HTML report in your browser:
```bash
make coverage-open
```

### Available Targets

#### `make coverage`
Runs tests with coverage instrumentation and displays a text summary in the terminal.

**Output:** Text table showing coverage per file
**Use case:** Quick coverage check during development

```bash
make coverage
```

#### `make coverage-html`
Generates a detailed HTML report with line-by-line coverage visualization.

**Output:** HTML files in `target/llvm-cov/html/`
**Use case:** Deep dive into which lines are covered/uncovered

```bash
make coverage-html
# View report at: target/llvm-cov/html/index.html
```

#### `make coverage-lcov`
Generates an LCOV-format report suitable for CI/CD integration.

**Output:** LCOV file at `target/llvm-cov/lcov.info`
**Use case:** CI/CD pipelines, integration with coverage services (Codecov, Coveralls)

```bash
make coverage-lcov
# Upload to coverage service:
# bash <(curl -s https://codecov.io/bash) -f target/llvm-cov/lcov.info
```

#### `make coverage-open`
Generates HTML report and opens it in your default browser.

**Output:** HTML report opened in browser
**Use case:** Interactive exploration of coverage data

```bash
make coverage-open
```

#### `make coverage-clean`
Removes all coverage artifacts and instrumentation data.

**Use case:** Clean build, troubleshooting

```bash
make coverage-clean
```

## Understanding Coverage Reports

### Text Report

The text report shows a table with these columns:

- **Filename**: Source file path
- **Regions**: Code regions (branches, functions)
- **Missed Regions**: Uncovered regions
- **Cover**: Region coverage percentage
- **Functions**: Total functions in file
- **Missed Functions**: Uncovered functions
- **Executed**: Functions with at least one covered line
- **Lines**: Total lines of code
- **Missed Lines**: Uncovered lines
- **Cover**: Line coverage percentage

Example:
```
Filename                           Lines  Missed Lines  Cover
rustnn/src/shape_inference.rs      2090          275   86.84%
rustnn/src/graph.rs                 175            5   97.14%
```

### HTML Report

The HTML report provides:

1. **Overview Page** (`index.html`)
   - Summary statistics
   - Coverage percentages
   - List of all source files with coverage

2. **File Details**
   - Line-by-line view of source code
   - Color-coded coverage:
     - Green: Line covered by tests
     - Red: Line not covered
     - Gray: Non-executable line (comments, blank)
   - Execution counts for each line
   - Branch coverage indicators

3. **Navigation**
   - Filter by coverage percentage
   - Sort by filename, coverage, etc.
   - Click files to see detailed view

### LCOV Report

The LCOV format is a text-based format that includes:
- Line coverage data (LH, LF metrics)
- Function coverage data (FNH, FNF metrics)
- Branch coverage data (BRH, BRF metrics)

This format is compatible with:
- [Codecov](https://codecov.io/)
- [Coveralls](https://coveralls.io/)
- [lcov](http://ltp.sourceforge.net/coverage/lcov.php) HTML generator
- Most CI/CD coverage tools

## Workflow Integration

### Development Workflow

**Before committing code:**
```bash
# Run tests
make test

# Check coverage
make coverage

# If coverage decreased significantly, consider adding tests
make coverage-html
# Open target/llvm-cov/html/index.html to see which lines need coverage
```

**Adding new features:**
1. Write tests alongside new code
2. Run `make coverage-open` to verify new code is covered
3. Aim for >80% coverage on new modules

### CI/CD Integration

For GitHub Actions or other CI systems:

```yaml
- name: Generate coverage report
  run: make coverage-lcov

- name: Upload to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: target/llvm-cov/lcov.info
    fail_ci_if_error: true
```

## Coverage Goals

**Project Goals:**
- Overall line coverage: >60% (current: 47.25%)
- Core modules (graph, shape_inference, validator): >80%
- New code: >70% coverage required

**Current Focus Areas:**
1. Converters (ONNX, CoreML) - Low coverage due to integration testing needs
2. Executors - Platform-specific, some modules not testable in all environments
3. Error handling paths - Often not covered in happy-path tests

## Tips for Improving Coverage

### 1. Write Unit Tests
Focus on testing individual functions in isolation:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        let result = my_function(input);
        assert_eq!(result, expected);
    }
}
```

### 2. Test Error Paths
Don't forget to test failure cases:

```rust
#[test]
fn test_invalid_input() {
    let result = my_function(invalid_input);
    assert!(result.is_err());
}
```

### 3. Use Test Fixtures
Create helper functions for common test setups:

```rust
fn create_test_graph() -> GraphInfo {
    // Build a standard test graph
}

#[test]
fn test_with_fixture() {
    let graph = create_test_graph();
    // Test graph operations
}
```

### 4. Parametrized Tests
Test multiple inputs efficiently:

```rust
#[test]
fn test_data_type_sizes() {
    let test_cases = vec![
        (DataType::Float32, 4),
        (DataType::Float16, 2),
        (DataType::Int32, 4),
    ];

    for (dtype, expected_size) in test_cases {
        assert_eq!(dtype.bytes_per_element(), expected_size);
    }
}
```

### 5. Integration Tests
For converters and executors, add integration tests that exercise the full pipeline.

## Excluding Code from Coverage

Use `#[cfg(not(coverage))]` to exclude code that shouldn't be covered:

```rust
#[cfg(not(coverage))]
fn debug_only_function() {
    // This code won't be included in coverage metrics
}
```

Common exclusions:
- Debug/print functions
- Platform-specific code not testable in CI
- Generated code
- Trivial getters/setters

## Troubleshooting

### Coverage Tool Not Found
```bash
error: no such command: `llvm-cov`
```

**Solution:** Install cargo-llvm-cov:
```bash
cargo install cargo-llvm-cov
```

### LLVM Tools Missing
```bash
error: llvm-tools-preview component is not installed
```

**Solution:** The tool will prompt to install, or run manually:
```bash
rustup component add llvm-tools-preview
```

### Coverage Data Stale
If coverage seems incorrect after code changes:

```bash
make coverage-clean
make coverage-html
```

### Build Errors
If you get build errors during coverage:

1. Ensure regular tests pass first: `make test`
2. Clean and rebuild: `cargo clean && make coverage`
3. Check for feature flag issues - coverage runs with `--all-features`

## Further Reading

- [cargo-llvm-cov documentation](https://github.com/taiki-e/cargo-llvm-cov)
- [LLVM coverage mapping](https://llvm.org/docs/CoverageMappingFormat.html)
- [Rust testing best practices](https://doc.rust-lang.org/book/ch11-00-testing.html)

## Summary

- Use `make coverage` for quick checks
- Use `make coverage-open` for detailed analysis
- Use `make coverage-lcov` for CI integration
- Aim for >70% coverage on new code
- Focus on testing core logic and error paths
- Run coverage before committing significant changes
