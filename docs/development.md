# Development Guide

## Prerequisites

- **Rust**: 1.70+ (install from [rustup.rs](https://rustup.rs/))
- **Python**: 3.11+ with pip
- **Maturin**: `pip install maturin`
- **Optional**: Graphviz for visualization (`brew install graphviz` on macOS)

## Building from Source

```bash
# Clone repository
git clone https://github.com/tarekziade/rustnn.git
cd rustnn

# See all available commands
make help

# Build Rust library
make build

# Build Python package (downloads ONNX Runtime automatically)
make python-dev

# Run tests
make test                     # Rust tests
make python-test              # Python tests (includes WPT conformance)

# Build documentation
make docs-serve               # Live preview at http://127.0.0.1:8000
make docs-build               # Build static site
```

## Running Examples

### Python Examples

```bash
# Install package first
make python-dev

# Run examples
make python-example           # Run all examples
make mobilenet-demo           # MobileNetV2 on all 3 backends
make text-gen-demo            # Text generation with attention
make text-gen-train           # Train model on sample data
make text-gen-trained         # Generate with trained weights

# Or run individual examples
python examples/python_simple.py
python examples/python_matmul.py
python examples/mobilenetv2_complete.py examples/images/test.jpg --backend cpu
```

### Rust Examples

```bash
# Validate a graph
make run

# Generate visualization
make viz

# Convert to ONNX
make onnx

# Convert to CoreML
make coreml
```

## Testing

### Python Tests

```bash
# All tests (includes WPT conformance tests)
make python-test

# WPT conformance tests only
make python-test-wpt

# Or use pytest directly
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_python_api.py::test_context_creation -v

# With coverage
python -m pytest tests/ --cov=webnn --cov-report=html
```

### Rust Tests

```bash
# All Rust tests
make test

# Or use cargo directly
cargo test

# Specific module
cargo test validator

# With output
cargo test -- --nocapture
```

## Feature Flags

The project uses Cargo feature flags to control optional functionality. The Makefile handles these automatically:

```bash
# Python bindings with ONNX Runtime (recommended)
make python-dev              # Includes python,onnx-runtime features

# Build Python wheel
make python-build            # Production build with all features

# Or use cargo/maturin directly if needed
cargo build --features python,onnx-runtime
maturin develop --features python,onnx-runtime,coreml-runtime
```

## Development Workflow

### 1. Make Changes

Edit Rust code in `src/` or Python code in `python/webnn/`.

### 2. Format Code

```bash
# Rust (automatically formats)
make fmt

# Python
black python/ tests/
```

### 3. Run Tests

```bash
# Full test suite
make test                    # Rust tests
make python-test             # Python tests

# Or run comprehensive validation
make validate-all-env        # Build, test, convert, validate
```

### 4. Build and Test Python Package

```bash
make python-dev              # Install in development mode
make python-test             # Run all tests
```

### 5. Update Documentation

Edit files in `docs/` and preview:

```bash
make docs-serve              # Live preview at http://127.0.0.1:8000
make docs-build              # Build static site
make ci-docs                 # Build in strict mode (CI)
```

## Debugging

### Rust

```bash
# Debug build
make build

# Run with visualization
make viz

# Run with backtrace
RUST_BACKTRACE=1 make run
```

### Python

```bash
# Run specific example with verbose output
python examples/python_simple.py

# Or enable debug logging in code
import webnn
import logging

logging.basicConfig(level=logging.DEBUG)

# Your code here
```

## Common Tasks

### Add a New Operation

1. Update `graph.rs` with new operation type
2. Add validation logic in `validator.rs`
3. Implement conversion in `converters/onnx.rs` and `converters/coreml.rs`
4. Add Python binding in `src/python/graph_builder.rs`
5. Add tests in `tests/test_python_api.py`

### Add a New Backend

1. Create new file in `src/executors/your_backend.rs`
2. Add feature flag in `Cargo.toml`
3. Implement executor trait/functions
4. Add conditional compilation in `src/executors/mod.rs`
5. Wire up in `src/python/context.rs` backend selection
6. Add tests

### Update Documentation

1. Edit markdown files in `docs/`
2. Preview with `make docs-serve`
3. Check links and formatting
4. Build with `make docs-build`
5. Test in strict mode with `make ci-docs`

## Troubleshooting

### Maturin Build Fails

```bash
# Update Rust
rustup update

# Clean all build artifacts
make clean-all

# Rebuild from scratch
make python-dev
```

### Import Errors

```bash
# Ensure you're in the right virtual environment
which python

# Clean and reinstall
make python-clean
make python-dev

# Verify installation
python -c "import webnn; print(webnn.__version__)"
```

### ONNX Runtime Issues

The Makefile automatically downloads ONNX Runtime for you:

```bash
# Download ONNX Runtime manually if needed
make onnxruntime-download

# Or install system-wide (optional)
brew install onnxruntime

# Build with system ONNX Runtime
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/opt/homebrew/lib
make python-dev
```

### Test Failures

```bash
# Run tests with verbose output
make python-test

# Run specific test
python -m pytest tests/test_python_api.py::test_name -xvs

# Check if backend is available
python -c "import webnn; ctx = webnn.ML().create_context(); print(ctx.accelerated)"
```

## Code Style

### Rust

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Write doc comments for public APIs

### Python

- Follow [PEP 8](https://pep8.org/)
- Use type hints
- Write docstrings for public APIs
- Use `black` for formatting

## Git Workflow

### Commits

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature X

- Detail 1
- Detail 2

[BOT] Generated with [Claude Code](https://claude.com/claude-code)"

# Push
git push origin main
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

- `cargo fmt --check` - Ensures Rust code is formatted
- Tests run automatically in CI

## CI/CD

### GitHub Actions

The project uses GitHub Actions for CI:

- `.github/workflows/ci.yml` - Main CI pipeline
  - Runs on push and pull requests
  - Tests on Linux and macOS
  - Builds Python wheels
  - Runs all tests

### Local CI Simulation

```bash
# Run the same checks as CI
make fmt                     # Format code
cargo clippy -- -D warnings  # Lint checks
make validate-all-env        # Full validation pipeline
make ci-docs                 # Documentation build (strict mode)
```

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 Guide](https://pyo3.rs/)
- [W3C WebNN Spec](https://www.w3.org/TR/webnn/)
- [ONNX Documentation](https://onnx.ai/)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
