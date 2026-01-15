CARGO := cargo
DOT ?= dot
GRAPH_FILE ?= examples/sample_graph.json
DOT_PATH ?= target/graph.dot
PNG_PATH ?= target/graph.png
ONNX_PATH ?= target/graph.onnx
COREML_PATH ?= target/graph.mlmodel
COREMLC_PATH ?= target/graph.mlmodelc
ORT_VERSION ?= 1.23.2
ORT_BASE ?= https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)
ORT_DIR ?= target/onnxruntime
MATURIN_ARGS ?=

# Platform detection
UNAME_S := $(shell uname)
UNAME_M := $(shell uname -m)

# Set platform-specific ONNX Runtime tarball name
ifeq ($(UNAME_S),Darwin)
	ifeq ($(UNAME_M),arm64)
		ORT_TARBALL ?= onnxruntime-osx-arm64-$(ORT_VERSION).tgz
	else
		ORT_TARBALL ?= onnxruntime-osx-x86_64-$(ORT_VERSION).tgz
	endif
	ORT_SHARED_GLOB ?= $(ORT_LIB_DIR)/libonnxruntime*.dylib
	ORT_DYLIB_FILE ?= $(ORT_LIB_DIR)/libonnxruntime.$(ORT_VERSION).dylib
	# Defer ORT_ENV_VARS assignment until after ORT_LIB_DIR is defined
	ORT_ENV_VARS_DEFERRED := 1
else ifeq ($(OS),Windows_NT)
	ORT_TARBALL ?= onnxruntime-win-x64-$(ORT_VERSION).zip
	ORT_SHARED_GLOB ?= $(ORT_LIB_DIR)/onnxruntime.dll
	ORT_DYLIB_FILE ?= $(ORT_LIB_DIR)/onnxruntime.dll
	# Defer ORT_ENV_VARS assignment until after ORT_LIB_DIR is defined
	ORT_ENV_VARS_DEFERRED := 1
else
	# Linux (including WSL)
	ifeq ($(UNAME_M),aarch64)
		ORT_TARBALL ?= onnxruntime-linux-aarch64-$(ORT_VERSION).tgz
	else
		ORT_TARBALL ?= onnxruntime-linux-x64-$(ORT_VERSION).tgz
	endif
	ORT_SHARED_GLOB ?= $(ORT_LIB_DIR)/libonnxruntime*.so*
	ORT_DYLIB_FILE ?= $(ORT_LIB_DIR)/libonnxruntime.so.$(ORT_VERSION)
	# Use absolute path for LD_LIBRARY_PATH (will be set after ORT_LIB_DIR_ABS is defined)
	ORT_ENV_VARS_NEEDS_ABS := 1
endif

# Derived paths (must come after ORT_TARBALL is set)
ORT_DIR_NAME_TMP := $(ORT_TARBALL:.tgz=)
ORT_DIR_NAME_TMP := $(ORT_DIR_NAME_TMP:.tar.gz=)
ORT_DIR_NAME ?= $(ORT_DIR_NAME_TMP:.zip=)
ORT_LIB_DIR ?= $(ORT_DIR)/$(ORT_DIR_NAME)/lib
ORT_LIB_LOCATION ?= $(ORT_LIB_DIR)

# Absolute path for library directory (needed for LD_LIBRARY_PATH on Linux)
ORT_LIB_DIR_ABS := $(shell pwd)/$(ORT_LIB_DIR)

# Set ORT_ENV_VARS now that paths are defined
ifeq ($(ORT_ENV_VARS_NEEDS_ABS),1)
	# Linux: Use absolute path in LD_LIBRARY_PATH for dynamic linker
	ORT_ENV_VARS := LD_LIBRARY_PATH=$(ORT_LIB_DIR_ABS):$$LD_LIBRARY_PATH ORT_DYLIB_PATH=$(ORT_DYLIB_FILE)
else ifeq ($(ORT_ENV_VARS_DEFERRED),1)
	# macOS/Windows: Use ORT_DYLIB_PATH with relative path
	ORT_ENV_VARS := ORT_DYLIB_PATH=$(ORT_DYLIB_FILE)
endif

.PHONY: build test fmt run viz onnx coreml coreml-validate onnx-validate validate-all-env \
	docs-serve docs-build docs-clean ci-docs \
	coverage coverage-html coverage-lcov coverage-open coverage-clean \
	help clean-all

clean:
	$(CARGO) clean
	rm -f target/graph.dot target/graph.png target/graph.onnx target/graph.mlmodel
	rm -rf target/graph.mlmodelc
	rm -rf $(ORT_DIR)

build:
	$(CARGO) build

test:
	@echo "Running clippy..."
	$(CARGO) clippy --all-targets -- -D warnings
	@echo "Running tests..."
	$(CARGO) test

fmt:
	$(CARGO) fmt

# ==============================================================================
# Code Coverage Targets
# ==============================================================================

coverage:
	@echo "Running tests with coverage instrumentation..."
	cargo llvm-cov --all-features --workspace --lib

coverage-html:
	@echo "Generating HTML coverage report..."
	cargo llvm-cov --all-features --workspace --lib --html
	@echo "[OK] HTML coverage report generated in target/llvm-cov/html/"

coverage-lcov:
	@echo "Generating LCOV coverage report..."
	cargo llvm-cov --all-features --workspace --lib --lcov --output-path target/llvm-cov/lcov.info
	@echo "[OK] LCOV coverage report generated at target/llvm-cov/lcov.info"

coverage-open: coverage-html
	@echo "Opening coverage report in browser..."
	open target/llvm-cov/html/index.html

coverage-clean:
	@echo "Cleaning coverage artifacts..."
	cargo llvm-cov clean --workspace
	rm -rf target/llvm-cov/
	@echo "[OK] Coverage artifacts cleaned"

run:
	$(CARGO) run -- examples/sample_graph.json

viz:
	$(CARGO) run -- $(GRAPH_FILE) --export-dot $(DOT_PATH)
	$(DOT) -Tpng $(DOT_PATH) -o $(PNG_PATH)
	open $(PNG_PATH)


onnxruntime-download:
	@if [ -d "$(ORT_LIB_DIR)" ]; then \
		echo "ONNX Runtime already downloaded at $(ORT_LIB_DIR)"; \
	else \
		echo "Downloading ONNX Runtime $(ORT_VERSION)..."; \
		mkdir -p $(ORT_DIR); \
		curl -L $(ORT_BASE)/$(ORT_TARBALL) -o $(ORT_DIR)/$(ORT_TARBALL); \
		if echo "$(ORT_TARBALL)" | grep -q '\.zip$$'; then \
			unzip -q $(ORT_DIR)/$(ORT_TARBALL) -d $(ORT_DIR); \
		else \
			tar -xzf $(ORT_DIR)/$(ORT_TARBALL) -C $(ORT_DIR); \
		fi; \
		echo "[OK] ONNX Runtime downloaded and extracted"; \
	fi

onnx: onnxruntime-download
	$(ORT_ENV_VARS) $(CARGO) run --features onnx-runtime -- $(GRAPH_FILE) --convert onnx --convert-output $(ONNX_PATH)
	@echo "ONNX graph written to $(ONNX_PATH)"

onnx-validate: onnx
	$(ORT_ENV_VARS) $(CARGO) run --features onnx-runtime -- $(GRAPH_FILE) --convert onnx --convert-output $(ONNX_PATH) --run-onnx

coreml:
	$(CARGO) run -- $(GRAPH_FILE) --convert coreml --convert-output $(COREML_PATH)
	@echo "CoreML graph written to $(COREML_PATH)"

coreml-validate: coreml
	$(CARGO) run --features coreml-runtime -- $(GRAPH_FILE) --convert coreml --convert-output $(COREML_PATH) --run-coreml --coreml-compiled-output $(COREMLC_PATH)

validate-all-env: build test onnx-validate coreml-validate
	@echo "Full pipeline (build/test/convert/validate) completed."

# ==============================================================================
# Documentation Targets
# ==============================================================================

docs-serve:
	@echo "Serving documentation with live reload..."
	@command -v mkdocs >/dev/null 2>&1 || { echo "Installing mkdocs..."; pip install -r docs/requirements.txt; }
	mkdocs serve

docs-build:
	@echo "Building documentation..."
	@command -v mkdocs >/dev/null 2>&1 || { echo "Installing mkdocs..."; pip install -r docs/requirements.txt; }
	mkdocs build
	@touch site/.nojekyll
	@echo "Created .nojekyll file to prevent GitHub Pages Jekyll processing"

ci-docs:
	@echo "Building documentation in strict mode (CI)..."
	@command -v mkdocs >/dev/null 2>&1 || { echo "Installing mkdocs..."; pip install -r docs/requirements.txt; }
	mkdocs build --strict
	@touch site/.nojekyll
	@echo "Created .nojekyll file to prevent GitHub Pages Jekyll processing"

docs-clean:
	@echo "Cleaning documentation build artifacts..."
	rm -rf site/

# ==============================================================================
# Comprehensive Clean
# ==============================================================================

clean-all: clean docs-clean coverage-clean
	@echo "All build artifacts cleaned."

# ==============================================================================
# Help Target
# ==============================================================================

help:
	@echo "rustnn - Available Targets"
	@echo "=========================="
	@echo ""
	@echo "Rust Targets:"
	@echo "  build              - Build the Rust project"
	@echo "  test               - Run Rust tests"
	@echo "  fmt                - Format Rust code"
	@echo "  run                - Run with sample graph"
	@echo "  clean              - Clean Rust build artifacts"
	@echo ""
	@echo "Code Coverage:"
	@echo "  coverage           - Run tests with coverage (text output)"
	@echo "  coverage-html      - Generate HTML coverage report"
	@echo "  coverage-lcov      - Generate LCOV report (for CI)"
	@echo "  coverage-open      - Generate and open HTML report in browser"
	@echo "  coverage-clean     - Clean coverage artifacts"
	@echo ""
	@echo "Visualization:"
	@echo "  viz                - Generate and open graph visualization"
	@echo ""
	@echo "ONNX Conversion:"
	@echo "  onnxruntime-download - Download ONNX Runtime"
	@echo "  onnx               - Convert graph to ONNX format"
	@echo "  onnx-validate      - Convert and validate ONNX graph"
	@echo ""
	@echo "CoreML Conversion:"
	@echo "  coreml             - Convert graph to CoreML format"
	@echo "  coreml-validate    - Convert and validate CoreML graph"
	@echo ""
	@echo "Documentation:"
	@echo "  docs-serve         - Serve documentation with live reload"
	@echo "  docs-build         - Build static documentation site"
	@echo "  ci-docs            - Build documentation in strict mode (CI)"
	@echo "  docs-clean         - Clean documentation artifacts"
	@echo ""
	@echo "Comprehensive:"
	@echo "  validate-all-env   - Run full pipeline (build/test/convert/validate)"
	@echo "  clean-all          - Clean all artifacts (Rust + docs)"
	@echo "  help               - Show this help message"
	@echo ""
	@echo "Note: Python bindings are available in the pywebnn package:"
	@echo "      https://github.com/rustnn/pywebnn"
	@echo ""
