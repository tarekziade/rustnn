CARGO := cargo
DOT ?= dot
GRAPH_FILE ?= examples/sample_graph.json
DOT_PATH ?= target/graph.dot
PNG_PATH ?= target/graph.png
ONNX_PATH ?= target/graph.onnx
COREML_PATH ?= target/graph.mlmodel
COREMLC_PATH ?= target/graph.mlmodelc
ORT_VERSION ?= 1.17.0
ORT_BASE ?= https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)
ORT_TARBALL ?= onnxruntime-osx-arm64-$(ORT_VERSION).tgz
ORT_DIR ?= target/onnxruntime
ORT_LIB_DIR ?= $(ORT_DIR)/onnxruntime-osx-arm64-$(ORT_VERSION)/lib
ORT_LIB_LOCATION ?= $(ORT_LIB_DIR)
.PHONY: build test fmt run viz onnx coreml coreml-validate onnx-validate validate-all-env \
	python-dev python-build python-test python-test-wpt python-clean python-example \
	docs-serve docs-build docs-clean ci-docs \
	help clean-all

clean:
	$(CARGO) clean
	rm -f target/graph.dot target/graph.png target/graph.onnx target/graph.mlmodel
	rm -rf target/graph.mlmodelc
	rm -rf $(ORT_DIR)

build:
	$(CARGO) build

test:
	$(CARGO) test

fmt:
	$(CARGO) fmt

run:
	$(CARGO) run -- examples/sample_graph.json

viz:
	$(CARGO) run -- $(GRAPH_FILE) --export-dot $(DOT_PATH)
	$(DOT) -Tpng $(DOT_PATH) -o $(PNG_PATH)
	open $(PNG_PATH)


onnxruntime-download:
	mkdir -p $(ORT_DIR)
	curl -L $(ORT_BASE)/$(ORT_TARBALL) -o $(ORT_DIR)/$(ORT_TARBALL)
	tar -xzf $(ORT_DIR)/$(ORT_TARBALL) -C $(ORT_DIR)

onnx: onnxruntime-download
	ORT_STRATEGY=system ORT_LIB_LOCATION=$(ORT_LIB_LOCATION) DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) RUSTFLAGS="-L $(ORT_LIB_DIR)" $(CARGO) run --features onnx-runtime -- $(GRAPH_FILE) --convert onnx --convert-output $(ONNX_PATH)
	@echo "ONNX graph written to $(ONNX_PATH)"

onnx-validate: onnx
	ORT_STRATEGY=system ORT_LIB_LOCATION=$(ORT_LIB_LOCATION) DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) RUSTFLAGS="-L $(ORT_LIB_DIR)" $(CARGO) run --features onnx-runtime -- $(GRAPH_FILE) --convert onnx --convert-output $(ONNX_PATH) --run-onnx

coreml:
	$(CARGO) run -- $(GRAPH_FILE) --convert coreml --convert-output $(COREML_PATH)
	@echo "CoreML graph written to $(COREML_PATH)"

coreml-validate: coreml
	$(CARGO) run --features coreml-runtime -- $(GRAPH_FILE) --convert coreml --convert-output $(COREML_PATH) --run-coreml --coreml-compiled-output $(COREMLC_PATH)

webnn-venv:
	python3 -m venv .venv-webnn
	. .venv-webnn/bin/activate && pip install --upgrade pip && pip install git+https://github.com/huningxin/onnx2webnn.git

validate-all-env: build test onnx-validate coreml-validate
	@echo "Full pipeline (build/test/convert/validate) completed."

# ==============================================================================
# Python Targets
# ==============================================================================

python-dev: onnxruntime-download
	@echo "Installing Python package in development mode..."
	pip install maturin
	ORT_STRATEGY=system \
	ORT_LIB_LOCATION=$(ORT_LIB_LOCATION) \
	DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) \
	RUSTFLAGS="-L $(ORT_LIB_DIR)" \
	maturin develop --features python,onnx-runtime

python-build: onnxruntime-download
	@echo "Building Python wheel..."
	pip install maturin
	ORT_STRATEGY=system \
	ORT_LIB_LOCATION=$(ORT_LIB_LOCATION) \
	DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) \
	RUSTFLAGS="-L $(ORT_LIB_DIR)" \
	maturin build --features python,onnx-runtime --release

python-test: python-dev
	@echo "Running Python tests (includes WPT conformance tests)..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python -m pytest tests/ -v; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python -m pytest tests/ -v; \
	fi

python-test-wpt: python-dev
	@echo "Running WPT conformance tests only..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python -m pytest tests/test_wpt_conformance.py -v; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python -m pytest tests/test_wpt_conformance.py -v; \
	fi

python-example: python-dev
	@echo "Running Python examples..."
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/python examples/python_simple.py; \
		.venv-webnn/bin/python examples/python_matmul.py; \
	else \
		python examples/python_simple.py; \
		python examples/python_matmul.py; \
	fi

python-clean:
	@echo "Cleaning Python artifacts..."
	rm -rf target/wheels
	rm -rf python/webnn/__pycache__
	rm -rf tests/__pycache__
	rm -rf examples/__pycache__
	find . -name "*.pyc" -delete
	find . -name "*.so" -delete
	find . -name "*.pyd" -delete

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

ci-docs:
	@echo "Building documentation in strict mode (CI)..."
	@command -v mkdocs >/dev/null 2>&1 || { echo "Installing mkdocs..."; pip install -r docs/requirements.txt; }
	mkdocs build --strict

docs-clean:
	@echo "Cleaning documentation build artifacts..."
	rm -rf site/

# ==============================================================================
# Comprehensive Clean
# ==============================================================================

clean-all: clean python-clean docs-clean
	@echo "All build artifacts cleaned."

# ==============================================================================
# Help Target
# ==============================================================================

help:
	@echo "Rust WebNN Graph - Available Targets"
	@echo "===================================="
	@echo ""
	@echo "Rust Targets:"
	@echo "  build              - Build the Rust project"
	@echo "  test               - Run Rust tests"
	@echo "  fmt                - Format Rust code"
	@echo "  run                - Run with sample graph"
	@echo "  clean              - Clean Rust build artifacts"
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
	@echo "Python API:"
	@echo "  python-dev         - Install Python package in development mode"
	@echo "  python-build       - Build Python wheel"
	@echo "  python-test        - Run all Python tests (includes WPT tests)"
	@echo "  python-test-wpt    - Run WPT conformance tests only"
	@echo "  python-example     - Run Python examples"
	@echo "  python-clean       - Clean Python artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  docs-serve         - Serve documentation with live reload"
	@echo "  docs-build         - Build static documentation site"
	@echo "  ci-docs            - Build documentation in strict mode (CI)"
	@echo "  docs-clean         - Clean documentation artifacts"
	@echo ""
	@echo "Comprehensive:"
	@echo "  validate-all-env   - Run full pipeline (build/test/convert/validate)"
	@echo "  clean-all          - Clean all artifacts (Rust + Python + docs)"
	@echo "  help               - Show this help message"
	@echo ""
