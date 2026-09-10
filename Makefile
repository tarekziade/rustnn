CARGO := cargo
DOT ?= dot
GRAPH_FILE ?= examples/sample_graph.json
DOT_PATH ?= target/graph.dot
PNG_PATH ?= target/graph.png
ONNX_PATH ?= target/graph.onnx
COREML_PATH ?= target/graph.mlmodel
COREMLC_PATH ?= target/graph.mlmodelc
LITERT_PATH ?= target/graph.tflite
CANN_PATH ?= target/graph.cann
OHOS_SDK_NATIVE ?=

ORT_VERSION ?= 1.29.0
ORT_BASE ?= https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)
ORT_DIR ?= target/onnxruntime
MATURIN_ARGS ?=
CHROMEDRIVER_CACHE ?= $(CURDIR)/.cache/chromedriver
CHROMEDRIVER ?= $(CHROMEDRIVER_CACHE)/chromedriver

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

# Bundled OHOS cross-compilation environment
# CANN_DDK points at the Huawei CANN-Kit DDK root (.../CANN-Kit-next/ddk/).
CANN_DDK ?=
CANN_CROSS_ENV = CC_aarch64_unknown_linux_ohos=$(OHOS_SDK_NATIVE)/llvm/bin/clang \
	CXX_aarch64_unknown_linux_ohos=$(OHOS_SDK_NATIVE)/llvm/bin/clang++ \
	AR_aarch64_unknown_linux_ohos=$(OHOS_SDK_NATIVE)/llvm/bin/llvm-ar \
	CFLAGS_aarch64_unknown_linux_ohos="--target=aarch64-linux-ohos --sysroot=$(OHOS_SDK_NATIVE)/sysroot" \
	CXXFLAGS_aarch64_unknown_linux_ohos="--target=aarch64-linux-ohos --sysroot=$(OHOS_SDK_NATIVE)/sysroot -std=c++17" \
	CARGO_TARGET_AARCH64_UNKNOWN_LINUX_OHOS_LINKER=$(OHOS_SDK_NATIVE)/llvm/bin/clang \
	CANN_DDK=$(CANN_DDK) \
	RUSTFLAGS="-Clink-arg=--target=aarch64-linux-ohos -Clink-arg=--sysroot=$(OHOS_SDK_NATIVE)/sysroot"

.PHONY: build test fmt run viz onnx coreml coreml-validate onnx-validate litert cann cann-build cann-device-test validate-cann-env validate-all-env \
	docs-serve docs-build docs-clean ci-docs docs-backend-ops docs-backend-ops-check \
	fmt-check lint \
	coverage coverage-html coverage-lcov coverage-open coverage-clean \
	help clean-all
	webnn-chromedriver require-wpt-cache test-webnn-wpt-chrome test-webnn-wpt-chrome-headless

clean:
	$(CARGO) clean
	rm -f target/graph.dot target/graph.png target/graph.onnx target/graph.mlmodel
	rm -rf target/graph.mlmodelc
	rm -rf $(ORT_DIR)

build:
	$(CARGO) build

test:
	@echo "Running rustfmt..."
	$(CARGO) fmt
	@echo "Running clippy..."
	$(CARGO) clippy --all-targets -- -D warnings
	@echo "Running tests..."
	$(CARGO) test
	@echo "Checking backend operator support report drift..."
	$(MAKE) docs-backend-ops-check

fetch-wpt:
	node scripts/fetch_wpt.mjs

require-wpt-cache:
	@test -d .cache/wpt/webnn || (echo "WPT cache is missing; run 'make fetch-wpt' first." >&2; exit 1)

# Download Chrome for Testing's newest driver compatible with the Chrome on PATH.
# Set CHROME=... to select a different installed Chrome binary, or
# CHROMEDRIVER_VERSION=... to pin a particular Chrome-for-Testing release.
webnn-chromedriver:
	@set -eu; \
	chrome="$(CHROME)"; \
	if [ -z "$$chrome" ]; then \
		for candidate in google-chrome google-chrome-stable google-chrome-unstable chromium chromium-browser; do \
			if command -v "$$candidate" >/dev/null 2>&1; then chrome="$$candidate"; break; fi; \
		done; \
	fi; \
	version="$(CHROMEDRIVER_VERSION)"; \
	if [ -z "$$version" ] && [ -n "$$chrome" ]; then \
		chrome_version="$$($$chrome --product-version 2>/dev/null || true)"; \
		major="$${chrome_version%%.*}"; \
		case "$$major" in (*[!0-9]*|'') ;; (*) version="$$(curl -fsSL "https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_$$major" || true)";; esac; \
	fi; \
	if [ -z "$$version" ]; then version="$$(curl -fsSL https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE)"; fi; \
	case "$$(uname -s):$$(uname -m)" in \
		Linux:x86_64) platform=linux64;; \
		Linux:aarch64) platform=linux64;; \
		Darwin:arm64) platform=mac-arm64;; \
		Darwin:x86_64) platform=mac-x64;; \
		*) echo "Unsupported platform for Chrome for Testing driver: $$(uname -s) $$(uname -m)" >&2; exit 1;; \
	esac; \
	mkdir -p "$(CHROMEDRIVER_CACHE)"; \
	tmp="$$(mktemp -d "$(CHROMEDRIVER_CACHE)/download.XXXXXX")"; \
	trap 'rm -rf "$$tmp"' EXIT; \
	curl -fsSL "https://storage.googleapis.com/chrome-for-testing-public/$$version/$$platform/chromedriver-$$platform.zip" -o "$$tmp/chromedriver.zip"; \
	unzip -oq "$$tmp/chromedriver.zip" -d "$(CHROMEDRIVER_CACHE)"; \
	ln -sfn "chromedriver-$$platform/chromedriver" "$(CHROMEDRIVER)"; \
	chmod +x "$(CHROMEDRIVER)"; \
	echo "ChromeDriver $$version: $(CHROMEDRIVER)"

# Requires wasm-pack, Node.js, curl, and unzip. webdriver.json enables Chrome's WebNN feature.
test-webnn-wpt-chrome: require-wpt-cache webnn-chromedriver
	@env -u CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER WASM_BINDGEN_TEST_ADDRESS=127.0.0.1:0 RUSTFLAGS='-C target-feature=+reference-types --cfg=web_sys_unstable_apis' wasm-pack test --chrome --chromedriver "$(CHROMEDRIVER)" -- --test webnn_wpt -F webnn-runtime,webnn-wpt-tests -- --nocapture

test-webnn-wpt-chrome-headless: require-wpt-cache webnn-chromedriver
	@env -u CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER RUSTFLAGS='-C target-feature=+reference-types --cfg=web_sys_unstable_apis' wasm-pack test --headless --chrome --chromedriver "$(CHROMEDRIVER)" -- --test webnn_wpt -F webnn-runtime,webnn-wpt-tests -- --nocapture

# WPT conformance (requires Node.js, WPT cache; set WPT_BACKEND=onnx|trtx to pick backend)
test-wpt: onnxruntime-download
	$(ORT_ENV_VARS) $(CARGO) test --test run_wpt_conformance --features onnx-runtime -- --test-threads 1

test-wpt-trtx:
	$(CARGO) test --test run_wpt_conformance --features "onnx-runtime,trtx-runtime" -- trtx --test-threads 1

test-wpt-litert:
	@HOST_TRIPLE=$$(rustc -vV 2>/dev/null | grep host: | cut -d' ' -f2); \
	LITERT_LIB_DIR="$${HOME}/.cache/litert-sys/v0.10.2/$${HOST_TRIPLE}"; \
	LD_LIBRARY_PATH="$$LITERT_LIB_DIR:$$LD_LIBRARY_PATH" \
	LIBRARY_PATH="$$LITERT_LIB_DIR:$$LIBRARY_PATH" \
	WPT_REPORT_JSON="$(WPT_REPORT_LITERT_JSON)" \
	$(CARGO) test --test run_wpt_conformance --features "litert-runtime" -- litert --test-threads=1

test-wpt-coreml:
	$(CARGO) test --test run_wpt_conformance --features coreml-runtime -- coreml --test-threads 1

test-wpt-coreml-report:
	@mkdir -p reports
	WPT_REPORT_JSON=reports/wpt-conformance.json $(CARGO) test --test run_wpt_conformance --features coreml-runtime -- coreml --test-threads 1

test-wpt-op: onnxruntime-download
	@test -n "$(OP)" || (echo "Usage: make test-wpt-op OP=add" && exit 1)
	$(ORT_ENV_VARS) $(CARGO) test --test run_wpt_conformance --features onnx-runtime -- $(OP) --test-threads 1

WPT_BACKEND ?= onnx
# Full WPT run with JSON/HTML reports; exits 0 even if trials fail (for nightly pages).
test-wpt-report: fetch-wpt onnxruntime-download
	@mkdir -p reports
	@HOST_TRIPLE=$$(rustc -vV 2>/dev/null | grep host: | cut -d' ' -f2); \
	LITERT_LIB_DIR="$${HOME}/.cache/litert-sys/v0.10.2/$${HOST_TRIPLE}"; \
	LD_LIBRARY_PATH="$$LITERT_LIB_DIR:$$LD_LIBRARY_PATH" \
	LIBRARY_PATH="$$LITERT_LIB_DIR:$$LIBRARY_PATH" \
	RUST_BACKTRACE=1 WPT_REPORT_JSON=reports/wpt-conformance.json $(ORT_ENV_VARS) $(CARGO) test --test run_wpt_conformance --features $(WPT_BACKEND)-runtime -- --test-threads 1

fmt:
	$(CARGO) fmt

fmt-check:
	$(CARGO) fmt --check

lint:
	$(CARGO) clippy --all-targets -- -D warnings

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
	$(CARGO) run --features coreml-runtime -- $(GRAPH_FILE) --convert coreml --convert-output $(COREML_PATH)
	@echo "CoreML graph written to $(COREML_PATH)"

coreml-validate: coreml
	$(CARGO) run --features coreml-runtime -- $(GRAPH_FILE) --convert coreml --convert-output $(COREML_PATH) --run-coreml --coreml-compiled-output $(COREMLC_PATH)

litert:
	$(CARGO) run --features litert-runtime -- $(GRAPH_FILE) --convert litert --convert-output $(LITERT_PATH)

cann:
	$(CARGO) run --features cann-runtime -- $(GRAPH_FILE) --convert cann --convert-output $(CANN_PATH)

validate-cann-env:
	@if ! rustup target list --installed | grep -qx aarch64-unknown-linux-ohos; then \
	    echo "Error: Rust target 'aarch64-unknown-linux-ohos' is not installed."; \
	    echo "  Install it with:  rustup target add aarch64-unknown-linux-ohos"; \
	    exit 1; \
	fi
	@if [ -z "$(OHOS_SDK_NATIVE)" ]; then \
	    echo "Error: OHOS_SDK_NATIVE not set. export OHOS_SDK_NATIVE=/path/to/OpenHarmony/<version>/sdk/native"; \
	    exit 1; \
	fi
	@if [ -z "$(CANN_DDK)" ]; then \
	    echo "Error: CANN_DDK not set. export CANN_DDK=/path/to/CANN-Kit-next/ddk/"; \
	    exit 1; \
	fi

cann-build: validate-cann-env
	$(CANN_CROSS_ENV) $(CARGO) build --target aarch64-unknown-linux-ohos --features cann-runtime --release

cann-device-test: validate-cann-env
	$(CANN_CROSS_ENV) $(CARGO) test --test test_cann_execution --no-run \
		--target aarch64-unknown-linux-ohos --features cann-runtime --release
	CANN_DDK=$(CANN_DDK) \
	./scripts/ohos-test-helper.sh $(filter-out $@,$(MAKECMDGOALS))

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

docs-backend-ops:
	@echo "Generating backend operator support report..."
	python3 scripts/generate_backend_operator_report.py

docs-backend-ops-check:
	@echo "Checking backend operator support report for drift..."
	python3 scripts/generate_backend_operator_report.py --check

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
	@echo "  test               - Run rustfmt, clippy, Rust tests, and backend report drift check"
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
	@echo "WPT Conformance:"
	@echo "  fetch-wpt          - Download/update WPT corpus (.cache/wpt)"
	@echo "  test-wpt           - Run full WPT suite (ONNX CPU, ~2482 cases)"
	@echo "  test-wpt-op OP=... - Run filtered WPT trials"
	@echo "  test-wpt-report    - Run full WPT suite and write JSON/HTML reports (ignores trial failures)"
	@echo "  test-wpt-trtx      - Run WPT suite via TensorRT (skips when GPU unavailable)"
	@echo "  webnn-chromedriver - Download a ChromeDriver compatible with installed Chrome"
	@echo "  test-webnn-wpt-chrome - Run browser WebNN WPT graph-build tests in Chrome"
	@echo "  test-webnn-wpt-chrome-headless - Run the browser WebNN WPT tests headlessly"
	@echo ""
	@echo "CoreML Conversion:"
	@echo "  coreml             - Convert graph to CoreML format"
	@echo "  coreml-validate    - Convert and validate CoreML graph"
	@echo ""
	@echo "LiteRT Conversion:"
	@echo "  litert             - Convert graph to LiteRT/TFLite format"
	@echo ""
	@echo "CANN:"
	@echo "  cann               - Convert graph to CANN/HiAI format"
	@echo "  cann-build    		- Cross-compile rustnn for OHOS via cargo"
	@echo "  cann-device-test   - Test on device via scripts/ohos-test-helper.sh"
	@echo ""
	@echo "Documentation:"
	@echo "  docs-serve         - Serve documentation with live reload"
	@echo "  docs-build         - Build static documentation site"
	@echo "  ci-docs            - Build documentation in strict mode (CI)"
	@echo "  docs-clean         - Clean documentation artifacts"
	@echo "  docs-backend-ops   - Generate backend operator support report"
	@echo "  docs-backend-ops-check - Verify backend operator report is up to date"
	@echo ""
	@echo "Comprehensive:"
	@echo "  validate-all-env   - Run full pipeline (build/test/convert/validate)"
	@echo "  clean-all          - Clean all artifacts (Rust + docs)"
	@echo "  help               - Show this help message"
	@echo ""
	@echo "Note: Python bindings are available in the pywebnn package:"
	@echo "      https://github.com/rustnn/pywebnn"
	@echo ""
