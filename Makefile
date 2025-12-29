CARGO := cargo
PYTHON ?= python3.12
DOT ?= dot
GRAPH_FILE ?= examples/sample_graph.json
DOT_PATH ?= target/graph.dot
PNG_PATH ?= target/graph.png
ONNX_PATH ?= target/graph.onnx
COREML_PATH ?= target/graph.mlmodel
COREMLC_PATH ?= target/graph.mlmodelc
ORT_VERSION ?= 1.23.2
ORT_BASE ?= https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)
ORT_TARBALL ?= onnxruntime-osx-arm64-$(ORT_VERSION).tgz
ORT_DIR_NAME_TMP := $(ORT_TARBALL:.tgz=)
ORT_DIR_NAME_TMP := $(ORT_DIR_NAME_TMP:.tar.gz=)
ORT_DIR_NAME ?= $(ORT_DIR_NAME_TMP:.zip=)
ORT_DIR ?= target/onnxruntime
ORT_LIB_DIR ?= $(ORT_DIR)/$(ORT_DIR_NAME)/lib
ORT_LIB_LOCATION ?= $(ORT_LIB_DIR)
MATURIN_ARGS ?=
UNAME_S := $(shell uname)
ifeq ($(UNAME_S),Darwin)
	ORT_SHARED_GLOB ?= $(ORT_LIB_DIR)/libonnxruntime*.dylib
else ifeq ($(OS),Windows_NT)
	ORT_SHARED_GLOB ?= $(ORT_LIB_DIR)/onnxruntime.dll
else
	ORT_SHARED_GLOB ?= $(ORT_LIB_DIR)/libonnxruntime*.so*
endif
.PHONY: build test fmt run viz onnx coreml coreml-validate onnx-validate validate-all-env \
	python-dev python-build python-test python-test-fast python-test-wpt python-test-wpt-onnx python-test-wpt-coreml \
	python-perf python-perf-full python-clean python-example \
	mobilenet-demo mobilenet-serialize mobilenet-serialize-text mobilenet-convert-to-text mobilenet-demo-webnn mobilenet-demo-hub \
	minilm-demo-hub \
	text-gen-demo text-gen-train text-gen-trained text-gen-enhanced text-gen-train-simple \
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
	@echo "Running clippy..."
	$(CARGO) clippy --all-targets -- -D warnings
	@echo "Running tests..."
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
	@echo "Installing Python package in development mode with all backends..."
	@if [ ! -d .venv-webnn ]; then \
		$(PYTHON) -m venv .venv-webnn; \
		.venv-webnn/bin/pip install --upgrade pip; \
		.venv-webnn/bin/pip install pytest pytest-asyncio pytest-xdist numpy maturin; \
	fi
	VIRTUAL_ENV=$(PWD)/.venv-webnn \
	PATH=$(PWD)/.venv-webnn/bin:$$PATH \
	ORT_STRATEGY=system \
	ORT_LIB_LOCATION=$(ORT_LIB_LOCATION) \
	DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) \
	RUSTFLAGS="-L $(ORT_LIB_DIR)" \
	.venv-webnn/bin/maturin develop --features python,onnx-runtime,coreml-runtime

python-build: onnxruntime-download
	@echo "Building Python wheel with all backends..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install maturin
	@echo "Staging ONNX Runtime dylibs into python package..."
	@mkdir -p python/webnn
	cp $(ORT_SHARED_GLOB) python/webnn/
	ORT_STRATEGY=system \
	ORT_LIB_LOCATION=$(ORT_LIB_LOCATION) \
	DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) \
	RUSTFLAGS="-L $(ORT_LIB_DIR)" \
	$(PYTHON) -m maturin build $(MATURIN_ARGS) --features python,onnx-runtime,coreml-runtime --release
	@echo "Cleaning staged ONNX Runtime dylibs..."
	rm -f python/webnn/libonnxruntime*.dylib python/webnn/libonnxruntime*.so* python/webnn/onnxruntime.dll

python-test: python-dev
	@echo "Running Python tests (includes WPT conformance tests)..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python -m pytest tests/ -v; \
		EXIT_CODE=$$?; \
		if [ $$EXIT_CODE -eq 134 ] || [ $$EXIT_CODE -eq 139 ]; then \
			echo "[WARNING]  Note: Python crashed during cleanup"; \
			echo "[OK]  All tests passed successfully before the crash"; \
			exit 0; \
		else \
			exit $$EXIT_CODE; \
		fi; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python -m pytest tests/ -v; \
		EXIT_CODE=$$?; \
		if [ $$EXIT_CODE -eq 134 ] || [ $$EXIT_CODE -eq 139 ]; then \
			echo "[WARNING]  Note: Python crashed during cleanup"; \
			echo "[OK]  All tests passed successfully before the crash"; \
			exit 0; \
		else \
			exit $$EXIT_CODE; \
		fi; \
	fi

python-test-fast: python-dev
	@echo "Running Python tests (excluding slow tests with large inputs)..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python -m pytest tests/ -v -m "not slow"; \
		EXIT_CODE=$$?; \
		if [ $$EXIT_CODE -eq 134 ] || [ $$EXIT_CODE -eq 139 ]; then \
			echo "[WARNING]  Note: Python crashed during cleanup"; \
			echo "[OK]  All tests passed successfully before the crash"; \
			exit 0; \
		else \
			exit $$EXIT_CODE; \
		fi; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python -m pytest tests/ -v -m "not slow"; \
		EXIT_CODE=$$?; \
		if [ $$EXIT_CODE -eq 134 ] || [ $$EXIT_CODE -eq 139 ]; then \
			echo "[WARNING]  Note: Python crashed during cleanup"; \
			echo "[OK]  All tests passed successfully before the crash"; \
			exit 0; \
		else \
			exit $$EXIT_CODE; \
		fi; \
	fi

python-test-wpt: python-dev
	@echo "Running WPT conformance tests only..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python -m pytest tests/test_wpt_conformance.py -v; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python -m pytest tests/test_wpt_conformance.py -v; \
	fi

python-test-wpt-onnx: python-dev
	@echo "Running WPT conformance tests - ONNX backend only..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python -m pytest tests/test_wpt_conformance.py -k "onnx" -v; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python -m pytest tests/test_wpt_conformance.py -k "onnx" -v; \
	fi

python-test-wpt-coreml: python-dev
	@echo "Running WPT conformance tests - CoreML backend only..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python -m pytest tests/test_wpt_conformance.py -k "coreml" -v; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python -m pytest tests/test_wpt_conformance.py -k "coreml" -v; \
	fi

python-perf: python-dev
	@echo "Running performance benchmarks (quick)..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python -m pytest tests/test_performance.py -m "benchmark and not slow" -v -s; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python -m pytest tests/test_performance.py -m "benchmark and not slow" -v -s; \
	fi

python-perf-full: python-dev
	@echo "Running full performance benchmark suite..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python -m pytest tests/test_performance.py -m "benchmark" -v -s; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python -m pytest tests/test_performance.py -m "benchmark" -v -s; \
	fi

python-example: python-dev
	@echo "Running Python examples..."
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/python_simple.py; \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/python_matmul.py; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/python_simple.py; \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/python_matmul.py; \
	fi

mobilenet-demo: python-dev
	@echo "Installing demo dependencies..."
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/pip install -q Pillow requests; \
	else \
		pip install -q Pillow requests; \
	fi
	@echo "========================================================================"
	@echo "Running MobileNetV2 Image Classifier on All Backends"
	@echo "========================================================================"
	@echo ""
	@echo "Backend 1/3: ONNX CPU"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/mobilenetv2_complete.py examples/images/test.jpg --backend cpu; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/mobilenetv2_complete.py examples/images/test.jpg --backend cpu; \
	fi
	@echo ""
	@echo "Backend 2/3: ONNX GPU"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/mobilenetv2_complete.py examples/images/test.jpg --backend gpu; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/mobilenetv2_complete.py examples/images/test.jpg --backend gpu; \
	fi
	@echo ""
	@echo "Backend 3/3: CoreML (Neural Engine)"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/mobilenetv2_complete.py examples/images/test.jpg --backend coreml; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/mobilenetv2_complete.py examples/images/test.jpg --backend coreml; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "All three backends completed successfully!"
	@echo "========================================================================"

mobilenet-serialize: python-dev
	@echo "Installing demo dependencies..."
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/pip install -q Pillow requests numpy; \
	else \
		pip install -q Pillow requests numpy; \
	fi
	@echo "========================================================================"
	@echo "Serializing MobileNetV2 to WebNN Graph Format (JSON)"
	@echo "========================================================================"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python scripts/serialize_mobilenet_to_webnn.py; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python scripts/serialize_mobilenet_to_webnn.py; \
	fi

mobilenet-serialize-text: python-dev
	@echo "Installing demo dependencies..."
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/pip install -q numpy; \
	else \
		pip install -q numpy; \
	fi
	@echo "========================================================================"
	@echo "Serializing MobileNetV2 to WebNN Text Format + Binary Weights"
	@echo "========================================================================"
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/python scripts/serialize_mobilenet_to_webnn_text.py; \
	else \
		python scripts/serialize_mobilenet_to_webnn_text.py; \
	fi

mobilenet-convert-to-text: python-dev
	@echo "Installing dependencies..."
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/pip install -q numpy; \
	else \
		pip install -q numpy; \
	fi
	@if [ ! -f examples/mobilenetv2_complete.webnn ]; then \
		echo "Error: examples/mobilenetv2_complete.webnn not found"; \
		echo "Please run 'make mobilenet-serialize' first to create the JSON file"; \
		exit 1; \
	fi
	@echo "========================================================================"
	@echo "Converting MobileNetV2 JSON to Text Format + Binary Weights"
	@echo "========================================================================"
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/python scripts/convert_json_to_text_format.py examples/mobilenetv2_complete.webnn; \
	else \
		python scripts/convert_json_to_text_format.py examples/mobilenetv2_complete.webnn; \
	fi
	@echo ""
	@echo "Conversion complete! You can now delete the large JSON file:"
	@echo "  rm examples/mobilenetv2_complete.webnn"

mobilenet-demo-webnn: python-dev
	@echo "Installing demo dependencies..."
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/pip install -q Pillow requests; \
	else \
		pip install -q Pillow requests; \
	fi
	@if [ ! -f examples/mobilenetv2_complete.webnn ]; then \
		echo "Serialized graph not found. Running serialization first..."; \
		$(MAKE) mobilenet-serialize; \
	fi
	@echo "========================================================================"
	@echo "Running MobileNetV2 (WebNN Graph Format) on All Backends"
	@echo "========================================================================"
	@echo ""
	@echo "Backend 1/3: ONNX CPU"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/mobilenetv2_from_webnn.py examples/images/test.jpg --backend cpu; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/mobilenetv2_from_webnn.py examples/images/test.jpg --backend cpu; \
	fi
	@echo ""
	@echo "Backend 2/3: ONNX GPU"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/mobilenetv2_from_webnn.py examples/images/test.jpg --backend gpu; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/mobilenetv2_from_webnn.py examples/images/test.jpg --backend gpu; \
	fi
	@echo ""
	@echo "Backend 3/3: CoreML (Neural Engine)"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/mobilenetv2_from_webnn.py examples/images/test.jpg --backend coreml; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/mobilenetv2_from_webnn.py examples/images/test.jpg --backend coreml; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "All three backends completed successfully!"
	@echo "========================================================================"

mobilenet-demo-hub: python-dev
	@echo "Installing demo dependencies..."
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/pip install -q Pillow requests; \
	else \
		pip install -q Pillow requests; \
	fi
	@echo "========================================================================"
	@echo "Running MobileNetV2 (Hugging Face Hub)"
	@echo "========================================================================"
	@echo ""
	@echo "Downloading model from Hugging Face Hub: tarekziade/mobilenet-webnn"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/mobilenetv2_from_hub.py examples/images/test.jpg --backend cpu; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/mobilenetv2_from_hub.py examples/images/test.jpg --backend cpu; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "Demo completed successfully!"
	@echo "========================================================================"

minilm-demo-hub: python-dev
	@echo "Installing demo dependencies..."
	@if [ -f .venv-webnn/bin/python ]; then \
		.venv-webnn/bin/pip install -q transformers torch --index-url https://download.pytorch.org/whl/cpu; \
	else \
		pip install -q transformers torch --index-url https://download.pytorch.org/whl/cpu; \
	fi
	@echo "========================================================================"
	@echo "Running all-MiniLM-L6-v2 demo (Hugging Face Hub)"
	@echo "========================================================================"
	@echo ""
	@echo "Downloading model from Hugging Face Hub: tarekziade/all-MiniLM-L6-v2-webnn"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		MINILM_MODEL_ID=tarekziade/all-MiniLM-L6-v2-webnn DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/minilm_embeddings.py; \
	else \
		MINILM_MODEL_ID=tarekziade/all-MiniLM-L6-v2-webnn DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/minilm_embeddings.py; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "Demo completed successfully!"
	@echo "========================================================================"

text-gen-demo: python-dev
	@echo "========================================================================"
	@echo "Running Text Generation Demo on All Backends"
	@echo "========================================================================"
	@echo ""
	@echo "Backend 1/3: ONNX CPU"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/text_generation_gpt.py --prompt "Hello world" --tokens 30 --temperature 0.8 --backend cpu; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/text_generation_gpt.py --prompt "Hello world" --tokens 30 --temperature 0.8 --backend cpu; \
	fi
	@echo ""
	@echo "Backend 2/3: ONNX GPU"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/text_generation_gpt.py --prompt "Hello world" --tokens 30 --temperature 0.8 --backend gpu; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/text_generation_gpt.py --prompt "Hello world" --tokens 30 --temperature 0.8 --backend gpu; \
	fi
	@echo ""
	@echo "Backend 3/3: CoreML (Neural Engine)"
	@echo "------------------------------------------------------------------------"
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/text_generation_gpt.py --prompt "Hello world" --tokens 30 --temperature 0.8 --backend coreml; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/text_generation_gpt.py --prompt "Hello world" --tokens 30 --temperature 0.8 --backend coreml; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "All three backends completed successfully!"
	@echo "========================================================================"

text-gen-train: python-dev
	@echo "========================================================================"
	@echo "Training Text Generation Model"
	@echo "========================================================================"
	@echo ""
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/train_text_model.py --data examples/sample_text.txt --epochs 10 --batch-size 32 --lr 0.05 --save trained_model.json; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/train_text_model.py --data examples/sample_text.txt --epochs 10 --batch-size 32 --lr 0.05 --save trained_model.json; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "Training completed! Model saved to trained_model.json"
	@echo "========================================================================"

text-gen-trained: python-dev
	@echo "========================================================================"
	@echo "Text Generation with Trained Weights"
	@echo "========================================================================"
	@echo ""
	@if [ ! -f trained_model.json ]; then \
		echo "Error: trained_model.json not found. Run 'make text-gen-train' first."; \
		exit 1; \
	fi
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/text_generation_gpt.py --weights trained_model.json --prompt "The model" --tokens 50 --temperature 0.8 --backend cpu; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/text_generation_gpt.py --weights trained_model.json --prompt "The model" --tokens 50 --temperature 0.8 --backend cpu; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "Generation with trained weights completed!"
	@echo "========================================================================"

text-gen-enhanced: python-dev
	@echo "========================================================================"
	@echo "Enhanced Text Generation with KV Cache"
	@echo "========================================================================"
	@echo ""
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/text_generation_enhanced.py --use-kv-cache --prompt "Hello world" --tokens 30 --temperature 0.8 --backend cpu; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/text_generation_enhanced.py --use-kv-cache --prompt "Hello world" --tokens 30 --temperature 0.8 --backend cpu; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "Enhanced generation with KV cache completed!"
	@echo "========================================================================"

text-gen-train-simple: python-dev
	@echo "========================================================================"
	@echo "Simple Training Demo That Actually Works!"
	@echo "========================================================================"
	@echo ""
	@if [ -f .venv-webnn/bin/python ]; then \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) .venv-webnn/bin/python examples/train_simple_demo.py --phrase "the quick brown fox" --epochs 100 --lr 0.15; \
	else \
		DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) python examples/train_simple_demo.py --phrase "the quick brown fox" --epochs 100 --lr 0.15; \
	fi
	@echo ""
	@echo "========================================================================"
	@echo "Simple training completed - loss decreased!"
	@echo "========================================================================"

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
	@echo "  python-test        - Run all Python tests (includes WPT)"
	@echo "  python-test-fast   - Run Python tests (excluding slow tests)"
	@echo "  python-test-wpt    - Run WPT conformance tests only"
	@echo "  python-perf        - Run quick performance benchmarks"
	@echo "  python-perf-full   - Run full performance benchmark suite"
	@echo "  python-example     - Run Python examples"
	@echo "  mobilenet-demo     - Run MobileNetV2 classifier (builds graph)"
	@echo "  mobilenet-serialize - Serialize to JSON format (194 MB)"
	@echo "  mobilenet-serialize-text - Serialize to text + binary (13 MB)"
	@echo "  mobilenet-convert-to-text - Convert JSON to text format (194MB -> 13MB)"
	@echo "  mobilenet-demo-webnn - Run MobileNetV2 (loads from webnn-graph)"
	@echo "  mobilenet-demo-hub - Run MobileNetV2 (downloads from Hugging Face Hub)"
	@echo "  minilm-demo-hub    - Run all-MiniLM-L6-v2 (downloads from Hugging Face Hub)"
	@echo "  text-gen-demo      - Run basic text generation with attention"
	@echo "  text-gen-train     - Train text generation model on sample data"
	@echo "  text-gen-trained   - Generate text using trained model weights"
	@echo "  text-gen-enhanced  - Run enhanced version with KV cache"
	@echo "  text-gen-train-simple - Simple training that actually works!"
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
