CARGO := cargo
DOT ?= dot
GRAPH_FILE ?= examples/sample_graph.json
DOT_PATH ?= target/graph.dot
PNG_PATH ?= target/graph.png
ONNX_PATH ?= target/graph.onnx.json
COREML_PATH ?= target/graph.coreml.json
PYTHON ?= python3
VENV ?= .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

.PHONY: build test fmt run viz onnx coreml coreml-validate coreml-env coreml-validate-env onnx-validate onnx-env onnx-validate-env validate-all-env

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

onnx:
	$(CARGO) run -- $(GRAPH_FILE) --convert onnx --convert-output $(ONNX_PATH)
	@echo "ONNX graph written to $(ONNX_PATH)"

onnx-validate: onnx
	python scripts/validate_onnx.py $(ONNX_PATH)

onnx-env:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install onnx onnxruntime numpy

onnx-validate-env: onnx onnx-env
	$(PYTHON_VENV) scripts/validate_onnx.py $(ONNX_PATH)

coreml:
	$(CARGO) run -- $(GRAPH_FILE) --convert coreml --convert-output $(COREML_PATH)
	@echo "CoreML graph written to $(COREML_PATH)"

coreml-validate: coreml
	python scripts/validate_coreml.py $(COREML_PATH)

coreml-env:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install coremltools

coreml-validate-env: coreml coreml-env
	$(PYTHON_VENV) scripts/validate_coreml.py $(COREML_PATH)

validate-all-env: build test onnx-validate-env coreml-validate-env
	@echo "Full pipeline (build/test/convert/validate) completed."
