# Makefile for WebNN Python API Examples
#
# This Makefile provides convenient targets for running example scripts.
# Run 'make help' to see all available targets.

.PHONY: help setup simple matmul classify all clean

# Default target
help:
	@echo "WebNN Python API Examples"
	@echo "========================="
	@echo ""
	@echo "Available targets:"
	@echo "  make setup              - Install Python dependencies (numpy, pillow)"
	@echo "  make simple             - Run simple example (graph building basics)"
	@echo "  make matmul             - Run matrix multiplication example"
	@echo "  make classify IMAGE=... - Run image classification example"
	@echo "  make all IMAGE=...      - Run all examples"
	@echo "  make clean              - Remove generated ONNX files"
	@echo "  make help               - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make simple"
	@echo "  make classify IMAGE=~/Downloads/cat.jpg"
	@echo "  make all IMAGE=~/Downloads/cat.jpg"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Build the Python package first (from project root): make python-test"
	@echo "  - Or run: maturin develop --features python,onnx-runtime"

# Install Python dependencies
setup:
	@echo "Installing Python dependencies..."
	pip install numpy pillow
	@echo "✓ Dependencies installed"

# Run simple example
simple:
	@echo "=========================================="
	@echo "Running Simple Example"
	@echo "=========================================="
	python python_simple.py
	@echo ""

# Run matrix multiplication example
matmul:
	@echo "=========================================="
	@echo "Running Matrix Multiplication Example"
	@echo "=========================================="
	python python_matmul.py
	@echo ""

# Run image classification example
# Requires IMAGE parameter
classify:
ifndef IMAGE
	$(error IMAGE parameter is required. Usage: make classify IMAGE=path/to/image.jpg)
endif
	@echo "=========================================="
	@echo "Running Image Classification Example"
	@echo "=========================================="
	python image_classification.py $(IMAGE)
	@echo ""

# Run all examples
all: simple matmul classify
	@echo "=========================================="
	@echo "All examples completed successfully!"
	@echo "=========================================="

# Clean generated files
clean:
	@echo "Cleaning generated ONNX files..."
	rm -f *.onnx
	@echo "✓ Clean complete"
