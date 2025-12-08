# WebNN Python API Examples

This directory contains example scripts demonstrating the WebNN Python API functionality.

## Prerequisites

1. **Build the Python package** (from project root):
   ```bash
   make python-test
   ```

   Or manually with ONNX Runtime support:
   ```bash
   ORT_STRATEGY=system \
     ORT_LIB_LOCATION=target/onnxruntime/onnxruntime-osx-arm64-1.17.0/lib \
     DYLD_LIBRARY_PATH=target/onnxruntime/onnxruntime-osx-arm64-1.17.0/lib \
     maturin develop --features python,onnx-runtime
   ```

2. **Install Python dependencies**:
   ```bash
   pip install numpy pillow
   ```

   Or using the Makefile in this directory:
   ```bash
   cd examples
   make setup
   ```

## Examples

### 1. Simple Example (`python_simple.py`)

**Description**: Basic introduction to the WebNN Python API

**What it demonstrates**:
- Creating a context and graph builder
- Defining input operands
- Building a simple graph: `z = relu(x + y)`
- Converting to ONNX format

**How to run**:
```bash
python examples/python_simple.py
# Or from examples directory:
make simple
```

**Expected output**:
- Context information
- Graph structure details
- ONNX conversion confirmation

---

### 2. Matrix Multiplication Example (`python_matmul.py`)

**Description**: Demonstrates matrix operations and constant tensors

**What it demonstrates**:
- Creating constant operands from NumPy arrays
- Matrix multiplication with `matmul()`
- Combining multiple operations
- Graph: `output = relu(matmul(input, weights) + bias)`

**How to run**:
```bash
python examples/python_matmul.py
# Or from examples directory:
make matmul
```

**Expected output**:
- Matrix dimensions and shapes
- Forward pass computation steps
- ONNX conversion details

---

### 3. Image Classification Example (`image_classification.py`)

**Description**: Complete image classification pipeline using modern WebNN operations

**What it demonstrates**:
- Image preprocessing (resize, normalize)
- Convolutional neural network operations:
  - `conv2d`: Standard and depthwise convolutions
  - `clamp`: ReLU6 activation (min=0, max=6)
  - `global_average_pool`: Spatial dimension reduction
  - `gemm`: Fully connected layer with transpose
  - `softmax`: Class probability distribution
- Real-world inference workflow
- Performance metrics

**How to run**:
```bash
python examples/image_classification.py path/to/image.jpg

# Example with a specific image:
python examples/image_classification.py ~/Downloads/cat.jpg

# Or from examples directory:
make classify IMAGE=path/to/image.jpg
```

**Requirements**:
- PIL/Pillow for image loading
- NumPy for numerical operations
- JPEG or PNG image file

**Expected output**:
```
============================================================
WebNN Image Classification Demo
============================================================
Image: /path/to/image.jpg

1. Loading and preprocessing image...
   ✓ Preprocessed to shape (1, 3, 224, 224) (15.32ms)

2. Creating WebNN context...
   ✓ Context created (device: cpu)

3. Building neural network graph...
   ✓ Graph built (8.45ms)

4. Running inference...
   ✓ Inference complete (125.67ms)

5. Top 5 Predictions:
------------------------------------------------------------
   1. Class 42                      12.34%
   2. Class 156                     10.89%
   3. Class 789                      9.45%
   4. Class 23                       8.12%
   5. Class 567                      7.89%

============================================================
Performance Summary:
  - Preprocessing: 15.32ms
  - Graph Build:   8.45ms
  - Inference:     125.67ms
============================================================

Note: This demo uses random weights for demonstration.
For real classification, load pretrained model weights.

Operations demonstrated:
  - conv2d: Standard and depthwise convolutions
  - clamp: ReLU6 activation (min=0, max=6)
  - global_average_pool: Spatial dimension reduction
  - gemm: Fully connected layer with transpose
  - softmax: Class probability distribution
```

**Note**: The example uses random weights, so predictions are not meaningful. For real classification, you would load pretrained model weights (e.g., MobileNet).

---

## Running All Examples

From the examples directory:
```bash
make all IMAGE=path/to/image.jpg
```

This runs all examples in sequence.

## Makefile Targets

From the `examples/` directory:

- `make setup` - Install Python dependencies (numpy, pillow)
- `make simple` - Run the simple example
- `make matmul` - Run the matrix multiplication example
- `make classify IMAGE=<path>` - Run image classification (requires IMAGE parameter)
- `make all IMAGE=<path>` - Run all examples
- `make help` - Show available targets
- `make clean` - Remove generated ONNX files

## Graph Samples

This directory also contains sample graph files for reference:

- `sample_graph.json` - Basic WebNN graph structure
- `toy_transformer.json` - Transformer-style architecture example
- `example_graph.onnx` - ONNX format output example
- `matmul_graph.onnx` - Matrix multiplication in ONNX format

## Additional Resources

- **Python API Documentation**: See [../README.md](../README.md#python-api-reference)
- **WebNN Specification**: [W3C WebNN API](https://www.w3.org/TR/webnn/)
- **Project Documentation**: [../docs/](../docs/)

## Troubleshooting

### Import Error: `No module named 'webnn'`

Build the Python package first:
```bash
cd ..
make python-test
```

### ONNX Runtime Not Available

The examples will work for graph building and ONNX conversion even without ONNX Runtime. However, actual inference (`context.compute()`) requires ONNX Runtime. Build with:
```bash
make python-test  # From project root
```

### Missing Dependencies

Install required packages:
```bash
pip install numpy pillow
```

### Image Classification Example Issues

Make sure you provide a valid image path:
```bash
python image_classification.py /absolute/path/to/image.jpg
```

Supported formats: JPEG, PNG
