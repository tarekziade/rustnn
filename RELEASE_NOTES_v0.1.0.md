## [WARNING] EXPERIMENTAL RELEASE - NOT FOR PRODUCTION

**DO NOT USE IN PRODUCTION.** This is a proof-of-concept for research and experimentation only.

This is the first experimental release of PyWebNN, a Python implementation of the W3C WebNN API with Rust bindings.

---

### [TARGET] Features

- Python **Python WebNN API** - Complete W3C WebNN specification implementation
- [STATS] **ONNX Conversion** - Full operation support, cross-platform
-  **CoreML Conversion** - Basic operations (add, matmul) for macOS
- [OK] **Graph Validation** - Comprehensive validation matching Chromium's WebNN
- [SEARCH] **Visualization** - Generate Graphviz diagrams
- [PACKAGE] **NumPy Integration** - Seamless tensor data conversion
-  **Cross-platform** - Linux, macOS, Windows wheels

###  Installation

```bash
pip install pywebnn
```

### [DEPLOY] Quick Start

```python
import webnn
import numpy as np

# Create context and builder
ml = webnn.ML()
context = ml.create_context(device_type="cpu")
builder = context.create_graph_builder()

# Build a simple neural network layer
input_tensor = builder.input("input", [1, 4], "float32")
weights = np.random.randn(4, 3).astype(np.float32)
weights_const = builder.constant(weights)
output = builder.matmul(input_tensor, weights_const)

# Compile and convert
graph = builder.build({"output": output})
context.convert_to_onnx(graph, "model.onnx")
print("[OK] Model saved to model.onnx")
```

###  Documentation

- [Getting Started Guide](https://github.com/tarekziade/rustnn/blob/main/docs/getting-started.md)
- [API Reference](https://github.com/tarekziade/rustnn/blob/main/docs/api-reference.md)
- [Examples](https://github.com/tarekziade/rustnn/blob/main/docs/examples.md)
- [Full Documentation](https://tarekziade.github.io/rustnn/)

### [WARNING] Known Limitations

- **CoreML**: Only supports add and matmul operations (activations coming soon)
- **Execution**: No actual tensor computation in `compute()` yet - only conversion
- **Operations**: Subset of WebNN spec implemented (see [TODO.txt](https://github.com/tarekziade/rustnn/blob/main/TODO.txt))
- **Testing**: Limited test coverage, use for experimentation only

###  What's Included

This release includes pre-built wheels for:
- **Linux**: x86_64, aarch64 (manylinux)
- **macOS**: x86_64 (Intel), aarch64 (Apple Silicon)
- **Windows**: x64, x86
- **Python**: 3.12, 3.13

###  Links

- **PyPI**: https://pypi.org/project/pywebnn/
- **Repository**: https://github.com/tarekziade/rustnn
- **Issues**: https://github.com/tarekziade/rustnn/issues
- **W3C WebNN Spec**: https://www.w3.org/TR/webnn/

###  Acknowledgments

Built with PyO3, Maturin, and inspired by the W3C WebNN Community Group.

---

**Made with  by Tarek Ziadé**
