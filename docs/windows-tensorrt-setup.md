# Windows Setup Guide: rustnn with TensorRT

This guide provides step-by-step instructions for setting up rustnn with TensorRT support on Windows for high-performance GPU inference.

## Overview

When properly configured, rustnn will automatically use TensorRT as the highest-priority backend for accelerated execution on NVIDIA GPUs, providing significantly better performance than CPU or standard ONNX Runtime execution.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with compute capability 7.0 or higher
  - Recommended: T4, RTX 20/30/40 series, A10, A100
  - Minimum: GTX 1080, Quadro P4000
- 8GB+ system RAM
- 20GB+ free disk space for dependencies

### Software Requirements
- Windows 10 (64-bit) or Windows 11
- Administrator access for installation

## Installation Steps

### Step 1: Install NVIDIA GPU Driver

1. Check your current driver version:
   ```powershell
   nvidia-smi
   ```
   If this command works, you already have drivers installed.

2. Download the latest driver:
   - Visit [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Select your GPU model
   - Download and run the installer

3. Reboot your system after installation

4. Verify installation:
   ```powershell
   nvidia-smi
   ```
   You should see your GPU information displayed.

### Step 2: Install CUDA Toolkit

TensorRT requires the CUDA runtime libraries.

1. Download CUDA Toolkit:
   - Visit [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select Windows → x86_64 → your Windows version
   - Download the installer (network or local installer)
   - Recommended version: CUDA 12.x (check TensorRT-RTX compatibility)

2. Run the installer:
   - Choose "Custom Installation"
   - At minimum, select:
     - CUDA Toolkit
     - CUDA Runtime Libraries
     - CUDA Development Libraries (if you plan to build from source)
   - Install to default location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

3. Verify installation:
   ```powershell
   nvcc --version
   ```
   You should see CUDA compiler version information.

4. Verify environment variable (automatically set by installer):
   ```powershell
   echo $env:CUDA_PATH
   ```
   Should output: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

### Step 3: Install TensorRT-RTX

1. Download TensorRT-RTX:
   - Visit [NVIDIA Developer TensorRT Downloads](https://developer.nvidia.com/tensorrt)
   - You may need to create a free NVIDIA Developer account
   - Download TensorRT-RTX for Windows (zip archive)
   - Choose the version compatible with your CUDA installation

2. Extract TensorRT-RTX:
   - Extract the zip file to a permanent location
   - Recommended: `C:\TensorRT-RTX`
   - The directory structure should look like:
     ```
     C:\TensorRT-RTX\
     ├── bin\
     ├── include\
     ├── lib\
     └── doc\
     ```

3. Set environment variable:
   ```powershell
   # Run PowerShell as Administrator
   [System.Environment]::SetEnvironmentVariable('TENSORRT_RTX_DIR', 'C:\TensorRT-RTX', 'Machine')
   ```

4. Add TensorRT to PATH:
   ```powershell
   # Run PowerShell as Administrator
   $oldPath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
   $newPath = "$oldPath;C:\TensorRT-RTX\lib"
   [System.Environment]::SetEnvironmentVariable('Path', $newPath, 'Machine')
   ```

5. Restart your terminal or reboot for changes to take effect

6. Verify installation:
   ```powershell
   dir $env:TENSORRT_RTX_DIR\include
   dir $env:TENSORRT_RTX_DIR\lib
   ```
   You should see TensorRT header files and library files.

### Step 4: Install Rust Toolchain

1. Download Rust:
   - Visit [rustup.rs](https://rustup.rs/)
   - Download and run `rustup-init.exe`

2. Install with default settings:
   - Choose option 1 (default installation)
   - This installs:
     - Rust compiler (rustc)
     - Cargo package manager
     - Standard library

3. Verify installation:
   ```powershell
   rustc --version
   cargo --version
   ```

4. Install Visual Studio Build Tools (required for linking):
   - Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
   - Install "Desktop development with C++"
   - Or use full Visual Studio 2019/2022 with C++ workload

### Step 5: Install Python (for Python bindings)

If you plan to use rustnn from Python:

1. Download Python 3.8 or later:
   - Visit [python.org](https://www.python.org/downloads/)
   - Download Windows installer (64-bit)

2. Install Python:
   - Check "Add Python to PATH" during installation
   - Choose "Install for all users" (recommended)

3. Verify installation:
   ```powershell
   python --version
   pip --version
   ```

### Step 6: Build rustnn with TensorRT Support

1. Clone the rustnn repository:
   ```powershell
   git clone https://github.com/tarekziade/rustnn.git
   cd rustnn
   ```

2. Build Rust library with TensorRT:
   ```powershell
   # Build with TensorRT support
   cargo build --release --features trtx-runtime
   ```

   This will:
   - Download and compile dependencies
   - Link against TensorRT-RTX libraries
   - Create optimized release build
   - Take 5-15 minutes on first build

3. Run tests to verify:
   ```powershell
   cargo test --lib --features trtx-runtime
   ```

4. Build Python package (if using Python bindings):
   ```powershell
   # Install maturin
   pip install maturin

   # Build Python wheel with TensorRT support
   maturin build --release --features "python,trtx-runtime"

   # Install the wheel
   pip install target/wheels/rustnn-*.whl
   ```

### Step 7: Verify TensorRT Integration

1. Create a test Python script (`test_trt.py`):
   ```python
   import webnn
   import numpy as np

   # Create context - should select TensorRT backend
   ml = webnn.ML()
   context = ml.create_context(
       power_preference="high-performance",
       accelerated=True
   )

   print(f"Backend selected: {context.accelerated}")
   print("TensorRT backend is active!" if context.accelerated else "Fallback backend")

   # Create a simple graph
   builder = context.create_graph_builder()
   x = builder.input("x", [2, 3], "float32")
   y = builder.relu(x)
   graph = builder.build({"output": y})

   # Execute
   inputs = {"x": np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)}
   outputs = context.compute(graph, inputs)

   print("Output:", outputs["output"])
   print("Success! TensorRT is working.")
   ```

2. Run the test:
   ```powershell
   python test_trt.py
   ```

3. Expected output:
   ```
   Backend selected: True
   TensorRT backend is active!
   Output: [[0. 2. 0.]
            [4. 0. 6.]]
   Success! TensorRT is working.
   ```

## Troubleshooting

### Build Errors

**Error: "Cannot find TensorRT headers"**
```
Solution:
1. Verify TENSORRT_RTX_DIR is set: echo $env:TENSORRT_RTX_DIR
2. Check the directory exists and contains include/ folder
3. Restart terminal after setting environment variables
```

**Error: "Linking error: cannot find -lnvinfer_10"**
```
Solution:
1. Verify TensorRT lib directory is in PATH
2. Check lib files exist: dir $env:TENSORRT_RTX_DIR\lib
3. Ensure you downloaded the correct Windows version of TensorRT-RTX
4. Try adding to PATH manually:
   $env:PATH += ";C:\TensorRT-RTX\lib"
```

**Error: "CUDA not found"**
```
Solution:
1. Verify CUDA_PATH is set: echo $env:CUDA_PATH
2. Run: nvcc --version (should work)
3. Reinstall CUDA Toolkit if necessary
```

### Runtime Errors

**Error: "TensorRT execution failed: CUDA initialization failed"**
```
Solution:
1. Check GPU is accessible: nvidia-smi
2. Update GPU drivers to latest version
3. Ensure no other process is using the GPU exclusively
4. Restart your computer
```

**Error: "DLL not found" when running Python**
```
Solution:
1. Ensure TensorRT lib directory is in PATH
2. Copy required DLLs to Python script directory:
   - nvinfer_10.dll
   - nvonnxparser_10.dll
   - cudart64_12.dll (or your CUDA version)
3. Or add to PATH for current session:
   $env:PATH += ";C:\TensorRT-RTX\lib;$env:CUDA_PATH\bin"
```

**Backend falls back to ONNX instead of TensorRT**
```
Solution:
1. Verify you built with trtx-runtime feature:
   cargo build --features trtx-runtime
2. Check Python package includes TensorRT:
   pip show rustnn (should list trtx in dependencies)
3. Rebuild Python package with correct features:
   maturin develop --features "python,trtx-runtime"
```

### Performance Issues

**TensorRT is slower than expected**
```
Tips:
1. TensorRT optimizes on first run (engine building)
   - First inference may take 10-60 seconds
   - Subsequent runs should be much faster
2. Use larger batch sizes when possible
3. Ensure GPU has adequate cooling (check temps with nvidia-smi)
4. Close other GPU-intensive applications
```

## Development Without TensorRT (Mock Mode)

If you want to develop on a machine without an NVIDIA GPU, you can use mock mode:

```powershell
# Build with mock feature
cargo build --features trtx-runtime-mock

# Run tests with mock
cargo test --lib --features trtx-runtime-mock

# Build Python package with mock
maturin develop --features "python,trtx-runtime-mock"
```

Mock mode:
- Compiles and runs without GPU
- Useful for development and CI/CD
- Does NOT perform actual inference
- Returns dummy results

## Environment Variable Summary

For quick reference, here are all the environment variables you need:

```powershell
# Run as Administrator
[System.Environment]::SetEnvironmentVariable('CUDA_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x', 'Machine')
[System.Environment]::SetEnvironmentVariable('TENSORRT_RTX_DIR', 'C:\TensorRT-RTX', 'Machine')

# Add to PATH
$oldPath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
$newPath = "$oldPath;C:\TensorRT-RTX\lib;$env:CUDA_PATH\bin"
[System.Environment]::SetEnvironmentVariable('Path', $newPath, 'Machine')
```

After setting these, restart your terminal or reboot.

## Performance Expectations

With TensorRT properly configured, you should see:

| Operation | CPU (ONNX) | GPU (ONNX) | GPU (TensorRT) |
|-----------|------------|------------|----------------|
| Small models (<10 ops) | ~10ms | ~5ms | ~2ms |
| Medium models (10-100 ops) | ~100ms | ~20ms | ~5ms |
| Large models (>100 ops) | ~1000ms | ~100ms | ~20ms |

Note: First-run times include engine building overhead (10-60 seconds).

## Next Steps

Once TensorRT is working:

1. Explore examples in `examples/` directory
2. Read the [API Reference](api-reference.md) for detailed usage
3. Check [Implementation Status](implementation-status.md) for supported operations
4. See [Development Guide](development.md) for contributing

## Additional Resources

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [rustnn Python API Reference](api-reference.md)
- [trtx-rs GitHub](https://github.com/tarekziade/trtx-rs)

## Support

If you encounter issues not covered in this guide:

1. Check existing [GitHub Issues](https://github.com/tarekziade/rustnn/issues)
2. Create a new issue with:
   - Your Windows version
   - GPU model (from nvidia-smi)
   - CUDA version (from nvcc --version)
   - TensorRT version
   - Full error message and stack trace
   - Steps to reproduce
