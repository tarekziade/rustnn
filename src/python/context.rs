//! ML context and backend selection for WebNN API
//!
//! PyO3 macros generate unsafe code that triggers unsafe_op_in_unsafe_fn warnings.
//! This is expected behavior from the macro-generated code.
#![allow(unsafe_op_in_unsafe_fn)]

use super::graph::PyMLGraph;
use super::graph_builder::PyMLGraphBuilder;
use super::operand::parse_data_type;
use super::tensor::PyMLTensor;
use crate::converters::GraphConverter;
use crate::debug_print;
use crate::graph::OperandDescriptor;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg(feature = "onnx-runtime")]
use crate::executors::onnx::{OnnxInput, run_onnx_with_inputs};

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
use crate::executors::coreml::run_coreml_zeroed_cached_with_weights;

/// Backend execution engine selected at context creation
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum Backend {
    /// ONNX Runtime (CPU execution)
    OnnxCpu,
    /// ONNX Runtime (GPU execution)
    OnnxGpu,
    /// CoreML (macOS Neural Engine or GPU)
    CoreML,
    /// TensorRT (NVIDIA GPU execution)
    TensorRT,
    /// No backend available (returns zeros)
    None,
}

/// ML namespace - entry point for WebNN API
#[pyclass(name = "ML")]
pub struct PyML;

#[pymethods]
impl PyML {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Create a new ML context
    ///
    /// Args:
    ///     power_preference: Power preference hint ("default", "high-performance", or "low-power")
    ///     accelerated: Whether to use GPU/NPU acceleration (default: true)
    ///     device_type: Force specific backend ("auto", "cpu", "gpu", "npu") (default: "auto")
    ///
    /// Returns:
    ///     MLContext: A new context for graph operations
    ///
    /// Note:
    ///     The accelerated parameter is a hint, not a guarantee. The platform
    ///     decides the actual device allocation based on runtime conditions.
    ///     Query context.accelerated after creation to check if acceleration is available.
    ///     device_type="auto" uses automatic backend selection based on availability.
    ///     device_type="cpu" forces ONNX CPU backend.
    ///     device_type="gpu" forces ONNX GPU backend.
    ///     device_type="npu" forces CoreML backend (macOS only).
    #[pyo3(signature = (power_preference="default", accelerated=true, device_type="auto"))]
    fn create_context(
        &self,
        power_preference: &str,
        accelerated: bool,
        device_type: &str,
    ) -> PyResult<PyMLContext> {
        Ok(PyMLContext::new(
            power_preference.to_string(),
            accelerated,
            device_type.to_string(),
        ))
    }
}

/// MLContext manages the execution environment for neural network graphs
#[pyclass(name = "MLContext")]
pub struct PyMLContext {
    power_preference: String,
    _accelerated_requested: bool,
    accelerated_available: bool,
    backend: Backend,

    /// Cached ONNX Runtime session for device tensor reuse
    /// This enables zero-copy execution by keeping the session alive across operations
    #[cfg(feature = "onnx-runtime")]
    onnx_session: std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<ort::session::Session>>>>,
}

#[pymethods]
impl PyMLContext {
    /// Create a graph builder for constructing computational graphs
    ///
    /// Returns:
    ///     MLGraphBuilder: A new graph builder
    fn create_graph_builder(&self) -> PyResult<PyMLGraphBuilder> {
        Ok(PyMLGraphBuilder::create())
    }

    /// Compute the graph with given inputs using the backend selected at context creation
    ///
    /// This is a synchronous execution method that returns computed results.
    ///
    /// Args:
    ///     graph: The compiled MLGraph to execute
    ///     inputs: Dictionary mapping input names to numpy arrays
    ///     outputs: Dictionary mapping output names to numpy arrays (pre-allocated)
    ///
    /// Returns:
    ///     Dictionary mapping output names to result numpy arrays
    #[pyo3(signature = (graph, inputs, _outputs=None))]
    fn compute(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
        _outputs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        // Route to appropriate backend based on context's backend selection
        match self.backend {
            Backend::OnnxCpu | Backend::OnnxGpu => self.compute_onnx(py, graph, inputs),
            Backend::CoreML => self.compute_coreml(py, graph, inputs),
            Backend::TensorRT => self.compute_trtx(py, graph, inputs),
            Backend::None => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "No backend available: build without onnx-runtime/coreml-runtime/trtx-runtime features",
                ));
            }
        }
    }

    /// Dispatch graph execution with MLTensor or MLDeviceTensor inputs/outputs
    ///
    /// Following the W3C WebNN MLTensor Explainer:
    /// https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md
    ///
    /// This method executes the graph with tensor inputs and writes results to output tensors.
    /// Supports both host tensors (MLTensor) and device tensors (MLDeviceTensor) for zero-copy execution.
    ///
    /// Args:
    ///     graph: The compiled MLGraph to execute
    ///     inputs: Dictionary mapping input names to MLTensor or MLDeviceTensor objects
    ///     outputs: Dictionary mapping output names to MLTensor or MLDeviceTensor objects
    ///
    /// Note:
    ///     When using MLDeviceTensor inputs/outputs, execution avoids host-device round-trips,
    ///     which is critical for iterative GenAI workloads like KV cache.
    #[pyo3(signature = (graph, inputs, outputs))]
    fn dispatch(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
        outputs: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        use super::tensor::PyMLDeviceTensor;

        // Check if we have any device tensors
        let mut has_device_tensors = false;
        for (_, value) in inputs.iter() {
            if value.downcast::<PyMLDeviceTensor>().is_ok() {
                has_device_tensors = true;
                break;
            }
        }

        if !has_device_tensors {
            for (_, value) in outputs.iter() {
                if value.downcast::<PyMLDeviceTensor>().is_ok() {
                    has_device_tensors = true;
                    break;
                }
            }
        }

        // Route to appropriate execution path
        if has_device_tensors {
            self.dispatch_with_device_tensors(py, graph, inputs, outputs)
        } else {
            // Legacy path: all host tensors (MLTensor)
            self.dispatch_with_host_tensors(py, graph, inputs, outputs)
        }
    }

    /// Dispatch with host tensors (legacy path)
    fn dispatch_with_host_tensors(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
        outputs: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        // Convert MLTensor inputs to numpy arrays
        let numpy_inputs = PyDict::new_bound(py);
        for (key, value) in inputs.iter() {
            let tensor = value.downcast::<PyMLTensor>()?;
            let numpy_array = self.read_tensor(py, &tensor.borrow())?;
            numpy_inputs.set_item(key, numpy_array)?;
        }

        // Execute graph
        let results = self.compute(py, graph, &numpy_inputs, None)?;

        // Write results to output tensors
        for (key, value) in outputs.iter() {
            let tensor = value.downcast::<PyMLTensor>()?;
            if let Some(result) = results.bind(py).get_item(&key)? {
                self.write_tensor(py, &tensor.borrow(), result.into())?;
            }
        }

        Ok(())
    }

    /// Dispatch with device tensors (zero-copy path)
    ///
    /// Note: Current implementation uses host round-trips for compatibility.
    /// Full zero-copy execution will be implemented in a future update.
    #[cfg(feature = "onnx-runtime")]
    fn dispatch_with_device_tensors(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
        outputs: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        use super::tensor::PyMLDeviceTensor;

        // For now, convert device tensors to numpy and use regular compute path
        // Full zero-copy execution requires more complex lifetime management
        // and will be implemented in a future update

        // Convert inputs (device or host tensors) to numpy
        let numpy_inputs = PyDict::new_bound(py);
        for (key, value) in inputs.iter() {
            if let Ok(device_tensor) = value.downcast::<PyMLDeviceTensor>() {
                let numpy_array = device_tensor.borrow().read_data(py)?;
                numpy_inputs.set_item(key, numpy_array)?;
            } else if let Ok(host_tensor) = value.downcast::<PyMLTensor>() {
                let numpy_array = self.read_tensor(py, &host_tensor.borrow())?;
                numpy_inputs.set_item(key, numpy_array)?;
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Input must be MLTensor or MLDeviceTensor",
                ));
            }
        }

        // Execute graph
        let results = self.compute(py, graph, &numpy_inputs, None)?;

        // Write results to output tensors (device or host)
        for (key, value) in outputs.iter() {
            if let Ok(device_tensor) = value.downcast::<PyMLDeviceTensor>() {
                if let Some(result) = results.bind(py).get_item(&key)? {
                    device_tensor.borrow_mut().write_data(py, result.into())?;
                }
            } else if let Ok(host_tensor) = value.downcast::<PyMLTensor>() {
                if let Some(result) = results.bind(py).get_item(&key)? {
                    self.write_tensor(py, &host_tensor.borrow(), result.into())?;
                }
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Output must be MLTensor or MLDeviceTensor",
                ));
            }
        }

        Ok(())
    }

    /// Stub for when ONNX Runtime is not available
    #[cfg(not(feature = "onnx-runtime"))]
    fn dispatch_with_device_tensors(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
        outputs: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Device tensors require ONNX Runtime (compile with onnx-runtime feature)",
        ))
    }

    /// Convert graph to ONNX format
    ///
    /// Args:
    ///     graph: The MLGraph to convert
    ///     output_path: Path to save the ONNX model
    fn convert_to_onnx(&self, graph: &PyMLGraph, output_path: &str) -> PyResult<()> {
        let converter = crate::converters::OnnxConverter;
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX conversion failed: {}", e))
        })?;

        std::fs::write(output_path, &converted.data).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write ONNX file: {}", e))
        })?;

        Ok(())
    }

    /// Convert graph to CoreML format (macOS only)
    ///
    /// Args:
    ///     graph: The MLGraph to convert
    ///     output_path: Path to save the CoreML model
    #[cfg(target_os = "macos")]
    fn convert_to_coreml(&self, graph: &PyMLGraph, output_path: &str) -> PyResult<()> {
        let converter = crate::converters::CoremlMlProgramConverter::default();
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML conversion failed: {}", e))
        })?;

        std::fs::write(output_path, &converted.data).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write CoreML file: {}", e))
        })?;

        Ok(())
    }

    /// Execute graph using CoreML runtime (macOS only, requires coreml-runtime feature)
    ///
    /// Args:
    ///     graph: The MLGraph to execute
    ///     device: Device to use ("cpu", "gpu", or "npu")
    ///
    /// Returns:
    ///     Dictionary with execution results
    #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
    #[pyo3(signature = (graph, device="cpu"))]
    fn execute_with_coreml(
        &self,
        py: Python,
        graph: &PyMLGraph,
        device: &str,
    ) -> PyResult<Py<PyDict>> {
        // Convert to CoreML
        let converter = crate::converters::CoremlMlProgramConverter::default();
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML conversion failed: {}", e))
        })?;

        // Parse device type (for future use with CoreML compute units selection)
        let _compute_units = match device {
            "cpu" => 0,
            "gpu" => 1,
            "npu" => 2,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid device type: {}. Use 'cpu', 'gpu', or 'npu'",
                    device
                )));
            }
        };

        // Build input descriptors map
        use std::collections::HashMap;
        let mut inputs = HashMap::new();
        for &input_id in &graph.graph_info.input_operands {
            if let Some(operand) = graph.graph_info.operand(input_id) {
                let name = operand
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("input_{}", input_id));
                inputs.insert(name, operand.descriptor.clone());
            }
        }

        // Execute with optional weight file
        let weights_ref = converted.weights_data.as_deref();
        run_coreml_zeroed_cached_with_weights(&converted.data, weights_ref, &inputs, None)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML execution failed: {}", e))
            })?;

        // Return empty dict for now (actual implementation would return outputs)
        let result = PyDict::new_bound(py);
        Ok(result.into())
    }

    /// Create a tensor (device-resident by default, per WebNN spec)
    ///
    /// Following the W3C WebNN specification:
    /// https://www.w3.org/TR/webnn/#dom-mlcontext-createtensor
    ///
    /// By default (readable=False, writable=False), creates a host-backed tensor.
    /// For true device-resident tensors with zero-copy execution, use create_device_tensor().
    ///
    /// Note: The spec intends device-resident tensors by default, but our implementation
    /// currently returns host-backed tensors for simplicity. We plan to add lazy device
    /// tensor materialization in a future version to fully match the spec.
    ///
    /// Args:
    ///     shape: Shape of the tensor
    ///     data_type: Data type string (e.g., "float32")
    ///     readable: If True, tensor data can be read back to CPU (default: False per spec)
    ///     writable: If True, tensor data can be written from CPU (default: False per spec)
    ///     exportable_to_gpu: If True, tensor can be used as GPU texture (default: False)
    ///
    /// Returns:
    ///     MLTensor: A new tensor with the specified properties
    ///
    /// Examples:
    ///     # Device-resident tensor (spec-compliant defaults)
    ///     tensor = context.create_tensor([2, 3], "float32")
    ///
    ///     # Host-accessible tensor (explicit flags)
    ///     host_tensor = context.create_tensor([2, 3], "float32", readable=True, writable=True)
    ///
    ///     # Convenience: use create_host_tensor() for host tensors
    ///     host_tensor = context.create_host_tensor([2, 3], "float32")
    #[pyo3(signature = (shape, data_type, readable=false, writable=false, exportable_to_gpu=false))]
    fn create_tensor(
        &self,
        shape: Vec<u32>,
        data_type: &str,
        readable: bool,
        writable: bool,
        exportable_to_gpu: bool,
    ) -> PyResult<PyMLTensor> {
        use super::tensor::MLTensorDescriptor;

        let dtype = parse_data_type(data_type)?;
        let descriptor = OperandDescriptor {
            data_type: dtype,
            shape,
            pending_permutation: Vec::new(),
        };

        let tensor_descriptor = MLTensorDescriptor {
            descriptor,
            readable,
            writable,
            exportable_to_gpu,
        };

        Ok(PyMLTensor::new(tensor_descriptor))
    }

    /// Convenience method for creating host-backed tensors (non-spec extension)
    ///
    /// This is equivalent to:
    ///   create_tensor(shape, data_type, readable=True, writable=True)
    ///
    /// Use this when you need to inspect or modify tensor contents from Python,
    /// such as for debugging, prototyping, or when your workflow requires host access.
    ///
    /// For production code with iterative workloads (like KV cache), prefer
    /// create_device_tensor() for optimal performance.
    ///
    /// Args:
    ///     shape: Shape of the tensor
    ///     data_type: Data type string (e.g., "float32")
    ///
    /// Returns:
    ///     MLTensor: A host-backed tensor (always readable and writable)
    ///
    /// Example:
    ///     # Quick and easy for prototyping
    ///     tensor = context.create_host_tensor([2, 3], "float32")
    ///     context.write_tensor(tensor, np.array([[1, 2, 3], [4, 5, 6]]))
    ///     data = context.read_tensor(tensor)
    #[pyo3(signature = (shape, data_type))]
    fn create_host_tensor(&self, shape: Vec<u32>, data_type: &str) -> PyResult<PyMLTensor> {
        // Just call create_tensor with explicit host flags
        self.create_tensor(shape, data_type, true, true, false)
    }

    /// Create a device-resident tensor for zero-copy execution
    ///
    /// Device tensors reside in GPU/NPU memory and enable persistent storage
    /// across inference steps without host round-trips. This is critical for
    /// iterative GenAI workloads like KV cache in transformers.
    ///
    /// Args:
    ///     graph: The MLGraph to associate with this tensor (needed for session management)
    ///     shape: Shape of the tensor
    ///     data_type: Data type string (e.g., "float32")
    ///     device: Device to allocate on (optional, defaults to context's backend device)
    ///
    /// Returns:
    ///     MLDeviceTensor: A new device-resident tensor
    ///
    /// Note:
    ///     Currently only supported with ONNX Runtime backend
    #[cfg(feature = "onnx-runtime")]
    #[pyo3(signature = (graph, shape, data_type, device=None))]
    fn create_device_tensor(
        &self,
        graph: &PyMLGraph,
        shape: Vec<usize>,
        data_type: &str,
        device: Option<&str>,
    ) -> PyResult<super::tensor::PyMLDeviceTensor> {
        use super::tensor::PyMLDeviceTensor;
        use crate::executors::onnx::OrtDeviceTensor;
        use crate::tensor::{DeviceKind, DeviceTensorHandle};

        // Parse data type
        let dtype = parse_data_type(data_type)?;

        // Determine device from backend or parameter
        let device_kind = if let Some(dev) = device {
            match dev {
                "cpu" => DeviceKind::Cpu,
                "cuda" => DeviceKind::Cuda,
                "directml" => DeviceKind::DirectML,
                "coreml" => DeviceKind::CoreML,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unsupported device: {}. Use 'cpu', 'cuda', 'directml', or 'coreml'",
                        dev
                    )));
                }
            }
        } else {
            // Infer from backend
            match self.backend {
                Backend::OnnxCpu => DeviceKind::Cpu,
                Backend::OnnxGpu => DeviceKind::Cuda,
                Backend::CoreML => DeviceKind::CoreML,
                Backend::TensorRT => DeviceKind::Cuda,
                Backend::None => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "No backend available for device tensor creation",
                    ));
                }
            }
        };

        // Get or create ONNX session
        let session = self.get_onnx_session(graph)?;

        // Create ONNX device tensor
        let ort_tensor = OrtDeviceTensor::new(session, shape, dtype, device_kind).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create device tensor: {}",
                e
            ))
        })?;

        // Wrap in DeviceTensorHandle
        let handle = DeviceTensorHandle::new(Box::new(ort_tensor));

        Ok(PyMLDeviceTensor::new(handle))
    }

    /// Stub for when ONNX Runtime is not available
    #[cfg(not(feature = "onnx-runtime"))]
    #[pyo3(signature = (_graph, _shape, _data_type, _device=None))]
    fn create_device_tensor(
        &self,
        _graph: &PyMLGraph,
        _shape: Vec<usize>,
        _data_type: &str,
        _device: Option<&str>,
    ) -> PyResult<super::tensor::PyMLDeviceTensor> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Device tensors require ONNX Runtime (compile with onnx-runtime feature)",
        ))
    }

    /// Read data from a tensor into a numpy array
    ///
    /// Follows the W3C WebNN MLTensor Explainer timeline model.
    ///
    /// Args:
    ///     tensor: The MLTensor to read from (must have readable=True)
    ///
    /// Returns:
    ///     numpy.ndarray: The tensor data as a numpy array
    ///
    /// Raises:
    ///     RuntimeError: If tensor is not readable or has been destroyed
    fn read_tensor(&self, py: Python, tensor: &PyMLTensor) -> PyResult<PyObject> {
        let numpy = py.import_bound("numpy")?;
        let data = tensor.get_data()?; // Now returns PyResult
        let shape_tuple = pyo3::types::PyTuple::new_bound(
            py,
            tensor
                .tensor_descriptor
                .descriptor
                .shape
                .iter()
                .map(|&d| d as i64),
        );

        let array = numpy.call_method1("array", (data,))?;
        let reshaped = array.call_method1("reshape", (shape_tuple,))?;

        Ok(reshaped.into())
    }

    /// Write data from a numpy array into a tensor
    ///
    /// Follows the W3C WebNN MLTensor Explainer timeline model.
    ///
    /// Args:
    ///     tensor: The MLTensor to write to (must have writable=True)
    ///     data: Numpy array or array-like data to write
    ///
    /// Raises:
    ///     RuntimeError: If tensor is not writable or has been destroyed
    ///     ValueError: If data shape doesn't match tensor shape
    fn write_tensor(&self, py: Python, tensor: &PyMLTensor, data: PyObject) -> PyResult<()> {
        let numpy = py.import_bound("numpy")?;

        // Convert to numpy array
        let array = numpy.call_method1("asarray", (data,))?;

        // Convert to float32
        let array_f32 = array.call_method1("astype", ("float32",))?;

        // Get shape
        let shape_obj = array_f32.getattr("shape")?;
        let shape: Vec<u32> = shape_obj
            .extract::<Vec<usize>>()?
            .iter()
            .map(|&d| d as u32)
            .collect();

        // Validate shape
        if shape != tensor.tensor_descriptor.descriptor.shape {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch: tensor has shape {:?}, but data has shape {:?}",
                tensor.tensor_descriptor.descriptor.shape, shape
            )));
        }

        // Get flattened data
        let flat = array_f32.call_method0("flatten")?;
        let data: Vec<f32> = flat.call_method0("tolist")?.extract()?;

        tensor.set_data(data)?;

        Ok(())
    }

    /// Get power preference hint
    #[getter]
    fn power_preference(&self) -> String {
        self.power_preference.clone()
    }

    /// Check if GPU/NPU acceleration is available
    ///
    /// Returns:
    ///     bool: True if the platform can provide GPU or NPU resources
    ///
    /// Note:
    ///     This indicates platform capability, not a guarantee of device allocation.
    ///     The actual execution may still use CPU if needed.
    #[getter]
    fn accelerated(&self) -> bool {
        self.accelerated_available
    }

    /// Get operation support limits for this context
    ///
    /// Returns a dictionary describing what operations and parameter types
    /// are supported by the backend implementation. This allows applications
    /// to query feature support and adapt their models accordingly.
    ///
    /// Returns:
    ///     dict: Dictionary with support limits for each operation
    ///
    /// Example:
    ///     >>> limits = context.op_support_limits()
    ///     >>> print(limits['preferredInputLayout'])
    ///     'nchw'
    ///     >>> print(limits['input']['dataTypes'])
    ///     ['float32', 'float16', 'int32', ...]
    fn op_support_limits(&self, py: Python) -> PyResult<Py<PyDict>> {
        let result = PyDict::new_bound(py);

        // Helper function to create data type lists
        let create_float_types = || -> Vec<&str> { vec!["float32", "float16"] };

        let create_all_types = || -> Vec<&str> {
            vec![
                "float32", "float16", "int32", "uint32", "int8", "uint8", "int64", "uint64",
            ]
        };

        // Helper function to create rank range
        let create_rank_range = |py: Python| -> PyResult<Py<PyDict>> {
            let rank = PyDict::new_bound(py);
            rank.set_item("min", 0)?;
            rank.set_item("max", 4)?; // Support up to 4D tensors
            Ok(rank.into())
        };

        // Helper function to create tensor limits
        let create_tensor_limits = |py: Python, float_only: bool| -> PyResult<Py<PyDict>> {
            let limits = PyDict::new_bound(py);
            let types = if float_only {
                create_float_types()
            } else {
                create_all_types()
            };
            limits.set_item("dataTypes", types)?;
            limits.set_item("rankRange", create_rank_range(py)?)?;
            Ok(limits.into())
        };

        // Helper function to create single input limits
        let create_single_input_limits = |py: Python| -> PyResult<Py<PyDict>> {
            let limits = PyDict::new_bound(py);
            limits.set_item("input", create_tensor_limits(py, true)?)?;
            limits.set_item("output", create_tensor_limits(py, true)?)?;
            Ok(limits.into())
        };

        // Helper function to create binary limits
        let create_binary_limits = |py: Python| -> PyResult<Py<PyDict>> {
            let limits = PyDict::new_bound(py);
            limits.set_item("a", create_tensor_limits(py, true)?)?;
            limits.set_item("b", create_tensor_limits(py, true)?)?;
            limits.set_item("output", create_tensor_limits(py, true)?)?;
            Ok(limits.into())
        };

        // Top-level properties
        result.set_item("preferredInputLayout", "nchw")?;
        result.set_item("maxTensorByteLength", 4294967295u64)?; // 4GB max

        // Input, constant, output limits
        result.set_item("input", create_tensor_limits(py, false)?)?;
        result.set_item("constant", create_tensor_limits(py, false)?)?;
        result.set_item("output", create_tensor_limits(py, false)?)?;

        // Binary operations
        result.set_item("add", create_binary_limits(py)?)?;
        result.set_item("sub", create_binary_limits(py)?)?;
        result.set_item("mul", create_binary_limits(py)?)?;
        result.set_item("div", create_binary_limits(py)?)?;
        result.set_item("pow", create_binary_limits(py)?)?;
        result.set_item("matmul", create_binary_limits(py)?)?;

        // Comparison operations
        result.set_item("equal", create_binary_limits(py)?)?;
        result.set_item("greater", create_binary_limits(py)?)?;
        result.set_item("greaterOrEqual", create_binary_limits(py)?)?;
        result.set_item("lesser", create_binary_limits(py)?)?;
        result.set_item("lesserOrEqual", create_binary_limits(py)?)?;

        // Logical operations
        result.set_item("logicalAnd", create_binary_limits(py)?)?;
        result.set_item("logicalOr", create_binary_limits(py)?)?;
        result.set_item("logicalXor", create_binary_limits(py)?)?;
        result.set_item("logicalNot", create_single_input_limits(py)?)?;

        // Unary/activation operations
        result.set_item("relu", create_single_input_limits(py)?)?;
        result.set_item("sigmoid", create_single_input_limits(py)?)?;
        result.set_item("tanh", create_single_input_limits(py)?)?;
        result.set_item("softmax", create_single_input_limits(py)?)?;
        result.set_item("gelu", create_single_input_limits(py)?)?;
        result.set_item("elu", create_single_input_limits(py)?)?;
        result.set_item("leakyRelu", create_single_input_limits(py)?)?;
        result.set_item("hardSwish", create_single_input_limits(py)?)?;
        result.set_item("hardSigmoid", create_single_input_limits(py)?)?;
        result.set_item("clamp", create_single_input_limits(py)?)?;
        result.set_item("prelu", create_binary_limits(py)?)?;
        result.set_item("softplus", create_single_input_limits(py)?)?;
        result.set_item("softsign", create_single_input_limits(py)?)?;
        result.set_item("identity", create_single_input_limits(py)?)?;

        // Element-wise unary operations
        result.set_item("abs", create_single_input_limits(py)?)?;
        result.set_item("ceil", create_single_input_limits(py)?)?;
        result.set_item("floor", create_single_input_limits(py)?)?;
        result.set_item("round", create_single_input_limits(py)?)?;
        result.set_item("neg", create_single_input_limits(py)?)?;
        result.set_item("sign", create_single_input_limits(py)?)?;
        result.set_item("reciprocal", create_single_input_limits(py)?)?;
        result.set_item("exp", create_single_input_limits(py)?)?;
        result.set_item("log", create_single_input_limits(py)?)?;
        result.set_item("sqrt", create_single_input_limits(py)?)?;
        result.set_item("erf", create_single_input_limits(py)?)?;

        // Trigonometric operations
        result.set_item("sin", create_single_input_limits(py)?)?;
        result.set_item("cos", create_single_input_limits(py)?)?;
        result.set_item("tan", create_single_input_limits(py)?)?;
        result.set_item("asin", create_single_input_limits(py)?)?;
        result.set_item("acos", create_single_input_limits(py)?)?;
        result.set_item("atan", create_single_input_limits(py)?)?;

        // Hyperbolic operations
        result.set_item("sinh", create_single_input_limits(py)?)?;
        result.set_item("cosh", create_single_input_limits(py)?)?;
        result.set_item("tanh", create_single_input_limits(py)?)?;
        result.set_item("asinh", create_single_input_limits(py)?)?;
        result.set_item("acosh", create_single_input_limits(py)?)?;
        result.set_item("atanh", create_single_input_limits(py)?)?;

        // Type conversion
        let cast_limits = PyDict::new_bound(py);
        cast_limits.set_item("input", create_tensor_limits(py, false)?)?;
        cast_limits.set_item("output", create_tensor_limits(py, false)?)?;
        result.set_item("cast", cast_limits)?;

        // Shape operations
        result.set_item("reshape", create_single_input_limits(py)?)?;
        result.set_item("transpose", create_single_input_limits(py)?)?;
        result.set_item("squeeze", create_single_input_limits(py)?)?;
        result.set_item("unsqueeze", create_single_input_limits(py)?)?;
        result.set_item("expand", create_single_input_limits(py)?)?;
        result.set_item("slice", create_single_input_limits(py)?)?;
        result.set_item("tile", create_single_input_limits(py)?)?;

        // Concat
        let concat_limits = PyDict::new_bound(py);
        concat_limits.set_item("inputs", create_tensor_limits(py, true)?)?;
        concat_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("concat", concat_limits)?;

        // Split
        let split_limits = PyDict::new_bound(py);
        split_limits.set_item("input", create_tensor_limits(py, true)?)?;
        split_limits.set_item("outputs", create_tensor_limits(py, true)?)?;
        result.set_item("split", split_limits)?;

        // Convolution
        let conv2d_limits = PyDict::new_bound(py);
        conv2d_limits.set_item("input", create_tensor_limits(py, true)?)?;
        conv2d_limits.set_item("filter", create_tensor_limits(py, true)?)?;
        conv2d_limits.set_item("bias", create_tensor_limits(py, true)?)?;
        conv2d_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("conv2d", conv2d_limits)?;

        let conv_transpose_limits = PyDict::new_bound(py);
        conv_transpose_limits.set_item("input", create_tensor_limits(py, true)?)?;
        conv_transpose_limits.set_item("filter", create_tensor_limits(py, true)?)?;
        conv_transpose_limits.set_item("bias", create_tensor_limits(py, true)?)?;
        conv_transpose_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("convTranspose2d", conv_transpose_limits)?;

        // Pooling
        let pool2d_limits = PyDict::new_bound(py);
        pool2d_limits.set_item("input", create_tensor_limits(py, true)?)?;
        pool2d_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("averagePool2d", pool2d_limits.clone())?;
        result.set_item("maxPool2d", pool2d_limits)?;

        // Normalization
        let batch_norm_limits = PyDict::new_bound(py);
        batch_norm_limits.set_item("input", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("mean", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("variance", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("scale", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("bias", create_tensor_limits(py, true)?)?;
        batch_norm_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("batchNormalization", batch_norm_limits)?;

        let norm_limits = PyDict::new_bound(py);
        norm_limits.set_item("input", create_tensor_limits(py, true)?)?;
        norm_limits.set_item("scale", create_tensor_limits(py, true)?)?;
        norm_limits.set_item("bias", create_tensor_limits(py, true)?)?;
        norm_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("instanceNormalization", norm_limits.clone())?;
        result.set_item("layerNormalization", norm_limits)?;

        // Reduction operations
        result.set_item("reduceSum", create_single_input_limits(py)?)?;
        result.set_item("reduceMean", create_single_input_limits(py)?)?;
        result.set_item("reduceMax", create_single_input_limits(py)?)?;
        result.set_item("reduceMin", create_single_input_limits(py)?)?;
        result.set_item("reduceProduct", create_single_input_limits(py)?)?;
        result.set_item("reduceL1", create_single_input_limits(py)?)?;
        result.set_item("reduceL2", create_single_input_limits(py)?)?;
        result.set_item("reduceLogSum", create_single_input_limits(py)?)?;
        result.set_item("reduceLogSumExp", create_single_input_limits(py)?)?;
        result.set_item("reduceSumSquare", create_single_input_limits(py)?)?;

        // GEMM
        let gemm_limits = PyDict::new_bound(py);
        gemm_limits.set_item("a", create_tensor_limits(py, true)?)?;
        gemm_limits.set_item("b", create_tensor_limits(py, true)?)?;
        gemm_limits.set_item("c", create_tensor_limits(py, true)?)?;
        gemm_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("gemm", gemm_limits)?;

        // ArgMax/ArgMin
        let arg_limits = PyDict::new_bound(py);
        arg_limits.set_item("input", create_tensor_limits(py, true)?)?;
        arg_limits.set_item("output", create_tensor_limits(py, false)?)?; // Outputs int64/uint64
        result.set_item("argMax", arg_limits.clone())?;
        result.set_item("argMin", arg_limits)?;

        // Gather operations
        let gather_limits = PyDict::new_bound(py);
        gather_limits.set_item("input", create_tensor_limits(py, true)?)?;
        gather_limits.set_item("indices", create_tensor_limits(py, false)?)?;
        gather_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("gather", gather_limits)?;

        // Scatter operations
        let scatter_limits = PyDict::new_bound(py);
        scatter_limits.set_item("input", create_tensor_limits(py, true)?)?;
        scatter_limits.set_item("indices", create_tensor_limits(py, false)?)?;
        scatter_limits.set_item("updates", create_tensor_limits(py, true)?)?;
        scatter_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("scatterElements", scatter_limits.clone())?;
        result.set_item("scatterND", scatter_limits)?;

        // Where
        let where_limits = PyDict::new_bound(py);
        where_limits.set_item("condition", create_tensor_limits(py, false)?)?;
        where_limits.set_item("trueValue", create_tensor_limits(py, true)?)?;
        where_limits.set_item("falseValue", create_tensor_limits(py, true)?)?;
        where_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("where", where_limits)?;

        // Pad
        result.set_item("pad", create_single_input_limits(py)?)?;

        // Quantization
        let quant_limits = PyDict::new_bound(py);
        quant_limits.set_item("input", create_tensor_limits(py, true)?)?;
        quant_limits.set_item("scale", create_tensor_limits(py, true)?)?;
        quant_limits.set_item("zeroPoint", create_tensor_limits(py, false)?)?;
        quant_limits.set_item("output", create_tensor_limits(py, false)?)?;
        result.set_item("quantizeLinear", quant_limits.clone())?;

        let dequant_limits = PyDict::new_bound(py);
        dequant_limits.set_item("input", create_tensor_limits(py, false)?)?;
        dequant_limits.set_item("scale", create_tensor_limits(py, true)?)?;
        dequant_limits.set_item("zeroPoint", create_tensor_limits(py, false)?)?;
        dequant_limits.set_item("output", create_tensor_limits(py, true)?)?;
        result.set_item("dequantizeLinear", dequant_limits)?;

        // Triangular
        result.set_item("triangular", create_single_input_limits(py)?)?;

        Ok(result.into())
    }

    /// Return backend/feature diagnostics for this context
    ///
    /// Useful to understand which backend was selected, which runtime features
    /// were compiled into the wheel, and whether the selected backend is actually
    /// available (otherwise the fallback path returns zeros).
    fn backend_info(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let info = PyDict::new_bound(py);

        let backend = match self.backend {
            Backend::OnnxCpu => "onnx_cpu",
            Backend::OnnxGpu => "onnx_gpu",
            Backend::CoreML => "coreml",
            Backend::TensorRT => "tensorrt",
            Backend::None => "none",
        };

        let onnx_compiled = cfg!(feature = "onnx-runtime");
        let coreml_compiled = cfg!(all(target_os = "macos", feature = "coreml-runtime"));
        let trtx_compiled = cfg!(feature = "trtx-runtime");

        let backend_available = match self.backend {
            Backend::OnnxCpu | Backend::OnnxGpu => onnx_compiled,
            Backend::CoreML => coreml_compiled,
            Backend::TensorRT => trtx_compiled,
            Backend::None => false,
        };

        let fallback_reason = if backend == "none" {
            Some("no backend selected (fallback)")
        } else if !backend_available {
            Some("selected backend not compiled into this build")
        } else {
            None
        };

        info.set_item("backend", backend)?;
        info.set_item("accelerated_available", self.accelerated_available)?;
        info.set_item("compiled_features", {
            let compiled = PyDict::new_bound(py);
            compiled.set_item("onnx_runtime", onnx_compiled)?;
            compiled.set_item("coreml_runtime", coreml_compiled)?;
            compiled.set_item("trtx_runtime", trtx_compiled)?;
            compiled
        })?;
        info.set_item("backend_available", backend_available)?;
        if let Some(reason) = fallback_reason {
            info.set_item("fallback_reason", reason)?;
        }

        Ok(info.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "MLContext(accelerated={}, power='{}')",
            self.accelerated_available, self.power_preference
        )
    }
}

impl PyMLContext {
    fn new(power_preference: String, accelerated_requested: bool, device_type: String) -> Self {
        // Force specific backend if requested, otherwise use automatic selection
        let (backend, accelerated_available) = match device_type.as_str() {
            "cpu" => {
                #[cfg(feature = "onnx-runtime")]
                {
                    (Backend::OnnxCpu, false)
                }
                #[cfg(not(feature = "onnx-runtime"))]
                {
                    (Backend::None, false)
                }
            }
            "gpu" => {
                #[cfg(feature = "onnx-runtime")]
                {
                    (Backend::OnnxGpu, true)
                }
                #[cfg(not(feature = "onnx-runtime"))]
                {
                    (Backend::None, false)
                }
            }
            "npu" => {
                #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
                {
                    (Backend::CoreML, true)
                }
                #[cfg(not(all(target_os = "macos", feature = "coreml-runtime")))]
                {
                    (Backend::None, false)
                }
            }
            _ => {
                // "auto" or unrecognized - use automatic selection
                Self::select_backend(accelerated_requested, &power_preference)
            }
        };

        Self {
            power_preference,
            _accelerated_requested: accelerated_requested,
            accelerated_available,
            backend,

            #[cfg(feature = "onnx-runtime")]
            onnx_session: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Get or create cached ONNX Runtime session for the given graph
    ///
    /// This enables device tensor reuse by keeping the session alive across operations.
    /// The session is created on first access and cached for subsequent use.
    #[cfg(feature = "onnx-runtime")]
    fn get_onnx_session(
        &self,
        graph: &PyMLGraph,
    ) -> Result<std::sync::Arc<ort::session::Session>, pyo3::PyErr> {
        let mut session_guard = self.onnx_session.lock().unwrap();

        if let Some(session) = session_guard.as_ref() {
            return Ok(std::sync::Arc::clone(session));
        }

        // Create new session
        let converter = crate::converters::OnnxConverter;
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX conversion failed: {}", e))
        })?;

        let session = ort::session::Session::builder()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Session builder failed: {}", e))
            })?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Set opt level failed: {}", e))
            })?
            .commit_from_memory(&converted.data)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Load model failed: {}", e))
            })?;

        let session_arc = std::sync::Arc::new(session);
        *session_guard = Some(std::sync::Arc::clone(&session_arc));

        Ok(session_arc)
    }

    /// Execute graph using ONNX Runtime backend
    #[cfg(feature = "onnx-runtime")]
    fn compute_onnx(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        debug_print!("[COMPUTE] compute_onnx called, converting graph to ONNX");
        // Convert graph to ONNX
        let converter = crate::converters::OnnxConverter;
        debug_print!("[COMPUTE] about to call converter.convert()");
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX conversion failed: {}", e))
        })?;

        // Convert Python inputs to OnnxInput structs
        let numpy = py.import_bound("numpy")?;
        let mut onnx_inputs = Vec::new();

        for input_id in &graph.graph_info.input_operands {
            let input_op = graph
                .graph_info
                .operands
                .get(*input_id as usize)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Input operand {} not found in graph",
                        input_id
                    ))
                })?;

            let default_name = format!("input_{}", input_id);
            let input_name = input_op.name.as_deref().unwrap_or(&default_name);

            // Skip empty KV cache inputs (past_sequence_length=0)
            // These will be removed by the converter, so don't expect them in inputs dict
            let has_empty_dimension = input_op.descriptor.shape.iter().any(|&dim| dim == 0);
            let is_kv_input = input_name.starts_with("past_key_values_");
            if has_empty_dimension && is_kv_input {
                debug_print!(
                    "[COMPUTE] Skipping empty KV input: {} (shape: {:?})",
                    input_name,
                    input_op.descriptor.shape
                );
                continue;
            }

            // Get the numpy array from inputs dict
            let array = inputs.get_item(input_name)?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Missing input: {}", input_name))
            })?;

            // Get the data type from the operand descriptor
            let data_type = &input_op.descriptor.data_type;

            // Convert array to the correct type based on descriptor
            let dtype_str = match data_type {
                crate::graph::DataType::Float32 => "float32",
                crate::graph::DataType::Float16 => "float16",
                crate::graph::DataType::Int8 => "int8",
                crate::graph::DataType::Uint8 => "uint8",
                crate::graph::DataType::Int32 => "int32",
                crate::graph::DataType::Uint32 => "uint32",
                crate::graph::DataType::Int64 => "int64",
                crate::graph::DataType::Uint64 => "uint64",
            };

            let array_typed = array.call_method1("astype", (dtype_str,))?;

            // Get shape
            let shape_obj = array_typed.getattr("shape")?;
            let shape: Vec<usize> = shape_obj.extract()?;

            // Get flattened data and convert to appropriate TensorData
            let flat = array_typed.call_method0("flatten")?;

            let tensor_data = match data_type {
                crate::graph::DataType::Float32 => {
                    let data: Vec<f32> = flat.call_method0("tolist")?.extract()?;
                    crate::executors::onnx::TensorData::Float32(data)
                }
                crate::graph::DataType::Float16 => {
                    // Get as float32 list then convert to f16 bits
                    let data_f32: Vec<f32> = flat.call_method0("tolist")?.extract()?;
                    let data_u16: Vec<u16> = data_f32
                        .iter()
                        .map(|&f| half::f16::from_f32(f).to_bits())
                        .collect();
                    crate::executors::onnx::TensorData::Float16(data_u16)
                }
                crate::graph::DataType::Int8 => {
                    let data: Vec<i8> = flat.call_method0("tolist")?.extract()?;
                    crate::executors::onnx::TensorData::Int8(data)
                }
                crate::graph::DataType::Uint8 => {
                    let data: Vec<u8> = flat.call_method0("tolist")?.extract()?;
                    crate::executors::onnx::TensorData::Uint8(data)
                }
                crate::graph::DataType::Int32 => {
                    let data: Vec<i32> = flat.call_method0("tolist")?.extract()?;
                    crate::executors::onnx::TensorData::Int32(data)
                }
                crate::graph::DataType::Uint32 => {
                    let data: Vec<u32> = flat.call_method0("tolist")?.extract()?;
                    crate::executors::onnx::TensorData::Uint32(data)
                }
                crate::graph::DataType::Int64 => {
                    let data: Vec<i64> = flat.call_method0("tolist")?.extract()?;
                    crate::executors::onnx::TensorData::Int64(data)
                }
                crate::graph::DataType::Uint64 => {
                    let data: Vec<u64> = flat.call_method0("tolist")?.extract()?;
                    crate::executors::onnx::TensorData::Uint64(data)
                }
            };

            onnx_inputs.push(OnnxInput {
                name: input_name.to_string(),
                shape,
                data: tensor_data,
            });
        }

        // Execute with ONNX runtime
        let onnx_outputs = run_onnx_with_inputs(&converted.data, onnx_inputs).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX execution failed: {}", e))
        })?;

        // Convert outputs back to numpy arrays
        let result = PyDict::new_bound(py);
        for output in onnx_outputs {
            let shape_tuple =
                pyo3::types::PyTuple::new_bound(py, output.shape.iter().map(|&d| d as i64));
            let array = numpy.call_method1("array", (output.data,))?;
            let reshaped = array.call_method1("reshape", (shape_tuple,))?;
            result.set_item(output.name, reshaped)?;
        }

        Ok(result.into())
    }

    /// Stub for when ONNX Runtime is not available but backend was selected as ONNX
    #[cfg(not(feature = "onnx-runtime"))]
    fn compute_onnx(
        &self,
        py: Python,
        graph: &PyMLGraph,
        _inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "ONNX Runtime backend selected but not compiled with onnx-runtime feature",
        ))
    }

    /// Execute graph using CoreML backend
    #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
    fn compute_coreml(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        use crate::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};

        // Convert graph to CoreML
        let converter = crate::converters::CoremlMlProgramConverter::default();
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML conversion failed: {}", e))
        })?;

        // Convert Python inputs to CoremlInput structs
        let numpy = py.import_bound("numpy")?;
        let mut coreml_inputs = Vec::new();

        for input_id in &graph.graph_info.input_operands {
            let input_op = graph
                .graph_info
                .operands
                .get(*input_id as usize)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Input operand {} not found in graph",
                        input_id
                    ))
                })?;

            let default_name = format!("input_{}", input_id);
            let input_name = input_op.name.as_deref().unwrap_or(&default_name);

            // Skip empty KV cache inputs (past_sequence_length=0)
            // These will be removed by the converter, so don't expect them in inputs dict
            let has_empty_dimension = input_op.descriptor.shape.iter().any(|&dim| dim == 0);
            let is_kv_input = input_name.starts_with("past_key_values_");
            if has_empty_dimension && is_kv_input {
                debug_print!(
                    "[COMPUTE] Skipping empty KV input: {} (shape: {:?})",
                    input_name,
                    input_op.descriptor.shape
                );
                continue;
            }

            // Get the numpy array from inputs dict
            let array = inputs.get_item(input_name)?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Missing input: {}", input_name))
            })?;

            // Convert to float32 array (CoreML uses float32)
            let array_f32 = array.call_method1("astype", ("float32",))?;

            // Get shape
            let shape_obj = array_f32.getattr("shape")?;
            let mut shape: Vec<usize> = shape_obj.extract()?;

            // CoreML requires explicit shapes - convert scalars (0D) to 1D [1]
            // This matches the conversion done in the CoreML model
            if shape.is_empty() {
                shape = vec![1];
            }

            // Get flattened data
            let flat = array_f32.call_method0("flatten")?;
            let data: Vec<f32> = flat.call_method0("tolist")?.extract()?;

            coreml_inputs.push(CoremlInput {
                name: input_name.to_string(),
                shape,
                data,
            });
        }

        // Execute with CoreML runtime (with optional weight file)
        let weights_ref = converted.weights_data.as_deref();
        let attempts =
            run_coreml_with_inputs_with_weights(&converted.data, weights_ref, coreml_inputs)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "CoreML execution failed: {}",
                        e
                    ))
                })?;

        // Find first successful attempt
        let outputs = attempts
            .iter()
            .find_map(|attempt| attempt.result.as_ref().ok().cloned())
            .ok_or_else(|| {
                // Collect all error messages for debugging
                let error_messages: Vec<String> = attempts
                    .iter()
                    .map(|attempt| {
                        format!(
                            "{}: {}",
                            attempt.compute_unit,
                            attempt
                                .result
                                .as_ref()
                                .err()
                                .unwrap_or(&"unknown error".to_string())
                        )
                    })
                    .collect();
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "CoreML execution failed on all compute units:\n{}",
                    error_messages.join("\n")
                ))
            })?;

        // Convert outputs back to numpy arrays
        let result = PyDict::new_bound(py);
        for output in outputs {
            // Check if original graph output was scalar (0D)
            // Find the corresponding output operand in the graph
            let mut original_shape = output.shape.clone();
            for &output_id in &graph.graph_info.output_operands {
                if let Some(operand) = graph.graph_info.operand(output_id) {
                    // Get output name from operand or use default naming
                    let default_name = format!("output_{}", output_id);
                    let output_name_in_graph = operand.name.as_deref().unwrap_or(&default_name);

                    if output_name_in_graph == output.name {
                        // If original was 0D and we got [1], reshape back to []
                        if operand.descriptor.shape.is_empty() && output.shape == vec![1] {
                            original_shape = vec![];
                        }
                        break;
                    }
                }
            }

            let shape_tuple =
                pyo3::types::PyTuple::new_bound(py, original_shape.iter().map(|&d| d as i64));
            let array = numpy.call_method1("array", (output.data,))?;
            let reshaped = array.call_method1("reshape", (shape_tuple,))?;
            result.set_item(output.name, reshaped)?;
        }

        Ok(result.into())
    }

    /// Stub for when CoreML is not available but backend was selected as CoreML
    #[cfg(any(not(target_os = "macos"), not(feature = "coreml-runtime")))]
    fn compute_coreml(
        &self,
        _py: Python,
        _graph: &PyMLGraph,
        _inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CoreML backend selected but not available on this platform or not compiled with coreml-runtime feature",
        ))
    }

    /// Execute graph using TensorRT backend
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    fn compute_trtx(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        use crate::executors::trtx::{TrtxInput, run_trtx_with_inputs};

        // Convert graph to ONNX (TensorRT uses ONNX as input format)
        let converter = crate::converters::OnnxConverter;
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX conversion failed: {}", e))
        })?;

        // Convert Python inputs to TrtxInput structs
        let numpy = py.import_bound("numpy")?;
        let mut trtx_inputs = Vec::new();

        for input_id in &graph.graph_info.input_operands {
            let input_op = graph
                .graph_info
                .operands
                .get(*input_id as usize)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Input operand {} not found in graph",
                        input_id
                    ))
                })?;

            let default_name = format!("input_{}", input_id);
            let input_name = input_op.name.as_deref().unwrap_or(&default_name);

            // Skip empty KV cache inputs (past_sequence_length=0)
            // These will be removed by the converter, so don't expect them in inputs dict
            let has_empty_dimension = input_op.descriptor.shape.iter().any(|&dim| dim == 0);
            let is_kv_input = input_name.starts_with("past_key_values_");
            if has_empty_dimension && is_kv_input {
                debug_print!(
                    "[COMPUTE] Skipping empty KV input: {} (shape: {:?})",
                    input_name,
                    input_op.descriptor.shape
                );
                continue;
            }

            // Get the numpy array from inputs dict
            let array = inputs.get_item(input_name)?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Missing input: {}", input_name))
            })?;

            // Convert to float32 array
            let array_f32 = array.call_method1("astype", ("float32",))?;

            // Get shape
            let shape_obj = array_f32.getattr("shape")?;
            let shape: Vec<usize> = shape_obj.extract()?;

            // Get flattened data
            let flat = array_f32.call_method0("flatten")?;
            let data: Vec<f32> = flat.call_method0("tolist")?.extract()?;

            trtx_inputs.push(TrtxInput {
                name: input_name.to_string(),
                shape,
                data,
            });
        }

        // Execute with TensorRT
        let trtx_outputs = run_trtx_with_inputs(&converted.data, trtx_inputs).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("TensorRT execution failed: {}", e))
        })?;

        // Convert outputs back to numpy arrays
        let result = PyDict::new_bound(py);
        for output in trtx_outputs {
            let shape_tuple =
                pyo3::types::PyTuple::new_bound(py, output.shape.iter().map(|&d| d as i64));
            let array = numpy.call_method1("array", (output.data,))?;
            let reshaped = array.call_method1("reshape", (shape_tuple,))?;
            result.set_item(output.name, reshaped)?;
        }

        Ok(result.into())
    }

    /// Stub for when TensorRT is not available but backend was selected as TensorRT
    #[cfg(not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock")))]
    fn compute_trtx(
        &self,
        _py: Python,
        _graph: &PyMLGraph,
        _inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "TensorRT backend selected but not compiled with trtx-runtime feature",
        ))
    }

    /// Fallback computation removed: we now error when no backend is available.

    /// Select the appropriate backend based on accelerated preference and power hint
    ///
    /// Returns: (Backend, accelerated_available)
    /// - Backend: The selected backend to use
    /// - accelerated_available: true if GPU/NPU acceleration is available
    ///
    /// Selection logic per WebNN Device Selection Explainer:
    /// - accelerated=true + power="low-power" → NPU > GPU > CPU
    /// - accelerated=true + power="high-performance" → GPU > NPU > CPU
    /// - accelerated=true + power="default" → GPU > NPU > CPU
    /// - accelerated=false → CPU only
    fn select_backend(accelerated: bool, power_preference: &str) -> (Backend, bool) {
        if !accelerated {
            // User explicitly requested CPU-only execution
            #[cfg(feature = "onnx-runtime")]
            {
                return (Backend::OnnxCpu, false);
            }
            #[cfg(not(feature = "onnx-runtime"))]
            {
                return (Backend::None, false);
            }
        }

        // Accelerated execution requested - select based on power preference
        match power_preference {
            "low-power" => {
                // Prefer NPU (Neural Engine on macOS) for low power
                #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
                {
                    return (Backend::CoreML, true);
                }

                // Fallback to GPU if NPU not available
                #[cfg(all(not(target_os = "macos"), feature = "onnx-runtime"))]
                {
                    return (Backend::OnnxGpu, true);
                }
                #[cfg(all(
                    target_os = "macos",
                    not(feature = "coreml-runtime"),
                    feature = "onnx-runtime"
                ))]
                {
                    return (Backend::OnnxGpu, true);
                }

                // No acceleration available - only reachable when onnx-runtime is not enabled
                #[cfg(not(feature = "onnx-runtime"))]
                {
                    (Backend::None, false)
                }

                // When onnx-runtime is enabled, one of the above cfg blocks will match and return,
                // so this branch is never reached. We need this to satisfy the compiler in that case.
                #[cfg(feature = "onnx-runtime")]
                #[allow(unreachable_code)]
                {
                    (Backend::None, false)
                }
            }
            "high-performance" | "default" => {
                // Prefer TensorRT for NVIDIA GPU when available (highest performance)
                #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
                {
                    return (Backend::TensorRT, true);
                }

                // Prefer ONNX GPU for cross-platform consistency
                // TODO: Enable CoreML priority once CoreML executor bugs are fixed
                // (currently panics on multi-output operations and some data type mismatches)
                #[cfg(all(
                    feature = "onnx-runtime",
                    not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))
                ))]
                {
                    return (Backend::OnnxGpu, true);
                }

                // Fallback to CoreML on macOS if ONNX not available
                #[cfg(all(
                    target_os = "macos",
                    feature = "coreml-runtime",
                    not(feature = "onnx-runtime"),
                    not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))
                ))]
                {
                    return (Backend::CoreML, true);
                }

                // No acceleration available, fallback to CPU
                #[cfg(all(
                    not(feature = "onnx-runtime"),
                    not(feature = "coreml-runtime"),
                    not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))
                ))]
                {
                    return (Backend::None, false);
                }
            }
            _ => {
                // Unknown power preference, use default behavior (GPU preferred)
                #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
                {
                    return (Backend::TensorRT, true);
                }
                #[cfg(all(
                    feature = "onnx-runtime",
                    not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))
                ))]
                {
                    return (Backend::OnnxGpu, true);
                }
                #[cfg(all(
                    not(feature = "onnx-runtime"),
                    not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))
                ))]
                {
                    return (Backend::None, false);
                }
            }
        }
    }
}
