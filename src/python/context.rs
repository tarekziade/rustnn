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
use crate::graph::OperandDescriptor;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg(feature = "onnx-runtime")]
use crate::executors::onnx::{OnnxInput, run_onnx_with_inputs};

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
use crate::executors::coreml::run_coreml_zeroed_cached;

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
    ///
    /// Returns:
    ///     MLContext: A new context for graph operations
    ///
    /// Note:
    ///     The accelerated parameter is a hint, not a guarantee. The platform
    ///     decides the actual device allocation based on runtime conditions.
    ///     Query context.accelerated after creation to check if acceleration is available.
    #[pyo3(signature = (power_preference="default", accelerated=true))]
    fn create_context(&self, power_preference: &str, accelerated: bool) -> PyResult<PyMLContext> {
        Ok(PyMLContext::new(power_preference.to_string(), accelerated))
    }
}

/// MLContext manages the execution environment for neural network graphs
#[pyclass(name = "MLContext")]
pub struct PyMLContext {
    power_preference: String,
    _accelerated_requested: bool,
    accelerated_available: bool,
    backend: Backend,
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
            Backend::None => self.compute_fallback(py, graph),
        }
    }

    /// Dispatch graph execution asynchronously with MLTensor inputs/outputs
    ///
    /// Following the W3C WebNN MLTensor Explainer:
    /// https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md
    ///
    /// This method queues the graph for execution and returns immediately.
    /// Results are written to output tensors and can be read later with read_tensor().
    ///
    /// Args:
    ///     graph: The compiled MLGraph to execute
    ///     inputs: Dictionary mapping input names to MLTensor objects
    ///     outputs: Dictionary mapping output names to MLTensor objects
    ///
    /// Note:
    ///     This is currently implemented as synchronous execution.
    ///     True async execution will be added in future versions.
    #[pyo3(signature = (graph, inputs, outputs))]
    fn dispatch(
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

        // Execute
        run_coreml_zeroed_cached(&converted.data, &inputs, None).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML execution failed: {}", e))
        })?;

        // Return empty dict for now (actual implementation would return outputs)
        let result = PyDict::new_bound(py);
        Ok(result.into())
    }

    /// Create a tensor for explicit tensor management
    ///
    /// Following the W3C WebNN MLTensor Explainer:
    /// https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md
    ///
    /// Args:
    ///     shape: Shape of the tensor
    ///     data_type: Data type string (e.g., "float32")
    ///     readable: If True, tensor data can be read back to CPU (default: True)
    ///     writable: If True, tensor data can be written from CPU (default: True)
    ///     exportable_to_gpu: If True, tensor can be used as GPU texture (default: False)
    ///
    /// Returns:
    ///     MLTensor: A new tensor with the specified properties
    #[pyo3(signature = (shape, data_type, readable=true, writable=true, exportable_to_gpu=false))]
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

    fn __repr__(&self) -> String {
        format!(
            "MLContext(accelerated={}, power='{}')",
            self.accelerated_available, self.power_preference
        )
    }
}

impl PyMLContext {
    fn new(power_preference: String, accelerated_requested: bool) -> Self {
        // Select backend based on accelerated preference and power preference
        let (backend, accelerated_available) =
            Self::select_backend(accelerated_requested, &power_preference);

        Self {
            power_preference,
            _accelerated_requested: accelerated_requested,
            accelerated_available,
            backend,
        }
    }

    /// Execute graph using ONNX Runtime backend
    #[cfg(feature = "onnx-runtime")]
    fn compute_onnx(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        // Convert graph to ONNX
        let converter = crate::converters::OnnxConverter;
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
        _inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        // Convert graph to CoreML
        let converter = crate::converters::CoremlMlProgramConverter::default();
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML conversion failed: {}", e))
        })?;

        // Build input descriptors map
        use std::collections::HashMap;
        let mut input_descriptors = HashMap::new();
        for &input_id in &graph.graph_info.input_operands {
            if let Some(operand) = graph.graph_info.operand(input_id) {
                let name = operand
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("input_{}", input_id));
                input_descriptors.insert(name, operand.descriptor.clone());
            }
        }

        // Execute with CoreML (currently returns zeros)
        run_coreml_zeroed_cached(&converted.data, &input_descriptors, None).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML execution failed: {}", e))
        })?;

        // Return zeros for now (CoreML integration needs full implementation)
        self.compute_fallback(py, graph)
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

    /// Fallback computation that returns zeros (when no backend available)
    fn compute_fallback(&self, py: Python, graph: &PyMLGraph) -> PyResult<Py<PyDict>> {
        let result = PyDict::new_bound(py);

        for output_id in &graph.graph_info.output_operands {
            let output_op = graph
                .graph_info
                .operands
                .get(*output_id as usize)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Output operand {} not found in graph",
                        output_id
                    ))
                })?;

            let output_name = output_op.name.as_deref().unwrap_or("output");

            let numpy = py.import_bound("numpy")?;
            let shape = output_op.descriptor.shape.clone();
            let dtype_str = match output_op.descriptor.data_type {
                crate::graph::DataType::Float32 => "float32",
                crate::graph::DataType::Float16 => "float16",
                crate::graph::DataType::Int32 => "int32",
                crate::graph::DataType::Uint32 => "uint32",
                crate::graph::DataType::Int8 => "int8",
                crate::graph::DataType::Uint8 => "uint8",
                crate::graph::DataType::Int64 => "int64",
            };

            let zeros = numpy.call_method1("zeros", (shape, dtype_str))?;
            result.set_item(output_name, zeros)?;
        }

        Ok(result.into())
    }

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

                // Fallback to ONNX GPU
                #[cfg(all(
                    feature = "onnx-runtime",
                    not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))
                ))]
                {
                    return (Backend::OnnxGpu, true);
                }

                // Fallback to NPU if GPU not available
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
