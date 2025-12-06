use super::graph::PyMLGraph;
use super::graph_builder::PyMLGraphBuilder;
use crate::converters::GraphConverter;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[cfg(feature = "onnx-runtime")]
use crate::executors::onnx::{run_onnx_with_inputs, OnnxInput};

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
use crate::executors::coreml::run_coreml_zeroed_cached;

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
    ///     device_type: Device type ("cpu", "gpu", or "npu")
    ///     power_preference: Power preference ("default", "high-performance", or "low-power")
    ///
    /// Returns:
    ///     MLContext: A new context for graph operations
    #[pyo3(signature = (device_type="cpu", power_preference="default"))]
    fn create_context(&self, device_type: &str, power_preference: &str) -> PyResult<PyMLContext> {
        Ok(PyMLContext::new(
            device_type.to_string(),
            power_preference.to_string(),
        ))
    }
}

/// MLContext manages the execution environment for neural network graphs
#[pyclass(name = "MLContext")]
pub struct PyMLContext {
    device_type: String,
    power_preference: String,
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

    /// Compute the graph with given inputs
    ///
    /// Args:
    ///     graph: The compiled MLGraph to execute
    ///     inputs: Dictionary mapping input names to numpy arrays
    ///     outputs: Dictionary mapping output names to numpy arrays (pre-allocated)
    ///
    /// Returns:
    ///     Dictionary mapping output names to result numpy arrays
    #[pyo3(signature = (graph, inputs, outputs=None))]
    fn compute(
        &self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &Bound<'_, PyDict>,
        outputs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        #[cfg(feature = "onnx-runtime")]
        {
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

                let input_name = input_op.name.as_deref().unwrap_or(&format!("input_{}", input_id));

                // Get the numpy array from inputs dict
                let array = inputs.get_item(input_name)?.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Missing input: {}",
                        input_name
                    ))
                })?;

                // Convert to float32 array
                let array_f32 = array.call_method1("astype", ("float32",))?;

                // Get shape
                let shape_obj = array_f32.getattr("shape")?;
                let shape: Vec<usize> = shape_obj.extract()?;

                // Get flattened data
                let flat = array_f32.call_method0("flatten")?;
                let data: Vec<f32> = flat.call_method0("tolist")?.extract()?;

                onnx_inputs.push(OnnxInput {
                    name: input_name.to_string(),
                    shape,
                    data,
                });
            }

            // Execute with ONNX runtime
            let onnx_outputs = run_onnx_with_inputs(&converted.data, onnx_inputs).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX execution failed: {}", e))
            })?;

            // Convert outputs back to numpy arrays
            let result = PyDict::new_bound(py);
            for output in onnx_outputs {
                let shape_tuple = pyo3::types::PyTuple::new_bound(
                    py,
                    output.shape.iter().map(|&d| d as i64),
                );
                let array = numpy.call_method1("array", (output.data,))?;
                let reshaped = array.call_method1("reshape", (shape_tuple,))?;
                result.set_item(output.name, reshaped)?;
            }

            Ok(result.into())
        }

        #[cfg(not(feature = "onnx-runtime"))]
        {
            // Fallback: return zeros if ONNX runtime is not available
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
                };

                let zeros = numpy.call_method1("zeros", (shape, dtype_str))?;
                result.set_item(output_name, zeros)?;
            }

            Ok(result.into())
        }
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
        let converter = crate::converters::CoremlConverter;
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML conversion failed: {}", e))
        })?;

        std::fs::write(output_path, &converted.data).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write CoreML file: {}", e))
        })?;

        Ok(())
    }

    /// Execute graph using ONNX runtime (requires onnx-runtime feature)
    ///
    /// Args:
    ///     graph: The MLGraph to execute
    ///
    /// Returns:
    ///     Dictionary with execution results
    #[cfg(feature = "onnx-runtime")]
    fn execute_with_onnx(&self, py: Python, graph: &PyMLGraph) -> PyResult<Py<PyDict>> {
        // Convert to ONNX
        let converter = crate::converters::OnnxConverter;
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX conversion failed: {}", e))
        })?;

        // Execute with empty inputs map (for now)
        let inputs = HashMap::new();
        run_onnx_zeroed(&converted.data, &inputs).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX execution failed: {}", e))
        })?;

        // Return empty dict for now (actual implementation would return outputs)
        let result = PyDict::new_bound(py);
        Ok(result.into())
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
        let converter = crate::converters::CoremlConverter;
        let converted = converter.convert(&graph.graph_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML conversion failed: {}", e))
        })?;

        // Parse device type
        let compute_units = match device {
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

        // Execute
        run_coreml_zeroed_cached(&converted.data, compute_units).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("CoreML execution failed: {}", e))
        })?;

        // Return empty dict for now (actual implementation would return outputs)
        let result = PyDict::new_bound(py);
        Ok(result.into())
    }

    /// Get device type
    #[getter]
    fn device_type(&self) -> String {
        self.device_type.clone()
    }

    /// Get power preference
    #[getter]
    fn power_preference(&self) -> String {
        self.power_preference.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "MLContext(device='{}', power='{}')",
            self.device_type, self.power_preference
        )
    }
}

impl PyMLContext {
    fn new(device_type: String, power_preference: String) -> Self {
        Self {
            device_type,
            power_preference,
        }
    }
}
