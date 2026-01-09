#![cfg(all(feature = "onnx-runtime"))]

use std::collections::HashMap;
use std::sync::Once;

use ort::session::SessionInputValue;

use half;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

use crate::error::GraphError;
use crate::graph::OperandDescriptor;

static INIT: Once = Once::new();

fn ensure_ort_initialized() -> Result<(), GraphError> {
    let mut result = Ok(());
    INIT.call_once(|| {
        let success = ort::init()
            .with_name("rustnn")
            .with_execution_providers([
                ort::execution_providers::CPUExecutionProvider::default().build()
            ])
            .commit();

        if !success {
            result = Err(GraphError::OnnxRuntimeFailed {
                reason: "ort init failed - unable to initialize ONNX Runtime".to_string(),
            });
        }
    });
    result
}

#[derive(Debug, Clone)]
pub struct OnnxOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: String,
}

/// Tensor data for different types
pub enum TensorData {
    Float32(Vec<f32>),
    Float16(Vec<u16>), // f16 stored as u16 bits
    Int8(Vec<i8>),
    Uint8(Vec<u8>),
    Int32(Vec<i32>),
    Uint32(Vec<u32>),
    Int64(Vec<i64>),
    Uint64(Vec<u64>),
}

/// Input tensor data for ONNX execution
pub struct OnnxInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: TensorData,
}

/// Output tensor with actual data
pub struct OnnxOutputWithData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn run_onnx_zeroed(
    model_bytes: &[u8],
    _inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<OnnxOutput>, GraphError> {
    // Initialize ort global environment (only once per process)
    ensure_ort_initialized()?;

    let mut session = Session::builder()
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("session builder failed: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?
        .commit_from_memory(model_bytes)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("load model failed: {e}"),
        })?;

    // Build zero-filled inputs
    let mut input_values = Vec::new();
    for input_info in session.inputs().iter() {
        // Get shape from input type
        let shape: Vec<usize> = match input_info.dtype() {
            ort::value::ValueType::Tensor {
                ty: _,
                shape,
                dimension_symbols: _,
            } => shape.iter().map(|&d| d.max(1) as usize).collect(),
            _ => {
                return Err(GraphError::OnnxRuntimeFailed {
                    reason: format!("input '{}' is not a tensor", input_info.name()),
                });
            }
        };

        let total: usize = shape.iter().product();
        let zeros = vec![0f32; total.max(1)];

        // Convert shape to Vec<i64> for ort compatibility
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();

        let tensor = Value::from_array((shape_i64.as_slice(), zeros)).map_err(|e| {
            GraphError::OnnxRuntimeFailed {
                reason: format!(
                    "failed to create input tensor for {}: {e}",
                    input_info.name()
                ),
            }
        })?;
        input_values.push(tensor.into_dyn());
    }

    // Run inference - convert to Vec of SessionInputValue
    let input_session_values: Vec<SessionInputValue> = input_values
        .into_iter()
        .map(SessionInputValue::from)
        .collect();
    let outputs = session.run(input_session_values.as_slice()).map_err(|e| {
        GraphError::OnnxRuntimeFailed {
            reason: format!("run failed: {e}"),
        }
    })?;

    // Extract output metadata
    let mut results = Vec::new();
    for (idx, (_name, value)) in outputs.iter().enumerate() {
        // Get tensor shape and type
        let (shape, _data) =
            value
                .try_extract_tensor::<f32>()
                .map_err(|e| GraphError::OnnxRuntimeFailed {
                    reason: format!("failed to extract tensor: {e}"),
                })?;

        let shape_vec: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
        results.push(OnnxOutput {
            name: format!("output_{idx}"),
            shape: shape_vec,
            data_type: "f32".to_string(),
        });
    }
    Ok(results)
}

/// Run ONNX model with actual input tensors and return output tensors with data
pub fn run_onnx_with_inputs(
    model_bytes: &[u8],
    inputs: Vec<OnnxInput>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    // Initialize ort global environment (only once per process)
    ensure_ort_initialized()?;

    let mut session = Session::builder()
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("session builder failed: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?
        .commit_from_memory(model_bytes)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("load model failed: {e}"),
        })?;

    // Extract output names for later use
    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();

    // Build input tensors from provided inputs
    let mut input_session_values: Vec<SessionInputValue> = Vec::new();
    for input in inputs {
        let session_value = match input.data {
            TensorData::Float32(data) => {
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create float32 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Float16(data) => {
                // Convert u16 bits to half::f16
                let f16_data: Vec<half::f16> = data
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits))
                    .collect();
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), f16_data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create float16 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Int8(data) => {
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int8 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint8(data) => {
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint8 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Int32(data) => {
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int32 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint32(data) => {
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint32 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Int64(data) => {
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int64 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint64(data) => {
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint64 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
        };

        input_session_values.push(session_value);
    }

    // Run inference
    let outputs = session.run(input_session_values.as_slice()).map_err(|e| {
        GraphError::OnnxRuntimeFailed {
            reason: format!("run failed: {e}"),
        }
    })?;

    // Extract output tensors with data
    let mut results = Vec::new();
    for (idx, (_name, value)) in outputs.iter().enumerate() {
        let name = output_names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("output_{}", idx));

        // Try to extract tensor with different types
        // The order matches most common types first for performance
        let (shape_vec, data_vec) = if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            (shape_vec, data.to_vec())
        } else if let Ok((shape, data)) = value.try_extract_tensor::<half::f16>() {
            let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let data_vec: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
            (shape_vec, data_vec)
        } else if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
            let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let data_vec: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            (shape_vec, data_vec)
        } else if let Ok((shape, data)) = value.try_extract_tensor::<u32>() {
            let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let data_vec: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            (shape_vec, data_vec)
        } else if let Ok((shape, data)) = value.try_extract_tensor::<i8>() {
            let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let data_vec: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            (shape_vec, data_vec)
        } else if let Ok((shape, data)) = value.try_extract_tensor::<u8>() {
            let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let data_vec: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            (shape_vec, data_vec)
        } else if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
            let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let data_vec: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            (shape_vec, data_vec)
        } else if let Ok((shape, data)) = value.try_extract_tensor::<u64>() {
            let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let data_vec: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            (shape_vec, data_vec)
        } else {
            return Err(GraphError::OnnxRuntimeFailed {
                reason: "failed to extract output tensor: unsupported data type".to_string(),
            });
        };

        results.push(OnnxOutputWithData {
            name,
            shape: shape_vec,
            data: data_vec,
        });
    }

    Ok(results)
}

/// Device-resident tensor implementation for ONNX Runtime
///
/// This enables zero-copy execution by keeping tensors on device (GPU/NPU)
/// across multiple inference steps, eliminating host round-trips for
/// iterative workloads like KV cache in GenAI models.
#[derive(Debug)]
pub struct OrtDeviceTensor {
    /// ONNX Runtime value stored on device
    value: Value,
    /// Session reference to keep it alive
    _session: std::sync::Arc<Session>,
    /// Data type
    dtype: crate::graph::DataType,
    /// Tensor shape
    shape: Vec<usize>,
    /// Device kind (CPU, CUDA, etc.)
    device: crate::tensor::DeviceKind,
}

impl OrtDeviceTensor {
    /// Create a new device tensor with the given shape and data type
    pub fn new(
        session: std::sync::Arc<Session>,
        shape: Vec<usize>,
        dtype: crate::graph::DataType,
        device: crate::tensor::DeviceKind,
    ) -> Result<Self, GraphError> {
        // Convert shape to i64 for ort compatibility
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();

        // Create zero-filled tensor on device
        // Currently only supporting f32, will expand to other types
        let total_elements: usize = shape.iter().product();
        let zeros = vec![0.0f32; total_elements.max(1)];

        let value = Value::from_array((shape_i64.as_slice(), zeros))
            .map_err(|e| GraphError::DeviceTensorFailed {
                reason: format!("failed to create device tensor: {e}"),
            })?
            .into();

        Ok(Self {
            value,
            _session: session,
            dtype,
            shape,
            device,
        })
    }

    /// Get a reference to the underlying ORT value
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Get a mutable reference to the underlying ORT value
    pub fn value_mut(&mut self) -> &mut Value {
        &mut self.value
    }
}

impl crate::tensor::DeviceTensorBackend for OrtDeviceTensor {
    fn dtype(&self) -> crate::graph::DataType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn device_kind(&self) -> crate::tensor::DeviceKind {
        self.device
    }

    fn backend_kind(&self) -> crate::tensor::BackendKind {
        match self.device {
            crate::tensor::DeviceKind::Cpu => crate::tensor::BackendKind::OnnxCpu,
            crate::tensor::DeviceKind::Cuda => crate::tensor::BackendKind::OnnxGpu,
            crate::tensor::DeviceKind::DirectML => crate::tensor::BackendKind::OnnxGpu,
            crate::tensor::DeviceKind::CoreML => crate::tensor::BackendKind::OnnxCpu,
        }
    }

    fn read_to_host(&self) -> Result<Vec<f32>, GraphError> {
        // Extract tensor data from device to host
        // Currently only supporting f32, will expand to other types
        match self.dtype {
            crate::graph::DataType::Float32 => {
                let (_, data) = self.value.try_extract_tensor::<f32>().map_err(|e| {
                    GraphError::DeviceTensorFailed {
                        reason: format!("failed to read f32 tensor from device: {e}"),
                    }
                })?;
                Ok(data.to_vec())
            }
            crate::graph::DataType::Float16 => {
                let (_, data) = self.value.try_extract_tensor::<half::f16>().map_err(|e| {
                    GraphError::DeviceTensorFailed {
                        reason: format!("failed to read f16 tensor from device: {e}"),
                    }
                })?;
                // Convert f16 to f32
                Ok(data.iter().map(|&x| x.to_f32()).collect())
            }
            crate::graph::DataType::Int32 => {
                let (_, data) = self.value.try_extract_tensor::<i32>().map_err(|e| {
                    GraphError::DeviceTensorFailed {
                        reason: format!("failed to read i32 tensor from device: {e}"),
                    }
                })?;
                // Convert i32 to f32
                Ok(data.iter().map(|&x| x as f32).collect())
            }
            _ => Err(GraphError::DeviceTensorFailed {
                reason: format!("unsupported data type for device tensor: {:?}", self.dtype),
            }),
        }
    }

    fn write_from_host(&mut self, data: &[f32]) -> Result<(), GraphError> {
        // Write tensor data from host to device
        // Note: ORT doesn't support in-place writes, so we recreate the value
        let expected_size: usize = self.shape.iter().product();
        if data.len() != expected_size {
            return Err(GraphError::DeviceTensorFailed {
                reason: format!(
                    "data size mismatch: expected {} elements, got {}",
                    expected_size,
                    data.len()
                ),
            });
        }

        // Convert shape to i64 for ort compatibility
        let shape_i64: Vec<i64> = self.shape.iter().map(|&d| d as i64).collect();

        // Create new value from host data
        // Currently only supporting f32, will expand to other types
        match self.dtype {
            crate::graph::DataType::Float32 => {
                self.value = Value::from_array((shape_i64.as_slice(), data.to_vec()))
                    .map_err(|e| GraphError::DeviceTensorFailed {
                        reason: format!("failed to write f32 tensor to device: {e}"),
                    })?
                    .into();
                Ok(())
            }
            crate::graph::DataType::Float16 => {
                // Convert f32 to f16
                let f16_data: Vec<half::f16> =
                    data.iter().map(|&x| half::f16::from_f32(x)).collect();
                self.value = Value::from_array((shape_i64.as_slice(), f16_data))
                    .map_err(|e| GraphError::DeviceTensorFailed {
                        reason: format!("failed to write f16 tensor to device: {e}"),
                    })?
                    .into();
                Ok(())
            }
            crate::graph::DataType::Int32 => {
                // Convert f32 to i32
                let i32_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                self.value = Value::from_array((shape_i64.as_slice(), i32_data))
                    .map_err(|e| GraphError::DeviceTensorFailed {
                        reason: format!("failed to write i32 tensor to device: {e}"),
                    })?
                    .into();
                Ok(())
            }
            _ => Err(GraphError::DeviceTensorFailed {
                reason: format!("unsupported data type for device tensor: {:?}", self.dtype),
            }),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Run ONNX model with device tensor bindings (zero-copy execution)
///
/// This function uses ONNX Runtime IoBinding to execute the model with
/// device-resident tensors, eliminating host-device round-trips.
///
/// Note: This is a placeholder implementation. Full IoBinding support
/// will be added in a future update.
#[allow(dead_code)]
pub fn run_onnx_with_bindings(
    _session: &Session,
    _input_bindings: Vec<(&str, &OrtDeviceTensor)>,
    _output_bindings: Vec<(&str, &mut OrtDeviceTensor)>,
) -> Result<(), GraphError> {
    // Placeholder for future IoBinding implementation
    // Current dispatch() implementation uses regular compute path
    Err(GraphError::DeviceTensorFailed {
        reason: "IoBinding not yet implemented - use dispatch() instead".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{BackendKind, DeviceKind, DeviceTensorBackend};

    #[test]
    fn test_ort_device_tensor_lifecycle() {
        // This test requires ONNX Runtime to be initialized
        // Skip if initialization fails
        if ensure_ort_initialized().is_err() {
            return;
        }

        // Create a simple ONNX model (add operation: y = x + 1)
        // For now, we'll skip this test as it requires a real ONNX model
        // Will be added when we have a test model available
    }
}
