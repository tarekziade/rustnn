#![cfg(all(feature = "onnx-runtime"))]

use std::collections::HashMap;
use std::sync::Once;

use ort::session::SessionInputValue;

use half;
use ndarray::ArrayD;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

use crate::error::GraphError;
use crate::graph::OperandDescriptor;

static INIT: Once = Once::new();

fn ensure_ort_initialized() -> Result<(), GraphError> {
    let mut result = Ok(());
    INIT.call_once(|| {
        if let Err(e) = ort::init()
            .with_name("rustnn")
            .with_execution_providers([
                ort::execution_providers::CPUExecutionProvider::default().build()
            ])
            .commit()
        {
            result = Err(GraphError::OnnxRuntimeFailed {
                reason: format!("ort init failed: {e}"),
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
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?
        .commit_from_memory(model_bytes)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("load model failed: {e}"),
        })?;

    // Build zero-filled inputs
    let mut input_values = Vec::new();
    for input_info in session.inputs.iter() {
        // Get shape from input type
        let shape: Vec<usize> = match &input_info.input_type {
            ort::value::ValueType::Tensor {
                ty: _,
                shape,
                dimension_symbols: _,
            } => shape.iter().map(|&d| d.max(1) as usize).collect(),
            _ => {
                return Err(GraphError::OnnxRuntimeFailed {
                    reason: format!("input '{}' is not a tensor", input_info.name),
                });
            }
        };

        let total: usize = shape.iter().product();
        let zeros = vec![0f32; total.max(1)];

        let array = ArrayD::from_shape_vec(shape.clone(), zeros).map_err(|e| {
            GraphError::OnnxRuntimeFailed {
                reason: format!("failed to create input array for {}: {e}", input_info.name),
            }
        })?;

        let tensor = Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("failed to create input tensor for {}: {e}", input_info.name),
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
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?
        .commit_from_memory(model_bytes)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("load model failed: {e}"),
        })?;

    // Extract output names for later use
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();

    // Build input tensors from provided inputs
    let mut input_session_values: Vec<SessionInputValue> = Vec::new();
    for input in inputs {
        let session_value = match input.data {
            TensorData::Float32(data) => {
                let array = ArrayD::from_shape_vec(input.shape.clone(), data).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create float32 input array for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                let value =
                    Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create float32 input tensor for {}: {e}",
                            input.name
                        ),
                    })?;
                SessionInputValue::from(value)
            }
            TensorData::Float16(data) => {
                // Convert u16 bits to half::f16
                let f16_data: Vec<half::f16> = data
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits))
                    .collect();
                let array = ArrayD::from_shape_vec(input.shape.clone(), f16_data).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create float16 input array for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                let value =
                    Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create float16 input tensor for {}: {e}",
                            input.name
                        ),
                    })?;
                SessionInputValue::from(value)
            }
            TensorData::Int8(data) => {
                let array = ArrayD::from_shape_vec(input.shape.clone(), data).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int8 input array for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                let value =
                    Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int8 input tensor for {}: {e}",
                            input.name
                        ),
                    })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint8(data) => {
                let array = ArrayD::from_shape_vec(input.shape.clone(), data).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint8 input array for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                let value =
                    Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint8 input tensor for {}: {e}",
                            input.name
                        ),
                    })?;
                SessionInputValue::from(value)
            }
            TensorData::Int32(data) => {
                let array = ArrayD::from_shape_vec(input.shape.clone(), data).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int32 input array for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                let value =
                    Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int32 input tensor for {}: {e}",
                            input.name
                        ),
                    })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint32(data) => {
                let array = ArrayD::from_shape_vec(input.shape.clone(), data).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint32 input array for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                let value =
                    Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint32 input tensor for {}: {e}",
                            input.name
                        ),
                    })?;
                SessionInputValue::from(value)
            }
            TensorData::Int64(data) => {
                let array = ArrayD::from_shape_vec(input.shape.clone(), data).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int64 input array for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                let value =
                    Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int64 input tensor for {}: {e}",
                            input.name
                        ),
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
