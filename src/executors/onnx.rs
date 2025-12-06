#![cfg(all(feature = "onnx-runtime"))]

use std::collections::HashMap;

use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{ArrayD, IxDyn};
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::{GraphOptimizationLevel, LoggingLevel};

use crate::error::GraphError;
use crate::graph::OperandDescriptor;

#[derive(Debug, Clone)]
pub struct OnnxOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: String,
}

/// Input tensor data for ONNX execution
pub struct OnnxInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
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
    let env = Environment::builder()
        .with_name("rustnn")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("env build failed: {e}"),
        })?;

    let mut session = build_session(&env, model_bytes)?;
    let feeds = build_feeds(&session)?;

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session
        .run(feeds.into_iter().map(|(_, arr)| arr).collect::<Vec<_>>())
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("run failed: {e}"),
        })?;

    let mut results = Vec::new();
    for (idx, tensor) in outputs.into_iter().enumerate() {
        let shape: Vec<i64> = tensor.view().shape().iter().map(|d| *d as i64).collect();
        results.push(OnnxOutput {
            name: format!("output_{idx}"),
            shape,
            data_type: "f32".to_string(),
        });
    }
    Ok(results)
}

fn build_session<'a>(env: &'a Environment, model_bytes: &[u8]) -> Result<Session<'a>, GraphError> {
    let builder = env
        .new_session_builder()
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("session builder failed: {e}"),
        })?;
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?;
    builder
        .with_model_from_memory(model_bytes)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("load model failed: {e}"),
        })
}

fn build_feeds<'a>(session: &'a Session) -> Result<Vec<(&'a str, ArrayD<f32>)>, GraphError> {
    let mut feeds = Vec::new();
    for input_info in session.inputs.iter() {
        let name = input_info.name.as_str();
        let shape: Vec<usize> = input_info
            .dimensions
            .iter()
            .map(|d| d.unwrap_or(1) as usize)
            .collect();
        let total: usize = shape.iter().product();
        let zeros = vec![0f32; total.max(1)];
        let array = ArrayD::from_shape_vec(shape.clone(), zeros).map_err(|e| {
            GraphError::OnnxRuntimeFailed {
                reason: format!("shape build failed for {name}: {e}"),
            }
        })?;
        feeds.push((name, array));
    }
    Ok(feeds)
}

/// Run ONNX model with actual input tensors and return output tensors with data
pub fn run_onnx_with_inputs(
    model_bytes: &[u8],
    inputs: Vec<OnnxInput>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    let env = Environment::builder()
        .with_name("rustnn")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("env build failed: {e}"),
        })?;

    let mut session = build_session(&env, model_bytes)?;

    // Extract output names before running inference (before mutable borrow)
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();

    // Build array inputs from provided inputs
    let mut feed_arrays = Vec::new();
    for input in inputs {
        let array = ArrayD::from_shape_vec(IxDyn(&input.shape), input.data).map_err(|e| {
            GraphError::OnnxRuntimeFailed {
                reason: format!("shape build failed for {}: {e}", input.name),
            }
        })?;
        feed_arrays.push(array);
    }

    // Run inference
    let outputs: Vec<OrtOwnedTensor<f32, _>> =
        session
            .run(feed_arrays)
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("run failed: {e}"),
            })?;

    // Extract output tensors with data
    let mut results = Vec::new();
    for (idx, tensor) in outputs.into_iter().enumerate() {
        let shape: Vec<usize> = tensor.view().shape().iter().map(|d| *d as usize).collect();
        let data: Vec<f32> = tensor.view().iter().copied().collect();
        let name = output_names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("output_{}", idx));

        results.push(OnnxOutputWithData { name, shape, data });
    }

    Ok(results)
}
