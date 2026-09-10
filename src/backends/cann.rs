/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::fmt;

use crate::GraphInfo;
use crate::backend_selection::DeviceType;
use crate::converters::cann::encode_via_adapter;
use crate::error::{Error, Result};
use crate::mlcontext::MLBackendGraph::CannEngine;
use crate::mlcontext::{
    ListDevices, MLBackendBuilder, MLBackendContext, MLGraph, MLNamedTensors, MLTensor,
    MLTensorDescriptor, RustNNOptions,
};
#[cfg(feature = "cann-runtime")]
use crate::operator_enums::MLOperandDataType;
#[cfg(feature = "cann-runtime")]
use hiai_rs::{Session, TensorDesc};

/// Map WebNN operand data type to CANN adapter enum
#[cfg(feature = "cann-runtime")]
fn ml_operand_to_cann_dtype(data_type: MLOperandDataType) -> i32 {
    match data_type {
        MLOperandDataType::Float32 => 0, // CANN_DT_FLOAT
        MLOperandDataType::Float16 => 1, // CANN_DT_FLOAT16
        MLOperandDataType::Int32 => 3,   // CANN_DT_INT32
        MLOperandDataType::Int8 => 2,    // CANN_DT_INT8
        MLOperandDataType::Uint8 => 4,   // CANN_DT_UINT8
        MLOperandDataType::Int64 => 9,   // CANN_DT_INT64
        MLOperandDataType::Uint32 => 8,  // CANN_DT_UINT32
        _ => 0,                          // default CANN_DT_FLOAT
    }
}

#[derive(Debug)]
pub(crate) struct CannTensor {
    memory: Vec<u8>,
}

#[derive(Debug)]
// Fields are only read by the runtime `dispatch`; the mock build never
// constructs a graph.
#[allow(dead_code)]
pub(crate) struct CannGraph {
    pub(crate) model_bytes: Vec<u8>,
    // Input/output names in the model's canonical order (matching how the
    // graph was compiled). dispatch() relies on this to feed tensors to the
    // NPU positionally, since MLNamedTensors (a BTreeMap) sorts by name rather
    // than the model's operand order.
    pub(crate) input_names: Vec<String>,
    pub(crate) output_names: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct CannContext {
    tensors: Vec<CannTensor>,
    _device_type: DeviceType,
    /// Loaded models, keyed by their compiled offline model bytes, so repeated
    /// dispatches of the same graph skip the expensive `Session::load`. Each
    /// session is wrapped in a `Mutex` because `Session` is `Send` but not
    /// `Sync` (its `dispatch` mutates the DDK model manager internally), while
    /// `CannContext` must satisfy the `Sync` bound of `MLBackendContext`.
    #[cfg(feature = "cann-runtime")]
    sessions: std::collections::HashMap<Vec<u8>, std::sync::Mutex<Session>>,
}

impl CannContext {
    pub(crate) fn new_from_device_type(
        device_type: DeviceType,
        _options: Option<&RustNNOptions>,
    ) -> Result<Self> {
        Ok(Self {
            tensors: Vec::new(),
            _device_type: device_type,
            #[cfg(feature = "cann-runtime")]
            sessions: std::collections::HashMap::new(),
        })
    }
}

impl ListDevices for CannContext {
    fn list_devices() -> Vec<crate::backend_selection::BackendDevice> {
        vec![crate::backend_selection::BackendDevice::Cann {
            device_type: crate::backend_selection::DeviceType::Npu,
        }]
    }
}

impl<'context> MLBackendContext<'context> for CannContext {
    fn accelerated(&self) -> bool {
        true
    }

    fn create_builder<'builder>(
        &mut self,
    ) -> Result<Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder,
    {
        Ok(Box::new(CannBuilder { graph: None }))
    }

    fn create_tensor(&mut self, descriptor: &MLTensorDescriptor) -> Result<MLTensor> {
        let byte_count = descriptor.rustnn_required_bytes();
        let memory = vec![0u8; byte_count];
        self.tensors.push(CannTensor { memory });
        Ok(MLTensor {
            id: self.tensors.len() - 1,
            constant: false,
            descriptor: descriptor.clone(),
        })
    }

    fn create_constant_tensor(
        &mut self,
        descriptor: &MLTensorDescriptor,
        input_data: &[u8],
    ) -> Result<MLTensor> {
        let mut tensor = self.create_tensor(descriptor)?;
        tensor.constant = true;
        self.write_tensor(&tensor, input_data).map_err(|e| {
            crate::error::Error::TensorCreationError {
                source: e.into(),
                descriptor: descriptor.clone(),
            }
        })?;
        Ok(tensor)
    }

    fn read_tensor(&mut self, tensor: &MLTensor, array: &mut [u8]) -> Result<()> {
        let host = &self.tensors[tensor.id].memory;
        let logical = tensor.rustnn_required_bytes();
        if array.len() < logical {
            return Err(crate::error::Error::TensorReadError {
                source: format!(
                    "buffer too small: need {} logical bytes, got {}",
                    logical,
                    array.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        let slice = host
            .get(..logical)
            .ok_or_else(|| crate::error::Error::TensorReadError {
                source: format!("tensor storage shorter than logical size ({logical} bytes)")
                    .into(),
                tensor: tensor.clone(),
            })?;
        array[..logical].copy_from_slice(slice);
        Ok(())
    }

    fn write_tensor(&mut self, tensor: &MLTensor, array: &[u8]) -> Result<()> {
        let host = &mut self.tensors[tensor.id].memory;
        if array.len() > host.len() {
            return Err(crate::error::Error::TensorWriteError {
                source: format!(
                    "write exceeds tensor storage: {} bytes > {}",
                    array.len(),
                    host.len()
                )
                .into(),
                tensor: tensor.clone(),
            });
        }
        let byte_len = array.len();
        host[..byte_len].copy_from_slice(array);
        Ok(())
    }

    #[cfg(feature = "cann-runtime")]
    fn dispatch(
        &mut self,
        graph: &mut MLGraph,
        inputs: &MLNamedTensors,
        outputs: &MLNamedTensors,
    ) -> Result<()> {
        let cann_graph = if let CannEngine(ref cann_graph) = graph.backend {
            cann_graph
        } else {
            return Err(Error::GraphDispatchError {
                source: "graph is not a CANN graph".into(),
            });
        };

        let model_bytes = &cann_graph.model_bytes;

        let build_desc = |tensor: &&MLTensor| -> TensorDesc {
            TensorDesc {
                data: self.tensors[tensor.id].memory.clone(),
                shape: tensor.shape().iter().map(|dim| *dim as u32).collect(),
                dtype: ml_operand_to_cann_dtype(tensor.data_type()),
            }
        };

        // Feed tensors to the NPU positionally, in the model's canonical
        // input/output order (BTreeMap sorts by name).
        let mut input_descs = Vec::with_capacity(cann_graph.input_names.len());
        for name in &cann_graph.input_names {
            let tensor = inputs
                .get(name.as_str())
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!("missing input '{name}' for CANN dispatch").into(),
                })?;
            input_descs.push(build_desc(tensor));
        }

        let mut output_descs = Vec::with_capacity(cann_graph.output_names.len());
        for name in &cann_graph.output_names {
            let tensor = outputs
                .get(name.as_str())
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!("missing output '{name}' for CANN dispatch").into(),
                })?;
            output_descs.push(build_desc(tensor));
        }

        // Reuse a previously loaded session for this model; load once on the
        // first dispatch and cache it (the model load dominates per-call cost).
        let session = if let Some(session) = self.sessions.get(model_bytes) {
            session
        } else {
            let session = Session::load(model_bytes).map_err(|e| Error::GraphDispatchError {
                source: Box::new(e),
            })?;
            self.sessions
                .insert(model_bytes.clone(), std::sync::Mutex::new(session));
            self.sessions
                .get(model_bytes)
                .expect("session was just inserted")
        };

        session
            .lock()
            .expect("session mutex poisoned")
            .dispatch(&input_descs, &mut output_descs)
            .map_err(|e| Error::GraphDispatchError {
                source: Box::new(e),
            })?;

        for (name, output_desc) in cann_graph.output_names.iter().zip(output_descs.iter()) {
            let tensor = outputs
                .get(name.as_str())
                .ok_or_else(|| Error::GraphDispatchError {
                    source: format!("missing output '{name}' for CANN dispatch").into(),
                })?;
            // hiai-rs `dispatch` truncates `output_desc.data` to the actual
            // produced size, so resize the destination to match before copying
            // (avoids a length-mismatch panic when the model output is smaller
            // than the pre-allocated tensor buffer).
            let mem = &mut self.tensors[tensor.id].memory;
            mem.truncate(output_desc.data.len());
            mem.copy_from_slice(&output_desc.data);
        }

        Ok(())
    }

    #[cfg(not(feature = "cann-runtime"))]
    fn dispatch(
        &mut self,
        _graph: &mut MLGraph,
        _inputs: &MLNamedTensors,
        _outputs: &MLNamedTensors,
    ) -> Result<()> {
        Err(Error::GraphDispatchError {
            source: "CANN shim not available (mock mode)".into(),
        })
    }

    fn rustnn_resize_tensor(&mut self, _tensor: &mut MLTensor, _new_shape: &[u64]) -> Result<()> {
        Ok(())
    }

    fn rustnn_set_tensor_capacity(
        &mut self,
        _tensor: &mut MLTensor,
        _max_shape: &[u64],
    ) -> Result<()> {
        Ok(())
    }
}

pub(crate) struct CannBuilder {
    graph: Option<GraphInfo>,
}

impl fmt::Debug for CannBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CannBuilder")
            .field("has_graph", &self.graph.is_some())
            .finish()
    }
}

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for CannBuilder {
    fn build(&mut self, graph_info: GraphInfo) -> Result<MLGraph<'context>> {
        // Build the CANN graph and compile to offline model bytes.
        let model_bytes =
            encode_via_adapter(&graph_info).map_err(|e| Error::GraphDispatchError {
                source: format!("CANN graph build failed: {e}").into(),
            })?;

        // Record the input/output names in the model's canonical order. Names
        // are guaranteed present: MLGraph::new() below runs io_binding_maps(),
        // which errors on missing or duplicate names.
        let input_names = graph_info
            .input_operands
            .iter()
            .map(|&id| {
                graph_info.operands[id as usize]
                    .name
                    .clone()
                    .expect("input name validated by io_binding_maps")
            })
            .collect();
        let output_names = graph_info
            .output_operands
            .iter()
            .map(|&id| {
                graph_info.operands[id as usize]
                    .name
                    .clone()
                    .expect("output name validated by io_binding_maps")
            })
            .collect();

        let graph = CannGraph {
            model_bytes,
            input_names,
            output_names,
        };
        MLGraph::new(CannEngine(graph), &graph_info)
    }
}

#[cfg(test)]
mod tests {
    use super::CannContext;
    use crate::backend_selection::DeviceType;
    use crate::mlcontext::{MLBackendContext, MLTensorDescriptor};
    use crate::operator_enums::MLOperandDataType;

    #[test]
    fn test_context_new() {
        let context = CannContext::new_from_device_type(DeviceType::Npu, None).unwrap();
        assert!(context.accelerated());
    }

    #[test]
    fn test_create_tensor() {
        let mut context = CannContext::new_from_device_type(DeviceType::Npu, None).unwrap();
        let mut desc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        desc.set_readable(true);
        desc.set_writable(true);
        let tensor = context.create_tensor(&desc).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_write_and_read_tensor() {
        let mut context = CannContext::new_from_device_type(DeviceType::Npu, None).unwrap();
        let mut desc = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2]);
        desc.set_readable(true);
        desc.set_writable(true);
        let tensor = context.create_tensor(&desc).unwrap();

        let upload = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut download = vec![0.0f32; 4];
        context
            .write_tensor(&tensor, bytemuck::cast_slice(&upload))
            .unwrap();
        context
            .read_tensor(&tensor, bytemuck::cast_slice_mut(&mut download))
            .unwrap();
        assert_eq!(&upload, &download);
    }
}
