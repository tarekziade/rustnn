/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use crate::debug_print;
use crate::error::GraphError;
use crate::graph::{
    ConstantData, DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor,
    OperandKind, to_dimension_vector,
};
use crate::operator_enums::MLOperandDataType;
use crate::operators::Operation;
use std::collections::{BTreeMap, HashMap};
use webnn_graph::ast::{ConstDecl, ConstInit, GraphJson, Node, OperandDesc};

/// The name used for an operand in the exported AST (and, for constants, as the safetensors tensor
/// key and `@weights(...)` reference). Falls back to `operand_<idx>` when the operand is unnamed.
///
/// Shared by the AST builder and the `.webnn`/`.safetensors` exporter so both agree on the scheme.
pub(crate) fn operand_export_name(operand: &Operand, idx: usize) -> String {
    operand
        .name
        .clone()
        .unwrap_or_else(|| format!("operand_{}", idx))
}

/// Maps operand id -> the name it should be exported under, overriding [`operand_export_name`].
///
/// Used by the builder's save path to name graph outputs without mutating the operands: the same
/// override must apply everywhere the operand is referenced (node inputs, node outputs, and the
/// graph `outputs` section) so all references stay consistent.
pub(crate) type OutputNameOverrides = HashMap<u32, String>;

/// Like [`operand_export_name`], but consults `overrides` first so callers can rename operands
/// (e.g. mark them as named graph outputs) without touching the `GraphInfo`.
fn operand_export_name_with_overrides(
    operand: &Operand,
    idx: usize,
    overrides: Option<&OutputNameOverrides>,
) -> String {
    if let Some(name) = overrides.and_then(|o| o.get(&(idx as u32))) {
        return name.clone();
    }
    operand_export_name(operand, idx)
}

/// Convert our DataType to webnn-graph DataType
fn to_webnn_datatype(dt: &DataType) -> webnn_graph::ast::DataType {
    match dt {
        DataType::Int4 => webnn_graph::ast::DataType::Int4,
        DataType::Uint4 => webnn_graph::ast::DataType::Uint4,
        DataType::Float32 => webnn_graph::ast::DataType::Float32,
        DataType::Float16 => webnn_graph::ast::DataType::Float16,
        DataType::Int32 => webnn_graph::ast::DataType::Int32,
        DataType::Uint32 => webnn_graph::ast::DataType::Uint32,
        DataType::Int64 => webnn_graph::ast::DataType::Int64,
        DataType::Uint64 => webnn_graph::ast::DataType::Uint64,
        DataType::Int8 => webnn_graph::ast::DataType::Int8,
        DataType::Uint8 => webnn_graph::ast::DataType::Uint8,
    }
}

/// Convert webnn-graph DataType to our DataType
fn from_webnn_datatype(dt: &webnn_graph::ast::DataType) -> DataType {
    match dt {
        webnn_graph::ast::DataType::Float32 => DataType::Float32,
        webnn_graph::ast::DataType::Float16 => DataType::Float16,
        webnn_graph::ast::DataType::Int4 => DataType::Int4,
        webnn_graph::ast::DataType::Uint4 => DataType::Uint4,
        webnn_graph::ast::DataType::Int32 => DataType::Int32,
        webnn_graph::ast::DataType::Uint32 => DataType::Uint32,
        webnn_graph::ast::DataType::Int64 => DataType::Int64,
        webnn_graph::ast::DataType::Uint64 => DataType::Uint64,
        webnn_graph::ast::DataType::Int8 => DataType::Int8,
        webnn_graph::ast::DataType::Uint8 => DataType::Uint8,
    }
}

fn to_webnn_dimension(dim: &Dimension) -> webnn_graph::ast::Dimension {
    match dim {
        Dimension::Static(v) => webnn_graph::ast::Dimension::Static(*v),
        Dimension::Dynamic(d) => {
            webnn_graph::ast::Dimension::Dynamic(webnn_graph::ast::DynamicDimension {
                name: d.name.clone(),
                max_size: d.max_size,
            })
        }
    }
}

fn from_webnn_dimension(dim: &webnn_graph::ast::Dimension) -> Dimension {
    match dim {
        webnn_graph::ast::Dimension::Static(v) => Dimension::Static(*v),
        webnn_graph::ast::Dimension::Dynamic(d) => Dimension::Dynamic(DynamicDimension {
            name: d.name.clone(),
            max_size: d.max_size,
        }),
    }
}

/// Build one [`webnn_graph::ast::Node`] for the operation at `op_index`, matching the shape produced by [`to_graph_json`].
///
/// Useful for diagnostics (for example ONNX conversion failures) without serializing the full graph.
pub fn graph_operation_to_webnn_node(
    graph: &GraphInfo,
    op_index: usize,
) -> Result<Node, GraphError> {
    graph_operation_to_webnn_node_with_overrides(graph, op_index, None)
}

/// Same as [`graph_operation_to_webnn_node`], but operand names can be overridden (see
/// [`OutputNameOverrides`]). Used by the save path to name graph outputs without mutating operands.
fn graph_operation_to_webnn_node_with_overrides(
    graph: &GraphInfo,
    op_index: usize,
    overrides: Option<&OutputNameOverrides>,
) -> Result<Node, GraphError> {
    let operation = graph
        .operations
        .get(op_index)
        .ok_or_else(|| GraphError::ConversionFailed {
            format: "webnn-graph-json".to_string(),
            reason: format!("operation index {op_index} out of range"),
        })?;
    let id = format!("op_{}", op_index);

    let input_names: Vec<String> = operation
        .input_operands()
        .iter()
        .map(|&idx| {
            operand_export_name_with_overrides(
                &graph.operands[idx as usize],
                idx as usize,
                overrides,
            )
        })
        .collect();

    let output_operands = operation.output_operands_slice();
    let output_names: Option<Vec<String>> = if output_operands.is_empty() {
        None
    } else {
        Some(
            output_operands
                .iter()
                .map(|&idx| {
                    operand_export_name_with_overrides(
                        &graph.operands[idx as usize],
                        idx as usize,
                        overrides,
                    )
                })
                .collect(),
        )
    };

    let mut options: serde_json::Map<String, serde_json::Value> = operation
        .attributes_value()
        .as_object()
        .cloned()
        .unwrap_or_else(serde_json::Map::new);
    options.remove("kind");

    Ok(Node {
        id,
        op: operation.op_type().to_string(),
        inputs: input_names,
        options,
        outputs: output_names,
    })
}

/// Controls how constant operands are represented when building the AST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstExport {
    /// Embed constant bytes directly in the AST as `ConstInit::InlineBytes` (clones the data).
    Inline,
    /// Emit a `ConstInit::Weights { ref }` reference instead, leaving the bytes out of the AST.
    /// Callers are responsible for writing the weights out (e.g. to a sidecar safetensors file).
    ExternalWeights,
}

/// Convert GraphInfo to GraphJson, embedding constant data inline.
pub fn to_graph_json(graph: &GraphInfo, quantized: bool) -> Result<GraphJson, GraphError> {
    to_graph_json_with_consts(graph, quantized, ConstExport::Inline, None)
}

/// Convert GraphInfo to GraphJson, choosing how constants are represented.
///
/// With [`ConstExport::ExternalWeights`] the constant bytes are not copied into the AST; each
/// constant becomes a `@weights(...)` reference keyed by the same name used here (`operand.name`,
/// else `operand_<idx>`). This is the zero-copy path used by the `.webnn`/`.safetensors` exporter.
///
/// When `output_override` is `Some`, the graph `outputs` section (and the corresponding operand
/// names in nodes) is taken from the override map instead of `graph.output_operands`. This lets the
/// builder export a graph's outputs without mutating operand kinds/names first. When `None`, outputs
/// are read from `graph.output_operands` using each operand's own name.
pub(crate) fn to_graph_json_with_consts(
    graph: &GraphInfo,
    quantized: bool,
    const_export: ConstExport,
    output_override: Option<&OutputNameOverrides>,
) -> Result<GraphJson, GraphError> {
    let mut inputs = BTreeMap::new();
    let mut consts = BTreeMap::new();
    let mut nodes = Vec::new();
    let mut outputs = BTreeMap::new();

    // Process operands - separate inputs and constants
    for (idx, operand) in graph.operands.iter().enumerate() {
        let name = operand_export_name(operand, idx);

        match &operand.kind {
            OperandKind::Input => {
                inputs.insert(
                    name,
                    OperandDesc {
                        data_type: to_webnn_datatype(&operand.descriptor.data_type),
                        shape: operand
                            .descriptor
                            .shape
                            .iter()
                            .map(to_webnn_dimension)
                            .collect(),
                    },
                );
            }
            OperandKind::Constant => {
                // Get constant data from the map
                if let Some(constant) = graph.constant_operand_ids_to_handles.get(&(idx as u32)) {
                    let const_shape =
                        operand
                            .descriptor
                            .static_shape()
                            .ok_or(GraphError::ConversionFailed {
                                format: "webnn-graph-json".to_string(),
                                reason: format!(
                                    "constant operand {} has dynamic shape",
                                    operand.name.as_deref().unwrap_or("unknown")
                                ),
                            })?;
                    let init = match const_export {
                        ConstExport::Inline => ConstInit::InlineBytes {
                            bytes: constant.data.clone(),
                        },
                        ConstExport::ExternalWeights => ConstInit::Weights {
                            r#ref: name.clone(),
                        },
                    };

                    consts.insert(
                        name,
                        ConstDecl {
                            data_type: to_webnn_datatype(&operand.descriptor.data_type),
                            shape: const_shape,
                            init,
                        },
                    );
                }
            }
            OperandKind::Intermediate => {
                // Intermediates are not in graph json
            }
            OperandKind::Output => {
                // Outputs are handled separately below
            }
        }
    }

    // Process operations
    for op_idx in 0..graph.operations.len() {
        nodes.push(graph_operation_to_webnn_node_with_overrides(
            graph,
            op_idx,
            output_override,
        )?);
    }

    // Process outputs. Prefer the caller-supplied override (builder save path) so we don't need to
    // mutate operands; otherwise read the finalized `graph.output_operands`.
    match output_override {
        Some(override_names) => {
            for (&operand_idx, name) in override_names {
                if graph.operands.get(operand_idx as usize).is_some() {
                    outputs.insert(name.clone(), name.clone());
                }
            }
        }
        None => {
            for &operand_idx in &graph.output_operands {
                if let Some(operand) = graph.operands.get(operand_idx as usize) {
                    let name = operand_export_name(operand, operand_idx as usize);
                    outputs.insert(name.clone(), name);
                }
            }
        }
    }

    Ok(GraphJson {
        name: Some("graph".to_string()),
        format: "webnn-graph-json".to_string(),
        version: 2,
        quantized: graph.quantized || quantized,
        inputs,
        consts,
        nodes,
        outputs,
    })
}

/// Convert GraphJson to GraphInfo
pub fn from_graph_json(graph_json: &GraphJson) -> Result<GraphInfo, GraphError> {
    let mut operands = Vec::new();
    let mut operations = Vec::new();
    let mut operand_map: BTreeMap<String, u32> = BTreeMap::new();
    let mut constant_operand_ids_to_handles: HashMap<u32, ConstantData> = HashMap::new();
    let mut input_operands = Vec::new();
    let mut output_operands = Vec::new();

    // Process inputs
    for (name, desc) in &graph_json.inputs {
        let idx = operands.len() as u32;
        operand_map.insert(name.clone(), idx);
        input_operands.push(idx);

        operands.push(Operand {
            name: Some(name.clone()),
            descriptor: OperandDescriptor {
                data_type: from_webnn_datatype(&desc.data_type),
                shape: desc.shape.iter().map(from_webnn_dimension).collect(),
                pending_permutation: Vec::new(),
            },
            kind: OperandKind::Input,
        });
    }

    // Process constants
    for (name, const_decl) in &graph_json.consts {
        let idx = operands.len() as u32;
        operand_map.insert(name.clone(), idx);

        // Convert ConstInit to Vec<u8>
        let data = match &const_decl.init {
            ConstInit::InlineBytes { bytes } => bytes.clone(),
            // Allow weight references: downstream loaders will resolve these using the manifest/weights.
            ConstInit::Weights { r#ref: _ } => Vec::new(),
            ConstInit::Scalar { value } => {
                // Convert scalar to repeated bytes based on shape and data type
                let element_count: usize = const_decl.shape.iter().map(|&x| x as usize).product();
                let dt = from_webnn_datatype(&const_decl.data_type);

                // Parse scalar value as f32 (most common case)
                let scalar_f32: f32 = if let Some(num) = value.as_f64() {
                    num as f32
                } else if let Some(num) = value.as_i64() {
                    num as f32
                } else if let Some(num) = value.as_u64() {
                    num as f32
                } else {
                    return Err(GraphError::ConversionFailed {
                        format: "webnn-graph-json".to_string(),
                        reason: format!("Cannot parse scalar value: {:?}", value),
                    });
                };

                // Create repeated bytes based on data type
                let mut bytes = Vec::new();
                for _ in 0..element_count {
                    match dt {
                        DataType::Int4 | DataType::Uint4 => {
                            return Err(GraphError::ConversionFailed {
                                format: "webnn-graph-json".to_string(),
                                reason: "int4/uint4 constants not supported in scalar export"
                                    .to_string(),
                            });
                        }
                        DataType::Float32 => bytes.extend_from_slice(&scalar_f32.to_le_bytes()),
                        DataType::Float16 => {
                            let f16_bits = half::f16::from_f32(scalar_f32).to_bits();
                            bytes.extend_from_slice(&f16_bits.to_le_bytes());
                        }
                        DataType::Int32 => {
                            bytes.extend_from_slice(&(scalar_f32 as i32).to_le_bytes())
                        }
                        DataType::Uint32 => {
                            bytes.extend_from_slice(&(scalar_f32 as u32).to_le_bytes())
                        }
                        DataType::Int64 => {
                            bytes.extend_from_slice(&(scalar_f32 as i64).to_le_bytes())
                        }
                        DataType::Uint64 => {
                            bytes.extend_from_slice(&(scalar_f32 as u64).to_le_bytes())
                        }
                        DataType::Int8 => bytes.push(scalar_f32 as i8 as u8),
                        DataType::Uint8 => bytes.push(scalar_f32 as u8),
                    }
                }
                bytes
            }
        };

        operands.push(Operand {
            name: Some(name.clone()),
            descriptor: OperandDescriptor {
                data_type: from_webnn_datatype(&const_decl.data_type),
                shape: to_dimension_vector(&const_decl.shape),
                pending_permutation: Vec::new(),
            },
            kind: OperandKind::Constant,
        });

        constant_operand_ids_to_handles.insert(idx, ConstantData { data, label: None });
    }

    // Process nodes (operations)
    for node in &graph_json.nodes {
        // Resolve all input names to operand indices
        let resolved_inputs: Vec<u32> = node
            .inputs
            .iter()
            .map(|name| {
                operand_map
                    .get(name)
                    .copied()
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "webnn-graph-json".to_string(),
                        reason: format!("Input operand '{}' not found", name),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let input_operands: Vec<u32> = resolved_inputs.clone();

        // Determine output names from node.outputs or node.id
        let output_names_list: Vec<String> = if let Some(output_names) = &node.outputs {
            output_names.clone()
        } else {
            // If outputs is None, use the node ID as the output name
            // This is common in .webnn DSL where `sum = add(a, b)` creates an output named "sum"
            vec![node.id.clone()]
        };

        // Create output operands
        let output_operand_ids: Vec<u32> = output_names_list
            .iter()
            .map(|name| {
                // Check if operand already exists
                if let Some(&idx) = operand_map.get(name) {
                    Ok::<u32, GraphError>(idx)
                } else {
                    // Create new output operand
                    let idx = operands.len() as u32;
                    operand_map.insert(name.clone(), idx);

                    operands.push(Operand {
                        name: Some(name.clone()),
                        descriptor: OperandDescriptor {
                            data_type: DataType::Float32, // Default, will be inferred
                            shape: Vec::new(),            // Will be inferred
                            pending_permutation: Vec::new(),
                        },
                        kind: OperandKind::Intermediate,
                    });

                    Ok::<u32, GraphError>(idx)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let attrs_value = serde_json::Value::Object(node.options.clone());
        let operator = Operation::from_json_attributes(
            &node.op,
            &input_operands,
            &output_operand_ids,
            &attrs_value,
        )
        .expect("unknown op type in JSON");

        operations.push(operator);
    }

    // Build output operands list from outputs map and mark them as Output
    for operand_ref in graph_json.outputs.values() {
        if let Some(&idx) = operand_map.get(operand_ref) {
            output_operands.push(idx);
            // Mark this operand as an actual graph output
            if let Some(operand) = operands.get_mut(idx as usize) {
                operand.kind = OperandKind::Output;
            }
        }
    }
    output_operands.sort_unstable();
    output_operands.dedup();

    let mut graph_info = GraphInfo {
        operands,
        input_operands,
        output_operands,
        operations,
        constant_operand_ids_to_handles,
        id_to_constant_tensor_operand_map: HashMap::new(),
        quantized: graph_json.quantized,
    };

    // Run shape inference pass to fill in output shapes
    infer_output_shapes(&mut graph_info)?;

    Ok(graph_info)
}

/// Infer output shapes for all operations in the graph
fn infer_output_shapes(graph: &mut GraphInfo) -> Result<(), GraphError> {
    use crate::shape_inference::*;

    fn parse_dtype(value: &serde_json::Value) -> Option<DataType> {
        let raw = value.as_str()?.to_ascii_lowercase();
        match raw.as_str() {
            "float32" => Some(DataType::Float32),
            "float16" => Some(DataType::Float16),
            "int32" => Some(DataType::Int32),
            "uint32" => Some(DataType::Uint32),
            "int64" => Some(DataType::Int64),
            "uint64" => Some(DataType::Uint64),
            "int8" => Some(DataType::Int8),
            "uint8" => Some(DataType::Uint8),
            "int4" => Some(DataType::Int4),
            "uint4" => Some(DataType::Uint4),
            _ => None,
        }
    }

    fn data_type_from_ml_operand_dtype(dt: MLOperandDataType) -> DataType {
        match dt {
            MLOperandDataType::Float32 => DataType::Float32,
            MLOperandDataType::Float16 => DataType::Float16,
            MLOperandDataType::Int32 => DataType::Int32,
            MLOperandDataType::Uint32 => DataType::Uint32,
            MLOperandDataType::Int64 => DataType::Int64,
            MLOperandDataType::Uint64 => DataType::Uint64,
            MLOperandDataType::Int8 => DataType::Int8,
            MLOperandDataType::Uint8 => DataType::Uint8,
            MLOperandDataType::Int4 => DataType::Int4,
            MLOperandDataType::Uint4 => DataType::Uint4,
        }
    }

    // Reserved for extended JSON operand/attribute parsing.
    #[allow(dead_code)]
    fn parse_i64_array(value: &serde_json::Value) -> Option<Vec<i64>> {
        let arr = value.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            if let Some(n) = v.as_i64() {
                out.push(n);
            } else {
                let n = v.as_u64()?;
                out.push(n as i64);
            }
        }
        Some(out)
    }

    // Reserved for extended JSON operand/attribute parsing.
    #[allow(dead_code)]
    fn parse_dimension_array(value: &serde_json::Value) -> Option<Vec<Dimension>> {
        let arr = value.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            if let Some(n) = v.as_u64() {
                out.push(Dimension::Static(n as u32));
                continue;
            }
            if let Some(n) = v.as_i64() {
                if n < 0 {
                    return None;
                }
                out.push(Dimension::Static(n as u32));
                continue;
            }

            let obj = v.as_object()?;
            let name = obj.get("name")?.as_str()?.to_string();
            let max_size = obj
                .get("maxSize")
                .or_else(|| obj.get("max_size"))?
                .as_u64()? as u32;
            out.push(Dimension::Dynamic(DynamicDimension { name, max_size }));
        }
        Some(out)
    }

    // Run multiple passes until no more shapes can be inferred
    debug_print!(
        "[SHAPE INFERENCE] Starting shape inference with {} operations",
        graph.operations.len()
    );
    let max_passes = 10; // Prevent infinite loops
    for pass_num in 0..max_passes {
        debug_print!("[SHAPE INFERENCE] Pass {}/{}", pass_num + 1, max_passes);
        let mut made_progress = false;

        // Process operations in order (assumed to be in dependency order from WebNN parser)
        for op_idx in 0..graph.operations.len() {
            let op = &graph.operations[op_idx];
            let op_type = op.op_type().to_ascii_lowercase();

            // Normalize tile inputs: if shape rank is missing, set to repeats length (filled with 1s)
            if op_type == "tile"
                && let Some(repeats_len) = match &op {
                    Operation::Tile { repetitions, .. } => {
                        (!repetitions.is_empty()).then_some(repetitions.len())
                    }
                    _ => None,
                }
                && let Some(input_id) = op.input_operands().first()
            {
                let inp = &mut graph.operands[*input_id as usize];
                if inp.descriptor.shape.len() != repeats_len {
                    inp.descriptor.shape = vec![Dimension::Static(1); repeats_len];
                    made_progress = true;
                }
            }

            // Skip if output already has a shape
            if let Some(output_id) = op.output_operand()
                && !graph.operands[output_id as usize]
                    .descriptor
                    .shape
                    .is_empty()
            {
                continue;
            }

            // Get input shapes and types
            let input_shapes: Vec<Vec<Dimension>> = op
                .input_operands()
                .iter()
                .map(|&id| graph.operands[id as usize].descriptor.shape.clone())
                .collect();
            let input_types: Vec<DataType> = op
                .input_operands()
                .iter()
                .map(|&id| graph.operands[id as usize].descriptor.data_type)
                .collect();

            // Infer output shape based on operation type
            let output_shape = match op_type.as_str() {
                // Binary element-wise operations (including comparisons/logical)
                "add" | "sub" | "mul" | "div" | "pow" | "max" | "min" | "greater"
                | "greaterorequal" | "less" | "lesser" | "lessorequal" | "lesserorequal"
                | "equal" | "notequal" | "logical_and" | "logical_or" | "logical_xor" => {
                    if input_shapes.len() >= 2 {
                        broadcast_shapes_dimensions(&input_shapes[0], &input_shapes[1]).ok()
                    } else {
                        None
                    }
                }

                // Unary element-wise operations (shape unchanged)
                "abs" | "ceil" | "floor" | "neg" | "relu" | "sigmoid" | "tanh" | "exp" | "log"
                | "sqrt" | "erf" | "sin" | "cos" | "tan" | "sign" | "reciprocal" | "roundEven"
                | "softplus" | "softsign" | "softmax" | "gelu" | "linear" | "identity" | "cast"
                | "reverse" | "cumulativesum" | "cumulative_sum" | "logical_not" | "isnan"
                | "isinfinite" | "quantizelinear" | "dequantizelinear"
                // Normalization ops preserve the input tensor rank and extents.
                | "batchnormalization" | "instancenormalization" | "layernormalization" => {
                    input_shapes.first().cloned()
                }
                "grucell" | "gru_cell" => {
                    if let Some(hidden_state_shape) = input_shapes.get(3) {
                        Some(hidden_state_shape.clone())
                    } else if let Some(input_shape) = input_shapes.first() {
                        let hidden_size = match &op {
                            Operation::GruCell { hidden_size, .. } => {
                                (*hidden_size > 0).then_some(*hidden_size)
                            }
                            _ => None,
                        };
                        if input_shape.len() == 2 {
                            hidden_size.map(|h| vec![input_shape[0].clone(), Dimension::Static(h)])
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }

                "concat" => {
                    let axis_val = match &op {
                        Operation::Concat { axis, .. } => *axis,
                        _ => 0,
                    };
                    if input_shapes.iter().all(|s| s.is_empty()) && axis_val == 0 {
                        Some(vec![Dimension::Static(input_shapes.len() as u32)])
                    } else {
                        infer_concat_shape_dimensions(&input_shapes, axis_val).ok()
                    }
                }

                // Expand: `newShape` is a method argument (see expand() in the WebNN spec).
                "expand" => {
                    if input_shapes.len() == 1 {
                        let new_shape_opt: Option<Vec<Dimension>> = match &op {
                            Operation::Expand { new_shape, .. } if !new_shape.is_empty() => Some(
                                new_shape
                                    .iter()
                                    .map(|d| Dimension::from(d.clone()))
                                    .collect(),
                            ),
                            _ => None,
                        };
                        new_shape_opt.and_then(|new_shape| {
                            infer_expand_shape_dimensions(&input_shapes[0], &new_shape).ok()
                        })
                    } else {
                        None
                    }
                }

                // Reshape - use newShape from the operation (builder method argument).
                "reshape" => match &op {
                    Operation::Reshape { new_shape, .. } => (!new_shape.is_empty()).then(|| {
                        new_shape
                            .iter()
                            .map(|d| Dimension::from(d.clone()))
                            .collect::<Vec<_>>()
                    }),
                    _ => None,
                },

                "unsqueeze" => match (&op, input_shapes.first()) {
                    (Operation::Unsqueeze { options, .. }, Some(input_shape)) => {
                        let axes = options.as_ref().map(|o| o.axes.as_slice()).unwrap_or(&[]);
                        infer_unsqueeze_shape_dimensions(input_shape, axes).ok()
                    }
                    _ => None,
                },

                // Transpose
                "transpose" => {
                    if input_shapes.len() == 1 {
                        let perm = match &op {
                            Operation::Transpose { options, .. } => options
                                .as_ref()
                                .filter(|o| !o.permutation.is_empty())
                                .map(|o| o.permutation.clone()),
                            _ => None,
                        };
                        infer_transpose_shape_dimensions(&input_shapes[0], perm.as_deref()).ok()
                    } else {
                        None
                    }
                }

                // MatMul
                "matmul" => {
                    if input_shapes.len() >= 2 {
                        if input_shapes[0].len() < 2 || input_shapes[1].len() < 2 {
                            None
                        } else {
                            infer_matmul_shape_dimensions(&input_shapes[0], &input_shapes[1]).ok()
                        }
                    } else {
                        None
                    }
                }

                // Gemm: same trailing-2D layout as matmul; optional `c` must broadcast to output.
                "gemm" => {
                    if input_shapes.len() >= 2 {
                        let (a_transpose, b_transpose) = match &op {
                            Operation::Gemm { options, .. } => options
                                .as_ref()
                                .map(|o| (o.a_transpose, o.b_transpose))
                                .unwrap_or((false, false)),
                            _ => (false, false),
                        };
                        infer_gemm_shape_dimensions(
                            &input_shapes[0],
                            &input_shapes[1],
                            a_transpose,
                            b_transpose,
                        )
                        .ok()
                        .and_then(|inferred| match &op {
                            Operation::Gemm { options, .. } => {
                                if let Some(c_id) = options.as_ref().and_then(|o| o.c) {
                                    let c_shape =
                                        graph.operands[c_id as usize].descriptor.shape.clone();
                                    if c_shape.is_empty() {
                                        Some(inferred)
                                    } else {
                                        broadcast_shapes_dimensions(&inferred, &c_shape).ok()
                                    }
                                } else {
                                    Some(inferred)
                                }
                            }
                            _ => Some(inferred),
                        })
                    } else {
                        None
                    }
                }

                // Reduction operations
                "reducemean" | "reducesum" | "reducemax" | "reducemin" | "reduceproduct"
                | "reducel1" | "reducel2" | "reducelogsum" | "reducelogsumexp"
                | "reducesumsquare" => {
                    let opts = match &op {
                        Operation::ReduceSum { options, .. }
                        | Operation::ReduceMean { options, .. }
                        | Operation::ReduceMax { options, .. }
                        | Operation::ReduceMin { options, .. }
                        | Operation::ReduceProduct { options, .. }
                        | Operation::ReduceL1 { options, .. }
                        | Operation::ReduceL2 { options, .. }
                        | Operation::ReduceLogSum { options, .. }
                        | Operation::ReduceLogSumExp { options, .. }
                        | Operation::ReduceSumSquare { options, .. } => options.as_ref(),
                        _ => None,
                    };
                    let (axes_raw, keep_dimensions) = opts
                        .map(|o| (o.axes.clone(), o.keep_dimensions))
                        .unwrap_or((None, false));

                    if let Some(input_shape) = input_shapes.first() {
                        let rank = input_shape.len() as u32;
                        let axes = match &axes_raw {
                            None => (0..rank).collect(),
                            Some(v) => v.clone(),
                        };
                        if axes.iter().any(|&axis| axis >= rank) {
                            None
                        } else {
                            let options = ReduceOptions {
                                axes,
                                keep_dimensions,
                            };
                            infer_reduce_shape_dimensions(input_shape, &options).ok()
                        }
                    } else {
                        None
                    }
                }

                // Gather: forward inference (inputs -> output) or use existing output shape for
                // back-propagation (output -> indices) when output operand already has a shape.
                "gather" => {
                    let axis_from_op = match &op {
                        Operation::Gather { options, .. } => {
                            options.as_ref().map(|o| o.axis as i64).unwrap_or(0)
                        }
                        _ => 0,
                    };

                    // If output operand already has a shape (e.g. from JSON or earlier pass), use it
                    // to back-propagate indices shape and as the result.
                    let existing_output_shape = op.output_operand().and_then(|id| {
                        let o = graph.operands.get(id as usize)?;
                        (!o.descriptor.shape.is_empty()).then_some(o.descriptor.shape.clone())
                    });

                    if let Some(shape_override) = existing_output_shape {
                        // Back-propagate implied indices shape when we have data shape and axis.
                        if !input_shapes.is_empty() {
                            let data_shape = &input_shapes[0];
                            let rank = data_shape.len() as i64;
                            let mut axis = axis_from_op;
                            if axis < 0 {
                                axis += rank;
                            }
                            if axis >= 0 && (axis as usize) < data_shape.len() {
                                let tail_len = data_shape.len().saturating_sub(axis as usize + 1);
                                if let Some(&indices_id) = op.input_operands().get(1) {
                                    let indices_operand = &mut graph.operands[indices_id as usize];
                                    if indices_operand.descriptor.shape.is_empty()
                                        && shape_override.len() >= tail_len
                                    {
                                        indices_operand.descriptor.shape = shape_override
                                            [..shape_override.len() - tail_len]
                                            .to_vec();
                                        made_progress = true;
                                    }
                                }
                            }
                        }
                        Some(shape_override)
                    } else if input_shapes.len() >= 2 {
                        let mut axis = axis_from_op;
                        let rank = input_shapes[0].len() as i64;
                        if axis < 0 {
                            axis += rank;
                        }
                        if axis < 0 || axis >= rank {
                            None
                        } else {
                            infer_gather_shape_dimensions(
                                &input_shapes[0],
                                &input_shapes[1],
                                axis as u32,
                            )
                            .ok()
                        }
                    } else {
                        None
                    }
                }

                // Where (broadcast across condition/true/false)
                "where" => {
                    if input_shapes.len() >= 3 {
                        infer_where_shape_dimensions(
                            &input_shapes[0],
                            &input_shapes[1],
                            &input_shapes[2],
                        )
                        .ok()
                    } else {
                        None
                    }
                }

                // Slice: starts/sizes are operation parameters (not MLSliceOptions).
                "slice" => {
                    if let Some(input_shape) = input_shapes.first() {
                        match &op {
                            Operation::Slice {
                                starts,
                                sizes,
                                options,
                                ..
                            } => {
                                if starts.is_empty()
                                    || sizes.is_empty()
                                    || starts.len() != sizes.len()
                                {
                                    None
                                } else {
                                    let sizes_u32: Vec<u32> =
                                        sizes.iter().map(|d| d.static_or_max()).collect();
                                    let strides = options.as_ref().map(|o| o.strides.as_slice());
                                    let input_dims: Vec<u32> = input_shape
                                        .iter()
                                        .map(crate::graph::get_static_or_max_size)
                                        .collect();
                                    infer_slice_shape(
                                        &input_dims,
                                        starts,
                                        &sizes_u32,
                                        strides,
                                    )
                                    .ok()
                                    .map(|shape| {
                                        shape
                                            .into_iter()
                                            .map(Dimension::Static)
                                            .collect::<Vec<_>>()
                                    })
                                }
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                }

                // Shape
                "shape" => input_shapes
                    .first()
                    .map(|input_shape| vec![Dimension::Static(input_shape.len() as u32)]),

                // Pool2d (WebNN defaults for window / strides / dilations / padding)
                "averagepool2d" | "maxpool2d" | "l2pool2d" => {
                    if let Some(input_shape) = input_shapes.first() {
                        let default_pool = crate::operator_options::MLPool2dOptions::default();
                        let o = match &op {
                            Operation::AveragePool2d { options, .. }
                            | Operation::MaxPool2d { options, .. }
                            | Operation::L2Pool2d { options, .. } => {
                                options.as_ref().unwrap_or(&default_pool)
                            }
                            _ => &default_pool,
                        };
                        infer_pool2d_shape_dimensions(
                            input_shape,
                            &o.layout,
                            o.window_dimensions.as_deref(),
                            &o.strides,
                            &o.dilations,
                            &o.padding,
                            o.output_sizes.as_deref(),
                            o.output_shape_rounding.eq_ignore_ascii_case("ceil"),
                        )
                        .ok()
                    } else {
                        None
                    }
                }

                "globalaveragepool" | "globalmaxpool" => {
                    if let Some(input_shape) = input_shapes.first() {
                        let default_pool = crate::operator_options::MLPool2dOptions::default();
                        let o = match &op {
                            Operation::GlobalAveragePool { options, .. }
                            | Operation::GlobalMaxPool { options, .. } => {
                                options.as_ref().unwrap_or(&default_pool)
                            }
                            _ => &default_pool,
                        };
                        let layout_enum = if o.layout.eq_ignore_ascii_case("nhwc") {
                            InputLayout::Nhwc
                        } else {
                            InputLayout::Nchw
                        };
                        let input_u32: Vec<u32> = input_shape
                            .iter()
                            .map(crate::graph::get_static_or_max_size)
                            .collect();
                        infer_global_pool_shape(&input_u32, layout_enum)
                            .ok()
                            .map(|v| to_dimension_vector(&v))
                    } else {
                        None
                    }
                }

                // Constant: shape from operator options
                "constant" => match &op {
                    Operation::Constant { options, .. } => options
                        .as_ref()
                        .map(|o| {
                            o.shape
                                .iter()
                                .map(|&u| Dimension::Static(u))
                                .collect::<Vec<_>>()
                        })
                        .or_else(|| Some(Vec::new())),
                    _ => Some(Vec::new()),
                },

                "tile" => {
                    if let (Some(input_shape), Operation::Tile { repetitions, .. }) =
                        (input_shapes.first(), op)
                    {
                        let input_u32: Vec<u32> = input_shape
                            .iter()
                            .map(crate::graph::get_static_or_max_size)
                            .collect();
                        infer_tile_shape(&input_u32, repetitions)
                            .ok()
                            .map(|v| to_dimension_vector(&v))
                    } else {
                        None
                    }
                }

                // For other operations, leave shape empty (will be handled later or is dynamic)
                _ => None,
            };

            // Update output operand shape if we inferred it
            if let Some(shape) = output_shape
                && let Some(output_id) = op.output_operand()
            {
                graph.operands[output_id as usize].descriptor.shape = shape;
                made_progress = true;
            }

            // Propagate output data types where deterministically known
            if let Some(output_id) = op.output_operand() {
                let output_type = match op_type.as_str() {
                    "shape" => Some(DataType::Int64),
                    "constant" => match &op {
                        Operation::Constant { options, .. } => options.as_ref().and_then(|o| {
                            parse_dtype(&serde_json::Value::String(o.data_type.clone()))
                        }),
                        _ => None,
                    },
                    "cast" => match &op {
                        Operation::Cast { data_type: to, .. } => {
                            Some(data_type_from_ml_operand_dtype(*to))
                        }
                        _ => None,
                    },
                    "dequantizelinear" => input_types.get(1).cloned().or(Some(DataType::Float32)),
                    "quantizelinear" => input_types.get(2).cloned().or(Some(DataType::Uint8)),
                    "argmax" | "argmin" => match &op {
                        Operation::ArgMax { options, .. } | Operation::ArgMin { options, .. } => {
                            Some(
                                options
                                    .as_ref()
                                    .map(|o| data_type_from_ml_operand_dtype(o.output_data_type))
                                    .unwrap_or(DataType::Int32),
                            )
                        }
                        _ => Some(DataType::Int32),
                    },
                    "expand"
                    | "gather"
                    | "gatherelements"
                    | "gathernd"
                    | "concat"
                    | "slice"
                    | "reshape"
                    | "squeeze"
                    | "unsqueeze"
                    | "transpose"
                    | "matmul"
                    | "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "pow"
                    | "max"
                    | "min"
                    | "abs"
                    | "ceil"
                    | "floor"
                    | "neg"
                    | "relu"
                    | "sigmoid"
                    | "tanh"
                    | "exp"
                    | "log"
                    | "sqrt"
                    | "erf"
                    | "sin"
                    | "cos"
                    | "tan"
                    | "sign"
                    | "roundEven"
                    | "reciprocal"
                    | "softplus"
                    | "softsign"
                    | "softmax"
                    | "gelu"
                    | "linear"
                    | "identity"
                    | "gru"
                    | "grucell"
                    | "gru_cell"
                    | "lstm"
                    | "lstmcell"
                    | "lstm_cell"
                    | "cumulativesum"
                    | "cumulative_sum"
                    | "hardsigmoid"
                    | "hardswish"
                    | "elu"
                    | "leakyrelu"
                    | "prelu"
                    | "clamp"
                    | "pad"
                    | "tile"
                    | "reverse"
                    | "triangular"
                    | "conv2d"
                    | "convtranspose2d"
                    | "gemm"
                    | "batchnormalization"
                    | "instancenormalization"
                    | "layernormalization"
                    | "reducemean"
                    | "reducesum"
                    | "reducemax"
                    | "reducemin"
                    | "reduceproduct"
                    | "reducel1"
                    | "reducel2"
                    | "reducelogsum"
                    | "reducelogsumexp"
                    | "averagepool2d"
                    | "maxpool2d"
                    | "global_average_pool"
                    | "global_max_pool"
                    | "reducesumsquare" => input_types.first().cloned(),
                    "greater" | "greaterorequal" | "less" | "lesser" | "lessorequal"
                    | "lesserorequal" | "equal" | "notequal" | "logical_and" | "logical_or"
                    | "logical_xor" | "logicaland" | "logicalor" | "logicalxor" | "logicalnot"
                    | "isnan" | "isinfinite" => Some(DataType::Uint8),
                    "where" => input_types
                        .get(1)
                        .cloned()
                        .or_else(|| input_types.get(2).cloned()),
                    _ => None,
                };
                if let Some(dtype) = output_type {
                    graph.operands[output_id as usize].descriptor.data_type = dtype;
                    // LSTM has multiple outputs (Y_h, Y_c, optional sequence); all match input type.
                    if matches!(op_type.as_str(), "lstm" | "lstmcell" | "lstm_cell") {
                        for &oid in op.output_operands() {
                            if oid != output_id {
                                graph.operands[oid as usize].descriptor.data_type = dtype;
                            }
                        }
                    }
                }
            }
        }

        // Stop if no progress was made this pass
        if !made_progress {
            break;
        }
    }

    // Summary: count operands with empty shapes
    let empty_shape_count = graph
        .operands
        .iter()
        .filter(|op| op.descriptor.shape.is_empty())
        .count();
    debug_print!(
        "[SHAPE INFERENCE] Completed: {} operands still have empty shapes",
        empty_shape_count
    );
    if empty_shape_count > 0 {
        debug_print!("[SHAPE INFERENCE] WARNING: Some operands could not have shapes inferred!");
        for (idx, op) in graph.operands.iter().enumerate() {
            if op.descriptor.shape.is_empty() {
                debug_print!("  operand_{}: name={:?}, kind={:?}", idx, op.name, op.kind);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use webnn_graph::serialize::{SerializeOptions, serialize_graph_to_wg_text};

    fn wshape(shape: &[u32]) -> Vec<webnn_graph::ast::Dimension> {
        webnn_graph::ast::to_dimension_vector(shape)
    }

    fn ushape(shape: &[u32]) -> Vec<u32> {
        shape.to_vec()
    }

    #[test]
    fn infer_output_shapes_layer_normalization_preserves_input_shape() {
        let text = r#"
        webnn_graph "ln_test" v1 {
            inputs { x: f32[1, 64, 3072]; }
            nodes {
                [y] = layerNormalization(x, epsilon=1e-06);
            }
            outputs { y; }
        }"#;
        let graph_json = webnn_graph::parser::parse_wg_text(text).expect("parse");
        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        let y_idx = graph_info.output_operands[0];
        assert_eq!(
            graph_info.operands[y_idx as usize].descriptor.shape,
            to_dimension_vector(&[1, 64, 3072]),
            "layerNormalization output shape should match input"
        );
    }

    #[test]
    fn infer_shape_tensor_type_through_unsqueeze() {
        let text = r#"
        webnn_graph "shape_type_test" v2 {
            inputs { x: f32[2, 3, 4]; }
            nodes {
                shape_out = shape(x);
                expanded_shape = unsqueeze(shape_out, axes=[0]);
            }
            outputs { expanded_shape; }
        }"#;
        let graph_json = webnn_graph::parser::parse_wg_text(text).expect("parse");
        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        let shape_output = graph_info.operations[0]
            .output_operand()
            .expect("shape output");
        let unsqueeze_output = graph_info.operations[1]
            .output_operand()
            .expect("unsqueeze output");
        assert_eq!(
            graph_info.operands[shape_output as usize]
                .descriptor
                .data_type,
            DataType::Int64
        );
        assert_eq!(
            graph_info.operands[unsqueeze_output as usize]
                .descriptor
                .data_type,
            DataType::Int64
        );
        assert_eq!(
            graph_info.operands[unsqueeze_output as usize]
                .descriptor
                .shape,
            to_dimension_vector(&[1, 3])
        );
    }

    #[test]
    fn test_datatype_conversion() {
        let types = vec![
            DataType::Float32,
            DataType::Float16,
            DataType::Int32,
            DataType::Uint32,
            DataType::Int64,
            DataType::Uint64,
            DataType::Int8,
            DataType::Uint8,
            DataType::Int4,
            DataType::Uint4,
        ];

        for dt in types {
            let webnn_dt = to_webnn_datatype(&dt);
            let back = from_webnn_datatype(&webnn_dt);
            assert_eq!(dt, back);
        }
    }

    fn build_quantized_graph_info(dtype: DataType) -> GraphInfo {
        GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: dtype,
                    shape: to_dimension_vector(&[2, 3]),
                    pending_permutation: vec![],
                },
                name: Some("input".to_string()),
            }],
            input_operands: vec![0],
            output_operands: vec![0],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        }
    }

    #[test]
    fn quantized_flag_roundtrips_json() {
        let graph = build_quantized_graph_info(DataType::Int8);
        let json = to_graph_json(&graph, false).expect("to_graph_json");
        assert!(json.quantized);

        let graph_from_json = from_graph_json(&json).expect("from_graph_json");
        assert!(graph_from_json.quantized);
        assert_eq!(
            graph_from_json.operands[0].descriptor.data_type,
            DataType::Int8
        );
        assert_eq!(graph_from_json.output_operands, vec![0]);

        // Passing quantized=true should also set the flag even if the graph info is false.
        let mut graph_not_marked = graph.clone();
        graph_not_marked.quantized = false;
        let json_explicit = to_graph_json(&graph_not_marked, true).expect("to_graph_json");
        assert!(json_explicit.quantized);
    }

    #[test]
    fn quantized_flag_roundtrips_text() {
        let graph = build_quantized_graph_info(DataType::Uint4);
        let graph_json = to_graph_json(&graph, true).expect("to_graph_json");
        let text = serialize_graph_to_wg_text(&graph_json, SerializeOptions { quantized: true })
            .expect("serialize to text");
        let parsed = webnn_graph::parser::parse_wg_text(&text).expect("parse text");
        assert!(
            parsed.quantized,
            "text serialization preserves quantized marker"
        );

        let graph_info = from_graph_json(&parsed).expect("graph from text");
        assert!(graph_info.quantized);
        assert_eq!(graph_info.operands[0].descriptor.data_type, DataType::Uint4);
    }

    #[test]
    fn test_to_graph_json_with_constants() {
        // Test conversion with constant operands
        let constant_data = vec![1u8, 2, 3, 4];
        let mut constant_map = HashMap::new();
        constant_map.insert(
            1u32,
            ConstantData {
                data: constant_data.clone(),
                label: None,
            },
        );

        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 1]),
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 1]),
                        pending_permutation: vec![],
                    },
                    name: Some("weight".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 1]),
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![2],
            operations: vec![],
            constant_operand_ids_to_handles: constant_map,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let json = to_graph_json(&graph, false).expect("to_graph_json");

        assert_eq!(json.inputs.len(), 1);
        assert!(json.inputs.contains_key("input"));
        assert_eq!(json.consts.len(), 1);
        assert!(json.consts.contains_key("weight"));
        assert_eq!(json.outputs.len(), 1);
        assert!(json.outputs.contains_key("output"));
    }

    #[test]
    fn test_to_graph_json_with_operations() {
        // Test conversion with operations
        let mut attrs = serde_json::Map::new();
        attrs.insert("alpha".to_string(), serde_json::json!(0.01));

        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 3]),
                        pending_permutation: vec![],
                    },
                    name: Some("x".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1, 3]),
                        pending_permutation: vec![],
                    },
                    name: Some("y".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![{
                let attributes = crate::operator_options::OperatorOptions::from_json_with_op_type(
                    "leakyRelu",
                    &serde_json::Value::Object(attrs),
                )
                .expect("leakyRelu options");
                Operation::LeakyRelu {
                    input: 0,
                    options: attributes.as_leaky_relu().cloned(),
                    outputs: vec![1],
                }
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let json = to_graph_json(&graph, false).expect("to_graph_json");

        assert_eq!(json.nodes.len(), 1);
        assert_eq!(json.nodes[0].op, "leakyRelu");
        assert_eq!(json.nodes[0].inputs, vec!["x"]);
        assert!(json.nodes[0].options.contains_key("alpha"));
    }

    #[test]
    fn test_from_graph_json_creates_operands() {
        use webnn_graph::ast::{ConstDecl, ConstInit, OperandDesc};

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: wshape(&[1, 2, 3]),
            },
        );

        let mut consts = BTreeMap::new();
        consts.insert(
            "weight".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: ushape(&[3, 3]),
                init: ConstInit::InlineBytes {
                    bytes: vec![0u8; 36],
                },
            },
        );

        let mut outputs = BTreeMap::new();
        outputs.insert("y".to_string(), "y".to_string());

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts,
            nodes: vec![],
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        // Check we have operands: input and constant
        // Note: outputs in GraphJson don't create separate operands unless they're
        // referenced by nodes, they just mark existing operands as outputs
        assert!(graph_info.operands.len() >= 2);
        assert_eq!(graph_info.input_operands.len(), 1);

        // Check constant data was stored
        assert_eq!(graph_info.constant_operand_ids_to_handles.len(), 1);
    }

    #[test]
    fn test_from_graph_json_with_scalar_constant() {
        use webnn_graph::ast::{ConstDecl, ConstInit};

        let mut consts = BTreeMap::new();
        consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: ushape(&[]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(1.5),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        // Scalar constant should be created
        assert_eq!(graph_info.operands.len(), 1);
        assert!(matches!(graph_info.operands[0].kind, OperandKind::Constant));
        let empty_shape: Vec<Dimension> = vec![];
        assert_eq!(graph_info.operands[0].descriptor.shape, empty_shape);
    }

    #[test]
    fn test_operand_name_generation() {
        // Test that unnamed operands get generated names
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1]),
                        pending_permutation: vec![],
                    },
                    name: None, // Unnamed
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[1]),
                        pending_permutation: vec![],
                    },
                    name: None, // Unnamed
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let json = to_graph_json(&graph, false).expect("to_graph_json");

        // Generated names should be present
        assert!(json.inputs.contains_key("operand_0"));
        assert!(json.outputs.contains_key("operand_1"));
    }

    #[test]
    fn test_all_data_types_roundtrip() {
        let types = vec![
            DataType::Float32,
            DataType::Float16,
            DataType::Int32,
            DataType::Uint32,
            DataType::Int64,
            DataType::Uint64,
            DataType::Int8,
            DataType::Uint8,
            DataType::Int4,
            DataType::Uint4,
        ];

        for dtype in types {
            let graph = build_quantized_graph_info(dtype);
            let json = to_graph_json(&graph, false).expect("to_graph_json");
            let back = from_graph_json(&json).expect("from_graph_json");

            assert_eq!(
                back.operands[0].descriptor.data_type, dtype,
                "Data type {:?} should roundtrip correctly",
                dtype
            );
        }
    }

    #[test]
    fn test_scalar_constant_float16() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float16,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(2.5),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        assert_eq!(
            graph_info.operands[0].descriptor.data_type,
            DataType::Float16
        );
        assert!(graph_info.constant_operand_ids_to_handles.contains_key(&0));
    }

    #[test]
    fn test_scalar_constant_int_types() {
        // Test Int32
        let mut consts = BTreeMap::new();
        consts.insert(
            "int_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Int32,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(42),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts: consts.clone(),
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        result.unwrap();

        // Test Uint32, Int64, Uint64, Int8, Uint8
        let types_to_test = vec![
            webnn_graph::ast::DataType::Uint32,
            webnn_graph::ast::DataType::Int64,
            webnn_graph::ast::DataType::Uint64,
            webnn_graph::ast::DataType::Int8,
            webnn_graph::ast::DataType::Uint8,
        ];

        for dtype in types_to_test {
            let mut consts = BTreeMap::new();
            consts.insert(
                "val".to_string(),
                ConstDecl {
                    data_type: dtype.clone(),
                    shape: ushape(&[1]),
                    init: ConstInit::Scalar {
                        value: serde_json::json!(10),
                    },
                },
            );

            let graph_json = GraphJson {
                name: Some("test".to_string()),
                format: "webnn-graph-json".to_string(),
                version: 2,
                quantized: false,
                inputs: BTreeMap::new(),
                consts,
                nodes: vec![],
                outputs: BTreeMap::new(),
            };

            from_graph_json(&graph_json)
                .unwrap_or_else(|e| panic!("Failed for type {:?}: {e}", dtype));
        }
    }

    #[test]
    fn test_scalar_constant_int4_error() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "int4_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Int4,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(1),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::ConversionFailed { reason, .. } => {
                assert!(reason.contains("int4/uint4"));
            }
            _ => panic!("Expected ConversionFailed error"),
        }
    }

    #[test]
    fn test_scalar_constant_invalid_value() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "bad_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!({"not": "a number"}),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::ConversionFailed { reason, .. } => {
                assert!(reason.contains("Cannot parse scalar value"));
            }
            _ => panic!("Expected ConversionFailed error"),
        }
    }

    #[test]
    fn test_weights_reference_constant() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "weight_ref".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: ushape(&[2, 2]),
                init: ConstInit::Weights {
                    r#ref: "model_weight".to_string(),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        assert_eq!(graph_info.operands.len(), 1);
        assert!(matches!(graph_info.operands[0].kind, OperandKind::Constant));
        // Weight references should create empty data (to be filled by loader)
        let const_data = graph_info.constant_operand_ids_to_handles.get(&0).unwrap();
        assert_eq!(const_data.data.len(), 0);
    }

    #[test]
    fn test_operation_missing_input() {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: wshape(&[2]),
            },
        );

        let nodes = vec![Node {
            id: "relu_0".to_string(),
            op: "relu".to_string(),
            inputs: vec!["missing_input".to_string()],
            options: serde_json::Map::new(),
            outputs: None,
        }];

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::ConversionFailed { reason, .. } => {
                assert!(reason.contains("not found"));
            }
            _ => panic!("Expected ConversionFailed error"),
        }
    }

    #[test]
    fn test_operation_with_none_outputs() {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: wshape(&[2]),
            },
        );

        let nodes = vec![Node {
            id: "relu_output".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: None, // Will default to node.id
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("result".to_string(), "relu_output".to_string());

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        // Should create an output operand named after the node.id
        assert!(
            graph_info
                .operands
                .iter()
                .any(|op| op.name.as_deref() == Some("relu_output"))
        );
    }

    #[test]
    fn test_scalar_with_i64_value() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "int_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Int32,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(42_i64),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        result.unwrap();
    }

    #[test]
    fn test_scalar_with_u64_value() {
        let mut consts = BTreeMap::new();
        consts.insert(
            "uint_val".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Uint32,
                shape: ushape(&[1]),
                init: ConstInit::Scalar {
                    value: serde_json::json!(42_u64),
                },
            },
        );

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs: BTreeMap::new(),
            consts,
            nodes: vec![],
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        result.unwrap();
    }

    #[test]
    fn test_operation_empty_outputs() {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: wshape(&[2]),
            },
        );

        let nodes = vec![Node {
            id: "node_0".to_string(),
            op: "relu".to_string(),
            inputs: vec!["x".to_string()],
            options: serde_json::Map::new(),
            outputs: Some(vec![]), // Empty outputs vector
        }];

        let graph_json = GraphJson {
            name: Some("test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs: BTreeMap::new(),
        };

        let result = from_graph_json(&graph_json);
        // Should succeed but create no output operands for the operation
        let graph_info = result.unwrap();
        assert_eq!(graph_info.operations.len(), 1);
        assert_eq!(graph_info.operations[0].output_operands().len(), 0);
    }

    #[test]
    fn test_quantize_linear_infers_output_shape_and_dtype() {
        use webnn_graph::ast::{ConstDecl, ConstInit, OperandDesc};

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(2),
                    webnn_graph::ast::Dimension::Static(3),
                ],
            },
        );

        let mut consts = BTreeMap::new();
        consts.insert(
            "scale".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![],
                init: ConstInit::Scalar {
                    value: serde_json::json!(0.5),
                },
            },
        );
        consts.insert(
            "zero_point".to_string(),
            ConstDecl {
                data_type: webnn_graph::ast::DataType::Uint8,
                shape: vec![],
                init: ConstInit::Scalar {
                    value: serde_json::json!(128),
                },
            },
        );

        let nodes = vec![Node {
            id: "q".to_string(),
            op: "quantizeLinear".to_string(),
            inputs: vec![
                "x".to_string(),
                "scale".to_string(),
                "zero_point".to_string(),
            ],
            options: serde_json::Map::new(),
            outputs: None,
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("result".to_string(), "q".to_string());

        let graph_json = GraphJson {
            name: Some("q_test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: true,
            inputs,
            consts,
            nodes,
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        let out_id = graph_info.output_operands[0] as usize;
        let out_desc = &graph_info.operands[out_id].descriptor;
        assert_eq!(
            out_desc.shape,
            vec![Dimension::Static(2), Dimension::Static(3)]
        );
        assert_eq!(out_desc.data_type, DataType::Uint8);
    }

    #[test]
    fn test_cumulative_sum_infers_output_shape_and_dtype() {
        use webnn_graph::ast::OperandDesc;

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(2),
                    webnn_graph::ast::Dimension::Static(3),
                ],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert(
            "axis".to_string(),
            serde_json::Value::Number(serde_json::Number::from(1)),
        );
        options.insert("exclusive".to_string(), serde_json::Value::Bool(true));
        options.insert("reversed".to_string(), serde_json::Value::Bool(true));

        let nodes = vec![Node {
            id: "cumsum".to_string(),
            op: "cumulativeSum".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: None,
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("result".to_string(), "cumsum".to_string());

        let graph_json = GraphJson {
            name: Some("cumsum_test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        let out_id = graph_info.output_operands[0] as usize;
        let out_desc = &graph_info.operands[out_id].descriptor;
        assert_eq!(
            out_desc.shape,
            vec![Dimension::Static(2), Dimension::Static(3)]
        );
        assert_eq!(out_desc.data_type, DataType::Float32);
    }

    #[test]
    fn test_gru_cell_infers_output_shape_and_dtype() {
        use webnn_graph::ast::OperandDesc;

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(3),
                    webnn_graph::ast::Dimension::Static(2),
                ],
            },
        );
        inputs.insert(
            "w".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(12),
                    webnn_graph::ast::Dimension::Static(2),
                ],
            },
        );
        inputs.insert(
            "r".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(12),
                    webnn_graph::ast::Dimension::Static(4),
                ],
            },
        );
        inputs.insert(
            "h".to_string(),
            OperandDesc {
                data_type: webnn_graph::ast::DataType::Float32,
                shape: vec![
                    webnn_graph::ast::Dimension::Static(3),
                    webnn_graph::ast::Dimension::Static(4),
                ],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert(
            "hiddenSize".to_string(),
            serde_json::Value::Number(serde_json::Number::from(4)),
        );

        let nodes = vec![Node {
            id: "gru".to_string(),
            op: "gruCell".to_string(),
            inputs: vec![
                "x".to_string(),
                "w".to_string(),
                "r".to_string(),
                "h".to_string(),
            ],
            options,
            outputs: None,
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("result".to_string(), "gru".to_string());

        let graph_json = GraphJson {
            name: Some("gru_test".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");

        let out_id = graph_info.output_operands[0] as usize;
        let out_desc = &graph_info.operands[out_id].descriptor;
        assert_eq!(
            out_desc.shape,
            vec![Dimension::Static(3), Dimension::Static(4)]
        );
        assert_eq!(out_desc.data_type, DataType::Float32);
    }

    #[test]
    fn test_from_graph_json_parses_dynamic_input_dimensions() {
        use webnn_graph::ast::{
            DataType as WDataType, Dimension as WDimension, DynamicDimension as WDynamicDimension,
            OperandDesc,
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 16,
                    }),
                    WDimension::Static(64),
                ],
            },
        );

        let mut outputs = BTreeMap::new();
        outputs.insert("x_out".to_string(), "x".to_string());

        let graph_json = GraphJson {
            name: Some("dynamic_input".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes: vec![],
            outputs,
        };

        let graph_info = from_graph_json(&graph_json).expect("from_graph_json");
        assert_eq!(graph_info.input_operands.len(), 1);
        let input = &graph_info.operands[graph_info.input_operands[0] as usize];
        assert_eq!(input.name.as_deref(), Some("x"));
        assert_eq!(input.descriptor.shape.len(), 2);
        match &input.descriptor.shape[0] {
            Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 16);
            }
            _ => panic!("expected dynamic dimension at axis 0"),
        }
        assert_eq!(input.descriptor.shape[1], Dimension::Static(64));
    }

    #[test]
    fn test_to_graph_json_preserves_dynamic_dimensions() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            Dimension::Dynamic(DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            Dimension::Static(3),
                        ],
                        pending_permutation: vec![],
                    },
                    name: Some("x".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            Dimension::Dynamic(DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            Dimension::Static(3),
                        ],
                        pending_permutation: vec![],
                    },
                    name: Some("y".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let json = to_graph_json(&graph, false).expect("to_graph_json");
        let input_desc = json.inputs.get("x").expect("input x");
        assert_eq!(input_desc.shape.len(), 2);
        match &input_desc.shape[0] {
            webnn_graph::ast::Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 8);
            }
            _ => panic!("expected dynamic input dimension"),
        }
        assert_eq!(input_desc.shape[1], webnn_graph::ast::Dimension::Static(3));
    }

    #[test]
    fn test_to_graph_json_rejects_dynamic_constant_shape() {
        let mut constants = HashMap::new();
        constants.insert(
            0u32,
            ConstantData {
                data: vec![0u8; 4],
                label: None,
            },
        );

        let graph = GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Constant,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![Dimension::Dynamic(DynamicDimension {
                        name: "n".to_string(),
                        max_size: 4,
                    })],
                    pending_permutation: vec![],
                },
                name: Some("const_dynamic".to_string()),
            }],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: constants,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let err = to_graph_json(&graph, false).unwrap_err();
        match err {
            GraphError::ConversionFailed { reason, .. } => {
                assert!(reason.contains("constant operand"));
                assert!(reason.contains("has dynamic shape"));
            }
            _ => panic!("expected ConversionFailed for dynamic constant shape"),
        }
    }

    #[test]
    fn test_from_graph_json_dynamic_expand_shape_inference_subset() {
        use webnn_graph::ast::{
            DataType as WDataType, Dimension as WDimension, DynamicDimension as WDynamicDimension,
            OperandDesc,
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    WDimension::Static(1),
                ],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert(
            "newShape".to_string(),
            serde_json::json!([
                { "name": "batch", "maxSize": 8 },
                4
            ]),
        );

        let nodes = vec![Node {
            id: "n0".to_string(),
            op: "expand".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: Some(vec!["y".to_string()]),
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("y".to_string(), "y".to_string());

        let graph_json = GraphJson {
            name: Some("dynamic_expand_subset".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph = from_graph_json(&graph_json).expect("from_graph_json");
        let y_id = graph.output_operands[0] as usize;
        let y_shape = &graph.operands[y_id].descriptor.shape;
        assert_eq!(y_shape.len(), 2);
        match &y_shape[0] {
            Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 8);
            }
            _ => panic!("expected dynamic batch dimension"),
        }
        assert_eq!(y_shape[1], Dimension::Static(4));
    }

    #[test]
    fn test_from_graph_json_dynamic_reshape_shape_inference_subset() {
        use webnn_graph::ast::{
            DataType as WDataType, Dimension as WDimension, DynamicDimension as WDynamicDimension,
            OperandDesc,
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "x".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    WDimension::Static(2),
                    WDimension::Static(2),
                ],
            },
        );

        let mut options = serde_json::Map::new();
        options.insert(
            "newShape".to_string(),
            serde_json::json!([
                { "name": "batch", "maxSize": 8 },
                4
            ]),
        );

        let nodes = vec![Node {
            id: "n0".to_string(),
            op: "reshape".to_string(),
            inputs: vec!["x".to_string()],
            options,
            outputs: Some(vec!["y".to_string()]),
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("y".to_string(), "y".to_string());

        let graph_json = GraphJson {
            name: Some("dynamic_reshape_subset".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph = from_graph_json(&graph_json).expect("from_graph_json");
        let y_id = graph.output_operands[0] as usize;
        let y_shape = &graph.operands[y_id].descriptor.shape;
        assert_eq!(y_shape.len(), 2);
        match &y_shape[0] {
            Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 8);
            }
            _ => panic!("expected dynamic batch dimension"),
        }
        assert_eq!(y_shape[1], Dimension::Static(4));
    }

    #[test]
    fn test_from_graph_json_dynamic_where_broadcast_subset() {
        use webnn_graph::ast::{
            DataType as WDataType, Dimension as WDimension, DynamicDimension as WDynamicDimension,
            OperandDesc,
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            "cond".to_string(),
            OperandDesc {
                data_type: WDataType::Uint8,
                shape: vec![WDimension::Static(1), WDimension::Static(4)],
            },
        );
        inputs.insert(
            "a".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    WDimension::Static(4),
                ],
            },
        );
        inputs.insert(
            "b".to_string(),
            OperandDesc {
                data_type: WDataType::Float32,
                shape: vec![
                    WDimension::Dynamic(WDynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    WDimension::Static(1),
                ],
            },
        );

        let nodes = vec![Node {
            id: "n0".to_string(),
            op: "where".to_string(),
            inputs: vec!["cond".to_string(), "a".to_string(), "b".to_string()],
            options: serde_json::Map::new(),
            outputs: Some(vec!["y".to_string()]),
        }];

        let mut outputs = BTreeMap::new();
        outputs.insert("y".to_string(), "y".to_string());

        let graph_json = GraphJson {
            name: Some("dynamic_where_subset".to_string()),
            format: "webnn-graph-json".to_string(),
            version: 2,
            quantized: false,
            inputs,
            consts: BTreeMap::new(),
            nodes,
            outputs,
        };

        let graph = from_graph_json(&graph_json).expect("from_graph_json");
        let y_id = graph.output_operands[0] as usize;
        let y_shape = &graph.operands[y_id].descriptor.shape;
        assert_eq!(y_shape.len(), 2);
        match &y_shape[0] {
            Dimension::Dynamic(d) => {
                assert_eq!(d.name, "batch");
                assert_eq!(d.max_size, 8);
            }
            _ => panic!("expected dynamic batch dimension"),
        }
        assert_eq!(y_shape[1], Dimension::Static(4));
    }
}
