use crate::error::GraphError;
use crate::graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation,
};
use std::collections::{BTreeMap, HashMap};
use webnn_graph::ast::{ConstDecl, ConstInit, GraphJson, Node, OperandDesc};

/// Convert our DataType to webnn-graph DataType
fn to_webnn_datatype(dt: &DataType) -> webnn_graph::ast::DataType {
    match dt {
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
        webnn_graph::ast::DataType::Int32 => DataType::Int32,
        webnn_graph::ast::DataType::Uint32 => DataType::Uint32,
        webnn_graph::ast::DataType::Int64 => DataType::Int64,
        webnn_graph::ast::DataType::Uint64 => DataType::Uint64,
        webnn_graph::ast::DataType::Int8 => DataType::Int8,
        webnn_graph::ast::DataType::Uint8 => DataType::Uint8,
    }
}

/// Convert GraphInfo to GraphJson
pub fn to_graph_json(graph: &GraphInfo) -> Result<GraphJson, GraphError> {
    let mut inputs = BTreeMap::new();
    let mut consts = BTreeMap::new();
    let mut nodes = Vec::new();
    let mut outputs = BTreeMap::new();

    // Process operands - separate inputs and constants
    for (idx, operand) in graph.operands.iter().enumerate() {
        let name = operand
            .name
            .clone()
            .unwrap_or_else(|| format!("operand_{}", idx));

        match &operand.kind {
            OperandKind::Input => {
                inputs.insert(
                    name,
                    OperandDesc {
                        data_type: to_webnn_datatype(&operand.descriptor.data_type),
                        shape: operand.descriptor.shape.clone(),
                    },
                );
            }
            OperandKind::Constant => {
                // Get constant data from the map
                if let Some(constant) = graph.constant_operand_ids_to_handles.get(&(idx as u32)) {
                    let init = ConstInit::InlineBytes {
                        bytes: constant.data.clone(),
                    };

                    consts.insert(
                        name,
                        ConstDecl {
                            data_type: to_webnn_datatype(&operand.descriptor.data_type),
                            shape: operand.descriptor.shape.clone(),
                            init,
                        },
                    );
                }
            }
            OperandKind::Output => {
                // Outputs are handled separately below
            }
        }
    }

    // Process operations
    for (op_idx, operation) in graph.operations.iter().enumerate() {
        let id = format!("op_{}", op_idx);

        // Collect input names
        let input_names: Vec<String> = operation
            .input_operands
            .iter()
            .map(|&idx| {
                graph.operands[idx as usize]
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("operand_{}", idx))
            })
            .collect();

        // Collect output names
        let output_operands = operation.get_output_operands();
        let output_names: Option<Vec<String>> = if !output_operands.is_empty() {
            Some(
                output_operands
                    .iter()
                    .map(|&idx| {
                        graph.operands[idx as usize]
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("operand_{}", idx))
                    })
                    .collect(),
            )
        } else {
            None
        };

        // Convert attributes to JSON options
        let options: serde_json::Map<String, serde_json::Value> =
            if let serde_json::Value::Object(map) = &operation.attributes {
                map.clone()
            } else {
                serde_json::Map::new()
            };

        nodes.push(Node {
            id,
            op: operation.op_type.clone(),
            inputs: input_names,
            options,
            outputs: output_names,
        });
    }

    // Process outputs from graph.output_operands
    for &operand_idx in &graph.output_operands {
        if let Some(operand) = graph.operands.get(operand_idx as usize) {
            let name = operand
                .name
                .clone()
                .unwrap_or_else(|| format!("operand_{}", operand_idx));
            outputs.insert(name.clone(), name);
        }
    }

    Ok(GraphJson {
        name: Some("graph".to_string()),
        format: "webnn-graph-json".to_string(),
        version: 1,
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
                shape: desc.shape.clone(),
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
                shape: const_decl.shape.clone(),
                pending_permutation: Vec::new(),
            },
            kind: OperandKind::Constant,
        });

        constant_operand_ids_to_handles.insert(idx, ConstantData { data, label: None });
    }

    // Process nodes (operations)
    for node in &graph_json.nodes {
        // Resolve input indices
        let input_operands: Vec<u32> = node
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
                        kind: OperandKind::Output, // Mark as Output (intermediate results)
                    });

                    Ok::<u32, GraphError>(idx)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Convert options to attributes JSON value
        let attributes = serde_json::Value::Object(node.options.clone());

        operations.push(Operation {
            op_type: node.op.clone(),
            input_operands,
            output_operand: output_operand_ids.first().copied(),
            output_operands: output_operand_ids,
            attributes,
            label: None,
        });
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
            _ => None,
        }
    }

    fn parse_i64_array(value: &serde_json::Value) -> Option<Vec<i64>> {
        let arr = value.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            if let Some(n) = v.as_i64() {
                out.push(n);
            } else if let Some(n) = v.as_u64() {
                out.push(n as i64);
            } else {
                return None;
            }
        }
        Some(out)
    }

    fn parse_u32_array(value: &serde_json::Value) -> Option<Vec<u32>> {
        let arr = value.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            if let Some(n) = v.as_u64() {
                out.push(n as u32);
            } else if let Some(n) = v.as_i64() {
                if n < 0 {
                    return None;
                }
                out.push(n as u32);
            } else {
                return None;
            }
        }
        Some(out)
    }

    // Run multiple passes until no more shapes can be inferred
    let max_passes = 10; // Prevent infinite loops
    for _pass in 0..max_passes {
        let mut made_progress = false;

        // Process operations in order (assumed to be in dependency order from WebNN parser)
        for op_idx in 0..graph.operations.len() {
            let op = &graph.operations[op_idx];
            let op_type = op.op_type.to_ascii_lowercase();

            // Normalize tile inputs: if shape rank is missing, set to repeats length (filled with 1s)
            if op_type == "tile"
                && let Some(repeats_len) = op
                    .attributes
                    .get("repetitions")
                    .or_else(|| op.attributes.get("repeats"))
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.len())
                && let Some(input_id) = op.input_operands.first()
            {
                let inp = &mut graph.operands[*input_id as usize];
                if inp.descriptor.shape.len() != repeats_len {
                    inp.descriptor.shape = vec![1; repeats_len];
                    made_progress = true;
                }
            }

            // Skip if output already has a shape
            if let Some(output_id) = op.output_operand
                && !graph.operands[output_id as usize]
                    .descriptor
                    .shape
                    .is_empty()
            {
                continue;
            }

            // Get input shapes and types
            let input_shapes: Vec<Vec<u32>> = op
                .input_operands
                .iter()
                .map(|&id| graph.operands[id as usize].descriptor.shape.clone())
                .collect();
            let input_types: Vec<DataType> = op
                .input_operands
                .iter()
                .map(|&id| graph.operands[id as usize].descriptor.data_type)
                .collect();

            // Infer output shape based on operation type
            let output_shape = match op_type.as_str() {
                // Binary element-wise operations (including comparisons/logical)
                "add" | "sub" | "mul" | "div" | "pow" | "max" | "min" | "greater"
                | "greaterorequal" | "less" | "lesser" | "lessorequal" | "lesserorequal"
                | "equal" | "logical_and" | "logical_or" | "logical_xor" => {
                    if input_shapes.len() >= 2 {
                        broadcast_shapes(&input_shapes[0], &input_shapes[1]).ok()
                    } else {
                        None
                    }
                }

                // Unary element-wise operations (shape unchanged)
                "abs" | "ceil" | "floor" | "neg" | "relu" | "sigmoid" | "tanh" | "exp" | "log"
                | "sqrt" | "erf" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh"
                | "cosh" | "asinh" | "acosh" | "atanh" | "round" | "sign" | "reciprocal"
                | "softplus" | "softsign" | "softmax" | "gelu" | "identity" | "cast"
                | "logical_not" => input_shapes.first().cloned(),

                // Concat
                "concat" => {
                    if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
                        if input_shapes.iter().all(|s| s.is_empty()) && axis == 0 {
                            Some(vec![input_shapes.len() as u32])
                        } else {
                            infer_concat_shape(&input_shapes, axis as u32).ok()
                        }
                    } else {
                        None
                    }
                }

                // Expand: use axes for unsqueeze-style or newShape for broadcast
                "expand" => {
                    if input_shapes.len() == 1 {
                        if let Some(axes) = op.attributes.get("axes").and_then(parse_i64_array) {
                            let rank = input_shapes[0].len() as i64;
                            let mut normalized = Vec::with_capacity(axes.len());
                            let mut valid = true;
                            for axis in axes {
                                let mut axis = axis;
                                if axis < 0 {
                                    axis += rank + 1;
                                }
                                if axis < 0 || axis > rank {
                                    valid = false;
                                    break;
                                }
                                normalized.push(axis as u32);
                            }
                            if valid {
                                infer_unsqueeze_shape(&input_shapes[0], &normalized).ok()
                            } else {
                                None
                            }
                        } else if let Some(new_shape) =
                            op.attributes.get("newShape").and_then(parse_u32_array)
                        {
                            infer_expand_shape(&input_shapes[0], &new_shape).ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }

                // Reshape - use newShape if available (static), otherwise skip (dynamic)
                "reshape" => op
                    .attributes
                    .get("newShape")
                    .and_then(|v| v.as_array())
                    .map(|new_shape_array| {
                        new_shape_array
                            .iter()
                            .filter_map(|v| v.as_u64().map(|u| u as u32))
                            .collect()
                    }),

                // Transpose
                "transpose" => {
                    if input_shapes.len() == 1 {
                        if let Some(perm_array) =
                            op.attributes.get("permutation").and_then(|v| v.as_array())
                        {
                            let perm: Vec<u32> = perm_array
                                .iter()
                                .filter_map(|v| v.as_u64().map(|u| u as u32))
                                .collect();
                            infer_transpose_shape(&input_shapes[0], Some(&perm)).ok()
                        } else {
                            // Default permutation (None = reverse axes)
                            infer_transpose_shape(&input_shapes[0], None).ok()
                        }
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
                            infer_matmul_shape(&input_shapes[0], &input_shapes[1]).ok()
                        }
                    } else {
                        None
                    }
                }

                // Reduction operations
                "reducemean" | "reducesum" | "reducemax" | "reducemin" | "reduceproduct"
                | "reducel1" | "reducel2" | "reducelogsum" | "reducelogsumexp"
                | "reducesumsquare" => {
                    if let Some(input_shape) = input_shapes.first() {
                        let rank = input_shape.len() as i64;
                        let axes = op
                            .attributes
                            .get("axes")
                            .and_then(parse_i64_array)
                            .unwrap_or_default();
                        let mut normalized_axes = Vec::with_capacity(axes.len());
                        let mut valid = true;
                        for axis in axes {
                            let mut axis = axis;
                            if axis < 0 {
                                axis += rank;
                            }
                            if axis < 0 || axis >= rank {
                                valid = false;
                                break;
                            }
                            normalized_axes.push(axis as u32);
                        }
                        if !valid {
                            None
                        } else {
                            let keep_dimensions = op
                                .attributes
                                .get("keepDimensions")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            let options = ReduceOptions {
                                axes: normalized_axes,
                                keep_dimensions,
                            };
                            infer_reduce_shape(input_shape, &options).ok()
                        }
                    } else {
                        None
                    }
                }

                // Gather
                "gather" => {
                    if let Some(shape_override) =
                        op.attributes.get("shape").and_then(parse_u32_array)
                    {
                        // Also try to back-propagate the implied indices shape when we know the data
                        // shape and axis. This helps downstream ops (e.g., Where) get proper ranks.
                        if !input_shapes.is_empty() {
                            let data_shape = &input_shapes[0];
                            let mut axis = op
                                .attributes
                                .get("axis")
                                .and_then(|v| v.as_i64())
                                .unwrap_or(0);
                            let rank = data_shape.len() as i64;
                            if axis < 0 {
                                axis += rank;
                            }
                            if axis >= 0 && (axis as usize) < data_shape.len() {
                                let tail_len = data_shape.len().saturating_sub(axis as usize + 1);
                                if let Some(indices_id) = op.input_operands.get(1)
                                    && graph.operands[*indices_id as usize]
                                        .descriptor
                                        .shape
                                        .is_empty()
                                    && shape_override.len() >= tail_len
                                {
                                    let inferred_indices =
                                        shape_override[..shape_override.len() - tail_len].to_vec();
                                    graph.operands[*indices_id as usize].descriptor.shape =
                                        inferred_indices;
                                    made_progress = true;
                                }
                            }
                        }
                        Some(shape_override)
                    } else if input_shapes.len() >= 2 {
                        let axis = op
                            .attributes
                            .get("axis")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        let rank = input_shapes[0].len() as i64;
                        let mut axis = axis;
                        if axis < 0 {
                            axis += rank;
                        }
                        if axis < 0 || axis >= rank {
                            None
                        } else {
                            infer_gather_shape(&input_shapes[0], &input_shapes[1], axis as u32).ok()
                        }
                    } else {
                        None
                    }
                }

                // Where (broadcast across condition/true/false)
                "where" => {
                    if input_shapes.len() >= 3 {
                        infer_where_shape(&input_shapes[0], &input_shapes[1], &input_shapes[2]).ok()
                    } else {
                        None
                    }
                }

                // Slice
                "slice" => {
                    if let Some(input_shape) = input_shapes.first() {
                        let rank = input_shape.len() as i64;
                        let axes = op
                            .attributes
                            .get("axes")
                            .and_then(parse_i64_array)
                            .unwrap_or_else(|| (0..rank).collect());
                        let starts = op.attributes.get("starts").and_then(parse_i64_array);
                        let ends = op.attributes.get("ends").and_then(parse_i64_array);
                        if let (Some(starts), Some(ends)) = (starts, ends) {
                            let steps = op
                                .attributes
                                .get("steps")
                                .and_then(parse_i64_array)
                                .unwrap_or_else(|| vec![1; axes.len()]);
                            if axes.len() == starts.len()
                                && axes.len() == ends.len()
                                && axes.len() == steps.len()
                            {
                                let mut output = input_shape.clone();
                                let mut valid = true;
                                for i in 0..axes.len() {
                                    let mut axis = axes[i];
                                    if axis < 0 {
                                        axis += rank;
                                    }
                                    if axis < 0 || axis >= rank {
                                        valid = false;
                                        break;
                                    }
                                    let axis = axis as usize;
                                    let dim = output[axis] as i64;
                                    let mut start = starts[i];
                                    let mut end = ends[i];
                                    let step = steps[i];
                                    if step == 0 {
                                        valid = false;
                                        break;
                                    }
                                    if start < 0 {
                                        start += dim;
                                    }
                                    if end < 0 {
                                        end += dim;
                                    }
                                    start = start.max(0).min(dim);
                                    end = end.max(0).min(dim);
                                    let step_abs = step.abs();
                                    let span = if end <= start {
                                        0
                                    } else {
                                        (end - start + (step_abs - 1)) / step_abs
                                    };
                                    output[axis] = span as u32;
                                }
                                if valid { Some(output) } else { None }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }

                // Shape
                "shape" => {
                    if let Some(input_shape) = input_shapes.first() {
                        Some(vec![input_shape.len() as u32])
                    } else {
                        None
                    }
                }

                // Constant
                "constant" => op
                    .attributes
                    .get("shape")
                    .and_then(parse_u32_array)
                    .or_else(|| Some(Vec::new())),

                // For other operations, leave shape empty (will be handled later or is dynamic)
                _ => None,
            };

            // Update output operand shape if we inferred it
            if let Some(shape) = output_shape
                && let Some(output_id) = op.output_operand
            {
                graph.operands[output_id as usize].descriptor.shape = shape;
                made_progress = true;
            }

            // Propagate output data types where deterministically known
            if let Some(output_id) = op.output_operand {
                let output_type = match op_type.as_str() {
                    "shape" => Some(DataType::Int64),
                    "constant" => op.attributes.get("dataType").and_then(parse_dtype),
                    "cast" => op.attributes.get("to").and_then(parse_dtype),
                    "expand" | "gather" | "concat" | "slice" | "reshape" | "transpose"
                    | "matmul" | "add" | "sub" | "mul" | "div" | "pow" | "max" | "min" | "abs"
                    | "ceil" | "floor" | "neg" | "relu" | "sigmoid" | "tanh" | "exp" | "log"
                    | "sqrt" | "erf" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
                    | "sinh" | "cosh" | "asinh" | "acosh" | "atanh" | "round" | "sign"
                    | "reciprocal" | "softplus" | "softsign" | "softmax" | "gelu" | "identity"
                    | "reducemean" | "reducesum" | "reducemax" | "reducemin" | "reduceproduct"
                    | "reducel1" | "reducel2" | "reducelogsum" | "reducelogsumexp"
                    | "reducesumsquare" => input_types.first().cloned(),
                    "greater" | "greaterorequal" | "less" | "lesser" | "lessorequal"
                    | "lesserorequal" | "equal" | "logical_and" | "logical_or" | "logical_xor" => {
                        Some(DataType::Uint8)
                    }
                    "where" => input_types
                        .get(1)
                        .cloned()
                        .or_else(|| input_types.get(2).cloned()),
                    _ => None,
                };
                if let Some(dtype) = output_type {
                    graph.operands[output_id as usize].descriptor.data_type = dtype;
                }
            }
        }

        // Stop if no progress was made this pass
        if !made_progress {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        ];

        for dt in types {
            let webnn_dt = to_webnn_datatype(&dt);
            let back = from_webnn_datatype(&webnn_dt);
            assert_eq!(dt, back);
        }
    }
}
