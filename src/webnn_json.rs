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
            ConstInit::Weights { r#ref } => {
                return Err(GraphError::ConversionFailed {
                    format: "webnn-graph-json".to_string(),
                    reason: format!(
                        "Weights reference '{}' not supported in conversion - weights must be inline",
                        r#ref
                    ),
                });
            }
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
                        kind: OperandKind::Output,
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

    // Build output operands list from outputs map
    for operand_ref in graph_json.outputs.values() {
        if let Some(&idx) = operand_map.get(operand_ref) {
            output_operands.push(idx);
            // Mark operand as output if not already
            if let Some(operand) = operands.get_mut(idx as usize)
                && !matches!(operand.kind, OperandKind::Output)
            {
                operand.kind = OperandKind::Output;
            }
        }
    }

    Ok(GraphInfo {
        operands,
        input_operands,
        output_operands,
        operations,
        constant_operand_ids_to_handles,
        id_to_constant_tensor_operand_map: HashMap::new(),
    })
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
