use std::collections::{HashMap, HashSet};

use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, OperandDescriptor, OperandKind};

#[derive(Debug, Clone)]
pub struct ContextProperties {
    pub tensor_byte_length_limit: usize,
    pub allowed_io_data_types: HashSet<DataType>,
}

impl Default for ContextProperties {
    fn default() -> Self {
        let allowed_io_data_types = [
            DataType::Int4,
            DataType::Uint4,
            DataType::Float32,
            DataType::Float16,
            DataType::Int32,
            DataType::Uint32,
            DataType::Int8,
            DataType::Uint8,
            DataType::Int64,
            DataType::Uint64,
        ]
        .into_iter()
        .collect();
        Self {
            tensor_byte_length_limit: 256 * 1024 * 1024, // 256MB to support WPT large tensor tests
            allowed_io_data_types,
        }
    }
}

#[derive(Debug)]
pub struct ValidationArtifacts {
    pub input_names_to_descriptors: HashMap<String, OperandDescriptor>,
    pub output_names_to_descriptors: HashMap<String, OperandDescriptor>,
    pub operand_to_dependent_operations: HashMap<u32, Vec<String>>,
    pub operand_to_producing_operation: HashMap<u32, String>,
}

pub struct GraphValidator<'a> {
    graph: &'a GraphInfo,
    context: ContextProperties,
    processed_operands: HashSet<u32>,
    operand_to_dependents: HashMap<u32, Vec<String>>,
    operand_to_producer: HashMap<u32, String>,
}

impl<'a> GraphValidator<'a> {
    pub fn new(graph: &'a GraphInfo, context: ContextProperties) -> Self {
        Self {
            graph,
            context,
            processed_operands: HashSet::new(),
            operand_to_dependents: HashMap::new(),
            operand_to_producer: HashMap::new(),
        }
    }

    pub fn validate(mut self) -> Result<ValidationArtifacts, GraphError> {
        if self.graph.operands.is_empty()
            || self.graph.operations.is_empty()
            || self.graph.output_operands.is_empty()
        {
            return Err(GraphError::EmptyGraph);
        }
        if self.graph.operands.len() >= u32::MAX as usize {
            return Err(GraphError::TooManyOperands {
                count: self.graph.operands.len(),
            });
        }

        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();
        let mut graph_inputs = Vec::with_capacity(self.graph.input_operands.len());
        let mut graph_outputs = Vec::with_capacity(self.graph.output_operands.len());
        let mut constant_handles = self.graph.constant_operand_ids_to_handles.clone();
        let tensor_constants = &self.graph.id_to_constant_tensor_operand_map;
        for (idx, operand) in self.graph.operands.iter().enumerate() {
            let operand_id = idx as u32;
            let descriptor = &operand.descriptor;
            let byte_length =
                descriptor
                    .byte_length()
                    .ok_or(GraphError::OperandElementCountOverflow {
                        operand: operand_id,
                    })?;
            if byte_length > self.context.tensor_byte_length_limit {
                return Err(GraphError::TensorLimit {
                    operand: operand_id,
                    byte_length,
                    limit: self.context.tensor_byte_length_limit,
                });
            }

            match operand.kind {
                OperandKind::Input => {
                    let name = operand.name.as_ref().ok_or(GraphError::MissingInputName {
                        operand: operand_id,
                    })?;
                    if name.is_empty() {
                        return Err(GraphError::MissingInputName {
                            operand: operand_id,
                        });
                    }
                    if !self
                        .context
                        .allowed_io_data_types
                        .contains(&descriptor.data_type)
                    {
                        return Err(GraphError::UnsupportedIoDataType {
                            operand: operand_id,
                            data_type: descriptor.data_type,
                        });
                    }
                    if inputs.insert(name.clone(), descriptor.clone()).is_some() {
                        return Err(GraphError::DuplicateInputName { name: name.clone() });
                    }
                    graph_inputs.push(operand_id);
                    self.processed_operands.insert(operand_id);
                }
                OperandKind::Output => {
                    // Only treat as a graph output if it is listed in graph.output_operands
                    if self.graph.output_operands.contains(&operand_id)
                        && let Some(name) = operand.name.as_ref()
                    {
                        if name.is_empty() {
                            return Err(GraphError::MissingOutputName {
                                operand: operand_id,
                            });
                        }
                        if !self
                            .context
                            .allowed_io_data_types
                            .contains(&descriptor.data_type)
                        {
                            return Err(GraphError::UnsupportedIoDataType {
                                operand: operand_id,
                                data_type: descriptor.data_type,
                            });
                        }
                        if outputs.insert(name.clone(), descriptor.clone()).is_some() {
                            return Err(GraphError::DuplicateOutputName { name: name.clone() });
                        }
                        graph_outputs.push(operand_id);
                    }
                }
                OperandKind::Constant => {
                    if let Some(data) = constant_handles.remove(&operand_id) {
                        if data.data.len() != byte_length {
                            return Err(GraphError::ConstantLengthMismatch {
                                operand: operand_id,
                                expected: byte_length,
                                actual: data.data.len(),
                            });
                        }
                    } else if !tensor_constants.contains_key(&operand_id) {
                        return Err(GraphError::MissingConstantData {
                            operand: operand_id,
                        });
                    }
                    self.processed_operands.insert(operand_id);
                }
            }
        }

        if graph_inputs != self.graph.input_operands {
            return Err(GraphError::InputOperandListMismatch);
        }
        if graph_outputs != self.graph.output_operands {
            return Err(GraphError::OutputOperandListMismatch);
        }
        if !constant_handles.is_empty() {
            return Err(GraphError::UnusedConstantHandles);
        }

        self.validate_operations()?;
        self.validate_operand_usage()?;

        Ok(ValidationArtifacts {
            input_names_to_descriptors: inputs,
            output_names_to_descriptors: outputs,
            operand_to_dependent_operations: self.operand_to_dependents,
            operand_to_producing_operation: self.operand_to_producer,
        })
    }

    fn validate_operations(&mut self) -> Result<(), GraphError> {
        for operation in &self.graph.operations {
            let op_name = operation.display_name();
            for &input_id in &operation.input_operands {
                self.graph.operand(input_id).ok_or_else(|| {
                    GraphError::InvalidOperandReference {
                        operation: op_name.clone(),
                        operand: input_id,
                    }
                })?;
                if !self.processed_operands.contains(&input_id) {
                    return Err(GraphError::OperandNotReady {
                        operation: op_name.clone(),
                        operand: input_id,
                    });
                }
                self.operand_to_dependents
                    .entry(input_id)
                    .or_default()
                    .push(op_name.clone());
            }

            // Handle both single and multi-output operations
            for &output_id in operation.output_operands_slice() {
                self.graph.operand(output_id).ok_or_else(|| {
                    GraphError::InvalidOperandReference {
                        operation: op_name.clone(),
                        operand: output_id,
                    }
                })?;
                if self.operand_to_producer.contains_key(&output_id) {
                    return Err(GraphError::OperandProducedTwice {
                        operation: op_name.clone(),
                        operand: output_id,
                    });
                }
                self.operand_to_producer
                    .insert(output_id, operation.op_type.clone());
                self.processed_operands.insert(output_id);
            }

            // Operation-specific validation
            match operation.op_type.as_str() {
                "quantizeLinear" => self.validate_quantize_like(operation, true)?,
                "dequantizeLinear" => self.validate_quantize_like(operation, false)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn validate_operand_usage(&self) -> Result<(), GraphError> {
        for (idx, operand) in self.graph.operands.iter().enumerate() {
            let operand_id = idx as u32;
            match operand.kind {
                OperandKind::Output => {
                    if operand.name.is_some() && !self.processed_operands.contains(&operand_id) {
                        return Err(GraphError::OutputNotProduced {
                            operand: operand_id,
                        });
                    }
                }
                OperandKind::Constant => {
                    // Unused constants are harmless; skip error to allow conversion of graphs
                    // that carry extra weight entries.
                    continue;
                }
                _ => {
                    if !self.operand_to_dependents.contains_key(&operand_id) {
                        return Err(GraphError::OperandNeverUsed {
                            operand: operand_id,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_quantize_like(
        &self,
        operation: &crate::graph::Operation,
        is_quantize: bool,
    ) -> Result<(), GraphError> {
        let op_name = operation.display_name();
        let invalid = |reason: String| GraphError::QuantizationValidation {
            operation: op_name.clone(),
            reason,
        };

        if operation.input_operands.len() != 3 {
            return Err(invalid(format!(
                "expected 3 inputs (input, scale, zeroPoint), got {}",
                operation.input_operands.len()
            )));
        }
        let output_id = operation
            .output_operand
            .ok_or_else(|| invalid("missing output operand".to_string()))?;

        let input_desc = self
            .graph
            .operand(operation.input_operands[0])
            .map(|o| &o.descriptor)
            .ok_or_else(|| GraphError::InvalidOperandReference {
                operation: op_name.clone(),
                operand: operation.input_operands[0],
            })?;
        let scale_desc = self
            .graph
            .operand(operation.input_operands[1])
            .map(|o| &o.descriptor)
            .ok_or_else(|| GraphError::InvalidOperandReference {
                operation: op_name.clone(),
                operand: operation.input_operands[1],
            })?;
        let zero_point_desc = self
            .graph
            .operand(operation.input_operands[2])
            .map(|o| &o.descriptor)
            .ok_or_else(|| GraphError::InvalidOperandReference {
                operation: op_name.clone(),
                operand: operation.input_operands[2],
            })?;
        let output_desc = self
            .graph
            .operand(output_id)
            .map(|o| &o.descriptor)
            .ok_or_else(|| GraphError::InvalidOperandReference {
                operation: op_name.clone(),
                operand: output_id,
            })?;

        // Dtype constraints
        let scale_ok = matches!(scale_desc.data_type, DataType::Float16 | DataType::Float32);
        if !scale_ok {
            return Err(invalid(format!(
                "scale must be float16 or float32 (got {:?})",
                scale_desc.data_type
            )));
        }
        let zero_point_ok = matches!(
            zero_point_desc.data_type,
            DataType::Int4 | DataType::Uint4 | DataType::Int8 | DataType::Uint8 | DataType::Int32
        );
        if !zero_point_ok {
            return Err(invalid(format!(
                "zeroPoint must be int4/uint4/int8/uint8/int32 (got {:?})",
                zero_point_desc.data_type
            )));
        }

        if is_quantize {
            let input_ok = matches!(
                input_desc.data_type,
                DataType::Float16 | DataType::Float32 | DataType::Int32
            );
            if !input_ok {
                return Err(invalid(format!(
                    "quantize input must be float16/float32/int32 (got {:?})",
                    input_desc.data_type
                )));
            }
            if output_desc.data_type != zero_point_desc.data_type {
                return Err(invalid(format!(
                    "quantize output dtype {:?} must match zeroPoint dtype {:?}",
                    output_desc.data_type, zero_point_desc.data_type
                )));
            }
        } else {
            let input_ok = matches!(
                input_desc.data_type,
                DataType::Int4
                    | DataType::Uint4
                    | DataType::Int8
                    | DataType::Uint8
                    | DataType::Int32
            );
            if !input_ok {
                return Err(invalid(format!(
                    "dequantize input must be int4/uint4/int8/uint8/int32 (got {:?})",
                    input_desc.data_type
                )));
            }
            if !matches!(output_desc.data_type, DataType::Float32) {
                return Err(invalid(format!(
                    "dequantize output must be float32 (got {:?})",
                    output_desc.data_type
                )));
            }
        }

        // Shape constraints
        let input_shape = &input_desc.shape;
        let scale_shape = &scale_desc.shape;
        let zero_point_shape = &zero_point_desc.shape;

        if scale_shape.is_empty() {
            if !zero_point_shape.is_empty() {
                return Err(invalid(format!(
                    "zeroPoint shape {:?} must match scalar scale for per-tensor quantization",
                    zero_point_shape
                )));
            }
        } else {
            if scale_shape.len() != input_shape.len() {
                return Err(invalid(format!(
                    "scale rank {} must match input rank {}",
                    scale_shape.len(),
                    input_shape.len()
                )));
            }
            if zero_point_shape != scale_shape {
                return Err(invalid(format!(
                    "zeroPoint shape {:?} must match scale shape {:?}",
                    zero_point_shape, scale_shape
                )));
            }
        }
        if output_desc.shape != *input_shape {
            return Err(invalid(format!(
                "output shape {:?} must match input shape {:?}",
                output_desc.shape, input_shape
            )));
        }

        let mut non_one_dims = Vec::new();
        for (idx, &dim) in scale_shape.iter().enumerate() {
            if dim != 1 {
                non_one_dims.push(idx);
            }
        }

        let is_per_tensor = non_one_dims.is_empty();
        let is_per_axis = non_one_dims.len() == 1
            && scale_shape[non_one_dims[0]] == *input_shape.get(non_one_dims[0]).unwrap_or(&0);

        if !(is_per_tensor || is_per_axis) {
            // Blockwise: allow divisibility along differing dims
            for (i, (&scale_dim, &input_dim)) in
                scale_shape.iter().zip(input_shape.iter()).enumerate()
            {
                if scale_dim == 1 || scale_dim == input_dim {
                    continue;
                }
                if scale_dim == 0 || input_dim == 0 || input_dim % scale_dim != 0 {
                    return Err(invalid(format!(
                        "scale dim {} (value {}) must divide input dim {} (value {}) for blockwise quantization",
                        i, scale_dim, i, input_dim
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ConstantData, GraphInfo, Operand, Operation};

    fn constant_data_for(descriptor: &OperandDescriptor) -> ConstantData {
        let len = descriptor.byte_length().expect("valid byte length");
        ConstantData {
            data: vec![0u8; len],
            label: None,
        }
    }

    fn build_quantize_graph(
        op_type: &str,
        input_dtype: DataType,
        scale_dtype: DataType,
        zero_point_dtype: DataType,
        input_shape: Vec<u32>,
        scale_shape: Vec<u32>,
    ) -> GraphInfo {
        let input_operand = Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: input_dtype,
                shape: input_shape.clone(),
                pending_permutation: Vec::new(),
            },
            name: Some("input".to_string()),
        };

        let scale_descriptor = OperandDescriptor {
            data_type: scale_dtype,
            shape: scale_shape.clone(),
            pending_permutation: Vec::new(),
        };
        let scale_operand = Operand {
            kind: OperandKind::Constant,
            descriptor: scale_descriptor.clone(),
            name: None,
        };

        let zero_point_descriptor = OperandDescriptor {
            data_type: zero_point_dtype,
            shape: scale_shape.clone(),
            pending_permutation: Vec::new(),
        };
        let zero_point_operand = Operand {
            kind: OperandKind::Constant,
            descriptor: zero_point_descriptor.clone(),
            name: None,
        };

        let output_descriptor = OperandDescriptor {
            data_type: if op_type == "quantizeLinear" {
                zero_point_dtype
            } else {
                DataType::Float32
            },
            shape: input_shape.clone(),
            pending_permutation: Vec::new(),
        };
        let output_operand = Operand {
            kind: OperandKind::Output,
            descriptor: output_descriptor.clone(),
            name: Some("output".to_string()),
        };

        let operation = Operation {
            op_type: op_type.to_string(),
            input_operands: vec![0, 1, 2],
            output_operand: Some(3),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        let mut constants = HashMap::new();
        constants.insert(1, constant_data_for(&scale_descriptor));
        constants.insert(2, constant_data_for(&zero_point_descriptor));

        GraphInfo {
            operands: vec![
                input_operand,
                scale_operand,
                zero_point_operand,
                output_operand,
            ],
            input_operands: vec![0],
            output_operands: vec![3],
            operations: vec![operation],
            constant_operand_ids_to_handles: constants,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn quantize_per_axis_int8_validates() {
        let graph = build_quantize_graph(
            "quantizeLinear",
            DataType::Float32,
            DataType::Float32,
            DataType::Int8,
            vec![1, 3, 4],
            vec![1, 3, 1],
        );
        let validator = GraphValidator::new(&graph, ContextProperties::default());
        assert!(validator.validate().is_ok());
    }

    #[test]
    fn quantize_blockwise_int4_validates() {
        let graph = build_quantize_graph(
            "quantizeLinear",
            DataType::Float16,
            DataType::Float16,
            DataType::Uint4,
            vec![4, 4],
            vec![2, 2],
        );
        let validator = GraphValidator::new(&graph, ContextProperties::default());
        assert!(validator.validate().is_ok());
    }

    #[test]
    fn quantize_rank_mismatch_fails() {
        let graph = build_quantize_graph(
            "quantizeLinear",
            DataType::Float32,
            DataType::Float32,
            DataType::Uint8,
            vec![2, 2],
            vec![1],
        );
        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        match err {
            GraphError::QuantizationValidation { reason, .. } => {
                assert!(reason.contains("rank"), "unexpected reason: {}", reason);
            }
            other => panic!("unexpected error {:?}", other),
        }
    }

    #[test]
    fn dequantize_invalid_scale_type_fails() {
        let graph = build_quantize_graph(
            "dequantizeLinear",
            DataType::Uint8,
            DataType::Uint8,
            DataType::Uint8,
            vec![2, 2],
            vec![1, 1],
        );
        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        match err {
            GraphError::QuantizationValidation { reason, .. } => {
                assert!(reason.contains("scale"), "unexpected reason: {}", reason);
            }
            other => panic!("unexpected error {:?}", other),
        }
    }

    #[test]
    fn test_empty_graph_fails() {
        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::EmptyGraph));
    }

    #[test]
    fn test_missing_input_name_fails() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: None, // Missing name
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation {
                op_type: "relu".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: vec![],
                attributes: serde_json::json!({}),
                label: None,
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::MissingInputName { .. }));
    }

    #[test]
    fn test_empty_input_name_fails() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("".to_string()), // Empty name
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation {
                op_type: "relu".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: vec![],
                attributes: serde_json::json!({}),
                label: None,
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::MissingInputName { .. }));
    }

    #[test]
    fn test_duplicate_input_name_fails() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()), // Duplicate name
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation {
                op_type: "add".to_string(),
                input_operands: vec![0, 1],
                output_operand: Some(2),
                output_operands: vec![],
                attributes: serde_json::json!({}),
                label: None,
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::DuplicateInputName { .. }));
    }

    #[test]
    fn test_duplicate_output_name_fails() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()), // Duplicate name
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1, 2],
            operations: vec![
                Operation {
                    op_type: "relu".to_string(),
                    input_operands: vec![0],
                    output_operand: Some(1),
                    output_operands: vec![],
                    attributes: serde_json::json!({}),
                    label: None,
                },
                Operation {
                    op_type: "sigmoid".to_string(),
                    input_operands: vec![0],
                    output_operand: Some(2),
                    output_operands: vec![],
                    attributes: serde_json::json!({}),
                    label: None,
                },
            ],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::DuplicateOutputName { .. }));
    }

    #[test]
    fn test_missing_constant_data_fails() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![2, 2],
                        pending_permutation: vec![],
                    },
                    name: None,
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![2],
            operations: vec![Operation {
                op_type: "add".to_string(),
                input_operands: vec![0, 1],
                output_operand: Some(2),
                output_operands: vec![],
                attributes: serde_json::json!({}),
                label: None,
            }],
            constant_operand_ids_to_handles: HashMap::new(), // Missing constant data
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::MissingConstantData { .. }));
    }

    #[test]
    fn test_constant_length_mismatch_fails() {
        let constant_descriptor = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![2, 2],
            pending_permutation: vec![],
        };

        let mut constants = HashMap::new();
        constants.insert(
            1,
            ConstantData {
                data: vec![0u8; 8], // Wrong size - should be 16 bytes (4 floats * 4 bytes)
                label: None,
            },
        );

        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Constant,
                    descriptor: constant_descriptor,
                    name: None,
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![2],
            operations: vec![Operation {
                op_type: "add".to_string(),
                input_operands: vec![0, 1],
                output_operand: Some(2),
                output_operands: vec![],
                attributes: serde_json::json!({}),
                label: None,
            }],
            constant_operand_ids_to_handles: constants,
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::ConstantLengthMismatch { .. }));
    }

    #[test]
    fn test_tensor_byte_limit_exceeded_fails() {
        let mut context = ContextProperties::default();
        context.tensor_byte_length_limit = 10; // Very small limit

        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![100, 100], // 40,000 bytes - exceeds limit
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![100, 100],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation {
                op_type: "relu".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: vec![],
                attributes: serde_json::json!({}),
                label: None,
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, context);
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::TensorLimit { .. }));
    }

    #[test]
    fn test_invalid_operand_reference_fails() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation {
                op_type: "relu".to_string(),
                input_operands: vec![99], // Invalid operand ID
                output_operand: Some(1),
                output_operands: vec![],
                attributes: serde_json::json!({}),
                label: None,
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::InvalidOperandReference { .. }));
    }

    #[test]
    fn test_operand_produced_twice_fails() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![
                Operation {
                    op_type: "relu".to_string(),
                    input_operands: vec![0],
                    output_operand: Some(1),
                    output_operands: vec![],
                    attributes: serde_json::json!({}),
                    label: None,
                },
                Operation {
                    op_type: "sigmoid".to_string(),
                    input_operands: vec![0],
                    output_operand: Some(1), // Same output produced twice
                    output_operands: vec![],
                    attributes: serde_json::json!({}),
                    label: None,
                },
            ],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let err = validator.validate().unwrap_err();
        assert!(matches!(err, GraphError::OperandProducedTwice { .. }));
    }

    #[test]
    fn test_context_properties_default() {
        let context = ContextProperties::default();
        assert_eq!(context.tensor_byte_length_limit, 256 * 1024 * 1024);
        assert!(context.allowed_io_data_types.contains(&DataType::Float32));
        assert!(context.allowed_io_data_types.contains(&DataType::Int8));
        assert_eq!(context.allowed_io_data_types.len(), 10);
    }

    #[test]
    fn test_validation_artifacts_created() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("input".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 2],
                        pending_permutation: vec![],
                    },
                    name: Some("output".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation {
                op_type: "relu".to_string(),
                input_operands: vec![0],
                output_operand: Some(1),
                output_operands: vec![],
                attributes: serde_json::json!({}),
                label: None,
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let validator = GraphValidator::new(&graph, ContextProperties::default());
        let artifacts = validator.validate().unwrap();

        assert_eq!(artifacts.input_names_to_descriptors.len(), 1);
        assert_eq!(artifacts.output_names_to_descriptors.len(), 1);
        assert!(artifacts.input_names_to_descriptors.contains_key("input"));
        assert!(artifacts.output_names_to_descriptors.contains_key("output"));
        assert!(artifacts.operand_to_dependent_operations.contains_key(&0));
        assert!(artifacts.operand_to_producing_operation.contains_key(&1));
    }
}
