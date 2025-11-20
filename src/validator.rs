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
            DataType::Float32,
            DataType::Float16,
            DataType::Int32,
            DataType::Uint32,
            DataType::Int8,
            DataType::Uint8,
        ]
        .into_iter()
        .collect();
        Self {
            tensor_byte_length_limit: 64 * 1024 * 1024,
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
                    if let Some(name) = operand.name.as_ref() {
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

            let output_id = operation.output_operand;
            self.graph
                .operand(output_id)
                .ok_or_else(|| GraphError::InvalidOperandReference {
                    operation: op_name.clone(),
                    operand: output_id,
                })?;
            if self.operand_to_producer.contains_key(&output_id) {
                return Err(GraphError::OperandProducedTwice {
                    operation: op_name,
                    operand: output_id,
                });
            }
            self.operand_to_producer
                .insert(output_id, operation.op_type.clone());
            self.processed_operands.insert(output_id);
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
}
