use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde_json::json;

use crate::converters::ConvertedGraph;
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, OperandKind};

#[derive(Default)]
pub struct CoremlConverter;

impl CoremlConverter {
    fn operand_name(graph: &GraphInfo, id: u32) -> String {
        graph
            .operand(id)
            .and_then(|op| op.name.clone())
            .unwrap_or_else(|| format!("operand_{}", id))
    }

    fn data_type_code(data_type: DataType) -> Result<&'static str, GraphError> {
        // Simplified CoreML type names
        let code = match data_type {
            DataType::Float32 => "float32",
            DataType::Float16 => "float16",
            DataType::Int32 => "int32",
            DataType::Uint32 => "uint32",
            DataType::Int8 => "int8",
            DataType::Uint8 => "uint8",
        };
        Ok(code)
    }

    fn tensor_type(
        desc: &crate::graph::OperandDescriptor,
    ) -> Result<serde_json::Value, GraphError> {
        Ok(json!({
            "dataType": Self::data_type_code(desc.data_type)?,
            "shape": desc.shape,
        }))
    }
}

impl crate::converters::GraphConverter for CoremlConverter {
    fn format(&self) -> &'static str {
        "coreml"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let inputs = graph
            .input_operands
            .iter()
            .map(|id| -> Result<serde_json::Value, GraphError> {
                let operand = graph
                    .operand(*id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
                Ok(json!({
                    "name": Self::operand_name(graph, *id),
                    "type": Self::tensor_type(&operand.descriptor)?,
                }))
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let outputs = graph
            .output_operands
            .iter()
            .map(|id| -> Result<serde_json::Value, GraphError> {
                let operand = graph
                    .operand(*id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
                Ok(json!({
                    "name": Self::operand_name(graph, *id),
                    "type": Self::tensor_type(&operand.descriptor)?,
                }))
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let constants = graph
            .constant_operand_ids_to_handles
            .iter()
            .map(|(id, data)| -> Result<serde_json::Value, GraphError> {
                let operand = graph
                    .operand(*id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
                Ok(json!({
                    "name": Self::operand_name(graph, *id),
                    "type": Self::tensor_type(&operand.descriptor)?,
                    "rawData": BASE64_STANDARD.encode(&data.data),
                }))
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let layers = graph
            .operations
            .iter()
            .enumerate()
            .map(|(idx, op)| {
                let layer_name = op
                    .label
                    .as_ref()
                    .cloned()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| format!("{}_{}", op.op_type, idx));
                json!({
                    "name": layer_name,
                    "type": op.op_type,
                    "input": op.input_operands.iter().map(|id| Self::operand_name(graph, *id)).collect::<Vec<_>>(),
                    "output": vec![Self::operand_name(graph, op.output_operand)],
                    "attributes": op.attributes, // retained for downstream mapping
                })
            })
            .collect::<Vec<_>>();

        let intermediates = graph
            .operands
            .iter()
            .enumerate()
            .filter(|(_, operand)| operand.kind != OperandKind::Constant)
            .map(|(id, operand)| -> Result<serde_json::Value, GraphError> {
                Ok(json!({
                    "name": Self::operand_name(graph, id as u32),
                    "type": Self::tensor_type(&operand.descriptor)?,
                }))
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let model = json!({
            "specificationVersion": 7,
            "producer": "rust-webnn-graph",
            "description": {
                "input": inputs,
                "output": outputs,
                "intermediate": intermediates,
            },
            "neuralNetwork": {
                "layers": layers,
                "weights": constants,
            },
        });

        let data =
            serde_json::to_vec_pretty(&model).map_err(|err| GraphError::ConversionFailed {
                format: "coreml".to_string(),
                reason: err.to_string(),
            })?;

        Ok(ConvertedGraph {
            format: "coreml",
            content_type: "application/json",
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::CoremlConverter;
    use crate::converters::GraphConverter;
    use crate::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation};

    #[test]
    fn exports_minimal_coreml_json() {
        let converter = CoremlConverter::default();
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 3],
                        pending_permutation: vec![],
                    },
                    name: Some("x".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 3],
                        pending_permutation: vec![],
                    },
                    name: Some("y".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 3],
                        pending_permutation: vec![],
                    },
                    name: Some("z".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation {
                op_type: "add".to_string(),
                input_operands: vec![0, 1],
                output_operand: 2,
                attributes: serde_json::json!({"alpha": 1.0}),
                label: None,
            }],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
        };

        let converted = converter.convert(&graph).unwrap();
        let json: Value = serde_json::from_slice(&converted.data).unwrap();
        assert_eq!(json["description"]["input"][0]["name"], "x");
        assert_eq!(json["description"]["output"][0]["name"], "z");
        assert_eq!(json["neuralNetwork"]["layers"][0]["type"], "add");
    }
}
