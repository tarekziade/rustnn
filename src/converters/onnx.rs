use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde_json::json;

use crate::converters::ConvertedGraph;
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, OperandKind};

#[derive(Default)]
pub struct OnnxConverter;

impl OnnxConverter {
    fn operand_name(graph: &GraphInfo, id: u32) -> String {
        graph
            .operand(id)
            .and_then(|op| op.name.clone())
            .unwrap_or_else(|| format!("operand_{}", id))
    }

    fn data_type_code(data_type: DataType) -> Result<i32, GraphError> {
        // TensorProto.DataType enum values from ONNX 1.15
        let code = match data_type {
            DataType::Float32 => 1,
            DataType::Uint8 => 2,
            DataType::Int8 => 3,
            DataType::Int32 => 6,
            DataType::Float16 => 10,
            DataType::Uint32 => 12,
        };
        Ok(code)
    }

    fn tensor_type(
        desc: &crate::graph::OperandDescriptor,
    ) -> Result<serde_json::Value, GraphError> {
        Ok(json!({
            "elem_type": Self::data_type_code(desc.data_type)?,
            "shape": {
                "dim": desc.shape.iter().map(|d| json!({"dim_value": d})).collect::<Vec<_>>()
            }
        }))
    }
}

impl crate::converters::GraphConverter for OnnxConverter {
    fn format(&self) -> &'static str {
        "onnx"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut initializers = Vec::new();

        for &id in &graph.input_operands {
            let operand = graph
                .operand(id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: id })?;
            let name = Self::operand_name(graph, id);
            inputs.push(json!({
                "name": name,
                "type": {
                    "tensor_type": Self::tensor_type(&operand.descriptor)?
                }
            }));
        }

        for &id in &graph.output_operands {
            let operand = graph
                .operand(id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: id })?;
            let name = Self::operand_name(graph, id);
            outputs.push(json!({
                "name": name,
                "type": {
                    "tensor_type": Self::tensor_type(&operand.descriptor)?
                }
            }));
        }

        for (id, data) in &graph.constant_operand_ids_to_handles {
            let operand = graph
                .operand(*id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
            let name = Self::operand_name(graph, *id);
            let dims = operand
                .descriptor
                .shape
                .iter()
                .map(|d| json!({ "dim_value": d }))
                .collect::<Vec<_>>();
            let data_type = Self::data_type_code(operand.descriptor.data_type)?;
            initializers.push(json!({
                "name": name,
                "data_type": data_type,
                "dims": dims,
                "raw_data": BASE64_STANDARD.encode(&data.data)
            }));
        }

        let nodes = graph
            .operations
            .iter()
            .enumerate()
            .map(|(idx, op)| {
                let node_name = op
                    .label
                    .as_ref()
                    .cloned()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| format!("{}_{}", op.op_type, idx));
                json!({
                    "name": node_name,
                    "op_type": op.op_type,
                    "input": op.input_operands.iter().map(|id| Self::operand_name(graph, *id)).collect::<Vec<_>>(),
                    "output": vec![Self::operand_name(graph, op.output_operand)],
                    "attribute": vec![json!({"name": "webnn_attributes", "type": "STRING", "s": op.attributes.to_string()})],
                })
            })
            .collect::<Vec<_>>();

        let value_info = graph
            .operands
            .iter()
            .enumerate()
            .filter(|(_, operand)| operand.kind != OperandKind::Constant)
            .map(|(id, operand)| {
                Ok(json!({
                    "name": Self::operand_name(graph, id as u32),
                    "type": { "tensor_type": Self::tensor_type(&operand.descriptor)? }
                }))
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let model = json!({
            "ir_version": 8,
            "opset_import": [{"domain": "", "version": 13}],
            "producer_name": "rust-webnn-graph",
            "graph": {
                "name": "webnn_graph",
                "node": nodes,
                "input": inputs,
                "output": outputs,
                "initializer": initializers,
                "value_info": value_info,
            }
        });

        let data =
            serde_json::to_vec_pretty(&model).map_err(|err| GraphError::ConversionFailed {
                format: "onnx".to_string(),
                reason: err.to_string(),
            })?;

        Ok(ConvertedGraph {
            format: "onnx",
            content_type: "application/json",
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::OnnxConverter;
    use crate::converters::GraphConverter;
    use crate::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation};

    #[test]
    fn exports_minimal_onnx_json() {
        let converter = OnnxConverter::default();
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
                op_type: "Add".to_string(),
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
        assert_eq!(json["graph"]["node"][0]["op_type"], "Add");
        assert_eq!(json["graph"]["input"][0]["name"], "x");
        assert_eq!(json["graph"]["output"][0]["name"], "z");
        assert_eq!(json["graph"]["node"][0]["output"][0], "z");
    }
}
