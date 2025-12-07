use crate::converters::ConvertedGraph;
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, Operation};
use crate::protos::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, tensor_proto::DataType as ProtoDataType,
    type_proto::Tensor as TensorTypeProto,
};
use prost::Message;

#[derive(Default)]
pub struct OnnxConverter;

impl OnnxConverter {
    fn operand_name(graph: &GraphInfo, id: u32) -> String {
        graph
            .operand(id)
            .and_then(|op| op.name.clone())
            .unwrap_or_else(|| format!("operand_{}", id))
    }

    fn data_type_code(data_type: DataType) -> ProtoDataType {
        match data_type {
            DataType::Float32 => ProtoDataType::Float,
            DataType::Uint8 => ProtoDataType::Uint8,
            DataType::Int8 => ProtoDataType::Int8,
            DataType::Int32 => ProtoDataType::Int32,
            DataType::Float16 => ProtoDataType::Float16,
            DataType::Uint32 => ProtoDataType::Uint32,
        }
    }

    fn onnx_op_type(op_type: &str) -> String {
        // Handle special cases
        if op_type.eq_ignore_ascii_case("matmul") {
            return "MatMul".to_string();
        }
        if op_type.eq_ignore_ascii_case("convTranspose2d") {
            return "ConvTranspose".to_string();
        }
        if op_type.eq_ignore_ascii_case("averagePool2d") {
            return "AveragePool".to_string();
        }
        if op_type.eq_ignore_ascii_case("maxPool2d") {
            return "MaxPool".to_string();
        }

        // Default: capitalize first letter
        let mut chars = op_type.chars();
        if let Some(first) = chars.next() {
            let mut s = first.to_ascii_uppercase().to_string();
            s.push_str(&chars.collect::<String>());
            s
        } else {
            String::new()
        }
    }

    /// Create ONNX attributes for conv2d operation
    fn create_conv2d_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON
        if let Some(strides) = op.attributes.get("strides").and_then(|v| v.as_array()) {
            let strides_i64: Vec<i64> = strides
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !strides_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("strides".to_string()),
                    ints: strides_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(dilations) = op.attributes.get("dilations").and_then(|v| v.as_array()) {
            let dilations_i64: Vec<i64> = dilations
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !dilations_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("dilations".to_string()),
                    ints: dilations_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(pads) = op.attributes.get("pads").and_then(|v| v.as_array()) {
            let pads_i64: Vec<i64> = pads
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !pads_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("pads".to_string()),
                    ints: pads_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
            attributes.push(AttributeProto {
                name: Some("group".to_string()), // Note: ONNX uses "group" not "groups"
                i: Some(groups as i64),
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for convTranspose2d operation
    fn create_conv_transpose2d_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON
        if let Some(strides) = op.attributes.get("strides").and_then(|v| v.as_array()) {
            let strides_i64: Vec<i64> = strides
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !strides_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("strides".to_string()),
                    ints: strides_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(dilations) = op.attributes.get("dilations").and_then(|v| v.as_array()) {
            let dilations_i64: Vec<i64> = dilations
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !dilations_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("dilations".to_string()),
                    ints: dilations_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(pads) = op.attributes.get("pads").and_then(|v| v.as_array()) {
            let pads_i64: Vec<i64> = pads
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !pads_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("pads".to_string()),
                    ints: pads_i64,
                    ..Default::default()
                });
            }
        }

        // output_padding attribute (specific to transposed convolution)
        if let Some(output_padding) = op
            .attributes
            .get("outputPadding")
            .and_then(|v| v.as_array())
        {
            let output_padding_i64: Vec<i64> = output_padding
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !output_padding_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("output_padding".to_string()),
                    ints: output_padding_i64,
                    ..Default::default()
                });
            }
        }

        // output_shape attribute (optional, specific to transposed convolution)
        if let Some(output_sizes) = op.attributes.get("outputSizes").and_then(|v| v.as_array()) {
            let output_shape_i64: Vec<i64> = output_sizes
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !output_shape_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("output_shape".to_string()),
                    ints: output_shape_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
            attributes.push(AttributeProto {
                name: Some("group".to_string()), // Note: ONNX uses "group" not "groups"
                i: Some(groups as i64),
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for pool2d operations
    fn create_pool2d_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON
        if let Some(window_dimensions) = op
            .attributes
            .get("windowDimensions")
            .and_then(|v| v.as_array())
        {
            let kernel_shape: Vec<i64> = window_dimensions
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !kernel_shape.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("kernel_shape".to_string()),
                    ints: kernel_shape,
                    ..Default::default()
                });
            }
        }

        if let Some(strides) = op.attributes.get("strides").and_then(|v| v.as_array()) {
            let strides_i64: Vec<i64> = strides
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !strides_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("strides".to_string()),
                    ints: strides_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(dilations) = op.attributes.get("dilations").and_then(|v| v.as_array()) {
            let dilations_i64: Vec<i64> = dilations
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !dilations_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("dilations".to_string()),
                    ints: dilations_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(pads) = op.attributes.get("pads").and_then(|v| v.as_array()) {
            let pads_i64: Vec<i64> = pads
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !pads_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("pads".to_string()),
                    ints: pads_i64,
                    ..Default::default()
                });
            }
        }

        attributes
    }
}

impl crate::converters::GraphConverter for OnnxConverter {
    fn format(&self) -> &'static str {
        "onnx"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let mut initializers = Vec::new();
        let mut inputs_val = Vec::new();
        let mut outputs_val = Vec::new();

        for &id in &graph.input_operands {
            let operand = graph
                .operand(id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: id })?;
            inputs_val.push(value_info(
                &Self::operand_name(graph, id),
                &operand.descriptor,
            ));
        }

        for &id in &graph.output_operands {
            let operand = graph
                .operand(id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: id })?;
            outputs_val.push(value_info(
                &Self::operand_name(graph, id),
                &operand.descriptor,
            ));
        }

        for (id, data) in &graph.constant_operand_ids_to_handles {
            let operand = graph
                .operand(*id)
                .ok_or_else(|| GraphError::InvalidConversionOperand { operand: *id })?;
            initializers.push(TensorProto {
                name: Some(Self::operand_name(graph, *id)),
                data_type: Some(Self::data_type_code(operand.descriptor.data_type) as i32),
                dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                raw_data: Some(prost::bytes::Bytes::from(data.data.clone())),
                ..Default::default()
            });
        }

        let nodes = graph
            .operations
            .iter()
            .enumerate()
            .map(|(idx, op)| {
                // Create attributes based on operation type
                let attributes = if op.op_type == "conv2d" {
                    Self::create_conv2d_attributes(op)
                } else if op.op_type == "convTranspose2d" {
                    Self::create_conv_transpose2d_attributes(op)
                } else if op.op_type == "averagePool2d" || op.op_type == "maxPool2d" {
                    Self::create_pool2d_attributes(op)
                } else {
                    Vec::new()
                };

                NodeProto {
                    input: op
                        .input_operands
                        .iter()
                        .map(|id| Self::operand_name(graph, *id))
                        .collect(),
                    output: vec![Self::operand_name(graph, op.output_operand)],
                    name: Some(
                        op.label
                            .clone()
                            .unwrap_or_else(|| format!("{}_{}", op.op_type, idx)),
                    ),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                }
            })
            .collect();

        let graph_proto = GraphProto {
            name: Some("webnn_graph".to_string()),
            node: nodes,
            input: inputs_val,
            output: outputs_val,
            initializer: initializers,
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: Some(7), // IR version 7 = ONNX 1.6-1.9 (supports opset 12-13)
            model_version: Some(1),
            producer_name: Some("rustnn".to_string()),
            producer_version: Some("0.1.0".to_string()),
            graph: Some(graph_proto),
            opset_import: vec![OperatorSetIdProto {
                version: Some(13),
                domain: Some("".to_string()), // Empty string = default ONNX domain
                ..Default::default()
            }],
            ..Default::default()
        };

        let data = model.encode_to_vec();

        Ok(ConvertedGraph {
            format: "onnx",
            content_type: "application/onnx",
            data,
        })
    }
}

fn value_info(name: &str, desc: &crate::graph::OperandDescriptor) -> ValueInfoProto {
    ValueInfoProto {
        name: Some(name.to_string()),
        r#type: Some(TypeProto {
            value: Some(crate::protos::onnx::type_proto::Value::TensorType(
                TensorTypeProto {
                    elem_type: Some(OnnxConverter::data_type_code(desc.data_type) as i32),
                    shape: Some(TensorShapeProto {
                        dim: desc
                            .shape
                            .iter()
                            .map(|d| crate::protos::onnx::tensor_shape_proto::Dimension {
                                value: Some(
                                    crate::protos::onnx::tensor_shape_proto::dimension::Value::DimValue(
                                        *d as i64,
                                    ),
                                ),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                    ..Default::default()
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}
