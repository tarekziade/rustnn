use crate::converters::ConvertedGraph;
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, Operation};
use crate::protos::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, attribute_proto::AttributeType,
    tensor_proto::DataType as ProtoDataType, type_proto::Tensor as TensorTypeProto,
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
            DataType::Int64 => ProtoDataType::Int64,
        }
    }

    fn onnx_op_type(op_type: &str) -> String {
        // Handle special cases
        if op_type.eq_ignore_ascii_case("matmul") {
            return "MatMul".to_string();
        }
        if op_type.eq_ignore_ascii_case("conv2d") {
            return "Conv".to_string();
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
        if op_type.eq_ignore_ascii_case("globalAveragePool") {
            return "GlobalAveragePool".to_string();
        }
        if op_type.eq_ignore_ascii_case("globalMaxPool") {
            return "GlobalMaxPool".to_string();
        }
        if op_type.eq_ignore_ascii_case("batchNormalization") {
            return "BatchNormalization".to_string();
        }
        if op_type.eq_ignore_ascii_case("instanceNormalization") {
            return "InstanceNormalization".to_string();
        }
        if op_type.eq_ignore_ascii_case("layerNormalization") {
            return "LayerNormalization".to_string();
        }
        // Reduction operations
        if op_type.eq_ignore_ascii_case("reduceSum") {
            return "ReduceSum".to_string();
        }
        if op_type.eq_ignore_ascii_case("reduceMean") {
            return "ReduceMean".to_string();
        }
        if op_type.eq_ignore_ascii_case("reduceMax") {
            return "ReduceMax".to_string();
        }
        if op_type.eq_ignore_ascii_case("reduceMin") {
            return "ReduceMin".to_string();
        }
        if op_type.eq_ignore_ascii_case("reduceProduct") {
            return "ReduceProd".to_string();
        }
        if op_type.eq_ignore_ascii_case("reduceL1") {
            return "ReduceL1".to_string();
        }
        if op_type.eq_ignore_ascii_case("reduceL2") {
            return "ReduceL2".to_string();
        }
        if op_type.eq_ignore_ascii_case("reduceLogSum") {
            return "ReduceLogSum".to_string();
        }
        if op_type.eq_ignore_ascii_case("reduceLogSumExp") {
            return "ReduceLogSumExp".to_string();
        }
        if op_type.eq_ignore_ascii_case("reduceSumSquare") {
            return "ReduceSumSquare".to_string();
        }
        // Logic operations
        if op_type.eq_ignore_ascii_case("equal") {
            return "Equal".to_string();
        }
        if op_type.eq_ignore_ascii_case("greater") {
            return "Greater".to_string();
        }
        if op_type.eq_ignore_ascii_case("greaterOrEqual") {
            return "GreaterOrEqual".to_string();
        }
        if op_type.eq_ignore_ascii_case("lesser") {
            return "Less".to_string();
        }
        if op_type.eq_ignore_ascii_case("lesserOrEqual") {
            return "LessOrEqual".to_string();
        }
        if op_type.eq_ignore_ascii_case("logicalNot") {
            return "Not".to_string();
        }
        if op_type.eq_ignore_ascii_case("logicalAnd") {
            return "And".to_string();
        }
        if op_type.eq_ignore_ascii_case("logicalOr") {
            return "Or".to_string();
        }
        if op_type.eq_ignore_ascii_case("logicalXor") {
            return "Xor".to_string();
        }
        // Quantization operations
        if op_type.eq_ignore_ascii_case("dequantizeLinear") {
            return "DequantizeLinear".to_string();
        }
        if op_type.eq_ignore_ascii_case("quantizeLinear") {
            return "QuantizeLinear".to_string();
        }
        // Tensor operations
        if op_type.eq_ignore_ascii_case("triangular") {
            return "Trilu".to_string(); // ONNX Trilu (triangle lower/upper)
        }
        // Specialized activation functions
        if op_type.eq_ignore_ascii_case("prelu") {
            return "PRelu".to_string(); // ONNX uses PRelu (not Prelu)
        }
        if op_type.eq_ignore_ascii_case("leakyRelu") {
            return "LeakyRelu".to_string();
        }
        if op_type.eq_ignore_ascii_case("clamp") {
            return "Clip".to_string();
        }
        if op_type.eq_ignore_ascii_case("gemm") {
            return "Gemm".to_string();
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
                    r#type: Some(AttributeType::Ints as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
                    ints: pads_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
            attributes.push(AttributeProto {
                name: Some("group".to_string()), // Note: ONNX uses "group" not "groups"
                r#type: Some(AttributeType::Int as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
                    ints: output_shape_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
            attributes.push(AttributeProto {
                name: Some("group".to_string()), // Note: ONNX uses "group" not "groups"
                r#type: Some(AttributeType::Int as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
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
                    r#type: Some(AttributeType::Ints as i32),
                    ints: pads_i64,
                    ..Default::default()
                });
            }
        }

        attributes
    }

    /// Create ONNX attributes for reduction operations
    fn create_reduce_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON
        if let Some(axes) = op.attributes.get("axes").and_then(|v| v.as_array()) {
            let axes_i64: Vec<i64> = axes
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !axes_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("axes".to_string()),
                    r#type: Some(AttributeType::Ints as i32),
                    ints: axes_i64,
                    ..Default::default()
                });
            }
        }

        if let Some(keep_dims) = op
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
        {
            attributes.push(AttributeProto {
                name: Some("keepdims".to_string()),
                r#type: Some(AttributeType::Int as i32),
                i: Some(if keep_dims { 1 } else { 0 }),
                ..Default::default()
            });
        }

        attributes
    }

    fn create_cast_node(
        node_name: &str,
        input: String,
        output: String,
        to_data_type: ProtoDataType,
    ) -> NodeProto {
        NodeProto {
            input: vec![input],
            output: vec![output],
            name: Some(node_name.to_string()),
            op_type: Some("Cast".to_string()),
            attribute: vec![AttributeProto {
                name: Some("to".to_string()),
                r#type: Some(AttributeType::Int as i32),
                i: Some(to_data_type as i64),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// Create ONNX attributes for squeeze/unsqueeze operations
    fn create_squeeze_unsqueeze_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(axes) = op.attributes.get("axes").and_then(|v| v.as_array()) {
            let axes_i64: Vec<i64> = axes
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as i64))
                .collect();
            if !axes_i64.is_empty() {
                attributes.push(AttributeProto {
                    name: Some("axes".to_string()),
                    r#type: Some(AttributeType::Ints as i32),
                    ints: axes_i64,
                    ..Default::default()
                });
            }
        }

        attributes
    }

    /// Create ONNX attributes for argMax/argMin operations
    fn create_arg_reduce_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
            attributes.push(AttributeProto {
                name: Some("axis".to_string()),
                r#type: Some(AttributeType::Int as i32),
                i: Some(axis as i64),
                ..Default::default()
            });
        }

        if let Some(keep_dims) = op
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
        {
            attributes.push(AttributeProto {
                name: Some("keepdims".to_string()), // ONNX uses "keepdims" not "keepDimensions"
                r#type: Some(AttributeType::Int as i32),
                i: Some(if keep_dims { 1 } else { 0 }),
                ..Default::default()
            });
        }

        // Note: outputDataType is handled by the output tensor's data type, not as an attribute

        attributes
    }

    /// Create ONNX attributes for concat operation
    fn create_concat_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Concat requires an axis attribute in ONNX
        if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
            attributes.push(AttributeProto {
                name: Some("axis".to_string()),
                r#type: Some(AttributeType::Int as i32),
                i: Some(axis as i64),
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for cast operation
    fn create_cast_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(to_type) = op.attributes.get("to").and_then(|v| v.as_str()) {
            // Convert string data type to ONNX data type code
            let type_code = match to_type {
                "float32" => ProtoDataType::Float as i64,
                "float16" => ProtoDataType::Float16 as i64,
                "int32" => ProtoDataType::Int32 as i64,
                "uint32" => ProtoDataType::Uint32 as i64,
                "int8" => ProtoDataType::Int8 as i64,
                "uint8" => ProtoDataType::Uint8 as i64,
                "int64" => ProtoDataType::Int64 as i64,
                _ => ProtoDataType::Undefined as i64,
            };

            attributes.push(AttributeProto {
                name: Some("to".to_string()),
                r#type: Some(AttributeType::Int as i32),
                i: Some(type_code),
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for scatterElements operation
    fn create_scatter_elements_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_i64()) {
            attributes.push(AttributeProto {
                name: Some("axis".to_string()),
                r#type: Some(AttributeType::Int as i32),
                i: Some(axis),
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for tile operation
    fn create_tile_attributes(_op: &Operation) -> Vec<AttributeProto> {
        // For Tile operation, repetitions is provided as a separate input tensor in ONNX
        // Not as an attribute, so we return empty attributes
        Vec::new()
    }

    /// Create ONNX attributes for triangular operation
    fn create_triangular_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(upper) = op.attributes.get("upper").and_then(|v| v.as_bool()) {
            attributes.push(AttributeProto {
                name: Some("upper".to_string()),
                r#type: Some(AttributeType::Int as i32),
                i: Some(if upper { 1 } else { 0 }),
                ..Default::default()
            });
        }

        if let Some(diagonal) = op.attributes.get("diagonal").and_then(|v| v.as_i64()) {
            attributes.push(AttributeProto {
                name: Some("k".to_string()), // ONNX uses "k" for diagonal offset
                r#type: Some(AttributeType::Int as i32),
                i: Some(diagonal),
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for hardSigmoid operation
    fn create_hardsigmoid_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: Some("alpha".to_string()),
                r#type: Some(AttributeType::Float as i32),
                f: Some(alpha as f32),
                ..Default::default()
            });
        }

        if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: Some("beta".to_string()),
                r#type: Some(AttributeType::Float as i32),
                f: Some(beta as f32),
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for hardSwish operation
    fn create_hardswish_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: Some("alpha".to_string()),
                r#type: Some(AttributeType::Float as i32),
                f: Some(alpha as f32),
                ..Default::default()
            });
        }

        if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: Some("beta".to_string()),
                r#type: Some(AttributeType::Float as i32),
                f: Some(beta as f32),
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for elu operation
    fn create_elu_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: Some("alpha".to_string()),
                r#type: Some(AttributeType::Float as i32),
                f: Some(alpha as f32),
                ..Default::default()
            });
        }

        attributes
    }

    /// Create ONNX attributes for leakyRelu operation
    fn create_leakyrelu_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: Some("alpha".to_string()),
                r#type: Some(AttributeType::Float as i32),
                f: Some(alpha as f32),
                ..Default::default()
            });
        }

        attributes
    }

    /// Clamp operation doesn't use attributes - min/max are inputs in opset 11+
    /// Handled in convert() method as special case

    /// Create ONNX attributes for gemm operation
    fn create_gemm_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: Some("alpha".to_string()),
                r#type: Some(AttributeType::Float as i32),
                f: Some(alpha as f32),
                ..Default::default()
            });
        }

        if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: Some("beta".to_string()),
                r#type: Some(AttributeType::Float as i32),
                f: Some(beta as f32),
                ..Default::default()
            });
        }

        if let Some(a_transpose) = op.attributes.get("a_transpose").and_then(|v| v.as_bool()) {
            attributes.push(AttributeProto {
                name: Some("transA".to_string()),
                r#type: Some(AttributeType::Int as i32),
                i: Some(if a_transpose { 1 } else { 0 }),
                ..Default::default()
            });
        }

        if let Some(b_transpose) = op.attributes.get("b_transpose").and_then(|v| v.as_bool()) {
            attributes.push(AttributeProto {
                name: Some("transB".to_string()),
                r#type: Some(AttributeType::Int as i32),
                i: Some(if b_transpose { 1 } else { 0 }),
                ..Default::default()
            });
        }

        attributes
    }

    fn create_operation_attributes(op: &Operation) -> Vec<AttributeProto> {
        if op.op_type == "conv2d" {
            Self::create_conv2d_attributes(op)
        } else if op.op_type == "convTranspose2d" {
            Self::create_conv_transpose2d_attributes(op)
        } else if op.op_type == "averagePool2d" || op.op_type == "maxPool2d" {
            Self::create_pool2d_attributes(op)
        } else if op.op_type.starts_with("reduce") {
            Self::create_reduce_attributes(op)
        } else if op.op_type == "squeeze" || op.op_type == "unsqueeze" {
            Self::create_squeeze_unsqueeze_attributes(op)
        } else if op.op_type == "argMax" || op.op_type == "argMin" {
            Self::create_arg_reduce_attributes(op)
        } else if op.op_type == "concat" {
            Self::create_concat_attributes(op)
        } else if op.op_type == "cast" {
            Self::create_cast_attributes(op)
        } else if op.op_type == "scatterElements" {
            Self::create_scatter_elements_attributes(op)
        } else if op.op_type == "tile" {
            Self::create_tile_attributes(op)
        } else if op.op_type == "triangular" {
            Self::create_triangular_attributes(op)
        } else if op.op_type == "hardSigmoid" {
            Self::create_hardsigmoid_attributes(op)
        } else if op.op_type == "hardSwish" {
            Self::create_hardswish_attributes(op)
        } else if op.op_type == "elu" {
            Self::create_elu_attributes(op)
        } else if op.op_type == "leakyRelu" {
            Self::create_leakyrelu_attributes(op)
        } else if op.op_type == "gemm" {
            Self::create_gemm_attributes(op)
        } else if op.op_type == "clamp" {
            // Clamp (Clip in ONNX opset 11+) uses inputs for min/max, not attributes
            Vec::new()
        } else {
            Vec::new()
        }
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

            // Logic operations output uint8 in WebNN (matching Chromium)
            // ONNX models will correctly use uint8 for logical operation outputs
            // The executor handles uint8 → f32 conversion for Python compatibility
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

        // Generate nodes, inserting Cast nodes for logic operations
        let mut nodes = Vec::new();
        let mut cast_counter = 0;

        for (idx, op) in graph.operations.iter().enumerate() {
            let op_name = op
                .label
                .clone()
                .unwrap_or_else(|| format!("{}_{}", op.op_type, idx));

            // Check if this is a logic operation that needs type conversion
            let is_comparison_op = matches!(
                op.op_type.as_str(),
                "equal" | "greater" | "greaterOrEqual" | "lesser" | "lesserOrEqual"
            );
            let is_logical_op = matches!(
                op.op_type.as_str(),
                "logicalNot" | "logicalAnd" | "logicalOr" | "logicalXor"
            );

            if is_logical_op {
                // Logical operations: Cast inputs to bool, execute op, cast output to uint8
                let mut cast_inputs = Vec::new();

                for &input_id in &op.input_operands {
                    let input_name = Self::operand_name(graph, input_id);
                    let cast_output_name = format!("cast_to_bool_{}_{}", op_name, cast_counter);
                    cast_counter += 1;

                    // Create Cast node: input type -> bool
                    nodes.push(Self::create_cast_node(
                        &format!("cast_to_bool_{}", cast_counter - 1),
                        input_name,
                        cast_output_name.clone(),
                        ProtoDataType::Bool,
                    ));

                    cast_inputs.push(cast_output_name);
                }

                // Create the logical operation node (outputs bool)
                let bool_output_name = format!("{}_bool_output", op_name);
                let attributes = Self::create_operation_attributes(op);

                nodes.push(NodeProto {
                    input: cast_inputs,
                    output: vec![bool_output_name.clone()],
                    name: Some(op_name.clone()),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                });

                // Cast bool → uint8 (matching Chromium's WebNN implementation)
                nodes.push(Self::create_cast_node(
                    &format!("cast_to_uint8_{}", cast_counter),
                    bool_output_name,
                    Self::operand_name(graph, op.output_operand),
                    ProtoDataType::Uint8,
                ));
                cast_counter += 1;
            } else if is_comparison_op {
                // Comparison operations: Execute op (outputs bool), cast output to uint8
                let bool_output_name = format!("{}_bool_output", op_name);
                let attributes = Self::create_operation_attributes(op);

                // Create comparison node (outputs bool)
                nodes.push(NodeProto {
                    input: op
                        .input_operands
                        .iter()
                        .map(|id| Self::operand_name(graph, *id))
                        .collect(),
                    output: vec![bool_output_name.clone()],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                });

                // Cast bool → uint8 (matching Chromium's WebNN implementation)
                nodes.push(Self::create_cast_node(
                    &format!("cast_to_uint8_{}", cast_counter),
                    bool_output_name,
                    Self::operand_name(graph, op.output_operand),
                    ProtoDataType::Uint8,
                ));
                cast_counter += 1;
            } else if op.op_type == "clamp" {
                // Clamp (Clip in ONNX) uses min/max as inputs (not attributes) in opset 11+
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| Self::operand_name(graph, *id))
                    .collect();

                // Add min value as second input (optional in ONNX Clip)
                if let Some(min_value) = op.attributes.get("min_value").and_then(|v| v.as_f64()) {
                    let min_name = format!("{}_min", op_name);
                    inputs.push(min_name.clone());

                    // Add min initializer
                    initializers.push(TensorProto {
                        name: Some(min_name),
                        data_type: Some(ProtoDataType::Float as i32),
                        dims: vec![], // Scalar
                        float_data: vec![min_value as f32],
                        ..Default::default()
                    });
                }

                // Add max value as third input (optional in ONNX Clip)
                if let Some(max_value) = op.attributes.get("max_value").and_then(|v| v.as_f64()) {
                    let max_name = format!("{}_max", op_name);
                    inputs.push(max_name.clone());

                    // Add max initializer
                    initializers.push(TensorProto {
                        name: Some(max_name),
                        data_type: Some(ProtoDataType::Float as i32),
                        dims: vec![], // Scalar
                        float_data: vec![max_value as f32],
                        ..Default::default()
                    });
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![Self::operand_name(graph, op.output_operand)],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: vec![], // No attributes for Clip in opset 11+
                    ..Default::default()
                });
            } else if op.op_type == "reshape" {
                // Reshape requires shape as a second input tensor in ONNX (not as an attribute)
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| Self::operand_name(graph, *id))
                    .collect();

                // Extract the new shape from attributes
                if let Some(new_shape) = op.attributes.get("newShape").and_then(|v| v.as_array()) {
                    let shape_values: Vec<i64> = new_shape
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as i64))
                        .collect();

                    let shape_name = format!("{}_shape", op_name);
                    inputs.push(shape_name.clone());

                    // Add shape as an initializer (constant tensor)
                    initializers.push(TensorProto {
                        name: Some(shape_name),
                        data_type: Some(ProtoDataType::Int64 as i32),
                        dims: vec![shape_values.len() as i64], // 1D tensor
                        int64_data: shape_values,
                        ..Default::default()
                    });
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![Self::operand_name(graph, op.output_operand)],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: vec![], // No attributes for Reshape
                    ..Default::default()
                });
            } else if op.op_type == "expand" {
                // Expand requires shape as a second input tensor in ONNX (not as an attribute)
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| Self::operand_name(graph, *id))
                    .collect();

                // Extract the new shape from attributes (required)
                let new_shape = op
                    .attributes
                    .get("newShape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "Expand operation missing 'newShape' attribute in operation {}",
                            op_name
                        ),
                    })?;

                let shape_values: Vec<i64> = new_shape
                    .iter()
                    .filter_map(|v| v.as_u64().map(|u| u as i64))
                    .collect();

                let shape_name = format!("{}_shape", op_name);
                inputs.push(shape_name.clone());

                // Add shape as an initializer (constant tensor)
                initializers.push(TensorProto {
                    name: Some(shape_name),
                    data_type: Some(ProtoDataType::Int64 as i32),
                    dims: vec![shape_values.len() as i64], // 1D tensor
                    int64_data: shape_values,
                    ..Default::default()
                });

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![Self::operand_name(graph, op.output_operand)],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: vec![], // No attributes for Expand
                    ..Default::default()
                });
            } else {
                // Regular operation - no Cast nodes needed
                let attributes = Self::create_operation_attributes(op);

                nodes.push(NodeProto {
                    input: op
                        .input_operands
                        .iter()
                        .map(|id| Self::operand_name(graph, *id))
                        .collect(),
                    output: vec![Self::operand_name(graph, op.output_operand)],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                });
            }
        }

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
