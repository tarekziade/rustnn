use crate::converters::{ConvertedGraph, operand_name};
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, Operation};
use crate::protos::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, attribute_proto::AttributeType,
    tensor_proto::DataType as ProtoDataType, type_proto::Tensor as TensorTypeProto,
};
use crate::shape_inference::infer_transpose_shape;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use prost::Message;
use std::env;

#[derive(Default)]
pub struct OnnxConverter;

impl OnnxConverter {
    fn debug_enabled() -> bool {
        env::var("RUSTNN_ONNX_DEBUG")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    fn invalid_operand(
        context: &str,
        operand: u32,
        op_info: Option<(&Operation, usize)>,
    ) -> GraphError {
        if Self::debug_enabled() {
            if let Some((op, idx)) = op_info {
                eprintln!(
                    "[DEBUG] Invalid operand {} at {} (op #{} type={} label={:?} inputs={:?} outputs={:?})",
                    operand,
                    context,
                    idx,
                    op.op_type,
                    op.label,
                    op.input_operands,
                    op.output_operands
                );
            } else {
                eprintln!("[DEBUG] Invalid operand {} at {}", operand, context);
            }
        }
        GraphError::InvalidConversionOperand { operand }
    }

    fn data_type_code(data_type: DataType) -> ProtoDataType {
        match data_type {
            DataType::Float32 => ProtoDataType::Float,
            // Treat WebNN uint8 as ONNX uint8; comparison/logical ops explicitly insert casts.
            DataType::Uint8 => ProtoDataType::Uint8,
            DataType::Int8 => ProtoDataType::Int8,
            DataType::Int32 => ProtoDataType::Int32,
            DataType::Float16 => ProtoDataType::Float16,
            DataType::Uint32 => ProtoDataType::Uint32,
            DataType::Int64 => ProtoDataType::Int64,
            DataType::Uint64 => ProtoDataType::Uint64,
        }
    }

    fn create_scalar_initializer(name: String, dtype: ProtoDataType, value: f32) -> TensorProto {
        let mut tensor = TensorProto {
            name: Some(name),
            data_type: Some(dtype as i32),
            dims: vec![], // Scalar
            ..Default::default()
        };

        // Set data based on type
        match dtype {
            ProtoDataType::Float => {
                tensor.float_data = vec![value];
            }
            ProtoDataType::Float16 => {
                // Convert f32 to f16 using half crate's from_f32
                let f16_value = half::f16::from_f32(value);
                // Store as raw bytes
                tensor.raw_data = Some(prost::bytes::Bytes::from(f16_value.to_le_bytes().to_vec()));
            }
            ProtoDataType::Int8 => {
                tensor.int32_data = vec![value as i32];
            }
            ProtoDataType::Uint8 => {
                tensor.int32_data = vec![value as i32];
            }
            ProtoDataType::Int32 => {
                tensor.int32_data = vec![value as i32];
            }
            ProtoDataType::Uint32 => {
                tensor.uint64_data = vec![value as u64];
            }
            ProtoDataType::Int64 => {
                tensor.int64_data = vec![value as i64];
            }
            ProtoDataType::Uint64 => {
                tensor.uint64_data = vec![value as u64];
            }
            _ => {
                tensor.float_data = vec![value];
            }
        }

        tensor
    }

    /// Create a vector initializer with proper data type handling
    fn create_vector_initializer(
        name: String,
        dtype: ProtoDataType,
        shape: Vec<i64>,
        value: f32,
    ) -> TensorProto {
        let size = shape.iter().product::<i64>() as usize;
        let mut tensor = TensorProto {
            name: Some(name),
            data_type: Some(dtype as i32),
            dims: shape,
            ..Default::default()
        };

        // Set data based on type
        match dtype {
            ProtoDataType::Float => {
                tensor.float_data = vec![value; size];
            }
            ProtoDataType::Float16 => {
                // Convert f32 to f16 and store as raw bytes
                let f16_value = half::f16::from_f32(value);
                let bytes: Vec<u8> = (0..size).flat_map(|_| f16_value.to_le_bytes()).collect();
                tensor.raw_data = Some(prost::bytes::Bytes::from(bytes));
            }
            ProtoDataType::Int8 => {
                tensor.int32_data = vec![value as i32; size];
            }
            ProtoDataType::Uint8 => {
                tensor.int32_data = vec![value as i32; size];
            }
            ProtoDataType::Int32 => {
                tensor.int32_data = vec![value as i32; size];
            }
            ProtoDataType::Uint32 => {
                tensor.uint64_data = vec![value as u64; size];
            }
            ProtoDataType::Int64 => {
                tensor.int64_data = vec![value as i64; size];
            }
            _ => {
                tensor.float_data = vec![value; size];
            }
        }

        tensor
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

    /// Helper: Parse a JSON array attribute as Vec<i64>
    fn parse_i64_array(op: &Operation, json_key: &str) -> Option<Vec<i64>> {
        op.attributes
            .get(json_key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
                    .collect()
            })
            .filter(|vec: &Vec<i64>| !vec.is_empty())
    }

    /// Helper: Add an integer array attribute to the attributes vector
    fn add_ints_attribute(attributes: &mut Vec<AttributeProto>, name: &str, values: Vec<i64>) {
        if !values.is_empty() {
            attributes.push(AttributeProto {
                name: Some(name.to_string()),
                r#type: Some(AttributeType::Ints as i32),
                ints: values,
                ..Default::default()
            });
        }
    }

    /// Helper: Add an integer attribute to the attributes vector
    fn add_int_attribute(attributes: &mut Vec<AttributeProto>, name: &str, value: i64) {
        attributes.push(AttributeProto {
            name: Some(name.to_string()),
            r#type: Some(AttributeType::Int as i32),
            i: Some(value),
            ..Default::default()
        });
    }

    /// Create ONNX attributes for conv2d operation
    fn create_conv2d_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON using helpers
        if let Some(strides) = Self::parse_i64_array(op, "strides") {
            Self::add_ints_attribute(&mut attributes, "strides", strides);
        }
        if let Some(dilations) = Self::parse_i64_array(op, "dilations") {
            Self::add_ints_attribute(&mut attributes, "dilations", dilations);
        }
        if let Some(pads) = Self::parse_i64_array(op, "pads") {
            Self::add_ints_attribute(&mut attributes, "pads", pads);
        }
        if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
            Self::add_int_attribute(&mut attributes, "group", groups as i64); // Note: ONNX uses "group" not "groups"
        }

        attributes
    }

    /// Create ONNX attributes for convTranspose2d operation
    fn create_conv_transpose2d_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON using helpers
        if let Some(strides) = Self::parse_i64_array(op, "strides") {
            Self::add_ints_attribute(&mut attributes, "strides", strides);
        }
        if let Some(dilations) = Self::parse_i64_array(op, "dilations") {
            Self::add_ints_attribute(&mut attributes, "dilations", dilations);
        }
        if let Some(pads) = Self::parse_i64_array(op, "pads") {
            Self::add_ints_attribute(&mut attributes, "pads", pads);
        }
        if let Some(output_padding) = Self::parse_i64_array(op, "outputPadding") {
            Self::add_ints_attribute(&mut attributes, "output_padding", output_padding);
        }
        if let Some(output_shape) = Self::parse_i64_array(op, "outputSizes") {
            Self::add_ints_attribute(&mut attributes, "output_shape", output_shape);
        }
        if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
            Self::add_int_attribute(&mut attributes, "group", groups as i64); // Note: ONNX uses "group" not "groups"
        }

        attributes
    }

    /// Create ONNX attributes for pool2d operations
    fn create_pool2d_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON using helpers
        if let Some(kernel_shape) = Self::parse_i64_array(op, "windowDimensions") {
            Self::add_ints_attribute(&mut attributes, "kernel_shape", kernel_shape);
        }
        if let Some(strides) = Self::parse_i64_array(op, "strides") {
            Self::add_ints_attribute(&mut attributes, "strides", strides);
        }
        if let Some(dilations) = Self::parse_i64_array(op, "dilations") {
            Self::add_ints_attribute(&mut attributes, "dilations", dilations);
        }
        if let Some(pads) = Self::parse_i64_array(op, "pads") {
            Self::add_ints_attribute(&mut attributes, "pads", pads);
        }

        attributes
    }

    /// Create ONNX attributes for reduction operations
    fn create_reduce_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Parse attributes from JSON using helpers
        if let Some(axes) = Self::parse_i64_array(op, "axes") {
            Self::add_ints_attribute(&mut attributes, "axes", axes);
        }
        if let Some(keep_dims) = op
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
        {
            Self::add_int_attribute(&mut attributes, "keepdims", if keep_dims { 1 } else { 0 });
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

        if let Some(axes) = Self::parse_i64_array(op, "axes") {
            Self::add_ints_attribute(&mut attributes, "axes", axes);
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

    /// Create ONNX attributes for transpose operation
    fn create_transpose_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(perm) = Self::parse_i64_array(op, "permutation") {
            Self::add_ints_attribute(&mut attributes, "perm", perm);
        }

        attributes
    }

    /// Create ONNX attributes for cast operation
    fn create_cast_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        if let Some(to_type) = op.attributes.get("to").and_then(|v| v.as_str()) {
            // Convert string data type to ONNX data type code
            let type_code = match to_type.to_ascii_lowercase().as_str() {
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
    ///
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

    /// Create ONNX attributes for layerNormalization operation
    fn create_layernorm_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Add epsilon attribute
        let epsilon = op
            .attributes
            .get("epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);
        attributes.push(AttributeProto {
            name: Some("epsilon".to_string()),
            r#type: Some(AttributeType::Float as i32),
            f: Some(epsilon as f32),
            ..Default::default()
        });

        // Add axis attribute
        // ONNX LayerNormalization normalizes from axis to the end (consecutive dimensions)
        // WebNN allows arbitrary axes, so we need to check compatibility
        let axes = op
            .attributes
            .get("axes")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
            .unwrap_or_else(|| vec![-1]);

        // For now, use the first axis (or -1 for last dimension)
        // TODO: Validate that axes are consecutive and end at last dimension
        // If not, we should emulate using primitive operations (like Chromium does)
        let axis = axes.first().copied().unwrap_or(-1);
        attributes.push(AttributeProto {
            name: Some("axis".to_string()),
            r#type: Some(AttributeType::Int as i32),
            i: Some(axis),
            ..Default::default()
        });

        attributes
    }

    /// Create ONNX attributes for batchNormalization or instanceNormalization
    fn create_normalization_attributes(op: &Operation) -> Vec<AttributeProto> {
        let mut attributes = Vec::new();

        // Add epsilon attribute
        let epsilon = op
            .attributes
            .get("epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);
        attributes.push(AttributeProto {
            name: Some("epsilon".to_string()),
            r#type: Some(AttributeType::Float as i32),
            f: Some(epsilon as f32),
            ..Default::default()
        });

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
        } else if op.op_type == "transpose" {
            Self::create_transpose_attributes(op)
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
        } else if op.op_type == "layerNormalization" {
            Self::create_layernorm_attributes(op)
        } else if op.op_type == "batchNormalization" || op.op_type == "instanceNormalization" {
            Self::create_normalization_attributes(op)
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
        let mut value_infos = Vec::new();

        for &id in &graph.input_operands {
            let operand = graph.operand(id).ok_or_else(|| {
                if Self::debug_enabled() {
                    eprintln!(
                        "[DEBUG] Missing input operand {} while building ONNX graph",
                        id
                    );
                }
                Self::invalid_operand("graph input lookup", id, None)
            })?;
            inputs_val.push(value_info(&operand_name(graph, id), &operand.descriptor));
        }

        for &id in &graph.output_operands {
            let operand = graph.operand(id).ok_or_else(|| {
                if Self::debug_enabled() {
                    eprintln!(
                        "[DEBUG] Missing output operand {} while building ONNX graph",
                        id
                    );
                }
                Self::invalid_operand("graph output lookup", id, None)
            })?;

            // Logic operations output uint8 in WebNN (matching Chromium)
            // ONNX models will correctly use uint8 for logical operation outputs
            // The executor handles uint8 → f32 conversion for Python compatibility
            outputs_val.push(value_info(&operand_name(graph, id), &operand.descriptor));
        }

        // Build type overrides for ops where output type must match an input (e.g., expand)
        let mut type_overrides: std::collections::HashMap<u32, DataType> =
            std::collections::HashMap::new();
        let mut shape_overrides: std::collections::HashMap<u32, Vec<u32>> =
            std::collections::HashMap::new();
        let mut operand_shapes: std::collections::HashMap<u32, Vec<u32>> =
            std::collections::HashMap::new();
        let mut name_to_id: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();

        // Seed operand_shapes with known operand descriptors
        for (idx, operand) in graph.operands.iter().enumerate() {
            if !operand.descriptor.shape.is_empty() {
                operand_shapes.insert(idx as u32, operand.descriptor.shape.clone());
            }
            if let Some(name) = &operand.name {
                name_to_id.insert(name.clone(), idx as u32);
            }
        }

        // Helper to read constant int64 values by operand name
        let get_const_i64 = |name: &str| -> Option<Vec<i64>> {
            let id = name_to_id.get(name)?;
            let handle = graph.constant_operand_ids_to_handles.get(id)?;
            let bytes = &handle.data;
            if bytes.len() % 8 != 0 {
                return None;
            }
            Some(
                bytes
                    .chunks(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            )
        };

        for op in &graph.operations {
            if op.op_type.eq_ignore_ascii_case("expand") {
                if op.input_operands.len() >= 2
                    && let (Some(&input_id), Some(output_id)) =
                        (op.input_operands.first(), op.output_operand)
                    && let Some(input_operand) = graph.operand(input_id)
                {
                    type_overrides.insert(output_id, input_operand.descriptor.data_type);
                    if let Some(shape) = operand_shapes.get(&op.input_operands[1]) {
                        shape_overrides.insert(output_id, shape.clone());
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("shape") {
                if let Some(output_id) = op.output_operand {
                    type_overrides.insert(output_id, DataType::Int64);
                }
            } else if op.op_type.eq_ignore_ascii_case("where") {
                if let (Some(output_id), Some(val_input_id)) =
                    (op.output_operand, op.input_operands.get(1))
                    && let Some(input_operand) = graph.operand(*val_input_id)
                {
                    type_overrides.insert(output_id, input_operand.descriptor.data_type);
                }
            } else if op.op_type.eq_ignore_ascii_case("slice") {
                if let (Some(&input_id), Some(output_id)) =
                    (op.input_operands.first(), op.output_operand)
                    && let Some(mut in_shape) = operand_shapes.get(&input_id).cloned()
                {
                    // Preserve input dtype for the slice output
                    if let Some(input_operand) = graph.operand(input_id) {
                        type_overrides.insert(output_id, input_operand.descriptor.data_type);
                    }
                    let axes = op
                        .attributes
                        .get("axes")
                        .and_then(|v| v.as_str())
                        .and_then(get_const_i64);
                    let starts = op
                        .attributes
                        .get("starts")
                        .and_then(|v| v.as_str())
                        .and_then(get_const_i64);
                    let ends = op
                        .attributes
                        .get("ends")
                        .and_then(|v| v.as_str())
                        .and_then(get_const_i64);
                    let steps = op
                        .attributes
                        .get("steps")
                        .and_then(|v| v.as_str())
                        .and_then(get_const_i64);

                    if let (Some(axes), Some(starts), Some(ends)) = (axes, starts, ends) {
                        let steps_vec = steps.unwrap_or_else(|| vec![1; axes.len()]);
                        let len = axes
                            .len()
                            .min(starts.len())
                            .min(ends.len())
                            .min(steps_vec.len());
                        for i in 0..len {
                            let axis = axes[i] as isize;
                            if axis < 0 || (axis as usize) >= in_shape.len() {
                                continue;
                            }
                            let step = steps_vec[i];
                            if step == 0 {
                                continue;
                            }
                            let start = starts[i];
                            let end = ends[i];
                            let dim = in_shape[axis as usize] as i64;
                            let s = if start < 0 { dim + start } else { start }.max(0);
                            let e = if end < 0 { dim + end } else { end }.min(dim);
                            let span = (e - s + (step.abs() - 1)) / step.abs();
                            if span > 0 {
                                in_shape[axis as usize] = span as u32;
                            }
                        }
                        shape_overrides.insert(output_id, in_shape.clone());
                        operand_shapes.insert(output_id, in_shape);
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("add") {
                if let Some(output_id) = op.output_operand
                    && op.input_operands.len() == 2
                    && let (Some(lhs), Some(rhs)) = (
                        operand_shapes.get(&op.input_operands[0]),
                        operand_shapes.get(&op.input_operands[1]),
                    )
                    && lhs == rhs
                {
                    shape_overrides.insert(output_id, lhs.clone());
                }
            }
            // Reshape: if newShape is static, set output shape
            else if op.op_type.eq_ignore_ascii_case("reshape") {
                if let Some(output_id) = op.output_operand
                    && let Some(new_shape_val) = op.attributes.get("newShape")
                    && let Some(arr) = new_shape_val.as_array()
                {
                    let shape: Vec<u32> = arr
                        .iter()
                        .filter_map(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
                        .map(|v| v as u32)
                        .collect();
                    if !shape.is_empty() {
                        shape_overrides.insert(output_id, shape.clone());
                        operand_shapes.insert(output_id, shape);
                    }
                }
            }
            // Transpose: derive output shape from permutation (default reverse)
            else if op.op_type.eq_ignore_ascii_case("transpose") {
                if let (Some(&input_id), Some(output_id)) =
                    (op.input_operands.first(), op.output_operand)
                    && let Some(input_shape) = operand_shapes.get(&input_id).cloned()
                {
                    let perm: Option<Vec<u32>> = op
                        .attributes
                        .get("permutation")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| {
                                    v.as_i64()
                                        .or_else(|| v.as_u64().map(|u| u as i64))
                                        .map(|n| n as u32)
                                })
                                .collect()
                        });

                    if let Ok(out_shape) = infer_transpose_shape(&input_shape, perm.as_deref()) {
                        shape_overrides.insert(output_id, out_shape.clone());
                        operand_shapes.insert(output_id, out_shape);
                    }
                }
            }
            // Gather: infer output shape from data/indices if available
            else if op.op_type.eq_ignore_ascii_case("gather")
                && let Some(output_id) = op.output_operand
                && op.input_operands.len() >= 2
            {
                let data_shape = operand_shapes.get(&op.input_operands[0]);
                let indices_shape = operand_shapes.get(&op.input_operands[1]);
                let axis = op
                    .attributes
                    .get("axis")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                if let (Some(data_shape), Some(indices_shape)) = (data_shape, indices_shape) {
                    let rank = data_shape.len() as i64;
                    let mut axis = axis;
                    if axis < 0 {
                        axis += rank;
                    }
                    if axis >= 0 && (axis as usize) < data_shape.len() {
                        let mut out_shape = indices_shape.clone();
                        out_shape.extend_from_slice(&data_shape[(axis as usize + 1)..]);
                        shape_overrides.insert(output_id, out_shape.clone());
                        operand_shapes.insert(output_id, out_shape);
                    }
                }
            }

            // Update operand_shapes for outputs where we inferred a shape override
            if let Some(out_id) = op.output_operand
                && let Some(shape) = shape_overrides.get(&out_id)
            {
                operand_shapes.insert(out_id, shape.clone());
            }
        }

        for (id, data) in &graph.constant_operand_ids_to_handles {
            let operand = graph.operand(*id).ok_or_else(|| {
                if Self::debug_enabled() {
                    eprintln!(
                        "[DEBUG] Missing constant operand {} while building initializers",
                        id
                    );
                }
                Self::invalid_operand("initializer lookup", *id, None)
            })?;

            // Handle zero-length constants by creating zero-filled tensors
            // This is a defensive measure for malformed models where constants have no data
            let tensor_proto = if data.data.is_empty() {
                let element_count: usize = operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|&d| d as usize)
                    .product();
                let dtype = Self::data_type_code(operand.descriptor.data_type);

                // For Int64, use int64_data field; for other types, use raw_data with zeros
                match operand.descriptor.data_type {
                    DataType::Int64 => TensorProto {
                        name: Some(operand_name(graph, *id)),
                        data_type: Some(dtype as i32),
                        dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                        int64_data: vec![0i64; element_count],
                        ..Default::default()
                    },
                    DataType::Int32 => TensorProto {
                        name: Some(operand_name(graph, *id)),
                        data_type: Some(dtype as i32),
                        dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                        int32_data: vec![0i32; element_count],
                        ..Default::default()
                    },
                    DataType::Float32 => TensorProto {
                        name: Some(operand_name(graph, *id)),
                        data_type: Some(dtype as i32),
                        dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                        float_data: vec![0f32; element_count],
                        ..Default::default()
                    },
                    _ => {
                        // For other types, create zero-filled raw_data
                        let bytes_per_element = match operand.descriptor.data_type {
                            DataType::Float16 => 2,
                            DataType::Int8 | DataType::Uint8 => 1,
                            DataType::Uint32 => 4,
                            DataType::Uint64 => 8,
                            _ => 4, // Default to 4 bytes
                        };
                        let zero_data = vec![0u8; element_count * bytes_per_element];
                        TensorProto {
                            name: Some(operand_name(graph, *id)),
                            data_type: Some(dtype as i32),
                            dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                            raw_data: Some(prost::bytes::Bytes::from(zero_data)),
                            ..Default::default()
                        }
                    }
                }
            } else {
                // Normal case: use provided data
                TensorProto {
                    name: Some(operand_name(graph, *id)),
                    data_type: Some(Self::data_type_code(operand.descriptor.data_type) as i32),
                    dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                    raw_data: Some(prost::bytes::Bytes::from(data.data.clone())),
                    ..Default::default()
                }
            };

            initializers.push(tensor_proto);
        }

        // Generate nodes, inserting Cast nodes for logic operations
        let mut nodes = Vec::new();
        let mut cast_counter = 0;

        let debug = Self::debug_enabled();

        if debug {
            if let Some(opd) = graph.operands.get(34) {
                eprintln!(
                    "[DEBUG] Operand 34 name={:?} dtype={:?} shape={:?}",
                    opd.name, opd.descriptor.data_type, opd.descriptor.shape
                );
            } else {
                eprintln!("[DEBUG] Operand 34 not present in operands table");
            }
        }

        for (idx, op) in graph.operations.iter().enumerate() {
            // Debug guard: ensure all input operands exist
            for &input_id in &op.input_operands {
                if graph.operand(input_id).is_none() {
                    if debug {
                        let input_name = graph
                            .operands
                            .get(input_id as usize)
                            .and_then(|opd| opd.name.clone())
                            .unwrap_or_else(|| format!("<unnamed:{}>", input_id));
                        eprintln!(
                            "[DEBUG] Missing operand id {} name '{}' for op {} ({}) at index {}. Inputs: {:?}",
                            input_id,
                            input_name,
                            op.label.clone().unwrap_or_else(|| op.op_type.clone()),
                            op.op_type,
                            idx,
                            op.input_operands
                        );
                        eprintln!(
                            "[DEBUG] operands.len()={} valid ids 0..{}",
                            graph.operands.len(),
                            graph.operands.len().saturating_sub(1)
                        );
                        eprintln!(
                            "[DEBUG] Failing op detail: idx={} type={} label={:?} inputs={:?}",
                            idx, op.op_type, op.label, op.input_operands
                        );
                    }
                    return Err(Self::invalid_operand(
                        "op input lookup",
                        input_id,
                        Some((op, idx)),
                    ));
                }
            }

            // WebNN constant() op: encode as initializer, not a node
            if op.op_type.eq_ignore_ascii_case("constant") {
                let output_id = op.output_operand.ok_or_else(|| {
                    Self::invalid_operand("constant output", idx as u32, Some((op, idx)))
                })?;

                // Extract attributes
                let data_b64 = op
                    .attributes
                    .get("data")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: "Constant op missing 'data' attribute".to_string(),
                    })?;
                let data = STANDARD
                    .decode(data_b64)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!("Constant op base64 decode failed: {}", e),
                    })?;

                let dtype_str = op
                    .attributes
                    .get("dataType")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: "Constant op missing 'dataType' attribute".to_string(),
                    })?;
                let data_type = match dtype_str.to_ascii_lowercase().as_str() {
                    "float32" => DataType::Float32,
                    "float16" => DataType::Float16,
                    "int32" => DataType::Int32,
                    "uint32" => DataType::Uint32,
                    "int64" => DataType::Int64,
                    "uint64" => DataType::Uint64,
                    "int8" => DataType::Int8,
                    "uint8" => DataType::Uint8,
                    other => {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!("Unsupported constant dataType '{}'", other),
                        });
                    }
                };

                let shape: Vec<i64> = op
                    .attributes
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)))
                            .collect()
                    })
                    .unwrap_or_default();

                initializers.push(TensorProto {
                    name: Some(operand_name(graph, output_id)),
                    data_type: Some(Self::data_type_code(data_type) as i32),
                    dims: shape,
                    raw_data: Some(prost::bytes::Bytes::from(data)),
                    ..Default::default()
                });

                continue;
            }

            let op_name = op
                .label
                .clone()
                .unwrap_or_else(|| format!("{}_{}", op.op_type, idx));

            // Special-case concat: expand scalar inputs to 1D so ONNX axis validation passes
            if op.op_type.eq_ignore_ascii_case("concat") {
                let mut inputs: Vec<String> = Vec::new();

                for (input_idx, input_id) in op.input_operands.iter().enumerate() {
                    let operand = graph.operand(*input_id).ok_or_else(|| {
                        if Self::debug_enabled() {
                            eprintln!(
                                "[DEBUG] Missing operand {} in expand at op idx {}",
                                input_id, idx
                            );
                        }
                        Self::invalid_operand("concat input lookup", *input_id, Some((op, idx)))
                    })?;
                    let input_name = operand_name(graph, *input_id);

                    if operand.descriptor.shape.is_empty() {
                        if let Some(data) = graph.constant_operand_ids_to_handles.get(input_id) {
                            // Expand scalar constant to shape [1]
                            let expanded_name = format!("{}_scalar{}_expanded", op_name, input_idx);
                            initializers.push(TensorProto {
                                name: Some(expanded_name.clone()),
                                data_type: Some(
                                    Self::data_type_code(operand.descriptor.data_type) as i32
                                ),
                                dims: vec![1],
                                raw_data: Some(prost::bytes::Bytes::from(data.data.clone())),
                                ..Default::default()
                            });
                            inputs.push(expanded_name);
                            continue;
                        } else {
                            // Try cloning an existing initializer with the same name
                            let expanded_name = format!("{}_scalar{}_expanded", op_name, input_idx);
                            if let Some(cloned) = initializers
                                .iter()
                                .find(|t| t.name.as_deref() == Some(&input_name))
                                .map(|orig| {
                                    let mut cloned = orig.clone();
                                    cloned.name = Some(expanded_name.clone());
                                    cloned.dims = vec![1];
                                    cloned
                                })
                            {
                                initializers.push(cloned);
                                inputs.push(expanded_name);
                                continue;
                            }
                        }

                        {
                            // Fallback: insert Unsqueeze to lift scalar to 1D
                            let unsq_name = format!("{}_scalar{}_unsq", op_name, input_idx);
                            nodes.push(NodeProto {
                                input: vec![input_name.clone()],
                                output: vec![unsq_name.clone()],
                                name: Some(format!("{}_unsqueeze_{}", op_name, input_idx)),
                                op_type: Some("Unsqueeze".to_string()),
                                attribute: vec![AttributeProto {
                                    name: Some("axes".to_string()),
                                    r#type: Some(AttributeType::Ints as i32),
                                    ints: vec![0],
                                    ..Default::default()
                                }],
                                ..Default::default()
                            });
                            inputs.push(unsq_name);
                            continue;
                        }
                    }

                    inputs.push(input_name);
                }

                let attributes = Self::create_operation_attributes(op);

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                });

                continue;
            }

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
                    let input_name = operand_name(graph, input_id);
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
                    operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    ),
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
                        .map(|id| operand_name(graph, *id))
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
                    operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    ),
                    ProtoDataType::Uint8,
                ));
                cast_counter += 1;
            } else if op.op_type.eq_ignore_ascii_case("where") {
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Ensure condition is bool for ONNX Where (WebNN uses uint8 for comparisons)
                if !inputs.is_empty() {
                    let cast_name = format!("{}_cond_bool", op_name);
                    nodes.push(Self::create_cast_node(
                        &cast_name,
                        inputs[0].clone(),
                        cast_name.clone(),
                        ProtoDataType::Bool,
                    ));
                    inputs[0] = cast_name;
                }

                let attributes = Self::create_operation_attributes(op);

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("tile") {
                // ONNX Tile takes repeats as a second input tensor (INT64)
                let data_input = if let Some(data_id) = op.input_operands.first() {
                    operand_name(graph, *data_id)
                } else {
                    return Err(Self::invalid_operand(
                        "tile missing data input",
                        idx as u32,
                        Some((op, idx)),
                    ));
                };

                // Repeats come from attribute; ignore missing second operand and synthesize.
                let repeats: Vec<i64> =
                    op.attributes
                        .get("repetitions")
                        .or_else(|| op.attributes.get("repeats"))
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
                        .filter(|v: &Vec<i64>| !v.is_empty())
                        .ok_or_else(|| {
                            let operand_id =
                                op.input_operands.get(1).copied().unwrap_or_else(|| {
                                    op.input_operands.first().copied().unwrap_or(0)
                                });
                            Self::invalid_operand(
                                "tile repeats/repetitions attribute",
                                operand_id,
                                Some((op, idx)),
                            )
                        })?;

                // If all repeats are 1, Tile is a no-op. Emit Identity to avoid shape issues.
                if repeats.iter().all(|&r| r == 1) {
                    nodes.push(NodeProto {
                        input: vec![data_input],
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: Some(format!("{}_identity", op_name)),
                        op_type: Some("Identity".to_string()),
                        attribute: vec![],
                        ..Default::default()
                    });
                    continue;
                }

                let repeats_name = format!("{}_repeats", op_name);
                initializers.push(TensorProto {
                    name: Some(repeats_name.clone()),
                    data_type: Some(ProtoDataType::Int64 as i32),
                    dims: vec![repeats.len() as i64],
                    int64_data: repeats,
                    ..Default::default()
                });
                let inputs = vec![data_input, repeats_name];

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: vec![],
                    ..Default::default()
                });
            } else if op.op_type == "clamp" {
                // Clamp (Clip in ONNX) uses min/max as inputs (not attributes) in opset 11+
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Get input operand data type - min/max must match this type
                let input_operand = graph.operand(op.input_operands[0]).ok_or_else(|| {
                    Self::invalid_operand(
                        "clamp input lookup",
                        op.input_operands[0],
                        Some((op, idx)),
                    )
                })?;
                let input_dtype = input_operand.descriptor.data_type;
                let onnx_dtype = Self::data_type_code(input_dtype);

                // Add min value as second input (optional in ONNX Clip)
                if let Some(min_value) = op.attributes.get("min_value").and_then(|v| v.as_f64()) {
                    let min_name = format!("{}_min", op_name);
                    inputs.push(min_name.clone());

                    // Add min initializer with matching data type
                    initializers.push(Self::create_scalar_initializer(
                        min_name,
                        onnx_dtype,
                        min_value as f32,
                    ));
                }

                // Add max value as third input (optional in ONNX Clip)
                if let Some(max_value) = op.attributes.get("max_value").and_then(|v| v.as_f64()) {
                    let max_name = format!("{}_max", op_name);
                    inputs.push(max_name.clone());

                    // Add max initializer with matching data type
                    initializers.push(Self::create_scalar_initializer(
                        max_name,
                        onnx_dtype,
                        max_value as f32,
                    ));
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
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
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Handle newShape attribute - can be array (static), string (operand reference), or missing
                if let Some(new_shape_attr) = op.attributes.get("newShape") {
                    if let Some(new_shape_array) = new_shape_attr.as_array() {
                        // Case 1: newShape is an array (static shape) - create constant initializer
                        let shape_values: Vec<i64> = new_shape_array
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
                    } else if let Some(shape_operand_name) = new_shape_attr.as_str() {
                        // Case 2: newShape is a string (operand reference) - use referenced operand as second input
                        // This handles dynamic reshapes where the shape is computed at runtime

                        // Use the string as-is since operand names in the graph preserve their original format
                        // The loader's sanitization only affects certain identifier patterns (not output names on LHS of =)
                        inputs.push(shape_operand_name.to_string());
                    } else {
                        return Err(GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!(
                                "Reshape operation has invalid newShape attribute type (not array or string) in operation {}",
                                op_name
                            ),
                        });
                    }
                } else {
                    // Case 3: No newShape attribute - infer from output operand descriptor (static shape)
                    let output_id = op.output_operand.expect("Single-output operation expected");
                    let output_operand = graph.operand(output_id).ok_or_else(|| {
                        Self::invalid_operand("reshape output lookup", output_id, Some((op, idx)))
                    })?;
                    let shape_values: Vec<i64> = output_operand
                        .descriptor
                        .shape
                        .iter()
                        .map(|&dim| dim as i64)
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
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: vec![], // No attributes for Reshape
                    ..Default::default()
                });
            } else if op.op_type == "expand" {
                // WebNN expand has two variants:
                // 1. With 'axes' - adds dimensions (maps to ONNX Unsqueeze)
                // 2. With 'newShape' - expands shape (maps to ONNX Expand)

                if let Some(axes_val) = op.attributes.get("axes").and_then(|v| v.as_array()) {
                    // WebNN expand with axes -> ONNX Unsqueeze
                    // In ONNX opset 13+, axes must be provided as an input tensor, not attribute
                    let axes_values: Vec<i64> =
                        axes_val.iter().filter_map(|v| v.as_i64()).collect();

                    let mut inputs: Vec<String> = op
                        .input_operands
                        .iter()
                        .map(|id| operand_name(graph, *id))
                        .collect();

                    // Create axes tensor name and add as second input
                    let axes_name = format!("{}_axes", op_name);
                    inputs.push(axes_name.clone());

                    // Add axes as an initializer (constant tensor)
                    initializers.push(TensorProto {
                        name: Some(axes_name),
                        data_type: Some(ProtoDataType::Int64 as i32),
                        dims: vec![axes_values.len() as i64], // 1D tensor
                        int64_data: axes_values,
                        ..Default::default()
                    });

                    nodes.push(NodeProto {
                        input: inputs,
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: Some(op_name.clone()),
                        op_type: Some("Unsqueeze".to_string()),
                        attribute: vec![], // No attributes for Unsqueeze in opset 13+
                        ..Default::default()
                    });
                } else if let Some(new_shape) =
                    op.attributes.get("newShape").and_then(|v| v.as_array())
                {
                    // WebNN expand with newShape -> ONNX Expand
                    let mut inputs: Vec<String> = op
                        .input_operands
                        .iter()
                        .map(|id| operand_name(graph, *id))
                        .collect();

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
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: Some(op_name),
                        op_type: Some("Expand".to_string()),
                        attribute: vec![], // No attributes for Expand
                        ..Default::default()
                    });
                } else {
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "Expand operation requires either 'axes' or 'newShape' attribute in operation {}",
                            op_name
                        ),
                    });
                }
            } else if op.op_type.starts_with("reduce") {
                // Reduction operations - in ONNX opset 13, only ReduceSum supports axes as input
                // In opset 18+, ReduceMean, ReduceProd, ReduceMax, ReduceMin also support axes as input
                // But we're using opset 13, so only ReduceSum gets axes as input
                let supports_axes_as_input = matches!(op.op_type.as_str(), "reduceSum");

                // Check if input needs casting (uint32 not supported by ONNX Runtime for some reductions)
                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("reduction input lookup", input_id, Some((op, idx)))
                })?;

                let input_name = operand_name(graph, input_id);
                let needs_cast = matches!(
                    input_operand.descriptor.data_type,
                    DataType::Uint32 | DataType::Uint8
                );

                let actual_input_name = if needs_cast {
                    // Cast uint32/uint8 to float32 for reduction operations
                    let cast_output = format!("{}_cast_to_float", op_name);
                    nodes.push(NodeProto {
                        input: vec![input_name],
                        output: vec![cast_output.clone()],
                        name: Some(format!("{}_pre_cast", op_name)),
                        op_type: Some("Cast".to_string()),
                        attribute: vec![AttributeProto {
                            name: Some("to".to_string()),
                            r#type: Some(AttributeType::Int as i32),
                            i: Some(ProtoDataType::Float as i32 as i64),
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    cast_output
                } else {
                    input_name
                };

                let mut inputs: Vec<String> = vec![actual_input_name];
                // Add any additional inputs (though reductions typically have only one input)
                inputs.extend(
                    op.input_operands[1..]
                        .iter()
                        .map(|id| operand_name(graph, *id)),
                );

                let mut attributes = Vec::new();

                // Extract axes from attributes
                if let Some(axes_i64) = Self::parse_i64_array(op, "axes") {
                    if supports_axes_as_input {
                        // Add axes as an input tensor (opset 13+ for supported operations)
                        let axes_name = format!("{}_axes", op_name);
                        inputs.push(axes_name.clone());

                        initializers.push(TensorProto {
                            name: Some(axes_name),
                            data_type: Some(ProtoDataType::Int64 as i32),
                            dims: vec![axes_i64.len() as i64],
                            int64_data: axes_i64,
                            ..Default::default()
                        });
                    } else {
                        // Add axes as an attribute (for operations that don't support axes as input in opset 13)
                        attributes.push(AttributeProto {
                            name: Some("axes".to_string()),
                            r#type: Some(AttributeType::Ints as i32),
                            ints: axes_i64,
                            ..Default::default()
                        });
                    }
                }

                // Add keepDimensions attribute
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

                let final_output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );

                let reduce_output_name = if needs_cast {
                    // Output to temporary name, will cast back after
                    format!("{}_float_output", op_name)
                } else {
                    final_output_name.clone()
                };

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![reduce_output_name.clone()],
                    name: Some(op_name.clone()),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                });

                // Cast back to original type if needed
                if needs_cast {
                    let original_type = Self::data_type_code(input_operand.descriptor.data_type);
                    nodes.push(NodeProto {
                        input: vec![reduce_output_name],
                        output: vec![final_output_name],
                        name: Some(format!("{}_post_cast", op_name)),
                        op_type: Some("Cast".to_string()),
                        attribute: vec![AttributeProto {
                            name: Some("to".to_string()),
                            r#type: Some(AttributeType::Int as i32),
                            i: Some(original_type as i32 as i64),
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                }
            } else if op.op_type == "slice" {
                // Slice operation - ONNX requires starts, ends, axes, steps as input tensors
                // Special case: ONNX Runtime doesn't support slicing 0D tensors

                // Check if input is 0D (scalar)
                let input_operand_id = op.input_operands[0];
                let input_operand = graph.operand(input_operand_id).ok_or_else(|| {
                    Self::invalid_operand("slice input lookup", input_operand_id, Some((op, idx)))
                })?;
                let is_0d = input_operand.descriptor.shape.is_empty();

                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Extract starts, sizes/ends, axes, and steps from attributes
                let parse_attr_i64 = |key: &str| -> Option<Vec<i64>> {
                    op.attributes.get(key).and_then(|v| {
                        if let Some(arr) = v.as_array() {
                            let vals: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                            if !vals.is_empty() { Some(vals) } else { None }
                        } else if let Some(name) = v.as_str() {
                            get_const_i64(name)
                        } else {
                            None
                        }
                    })
                };

                let starts = parse_attr_i64("starts").unwrap_or_default();
                // Prefer explicit ends attribute; fall back to sizes (ONNX/WebNN use sizes)
                let mut ends = parse_attr_i64("ends").unwrap_or_default();
                let sizes = parse_attr_i64("sizes").unwrap_or_default();

                // Special case: 0D tensor with empty starts/sizes is a no-op
                // Use Identity node instead of Slice (ONNX Runtime doesn't support slicing scalars)
                if is_0d && starts.is_empty() && sizes.is_empty() {
                    nodes.push(NodeProto {
                        input: vec![inputs[0].clone()],
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: Some(op_name),
                        op_type: Some("Identity".to_string()),
                        ..Default::default()
                    });
                    continue; // Skip the rest of the slice handling
                }

                let axes = parse_attr_i64("axes");

                let steps = parse_attr_i64("strides");

                // Convert sizes to ends when explicit ends are not provided: ends[i] = starts[i] + sizes[i]
                if ends.is_empty() {
                    ends = starts
                        .iter()
                        .zip(sizes.iter())
                        .map(|(start, size)| start + size)
                        .collect();
                }

                // Capture length before moving starts
                let starts_len = starts.len();

                // Add starts as initializer
                let starts_name = format!("{}_starts", op_name);
                inputs.push(starts_name.clone());
                initializers.push(TensorProto {
                    name: Some(starts_name),
                    data_type: Some(ProtoDataType::Int64 as i32),
                    dims: vec![starts_len as i64],
                    int64_data: starts,
                    ..Default::default()
                });

                // Add ends as initializer
                let ends_name = format!("{}_ends", op_name);
                inputs.push(ends_name.clone());
                initializers.push(TensorProto {
                    name: Some(ends_name),
                    data_type: Some(ProtoDataType::Int64 as i32),
                    dims: vec![ends.len() as i64],
                    int64_data: ends,
                    ..Default::default()
                });

                // Add axes as initializer
                // If axes not provided, default to [0, 1, ..., len(starts)-1]
                let axes_data = axes.unwrap_or_else(|| (0..starts_len as i64).collect());

                let axes_name = format!("{}_axes", op_name);
                inputs.push(axes_name.clone());
                initializers.push(TensorProto {
                    name: Some(axes_name),
                    data_type: Some(ProtoDataType::Int64 as i32),
                    dims: vec![axes_data.len() as i64],
                    int64_data: axes_data,
                    ..Default::default()
                });

                // Add steps as initializer (if provided)
                if let Some(steps_data) = steps {
                    let steps_name = format!("{}_steps", op_name);
                    inputs.push(steps_name.clone());
                    initializers.push(TensorProto {
                        name: Some(steps_name),
                        data_type: Some(ProtoDataType::Int64 as i32),
                        dims: vec![steps_data.len() as i64],
                        int64_data: steps_data,
                        ..Default::default()
                    });
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: vec![], // No attributes for Slice in opset 13+
                    ..Default::default()
                });
            } else if op.op_type == "split" {
                // Split operation - multi-output operation
                let outputs: Vec<String> = op
                    .output_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Get axis attribute
                let mut attributes = Vec::new();
                if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
                    attributes.push(AttributeProto {
                        name: Some("axis".to_string()),
                        r#type: Some(AttributeType::Int as i32),
                        i: Some(axis as i64),
                        ..Default::default()
                    });
                }

                // Collect inputs
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Handle splits parameter - either count or sizes
                // ONNX Split opset 13+ takes split sizes as an optional input tensor
                if let Some(splits_val) = op.attributes.get("splits") {
                    if let Some(_count) = splits_val.as_u64() {
                        // Equal splits - ONNX Split without split input divides evenly
                        // based on the number of outputs. No input needed.
                    } else if let Some(sizes) = splits_val.as_array() {
                        // Explicit split sizes - create initializer
                        let split_sizes: Vec<i64> = sizes
                            .iter()
                            .filter_map(|v| v.as_u64().map(|n| n as i64))
                            .collect();

                        let splits_name = format!("{}_splits", op_name);

                        // Create initializer for split sizes
                        let splits_tensor = TensorProto {
                            name: Some(splits_name.clone()),
                            data_type: Some(ProtoDataType::Int64 as i32),
                            dims: vec![split_sizes.len() as i64],
                            int64_data: split_sizes,
                            ..Default::default()
                        };
                        initializers.push(splits_tensor);

                        // Add splits as second input
                        inputs.push(splits_name);
                    }
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: outputs,
                    name: Some(op_name),
                    op_type: Some("Split".to_string()),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type == "gather" {
                // Gather operation - ONNX only supports int32/int64 indices, need to cast uint32/uint8
                // Also need to clamp indices to prevent out-of-bounds errors (following Chromium's approach)
                let mut inputs: Vec<String> = Vec::new();

                // First input: data tensor
                let data_operand_id = op.input_operands[0];
                inputs.push(operand_name(graph, data_operand_id));

                // Get axis parameter (default is 0)
                let axis = op
                    .attributes
                    .get("axis")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as usize;

                // Get input shape and dimension size at axis
                let data_operand = graph.operand(data_operand_id).ok_or_else(|| {
                    Self::invalid_operand("gather data lookup", data_operand_id, Some((op, idx)))
                })?;

                // Check if axis is within bounds
                if axis >= data_operand.descriptor.shape.len() {
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: format!(
                            "Gather operation: axis {} is out of bounds for shape with {} dimensions (operand {})",
                            axis,
                            data_operand.descriptor.shape.len(),
                            data_operand_id
                        ),
                    });
                }

                let dim_size = data_operand.descriptor.shape[axis] as i64;

                // Second input: indices tensor - may need casting and clamping
                let indices_id = op.input_operands[1];
                let indices_name = operand_name(graph, indices_id);
                let indices_operand = graph.operand(indices_id).ok_or_else(|| {
                    Self::invalid_operand("gather indices lookup", indices_id, Some((op, idx)))
                })?;

                // Step 1: Cast indices to int64 if needed (required for Clamp operation)
                let indices_after_cast =
                    if !matches!(indices_operand.descriptor.data_type, DataType::Int64) {
                        // Cast to int64 for ONNX compatibility (Clamp requires all inputs to be same type)
                        let cast_output_name = format!("{}_indices_int64", op_name);
                        nodes.push(Self::create_cast_node(
                            &format!("{}_cast_indices", op_name),
                            indices_name,
                            cast_output_name.clone(),
                            ProtoDataType::Int64,
                        ));
                        cast_output_name
                    } else {
                        indices_name
                    };

                // Step 2: Clamp indices to valid range [-dim_size, dim_size - 1]
                // This prevents out-of-bounds errors from ONNX Runtime
                let clamp_min_name = format!("{}_clamp_min", op_name);
                let clamp_max_name = format!("{}_clamp_max", op_name);
                let clamped_indices_name = format!("{}_indices_clamped", op_name);

                // Create scalar initializers for min and max
                initializers.push(TensorProto {
                    name: Some(clamp_min_name.clone()),
                    data_type: Some(ProtoDataType::Int64 as i32),
                    dims: vec![],
                    int64_data: vec![-dim_size],
                    ..Default::default()
                });

                initializers.push(TensorProto {
                    name: Some(clamp_max_name.clone()),
                    data_type: Some(ProtoDataType::Int64 as i32),
                    dims: vec![],
                    int64_data: vec![dim_size - 1],
                    ..Default::default()
                });

                // Insert Clip node (Clamp was deprecated in favor of Clip in opset 11+)
                nodes.push(NodeProto {
                    input: vec![indices_after_cast, clamp_min_name, clamp_max_name],
                    output: vec![clamped_indices_name.clone()],
                    name: Some(format!("{}_clip_indices", op_name)),
                    op_type: Some("Clip".to_string()),
                    ..Default::default()
                });

                // Optionally reshape indices to match the expected rank derived from the
                // output shape override (if provided). This keeps ONNX shape inference in
                // sync with WebNN metadata when the upstream graph collapses to scalars.
                let mut final_indices = clamped_indices_name;
                if let Some(output_id) = op.output_operand {
                    let out_operand = graph.operand(output_id).ok_or_else(|| {
                        Self::invalid_operand("gather output lookup", output_id, Some((op, idx)))
                    })?;
                    let out_shape = &out_operand.descriptor.shape;
                    let tail_len = data_operand.descriptor.shape.len().saturating_sub(axis + 1);
                    if !out_shape.is_empty() && out_shape.len() >= tail_len {
                        let target_indices_shape = out_shape[..out_shape.len() - tail_len].to_vec();
                        if !target_indices_shape.is_empty()
                            && target_indices_shape.iter().all(|d| *d > 0)
                        {
                            let shape_const_name = format!("{}_indices_shape", op_name);
                            initializers.push(TensorProto {
                                name: Some(shape_const_name.clone()),
                                data_type: Some(ProtoDataType::Int64 as i32),
                                dims: vec![target_indices_shape.len() as i64],
                                int64_data: target_indices_shape
                                    .iter()
                                    .map(|d| *d as i64)
                                    .collect(),
                                ..Default::default()
                            });
                            let reshaped_name = format!("{}_indices_reshaped", op_name);
                            nodes.push(NodeProto {
                                input: vec![final_indices.clone(), shape_const_name],
                                output: vec![reshaped_name.clone()],
                                name: Some(format!("{}_reshape_indices", op_name)),
                                op_type: Some("Reshape".to_string()),
                                ..Default::default()
                            });
                            final_indices = reshaped_name;
                        }
                    }
                }

                inputs.push(final_indices);

                // Create Gather node
                let attributes = Self::create_operation_attributes(op);
                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some("Gather".to_string()),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type == "conv2d" || op.op_type == "convTranspose2d" {
                // Conv2d/ConvTranspose2d operations - handle layout transformations
                let mut conv_inputs: Vec<String> = Vec::new();

                // Handle input layout (NHWC → NCHW if needed)
                let input_name = operand_name(graph, op.input_operands[0]);
                let input_layout = op
                    .attributes
                    .get("inputLayout")
                    .and_then(|v| v.as_str())
                    .unwrap_or("nchw");

                let transposed_input = if input_layout == "nhwc" {
                    // Insert Transpose node: NHWC → NCHW
                    let transpose_output = format!("{}_input_transposed", op_name);
                    nodes.push(NodeProto {
                        input: vec![input_name],
                        output: vec![transpose_output.clone()],
                        name: Some(format!("{}_transpose_input", op_name)),
                        op_type: Some("Transpose".to_string()),
                        attribute: vec![AttributeProto {
                            name: Some("perm".to_string()),
                            r#type: Some(AttributeType::Ints as i32),
                            ints: vec![0, 3, 1, 2], // NHWC → NCHW
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    transpose_output
                } else {
                    input_name
                };
                conv_inputs.push(transposed_input);

                // Handle filter layout transformation
                let filter_name = operand_name(graph, op.input_operands[1]);
                let filter_layout = op
                    .attributes
                    .get("filterLayout")
                    .and_then(|v| v.as_str())
                    .unwrap_or(if op.op_type == "convTranspose2d" {
                        "iohw"
                    } else {
                        "oihw"
                    });

                let is_transpose = op.op_type == "convTranspose2d";
                let needs_transpose = if is_transpose {
                    // ConvTranspose: ONNX expects IOHW (Input, Output, H, W)
                    filter_layout != "iohw"
                } else {
                    // Conv: ONNX expects OIHW (Output, Input, H, W)
                    filter_layout != "oihw"
                };

                let transposed_filter = if needs_transpose {
                    let perm = if is_transpose {
                        // ConvTranspose filter layout conversions → IOHW
                        match filter_layout {
                            "hwoi" => vec![3, 2, 0, 1], // HWOI (H,W,O,I) → IOHW (I,O,H,W)
                            "ohwi" => vec![3, 0, 1, 2], // OHWI (O,H,W,I) → IOHW (I,O,H,W)
                            "oihw" => vec![1, 0, 2, 3], // OIHW (O,I,H,W) → IOHW (I,O,H,W)
                            _ => vec![0, 1, 2, 3],      // Default: no transpose
                        }
                    } else {
                        // Conv2d filter layout conversions → OIHW
                        match filter_layout {
                            "hwio" => vec![3, 2, 0, 1], // HWIO (H,W,I,O) → OIHW (O,I,H,W)
                            "ohwi" => vec![0, 3, 1, 2], // OHWI (O,H,W,I) → OIHW (O,I,H,W)
                            "ihwo" => vec![3, 0, 1, 2], // IHWO (I,H,W,O) → OIHW (O,I,H,W)
                            _ => vec![0, 1, 2, 3],      // Default: no transpose
                        }
                    };

                    let transpose_output = format!("{}_filter_transposed", op_name);
                    nodes.push(NodeProto {
                        input: vec![filter_name],
                        output: vec![transpose_output.clone()],
                        name: Some(format!("{}_transpose_filter", op_name)),
                        op_type: Some("Transpose".to_string()),
                        attribute: vec![AttributeProto {
                            name: Some("perm".to_string()),
                            r#type: Some(AttributeType::Ints as i32),
                            ints: perm,
                            ..Default::default()
                        }],
                        ..Default::default()
                    });
                    transpose_output
                } else {
                    filter_name
                };
                conv_inputs.push(transposed_filter);

                // Add bias if present (third input)
                if op.input_operands.len() > 2 {
                    conv_inputs.push(operand_name(graph, op.input_operands[2]));
                }

                // Create Conv/ConvTranspose node
                let attributes = Self::create_operation_attributes(op);
                nodes.push(NodeProto {
                    input: conv_inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if matches!(
                op.op_type.as_str(),
                "layerNormalization" | "batchNormalization" | "instanceNormalization"
            ) {
                // Normalization operations - ONNX requires scale/bias as inputs, not attributes
                // Following Chromium's approach: create default initializers when not provided

                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("normalization input lookup", input_id, Some((op, idx)))
                })?;
                let input_data_type = Self::data_type_code(input_operand.descriptor.data_type);

                let mut inputs: Vec<String> = vec![operand_name(graph, input_id)];

                // Check if scale and bias are provided via attributes
                let has_scale = op
                    .attributes
                    .get("has_scale")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let has_bias = op
                    .attributes
                    .get("has_bias")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                // For layer normalization, check if axes are empty or if input is 0D
                // When axes are empty, no normalization occurs (output = bias or 0)
                if op.op_type == "layerNormalization" {
                    let axes = op
                        .attributes
                        .get("axes")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                        .unwrap_or_else(std::vec::Vec::new);

                    // Handle empty axes or 0D tensor with axes=[-1] case
                    // When axes are empty or when input is 0D scalar, no normalization occurs
                    // This matches Chromium's implementation
                    let is_0d_with_default_axes =
                        input_operand.descriptor.shape.is_empty() && axes == vec![-1];
                    if axes.is_empty() || is_0d_with_default_axes {
                        let output_name = operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        );

                        if has_bias && op.input_operands.len() > 2 {
                            // output = bias + 0
                            let bias_name = operand_name(graph, op.input_operands[2]);
                            let zero_name = format!("{}_zero", op_name);
                            initializers.push(Self::create_scalar_initializer(
                                zero_name.clone(),
                                input_data_type,
                                0.0,
                            ));
                            nodes.push(NodeProto {
                                input: vec![bias_name, zero_name],
                                output: vec![output_name],
                                name: Some(op_name),
                                op_type: Some("Add".to_string()),
                                ..Default::default()
                            });
                        } else {
                            // output = input - input = 0
                            let input_name = operand_name(graph, input_id);
                            nodes.push(NodeProto {
                                input: vec![input_name.clone(), input_name],
                                output: vec![output_name],
                                name: Some(op_name),
                                op_type: Some("Sub".to_string()),
                                ..Default::default()
                            });
                        }
                        continue;
                    }
                }

                // Determine scale/bias shape based on normalization type
                let scale_bias_shape = if op.op_type == "layerNormalization" {
                    // For layer norm, ONNX LayerNormalization expects scale/bias shape to match
                    // X.shape[axis:], i.e., all dimensions from axis to the end
                    let axes = op
                        .attributes
                        .get("axes")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                        .unwrap_or_else(|| vec![-1]);

                    // Get the first axis (ONNX only supports a single axis parameter)
                    let first_axis = axes.first().copied().unwrap_or(-1);
                    let actual_axis = if first_axis < 0 {
                        ((input_operand.descriptor.shape.len() as i64 + first_axis) as usize)
                            .min(input_operand.descriptor.shape.len())
                    } else {
                        (first_axis as usize).min(input_operand.descriptor.shape.len())
                    };

                    // Calculate product of all dimensions from axis to end
                    let mut size = 1i64;
                    for i in actual_axis..input_operand.descriptor.shape.len() {
                        size *= input_operand.descriptor.shape[i] as i64;
                    }
                    vec![size]
                } else if op.op_type == "batchNormalization" {
                    // For batch norm, scale/bias/mean/variance shape is [channels]
                    // Channel dimension is specified by the axis parameter
                    let axis = op
                        .attributes
                        .get("axis")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(1);

                    let channel_dim = if axis < 0 {
                        ((input_operand.descriptor.shape.len() as i64 + axis) as usize)
                            .min(input_operand.descriptor.shape.len().saturating_sub(1))
                    } else {
                        (axis as usize).min(input_operand.descriptor.shape.len().saturating_sub(1))
                    };

                    let channels = if input_operand.descriptor.shape.len() > channel_dim {
                        input_operand.descriptor.shape[channel_dim] as i64
                    } else {
                        1
                    };
                    vec![channels]
                } else if op.op_type == "instanceNormalization" {
                    // For instance norm, scale/bias shape is [channels]
                    // Channel dimension depends on layout: NCHW=1, NHWC=last
                    let layout = op
                        .attributes
                        .get("layout")
                        .and_then(|v| v.as_str())
                        .unwrap_or("nchw");

                    // TODO: ONNX InstanceNormalization ALWAYS requires NCHW layout
                    // When layout='nhwc', we need to add transpose nodes:
                    // 1. Input: NHWC → NCHW (before operation)
                    // 2. Output: NCHW → NHWC (after operation)
                    // Currently failing 4 tests: instanceNormalization with layout='nhwc'
                    // See: https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/ort/graph_builder_ort.cc
                    // Chromium comment: "ONNX InstanceNormalization expects NCHW layout, channel is at index 1"

                    let channel_dim = if layout == "nhwc" {
                        // NHWC: channels at last dimension
                        input_operand.descriptor.shape.len().saturating_sub(1)
                    } else {
                        // NCHW: channels at dimension 1 (default)
                        1
                    };

                    let channels = if input_operand.descriptor.shape.len() > channel_dim {
                        input_operand.descriptor.shape[channel_dim] as i64
                    } else {
                        1
                    };
                    vec![channels]
                } else {
                    vec![1]
                };

                // Batch normalization has different input order than layer/instance normalization
                // Python API order: [input, mean, variance, scale?, bias?]
                // ONNX order: [input, scale, bias, mean, variance]
                if op.op_type == "batchNormalization" {
                    // Add scale (index 3 in Python API if provided, else default)
                    if has_scale && op.input_operands.len() > 3 {
                        inputs.push(operand_name(graph, op.input_operands[3]));
                    } else {
                        let scale_name = format!("{}_scale_default", op_name);
                        initializers.push(Self::create_vector_initializer(
                            scale_name.clone(),
                            input_data_type,
                            scale_bias_shape.clone(),
                            1.0,
                        ));
                        inputs.push(scale_name);
                    }

                    // Add bias (index 4 in Python API if provided, else default)
                    if has_bias && op.input_operands.len() > 4 {
                        inputs.push(operand_name(graph, op.input_operands[4]));
                    } else {
                        let bias_name = format!("{}_bias_default", op_name);
                        initializers.push(Self::create_vector_initializer(
                            bias_name.clone(),
                            input_data_type,
                            scale_bias_shape.clone(),
                            0.0,
                        ));
                        inputs.push(bias_name);
                    }

                    // Add mean (index 1 - required)
                    if op.input_operands.len() > 1 {
                        inputs.push(operand_name(graph, op.input_operands[1]));
                    }

                    // Add variance (index 2 - required)
                    if op.input_operands.len() > 2 {
                        inputs.push(operand_name(graph, op.input_operands[2]));
                    }
                } else {
                    // Layer normalization and instance normalization
                    // Python API order: [input, scale?, bias?]
                    // ONNX order: [input, scale, bias]

                    // Add scale input (from operand or create default with 1.0)
                    if has_scale && op.input_operands.len() > 1 {
                        inputs.push(operand_name(graph, op.input_operands[1]));
                    } else {
                        // Create default scale initializer (all 1.0) with proper dtype
                        let scale_name = format!("{}_scale_default", op_name);
                        initializers.push(Self::create_vector_initializer(
                            scale_name.clone(),
                            input_data_type,
                            scale_bias_shape.clone(),
                            1.0,
                        ));
                        inputs.push(scale_name);
                    }

                    // Add bias input (from operand or create default with 0.0) - optional for layer norm
                    if has_bias && op.input_operands.len() > 2 {
                        inputs.push(operand_name(graph, op.input_operands[2]));
                    } else if op.op_type != "layerNormalization" || has_bias {
                        // Batch/instance norm always need bias; layer norm only if explicitly requested
                        let bias_name = format!("{}_bias_default", op_name);
                        initializers.push(Self::create_vector_initializer(
                            bias_name.clone(),
                            input_data_type,
                            scale_bias_shape.clone(),
                            0.0,
                        ));
                        inputs.push(bias_name);
                    }
                }

                let attributes = Self::create_operation_attributes(op);
                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type == "hardSwish" {
                // HardSwish decomposition: x * clip(x + 3, 0, 6) / 6
                // ONNX opset 13 doesn't have HardSwish, so we decompose it
                let input_name = operand_name(graph, op.input_operands[0]);
                let output_name = operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );

                // Get input data type for scalar initializers
                let input_operand = graph.operand(op.input_operands[0]).ok_or_else(|| {
                    Self::invalid_operand(
                        "hardSwish input lookup",
                        op.input_operands[0],
                        Some((op, idx)),
                    )
                })?;
                let dtype = Self::data_type_code(input_operand.descriptor.data_type);

                // Step 1: Add 3 to x
                let three_name = format!("{}_const_3", op_name);
                initializers.push(Self::create_scalar_initializer(
                    three_name.clone(),
                    dtype,
                    3.0,
                ));

                let add_output = format!("{}_add_3", op_name);
                nodes.push(NodeProto {
                    input: vec![input_name.clone(), three_name],
                    output: vec![add_output.clone()],
                    name: Some(format!("{}_add", op_name)),
                    op_type: Some("Add".to_string()),
                    ..Default::default()
                });

                // Step 2: Clip to [0, 6]
                let zero_name = format!("{}_const_0", op_name);
                let six_name = format!("{}_const_6", op_name);
                initializers.push(Self::create_scalar_initializer(
                    zero_name.clone(),
                    dtype,
                    0.0,
                ));
                initializers.push(Self::create_scalar_initializer(
                    six_name.clone(),
                    dtype,
                    6.0,
                ));

                let clip_output = format!("{}_clip", op_name);
                nodes.push(NodeProto {
                    input: vec![add_output, zero_name, six_name],
                    output: vec![clip_output.clone()],
                    name: Some(format!("{}_clip", op_name)),
                    op_type: Some("Clip".to_string()),
                    ..Default::default()
                });

                // Step 3: Divide by 6
                let six_div_name = format!("{}_const_6_div", op_name);
                initializers.push(Self::create_scalar_initializer(
                    six_div_name.clone(),
                    dtype,
                    6.0,
                ));

                let div_output = format!("{}_div", op_name);
                nodes.push(NodeProto {
                    input: vec![clip_output, six_div_name],
                    output: vec![div_output.clone()],
                    name: Some(format!("{}_div", op_name)),
                    op_type: Some("Div".to_string()),
                    ..Default::default()
                });

                // Step 4: Multiply by x
                nodes.push(NodeProto {
                    input: vec![input_name, div_output],
                    output: vec![output_name],
                    name: Some(format!("{}_mul", op_name)),
                    op_type: Some("Mul".to_string()),
                    ..Default::default()
                });
            } else {
                // Check if operation requires float types (ONNX limitation)
                let has_float_inputs = op.input_operands.iter().any(|&input_id| {
                    graph
                        .operand(input_id)
                        .map(|operand| {
                            let dtype = type_overrides
                                .get(&input_id)
                                .copied()
                                .unwrap_or(operand.descriptor.data_type);
                            matches!(dtype, DataType::Float32 | DataType::Float16)
                        })
                        .unwrap_or(false)
                });
                let requires_float = matches!(
                    op.op_type.as_str(),
                    "relu"
                        | "sigmoid"
                        | "tanh"
                        | "softmax"
                        | "elu"
                        | "leakyRelu"
                        | "prelu"
                        | "hardSigmoid"
                        | "hardSwish"
                        | "softplus"
                        | "softsign"
                        | "sub"
                        | "div"
                        | "pow"
                );

                // Check if any inputs have integer types
                let has_integer_inputs = op.input_operands.iter().any(|&input_id| {
                    if let Some(operand) = graph.operand(input_id) {
                        let dtype = type_overrides
                            .get(&input_id)
                            .copied()
                            .unwrap_or(operand.descriptor.data_type);
                        matches!(
                            dtype,
                            DataType::Int8
                                | DataType::Uint8
                                | DataType::Int32
                                | DataType::Uint32
                                | DataType::Int64
                                | DataType::Uint64
                        )
                    } else {
                        false
                    }
                });

                let mixed_numeric_inputs = has_integer_inputs
                    && has_float_inputs
                    && matches!(op.op_type.as_str(), "mul" | "add");

                if (requires_float && has_integer_inputs) || mixed_numeric_inputs {
                    // Cast inputs to float32, execute operation, cast output back
                    let mut cast_inputs = Vec::new();
                    let mut original_types = Vec::new();

                    for &input_id in &op.input_operands {
                        let input_name = operand_name(graph, input_id);
                        let input_operand = graph.operand(input_id).ok_or_else(|| {
                            Self::invalid_operand(
                                "float-cast input lookup",
                                input_id,
                                Some((op, idx)),
                            )
                        })?;

                        let dtype = type_overrides
                            .get(&input_id)
                            .copied()
                            .unwrap_or(input_operand.descriptor.data_type);
                        original_types.push(dtype);

                        if matches!(
                            input_operand.descriptor.data_type,
                            DataType::Int8
                                | DataType::Uint8
                                | DataType::Int32
                                | DataType::Uint32
                                | DataType::Int64
                                | DataType::Uint64
                        ) {
                            // Cast to float32
                            let cast_output_name =
                                format!("{}_input_{}_float32", op_name, cast_counter);
                            cast_counter += 1;

                            nodes.push(Self::create_cast_node(
                                &format!("cast_to_float32_{}", cast_counter - 1),
                                input_name,
                                cast_output_name.clone(),
                                ProtoDataType::Float,
                            ));

                            cast_inputs.push(cast_output_name);
                        } else {
                            cast_inputs.push(input_name);
                        }
                    }

                    // Create the operation node (outputs float32)
                    let float_output_name = format!("{}_float32_output", op_name);
                    let output_operand_id =
                        op.output_operand.expect("Single-output operation expected");
                    let final_output_name = operand_name(graph, output_operand_id);
                    let op_output_name = if requires_float && !mixed_numeric_inputs {
                        float_output_name.clone()
                    } else {
                        final_output_name.clone()
                    };
                    let attributes = Self::create_operation_attributes(op);

                    nodes.push(NodeProto {
                        input: cast_inputs,
                        output: vec![op_output_name.clone()],
                        name: Some(op_name.clone()),
                        op_type: Some(Self::onnx_op_type(&op.op_type)),
                        attribute: attributes,
                        ..Default::default()
                    });

                    // Cast output back to original type (use first input's type as reference)
                    if requires_float && !mixed_numeric_inputs {
                        let output_type = original_types[0];
                        nodes.push(Self::create_cast_node(
                            &format!("{}_cast_output", op_name),
                            float_output_name,
                            final_output_name,
                            Self::data_type_code(output_type),
                        ));
                        type_overrides.insert(output_operand_id, output_type);
                    } else {
                        // Keep float output for mixed numeric inputs so downstream ops see a float
                        type_overrides.insert(output_operand_id, DataType::Float32);
                    }
                } else {
                    // Regular operation - no Cast nodes needed
                    let attributes = Self::create_operation_attributes(op);

                    nodes.push(NodeProto {
                        input: op
                            .input_operands
                            .iter()
                            .map(|id| operand_name(graph, *id))
                            .collect(),
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: Some(op_name),
                        op_type: Some(Self::onnx_op_type(&op.op_type)),
                        attribute: attributes,
                        ..Default::default()
                    });
                }
            }
        }

        // Add value_info for operands where we have inferred shapes
        let mut seen_names = std::collections::HashSet::new();
        for vi in inputs_val.iter().chain(outputs_val.iter()) {
            if let Some(name) = &vi.name {
                seen_names.insert(name.clone());
            }
        }
        for (idx, operand) in graph.operands.iter().enumerate() {
            let name = operand
                .name
                .clone()
                .unwrap_or_else(|| format!("operand_{}", idx as u32));
            if seen_names.contains(&name) {
                continue;
            }

            // Prefer shape_overrides, otherwise use any known operand_shapes entry
            let shape_override = shape_overrides
                .get(&(idx as u32))
                .cloned()
                .or_else(|| operand_shapes.get(&(idx as u32)).cloned());

            if let Some(shape) = shape_override
                && !shape.is_empty()
            {
                let mut desc = operand.descriptor.clone();
                desc.shape = shape;
                if let Some(dt) = type_overrides.get(&(idx as u32)) {
                    desc.data_type = *dt;
                }
                value_infos.push(value_info(&name, &desc));
            }
        }

        let graph_proto = GraphProto {
            name: Some("webnn_graph".to_string()),
            node: nodes,
            input: inputs_val,
            output: outputs_val,
            initializer: initializers,
            value_info: value_infos,
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
            }],
            ..Default::default()
        };

        let data = model.encode_to_vec();

        Ok(ConvertedGraph {
            format: "onnx",
            content_type: "application/onnx",
            data,
            weights_data: None, // ONNX doesn't use external weight files
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
                                denotation: None,
                            })
                            .collect(),
                    }),
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}
