use crate::converters::{ConvertedGraph, operand_name};
use crate::debug_print;
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, OperandKind, Operation};
use crate::protos::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, attribute_proto::AttributeType,
    tensor_proto::DataType as ProtoDataType, type_proto::Tensor as TensorTypeProto,
};
use crate::shape_inference::{broadcast_shapes, infer_matmul_shape, infer_transpose_shape};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use prost::Message;
use webnn_onnx_utils::{
    attributes::AttrBuilder, data_types as utils_data_types, tensor_data::TensorData,
};

#[derive(Default)]
pub struct OnnxConverter;

impl OnnxConverter {
    fn invalid_operand(
        context: &str,
        operand: u32,
        op_info: Option<(&Operation, usize)>,
    ) -> GraphError {
        if let Some((op, idx)) = op_info {
            debug_print!(
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
            debug_print!("[DEBUG] Invalid operand {} at {}", operand, context);
        }
        GraphError::InvalidConversionOperand { operand }
    }

    fn data_type_code(data_type: DataType) -> ProtoDataType {
        // Convert rust-webnn-graph DataType to webnn_onnx_utils DataType first
        let utils_dtype = match data_type {
            DataType::Float32 => utils_data_types::DataType::Float32,
            DataType::Float16 => utils_data_types::DataType::Float16,
            DataType::Int32 => utils_data_types::DataType::Int32,
            DataType::Uint32 => utils_data_types::DataType::Uint32,
            DataType::Int64 => utils_data_types::DataType::Int64,
            DataType::Uint64 => utils_data_types::DataType::Uint64,
            DataType::Int8 => utils_data_types::DataType::Int8,
            DataType::Uint8 => utils_data_types::DataType::Uint8,
        };
        // Use shared library conversion
        utils_data_types::webnn_to_onnx(utils_dtype)
    }

    fn create_scalar_initializer(name: String, dtype: ProtoDataType, value: f32) -> TensorProto {
        // Convert ProtoDataType to utils DataType
        let utils_dtype = utils_data_types::onnx_proto_to_webnn(dtype)
            .unwrap_or(utils_data_types::DataType::Float32);

        // Use shared library to create scalar tensor
        TensorData::scalar(utils_dtype.clone(), value).to_tensor_proto(name, utils_dtype, vec![])
    }

    /// Create a vector initializer with proper data type handling
    fn create_vector_initializer(
        name: String,
        dtype: ProtoDataType,
        shape: Vec<i64>,
        value: f32,
    ) -> TensorProto {
        // Convert ProtoDataType to utils DataType
        let utils_dtype = utils_data_types::onnx_proto_to_webnn(dtype)
            .unwrap_or(utils_data_types::DataType::Float32);

        // Use shared library to create filled tensor
        TensorData::filled(utils_dtype.clone(), &shape, value).to_tensor_proto(
            name,
            utils_dtype,
            shape,
        )
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
        // Use shared library's parse_json_ints
        webnn_onnx_utils::attributes::parse_json_ints(&op.attributes, json_key)
    }

    /// Helper: Add an integer array attribute using shared AttrBuilder
    fn add_ints_attribute(attributes: &mut Vec<AttributeProto>, name: &str, values: Vec<i64>) {
        if !values.is_empty() {
            let builder = AttrBuilder::new().add_ints(name, values);
            attributes.extend(builder.build());
        }
    }

    /// Helper: Add an integer attribute using shared AttrBuilder
    fn add_int_attribute(attributes: &mut Vec<AttributeProto>, name: &str, value: i64) {
        let builder = AttrBuilder::new().add_int(name, value);
        attributes.extend(builder.build());
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
            name: node_name.to_string(),
            op_type: "Cast".to_string(),
            attribute: vec![AttributeProto {
                name: "to".to_string(),
                r#type: AttributeType::Int as i32,
                i: to_data_type as i64,
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
                name: "axis".to_string(),
                r#type: AttributeType::Int as i32,
                i: axis as i64,
                ..Default::default()
            });
        }

        if let Some(keep_dims) = op
            .attributes
            .get("keepDimensions")
            .and_then(|v| v.as_bool())
        {
            attributes.push(AttributeProto {
                name: "keepdims".to_string(), // ONNX uses "keepdims" not "keepDimensions"
                r#type: AttributeType::Int as i32,
                i: if keep_dims { 1 } else { 0 },
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
        if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_i64()) {
            attributes.push(AttributeProto {
                name: "axis".to_string(),
                r#type: AttributeType::Int as i32,
                i: axis,
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
                name: "to".to_string(),
                r#type: AttributeType::Int as i32,
                i: type_code,
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
                name: "axis".to_string(),
                r#type: AttributeType::Int as i32,
                i: axis,
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
                name: "upper".to_string(),
                r#type: AttributeType::Int as i32,
                i: if upper { 1 } else { 0 },
                ..Default::default()
            });
        }

        if let Some(diagonal) = op.attributes.get("diagonal").and_then(|v| v.as_i64()) {
            attributes.push(AttributeProto {
                name: "k".to_string(), // ONNX uses "k" for diagonal offset
                r#type: AttributeType::Int as i32,
                i: diagonal,
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
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
                ..Default::default()
            });
        }

        if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "beta".to_string(),
                r#type: AttributeType::Float as i32,
                f: beta as f32,
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
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
                ..Default::default()
            });
        }

        if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "beta".to_string(),
                r#type: AttributeType::Float as i32,
                f: beta as f32,
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
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
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
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
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
                name: "alpha".to_string(),
                r#type: AttributeType::Float as i32,
                f: alpha as f32,
                ..Default::default()
            });
        }

        if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
            attributes.push(AttributeProto {
                name: "beta".to_string(),
                r#type: AttributeType::Float as i32,
                f: beta as f32,
                ..Default::default()
            });
        }

        if let Some(a_transpose) = op.attributes.get("a_transpose").and_then(|v| v.as_bool()) {
            attributes.push(AttributeProto {
                name: "transA".to_string(),
                r#type: AttributeType::Int as i32,
                i: if a_transpose { 1 } else { 0 },
                ..Default::default()
            });
        }

        if let Some(b_transpose) = op.attributes.get("b_transpose").and_then(|v| v.as_bool()) {
            attributes.push(AttributeProto {
                name: "transB".to_string(),
                r#type: AttributeType::Int as i32,
                i: if b_transpose { 1 } else { 0 },
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
            name: "epsilon".to_string(),
            r#type: AttributeType::Float as i32,
            f: epsilon as f32,
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
            name: "axis".to_string(),
            r#type: AttributeType::Int as i32,
            i: axis,
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
            name: "epsilon".to_string(),
            r#type: AttributeType::Float as i32,
            f: epsilon as f32,
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
        debug_print!("[DEBUG] Starting ONNX conversion");
        debug_print!("  Total operations: {}", graph.operations.len());
        let expand_count = graph
            .operations
            .iter()
            .filter(|op| op.op_type == "expand")
            .count();
        debug_print!("  Expand operations: {}", expand_count);

        let mut initializers = Vec::new();
        let mut inputs_val = Vec::new();
        let mut outputs_val = Vec::new();
        let mut value_infos = Vec::new(); // Will add entries for explicitly tracked shapes
        let mut skipped_inputs = std::collections::HashSet::new(); // Track skipped empty KV inputs
        let operand_remapping: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new(); // Map skipped outputs to replacements

        for &id in &graph.input_operands {
            let operand = graph.operand(id).ok_or_else(|| {
                debug_print!(
                    "[DEBUG] Missing input operand {} while building ONNX graph",
                    id
                );
                Self::invalid_operand("graph input lookup", id, None)
            })?;

            // Skip KV cache inputs with empty dimensions (past_sequence_length=0)
            // These inputs are never used in the computation - they're just concatenated
            // with new KV, and concat(empty, new) = new.
            let input_name = operand_name(graph, id);
            let has_empty_dimension = operand.descriptor.shape.contains(&0);
            let is_kv_input = input_name.starts_with("past_key_values_");

            // Debug: print all KV input shapes
            if is_kv_input {
                debug_print!(
                    "[ONNX CONVERTER] KV input: {} shape={:?} has_empty={}",
                    input_name,
                    operand.descriptor.shape,
                    has_empty_dimension
                );
            }

            if has_empty_dimension && is_kv_input {
                debug_print!(
                    "[DEBUG] Skipping empty KV cache input: {} (shape: {:?})",
                    input_name,
                    operand.descriptor.shape
                );
                skipped_inputs.insert(id);
                continue;
            }

            inputs_val.push(value_info(&input_name, &operand.descriptor));
        }

        // Sort outputs: "logits" first, then alphabetically by name
        // This ensures ONNX models have logits at output 0 (expected by users)
        let mut sorted_outputs: Vec<u32> = graph.output_operands.clone();
        sorted_outputs.sort_by_key(|&id| {
            let operand = graph.operand(id);
            let name = operand.and_then(|op| op.name.as_deref()).unwrap_or("");
            // Sort key: (priority, name)
            // "logits" gets priority 0 (first), everything else gets priority 1 (alphabetically)
            if name == "logits" {
                (0, String::new())
            } else {
                (1, name.to_string())
            }
        });

        for &id in &sorted_outputs {
            let operand = graph.operand(id).ok_or_else(|| {
                debug_print!(
                    "[DEBUG] Missing output operand {} while building ONNX graph",
                    id
                );
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
                if let (Some(&input_id), Some(output_id)) =
                    (op.input_operands.first(), op.output_operand)
                    && let Some(input_operand) = graph.operand(input_id)
                {
                    type_overrides.insert(output_id, input_operand.descriptor.data_type);

                    // Check for newShape attribute first
                    if let Some(new_shape_attr) = op.attributes.get("newShape") {
                        if let Some(new_shape_array) = new_shape_attr.as_array() {
                            let shape: Vec<u32> = new_shape_array
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
                    // Fall back to shape from second input operand (if present)
                    else if op.input_operands.len() >= 2
                        && let Some(shape) = operand_shapes.get(&op.input_operands[1])
                    {
                        shape_overrides.insert(output_id, shape.clone());
                        operand_shapes.insert(output_id, shape.clone());
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
            } else if op.op_type.eq_ignore_ascii_case("cos")
                || op.op_type.eq_ignore_ascii_case("sin")
                || op.op_type.eq_ignore_ascii_case("tan")
                || op.op_type.eq_ignore_ascii_case("exp")
                || op.op_type.eq_ignore_ascii_case("log")
                || op.op_type.eq_ignore_ascii_case("abs")
                || op.op_type.eq_ignore_ascii_case("neg")
                || op.op_type.eq_ignore_ascii_case("sqrt")
                || op.op_type.eq_ignore_ascii_case("relu")
                || op.op_type.eq_ignore_ascii_case("sigmoid")
                || op.op_type.eq_ignore_ascii_case("tanh")
                || op.op_type.eq_ignore_ascii_case("cast")
            {
                // Track unary element-wise operations (preserve input shape and type)
                if let Some(output_id) = op.output_operand
                    && let Some(&input_id) = op.input_operands.first()
                {
                    let output_name = graph
                        .operand(output_id)
                        .and_then(|op| op.name.as_ref())
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");

                    // Preserve shape from input
                    if let Some(input_shape) = operand_shapes.get(&input_id) {
                        debug_print!(
                            "[UNARY DEBUG] {} op {} preserves shape {:?} from input {}",
                            op.op_type,
                            output_name,
                            input_shape,
                            input_id
                        );
                        shape_overrides.insert(output_id, input_shape.clone());
                        operand_shapes.insert(output_id, input_shape.clone());
                    } else {
                        debug_print!(
                            "[UNARY WARNING] {} op {} has no input shape for input {}",
                            op.op_type,
                            output_name,
                            input_id
                        );
                    }

                    // Preserve type from input (except Cast which changes type)
                    if !op.op_type.eq_ignore_ascii_case("cast") {
                        let input_type = type_overrides
                            .get(&input_id)
                            .copied()
                            .or_else(|| graph.operand(input_id).map(|op| op.descriptor.data_type));

                        if let Some(dtype) = input_type {
                            type_overrides.insert(output_id, dtype);
                        }
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("add")
                || op.op_type.eq_ignore_ascii_case("sub")
                || op.op_type.eq_ignore_ascii_case("mul")
                || op.op_type.eq_ignore_ascii_case("div")
                || op.op_type.eq_ignore_ascii_case("pow")
            {
                // Track binary element-wise operation output shapes (use broadcasting)
                if let Some(output_id) = op.output_operand
                    && op.input_operands.len() >= 2
                {
                    // Try to compute broadcast shape from inputs
                    if let (Some(lhs), Some(rhs)) = (
                        operand_shapes.get(&op.input_operands[0]),
                        operand_shapes.get(&op.input_operands[1]),
                    ) && let Ok(result_shape) = broadcast_shapes(lhs, rhs)
                    {
                        shape_overrides.insert(output_id, result_shape.clone());
                        operand_shapes.insert(output_id, result_shape);
                    }

                    // Preserve type from first input (binary ops typically preserve input type)
                    if let Some(&first_input_id) = op.input_operands.first() {
                        let input_type =
                            type_overrides.get(&first_input_id).copied().or_else(|| {
                                graph
                                    .operand(first_input_id)
                                    .map(|op| op.descriptor.data_type)
                            });

                        if let Some(dtype) = input_type {
                            type_overrides.insert(output_id, dtype);
                        }
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("matmul") {
                // Track matmul output shapes
                if let Some(output_id) = op.output_operand
                    && op.input_operands.len() == 2
                {
                    let output_name = graph
                        .operand(output_id)
                        .and_then(|op| op.name.as_ref())
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");

                    // Get input shapes
                    let lhs_shape = operand_shapes.get(&op.input_operands[0]);
                    let rhs_shape = operand_shapes.get(&op.input_operands[1]);

                    if let (Some(lhs), Some(rhs)) = (lhs_shape, rhs_shape) {
                        if let Ok(out_shape) = infer_matmul_shape(lhs, rhs) {
                            debug_print!(
                                "[MATMUL DEBUG] Matmul {} tracked output shape {:?} from inputs {:?} @ {:?}",
                                output_name,
                                out_shape,
                                lhs,
                                rhs
                            );
                            shape_overrides.insert(output_id, out_shape.clone());
                            operand_shapes.insert(output_id, out_shape);
                        } else {
                            debug_print!(
                                "[MATMUL WARNING] Matmul {} failed to infer shape from inputs {:?} @ {:?}",
                                output_name,
                                lhs,
                                rhs
                            );
                        }
                    } else {
                        debug_print!(
                            "[MATMUL WARNING] Matmul {} missing input shapes: lhs={} rhs={}",
                            output_name,
                            lhs_shape.is_some(),
                            rhs_shape.is_some()
                        );
                    }

                    // Preserve type from first input
                    let input_type =
                        type_overrides
                            .get(&op.input_operands[0])
                            .copied()
                            .or_else(|| {
                                graph
                                    .operand(op.input_operands[0])
                                    .map(|op| op.descriptor.data_type)
                            });

                    if let Some(dtype) = input_type {
                        type_overrides.insert(output_id, dtype);
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("concat") {
                // Track concat output shapes
                if let Some(output_id) = op.output_operand
                    && op.input_operands.len() >= 2
                {
                    let output_name = graph
                        .operand(output_id)
                        .and_then(|op| op.name.as_ref())
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");

                    // Get axis attribute
                    let axis = op
                        .attributes
                        .get("axis")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);

                    // Get input shapes
                    let input_shapes: Vec<_> = op
                        .input_operands
                        .iter()
                        .filter_map(|id| operand_shapes.get(id))
                        .collect();

                    if input_shapes.len() == op.input_operands.len() && !input_shapes.is_empty() {
                        let rank = input_shapes[0].len() as i64;
                        let mut axis = axis;
                        if axis < 0 {
                            axis += rank;
                        }
                        if axis >= 0 && (axis as usize) < input_shapes[0].len() {
                            let mut out_shape = input_shapes[0].clone();
                            let concat_axis = axis as usize;
                            for shape in &input_shapes[1..] {
                                out_shape[concat_axis] += shape[concat_axis];
                            }
                            debug_print!(
                                "[CONCAT DEBUG] Concat {} tracked output shape {:?}",
                                output_name,
                                out_shape
                            );
                            shape_overrides.insert(output_id, out_shape.clone());
                            operand_shapes.insert(output_id, out_shape);
                        }
                    } else {
                        debug_print!(
                            "[CONCAT WARNING] Concat {} missing input shapes: have {}/{} inputs",
                            output_name,
                            input_shapes.len(),
                            op.input_operands.len()
                        );
                    }

                    // Preserve type from first input (concat requires all inputs have same type)
                    if let Some(&first_input_id) = op.input_operands.first() {
                        // Check if we have a type override for the first input
                        let input_type =
                            type_overrides.get(&first_input_id).copied().or_else(|| {
                                // Fall back to descriptor type
                                graph
                                    .operand(first_input_id)
                                    .map(|op| op.descriptor.data_type)
                            });

                        if let Some(dtype) = input_type {
                            type_overrides.insert(output_id, dtype);
                        }
                    }
                }
            } else if op.op_type.eq_ignore_ascii_case("unsqueeze") {
                // Track unsqueeze output shapes (adds dimensions)
                if let Some(output_id) = op.output_operand
                    && let Some(&input_id) = op.input_operands.first()
                    && let Some(input_shape) = operand_shapes.get(&input_id)
                    && let Some(axes_val) = op.attributes.get("axes")
                {
                    let axes_i64: Vec<i64> = if let Some(arr) = axes_val.as_array() {
                        arr.iter().filter_map(|v| v.as_i64()).collect()
                    } else {
                        vec![]
                    };

                    if !axes_i64.is_empty() {
                        use crate::shape_inference::infer_unsqueeze_shape;
                        let axes_u32: Vec<u32> = axes_i64
                            .iter()
                            .map(|&a| {
                                if a < 0 {
                                    ((input_shape.len() as i64 + 1) + a) as u32
                                } else {
                                    a as u32
                                }
                            })
                            .collect();

                        if let Ok(out_shape) = infer_unsqueeze_shape(input_shape, &axes_u32) {
                            shape_overrides.insert(output_id, out_shape.clone());
                            operand_shapes.insert(output_id, out_shape);

                            // Preserve input type for unsqueeze output
                            if let Some(input_operand) = graph.operand(input_id) {
                                type_overrides
                                    .insert(output_id, input_operand.descriptor.data_type);
                            }
                        }
                    }
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
                debug_print!(
                    "[DEBUG] Missing constant operand {} while building initializers",
                    id
                );
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
                        name: operand_name(graph, *id),
                        data_type: dtype as i32,
                        dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                        int64_data: vec![0i64; element_count],
                        ..Default::default()
                    },
                    DataType::Int32 => TensorProto {
                        name: operand_name(graph, *id),
                        data_type: dtype as i32,
                        dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                        int32_data: vec![0i32; element_count],
                        ..Default::default()
                    },
                    DataType::Float32 => TensorProto {
                        name: operand_name(graph, *id),
                        data_type: dtype as i32,
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
                            name: operand_name(graph, *id),
                            data_type: dtype as i32,
                            dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                            raw_data: zero_data,
                            ..Default::default()
                        }
                    }
                }
            } else {
                // Normal case: use provided data
                TensorProto {
                    name: operand_name(graph, *id),
                    data_type: Self::data_type_code(operand.descriptor.data_type) as i32,
                    dims: operand.descriptor.shape.iter().map(|d| *d as i64).collect(),
                    raw_data: data.data.clone(),
                    ..Default::default()
                }
            };

            initializers.push(tensor_proto);
        }

        // Generate nodes, inserting Cast nodes for logic operations
        let mut nodes = Vec::new();
        let mut cast_counter = 0;

        if let Some(opd) = graph.operands.get(34) {
            debug_print!(
                "[DEBUG] Operand 34 name={:?} dtype={:?} shape={:?}",
                opd.name,
                opd.descriptor.data_type,
                opd.descriptor.shape
            );
        } else {
            debug_print!("[DEBUG] Operand 34 not present in operands table");
        }

        for (idx, op) in graph.operations.iter().enumerate() {
            // Debug guard: ensure all input operands exist
            for &input_id in &op.input_operands {
                // Resolve remapping first
                let resolved_id = operand_remapping
                    .get(&input_id)
                    .copied()
                    .unwrap_or(input_id);
                if graph.operand(resolved_id).is_none() {
                    let input_name = graph
                        .operands
                        .get(input_id as usize)
                        .and_then(|opd| opd.name.clone())
                        .unwrap_or_else(|| format!("<unnamed:{}>", input_id));
                    debug_print!(
                        "[DEBUG] Missing operand id {} name '{}' for op {} ({}) at index {}. Inputs: {:?}",
                        input_id,
                        input_name,
                        op.label.clone().unwrap_or_else(|| op.op_type.clone()),
                        op.op_type,
                        idx,
                        op.input_operands
                    );
                    debug_print!(
                        "[DEBUG] operands.len()={} valid ids 0..{}",
                        graph.operands.len(),
                        graph.operands.len().saturating_sub(1)
                    );
                    debug_print!(
                        "[DEBUG] Failing op detail: idx={} type={} label={:?} inputs={:?}",
                        idx,
                        op.op_type,
                        op.label,
                        op.input_operands
                    );
                    return Err(Self::invalid_operand(
                        "op input lookup",
                        input_id,
                        Some((op, idx)),
                    ));
                }
            }

            // Replace concat operations with empty KV inputs with Identity nodes
            // For past_sequence_length=0, concat(empty, new) = new, so we just copy the input
            if op.op_type.eq_ignore_ascii_case("concat") {
                let has_skipped_input = op
                    .input_operands
                    .iter()
                    .any(|&id| skipped_inputs.contains(&id));

                // Debug: print all concat ops
                debug_print!(
                    "[ONNX CONVERTER] Concat op idx={} has {} inputs, has_skipped={}",
                    idx,
                    op.input_operands.len(),
                    has_skipped_input
                );

                if has_skipped_input {
                    // Find the non-skipped input (the actual new KV)
                    let remaining_inputs: Vec<_> = op
                        .input_operands
                        .iter()
                        .filter(|&&id| !skipped_inputs.contains(&id))
                        .copied()
                        .collect();

                    debug_print!(
                        "[ONNX CONVERTER]   Remaining inputs: {}",
                        remaining_inputs.len()
                    );

                    if remaining_inputs.len() == 1 {
                        // Perfect case: one skipped (empty past), one remaining (new KV)
                        let output_id = op.output_operand.ok_or_else(|| {
                            Self::invalid_operand("concat output", idx as u32, Some((op, idx)))
                        })?;

                        let input_id = remaining_inputs[0];
                        let resolved_input_id = operand_remapping
                            .get(&input_id)
                            .copied()
                            .unwrap_or(input_id);
                        let input_name = operand_name(graph, resolved_input_id);
                        let output_name = operand_name(graph, output_id);

                        debug_print!(
                            "[ONNX CONVERTER]   Creating Identity node: {} -> {}",
                            input_name,
                            output_name
                        );

                        // Create an Identity node: output = Identity(input)
                        let identity_node = NodeProto {
                            input: vec![input_name],
                            output: vec![output_name.clone()],
                            name: format!("identity_{}", output_id),
                            op_type: "Identity".to_string(),
                            ..Default::default()
                        };

                        nodes.push(identity_node);

                        // Track shape for the output
                        operand_shapes.insert(
                            output_id,
                            graph
                                .operand(resolved_input_id)
                                .map(|opd| opd.descriptor.shape.clone())
                                .unwrap_or_default(),
                        );

                        continue;
                    }
                }
            }

            // WebNN constant() op: encode as initializer, not a node
            if op.op_type.eq_ignore_ascii_case("constant") {
                let output_id = op.output_operand.ok_or_else(|| {
                    Self::invalid_operand("constant output", idx as u32, Some((op, idx)))
                })?;

                // Get constant data: try 'init' attribute first (reference to named constant),
                // then fall back to 'data' attribute (inline base64)
                let data = if let Some(init_ref) =
                    op.attributes.get("init").and_then(|v| v.as_str())
                {
                    // 'init' attribute references a named constant declaration (e.g., "$_name")
                    // The operand name in the graph keeps the '$' prefix
                    debug_print!("[DEBUG] Constant operation with 'init' reference:");
                    debug_print!("  Operation index: {}", idx);
                    debug_print!("  Output operand: {}", output_id);
                    debug_print!("  Init reference: {}", init_ref);
                    debug_print!("  Looking for constant operand named: {}", init_ref);

                    // Find the constant operand with matching name
                    // Note: Named constants from the constants{} section have OperandKind::Constant
                    let const_operand_id = graph
                        .operands
                        .iter()
                        .enumerate()
                        .find(|(_, op)| {
                            op.name.as_deref() == Some(init_ref) && op.kind == OperandKind::Constant
                        })
                        .map(|(id, _)| id as u32)
                        .ok_or_else(|| {
                            debug_print!("[DEBUG] Failed to find constant operand:");
                            debug_print!("  All constant operands:");
                            for (id, op) in graph.operands.iter().enumerate() {
                                if op.kind == OperandKind::Constant {
                                    debug_print!("    ID {}: name={:?}", id, op.name);
                                }
                            }
                            GraphError::ConversionFailed {
                                format: "onnx".to_string(),
                                reason: format!(
                                    "Constant op init='{}' references unknown constant operand",
                                    init_ref
                                ),
                            }
                        })?;

                    // Look up the constant data
                    graph
                        .constant_operand_ids_to_handles
                        .get(&const_operand_id)
                        .map(|const_data| const_data.data.clone())
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!(
                                "Constant op init='{}' found operand {} but no data in constant_operand_ids_to_handles",
                                init_ref, const_operand_id
                            ),
                        })?
                } else if let Some(data_b64) = op.attributes.get("data").and_then(|v| v.as_str()) {
                    // 'data' attribute contains inline base64-encoded data
                    STANDARD
                        .decode(data_b64)
                        .map_err(|e| GraphError::ConversionFailed {
                            format: "onnx".to_string(),
                            reason: format!("Constant op base64 decode failed: {}", e),
                        })?
                } else {
                    // Neither 'init' nor 'data' found
                    debug_print!("[DEBUG] Constant operation missing 'data' or 'init' attribute:");
                    debug_print!("  Operation index: {}", idx);
                    debug_print!("  Output operand: {}", output_id);
                    if let Some(obj) = op.attributes.as_object() {
                        debug_print!(
                            "  Available attributes: {:?}",
                            obj.keys().collect::<Vec<_>>()
                        );
                    } else {
                        debug_print!("  Attributes: {:?}", op.attributes);
                    }
                    debug_print!("  Label: {:?}", op.label);
                    return Err(GraphError::ConversionFailed {
                        format: "onnx".to_string(),
                        reason: "Constant op missing both 'data' and 'init' attributes".to_string(),
                    });
                };

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
                    name: operand_name(graph, output_id),
                    data_type: Self::data_type_code(data_type) as i32,
                    dims: shape,
                    raw_data: data,
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
                    // Resolve any remapping first (for skipped concat outputs used as inputs)
                    let resolved_id = operand_remapping
                        .get(input_id)
                        .copied()
                        .unwrap_or(*input_id);

                    let operand = graph.operand(resolved_id).ok_or_else(|| {
                        debug_print!(
                            "[DEBUG] Missing operand {} in concat at op idx {}",
                            resolved_id,
                            idx
                        );
                        Self::invalid_operand("concat input lookup", resolved_id, Some((op, idx)))
                    })?;
                    // Use remapped operand name if this input was a skipped concat output
                    let input_name = {
                        let resolved_id = operand_remapping
                            .get(input_id)
                            .copied()
                            .unwrap_or(*input_id);
                        operand_name(graph, resolved_id)
                    };

                    // Use tracked shape if available, otherwise fall back to descriptor
                    // Check both original and resolved IDs for shape tracking
                    let input_shape = operand_shapes
                        .get(&resolved_id)
                        .or_else(|| operand_shapes.get(input_id))
                        .cloned()
                        .unwrap_or_else(|| operand.descriptor.shape.clone());

                    if input_shape.is_empty() {
                        if let Some(data) = graph.constant_operand_ids_to_handles.get(&resolved_id)
                        {
                            // Expand scalar constant to shape [1]
                            let expanded_name = format!("{}_scalar{}_expanded", op_name, input_idx);
                            initializers.push(TensorProto {
                                name: expanded_name.clone(),
                                data_type: Self::data_type_code(operand.descriptor.data_type)
                                    as i32,
                                dims: vec![1],
                                raw_data: data.data.clone(),
                                ..Default::default()
                            });
                            inputs.push(expanded_name);
                            continue;
                        } else {
                            // Try cloning an existing initializer with the same name
                            let expanded_name = format!("{}_scalar{}_expanded", op_name, input_idx);
                            if let Some(cloned) = initializers
                                .iter()
                                .find(|t| t.name == input_name)
                                .map(|orig| {
                                    let mut cloned = orig.clone();
                                    cloned.name = expanded_name.clone();
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
                            // Opset 14: axes must be provided as input tensor
                            let unsq_name = format!("{}_scalar{}_unsq", op_name, input_idx);
                            let axes_name = format!("{}_unsqueeze_{}_axes", op_name, input_idx);

                            // Create axes initializer
                            initializers.push(TensorProto {
                                name: axes_name.clone(),
                                data_type: ProtoDataType::Int64 as i32,
                                dims: vec![1],
                                int64_data: vec![0],
                                ..Default::default()
                            });

                            nodes.push(NodeProto {
                                input: vec![input_name.clone(), axes_name],
                                output: vec![unsq_name.clone()],
                                name: format!("{}_unsqueeze_{}", op_name, input_idx),
                                op_type: "Unsqueeze".to_string(),
                                attribute: vec![], // No attributes in opset 14
                                ..Default::default()
                            });
                            inputs.push(unsq_name);
                            continue;
                        }
                    }

                    inputs.push(input_name);
                }

                let attributes = Self::create_operation_attributes(op);

                // Debug: trace concat operations to find rank mismatches
                if op_name.contains("concat") {
                    debug_print!(
                        "[RUST DEBUG] Concat {} has {} inputs:",
                        op_name,
                        op.input_operands.len()
                    );
                    for (input_idx, input_id) in op.input_operands.iter().enumerate() {
                        if let Some(operand) = graph.operand(*input_id) {
                            debug_print!(
                                "  Input {}: operand_{} shape={:?} rank={}",
                                input_idx,
                                input_id,
                                operand.descriptor.shape,
                                operand.descriptor.shape.len()
                            );
                        }
                    }
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
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
                    name: op_name.clone(),
                    op_type: Self::onnx_op_type(&op.op_type),
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
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
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
            } else if op.op_type.eq_ignore_ascii_case("triangular") {
                // Triangular operation: Cast integer inputs to float32
                let input_id = op.input_operands[0];
                let input_operand = graph.operand(input_id).ok_or_else(|| {
                    Self::invalid_operand("triangular input", input_id, Some((op, idx)))
                })?;

                let needs_cast = matches!(
                    input_operand.descriptor.data_type,
                    DataType::Int32 | DataType::Int64 | DataType::Uint32 | DataType::Uint8
                );

                let input_name = if needs_cast {
                    let cast_output = format!("{}_input_float", op_name);
                    debug_print!(
                        "[FIX] Triangular op {} has {:?} input, casting to Float32",
                        op_name,
                        input_operand.descriptor.data_type
                    );
                    nodes.push(Self::create_cast_node(
                        &format!("{}_pre_cast", op_name),
                        operand_name(graph, input_id),
                        cast_output.clone(),
                        ProtoDataType::Float,
                    ));
                    cast_output
                } else {
                    operand_name(graph, input_id)
                };

                let attributes = Self::create_operation_attributes(op);

                nodes.push(NodeProto {
                    input: vec![input_name],
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name.clone(),
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });
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
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
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
                        name: format!("{}_identity", op_name),
                        op_type: "Identity".to_string(),
                        attribute: vec![],
                        ..Default::default()
                    });
                    continue;
                }

                let repeats_name = format!("{}_repeats", op_name);
                initializers.push(TensorProto {
                    name: repeats_name.clone(),
                    data_type: ProtoDataType::Int64 as i32,
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
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
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
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
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
                            name: shape_name,
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![shape_values.len() as i64], // 1D tensor
                            int64_data: shape_values.clone(),
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
                        name: shape_name,
                        data_type: ProtoDataType::Int64 as i32,
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
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: vec![], // No attributes for Reshape
                    ..Default::default()
                });
            } else if op.op_type == "expand" {
                // WebNN expand has two variants:
                // 1. With 'axes' - adds dimensions (maps to ONNX Unsqueeze)
                // 2. With 'newShape' - expands shape (maps to ONNX Expand or Reshape)

                debug_print!("[DEBUG] Processing WebNN expand operation:");
                debug_print!("  Op name: {}", op_name);
                if let Some(obj) = op.attributes.as_object() {
                    debug_print!("  Attributes: {:?}", obj.keys().collect::<Vec<_>>());
                }

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
                        name: axes_name,
                        data_type: ProtoDataType::Int64 as i32,
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
                        name: op_name.clone(),
                        op_type: "Unsqueeze".to_string(),
                        attribute: vec![], // No attributes for Unsqueeze in opset 13+
                        ..Default::default()
                    });
                } else if let Some(new_shape) =
                    op.attributes.get("newShape").and_then(|v| v.as_array())
                {
                    // WebNN expand with newShape can be either:
                    // 1. ONNX Expand (broadcasting-compatible shapes)
                    // 2. ONNX Reshape (arbitrary shape changes)

                    let mut inputs: Vec<String> = op
                        .input_operands
                        .iter()
                        .map(|id| operand_name(graph, *id))
                        .collect();

                    let shape_values: Vec<i64> = new_shape
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as i64))
                        .collect();

                    // Get input operand shape to determine if this is broadcasting or reshaping
                    let input_id = op.input_operands[0];
                    // Use tracked shape if available, otherwise fall back to descriptor
                    let input_shape = operand_shapes.get(&input_id).cloned().unwrap_or_else(|| {
                        graph.operands[input_id as usize].descriptor.shape.clone()
                    });

                    // Check if shapes are broadcasting-compatible (ONNX Expand rules):
                    // - Align shapes from the right
                    // - Each dimension pair must be equal or one must be 1
                    // - SPECIAL CASE: Scalars (rank 0) from constants should use Reshape, not Expand
                    let is_broadcast_compatible = {
                        let mut compatible = true;
                        let input_rank = input_shape.len();
                        let target_rank = shape_values.len();

                        debug_print!("[DEBUG] Expand operation:");
                        debug_print!("  Op name: {}", op_name);
                        debug_print!("  Input operand ID: {}", input_id);
                        debug_print!("  Input shape: {:?} (rank={})", input_shape, input_rank);
                        debug_print!("  Target shape: {:?} (rank={})", shape_values, target_rank);

                        // Only apply scalar handling to actual constant operands
                        // Runtime computed values may have different shapes than static descriptors
                        let is_constant = graph
                            .constant_operand_ids_to_handles
                            .contains_key(&input_id);

                        // Scalars (rank 0) need special handling regardless of whether they're constants:
                        // Reshape to target rank with all 1s, then expand will broadcast properly
                        if input_rank == 0 {
                            debug_print!(
                                "  Scalar input (constant={}) - will reshape to match target rank with all 1s",
                                is_constant
                            );

                            // Step 1: Reshape scalar to [1,1,...,1] with same rank as target
                            let reshape_intermediate =
                                format!("{}_scalar_to_rank{}", op_name, target_rank);
                            let reshape_shape_name = format!("{}_reshape_shape", op_name);

                            // Create shape [1,1,...,1] with target_rank dimensions
                            let intermediate_shape = vec![1i64; target_rank];

                            initializers.push(TensorProto {
                                name: reshape_shape_name.clone(),
                                data_type: ProtoDataType::Int64 as i32,
                                dims: vec![target_rank as i64],
                                int64_data: intermediate_shape,
                                ..Default::default()
                            });

                            nodes.push(NodeProto {
                                input: vec![inputs[0].clone(), reshape_shape_name],
                                output: vec![reshape_intermediate.clone()],
                                name: format!("{}_scalar_reshape", op_name),
                                op_type: "Reshape".to_string(),
                                attribute: vec![],
                                ..Default::default()
                            });

                            // Step 2: Update input for subsequent Expand to use reshaped tensor
                            inputs[0] = reshape_intermediate;

                            // Now it's compatible for expand (from [1,1,...,1] to target shape)
                            compatible = true;
                        } else {
                            // Align from right and check each dimension
                            for i in 0..input_rank.min(target_rank) {
                                let input_dim = input_shape[input_rank - 1 - i];
                                let target_dim = shape_values[target_rank - 1 - i] as u32;

                                // Dimensions are compatible if they're equal or either is 1
                                if input_dim != target_dim && input_dim != 1 && target_dim != 1 {
                                    debug_print!(
                                        "  Incompatible at dim {}: {} vs {}",
                                        i,
                                        input_dim,
                                        target_dim
                                    );
                                    compatible = false;
                                    break;
                                }
                            }
                        }

                        debug_print!("  Broadcasting compatible: {}", compatible);
                        compatible
                    };

                    // Use Expand for broadcasting, Reshape for arbitrary shape changes
                    let op_type = if is_broadcast_compatible {
                        "Expand"
                    } else {
                        debug_print!(
                            "[FIX] Using Reshape instead of Expand for {} -> {:?}",
                            op_name,
                            shape_values
                        );
                        "Reshape"
                    };

                    let shape_name = format!("{}_shape", op_name);
                    inputs.push(shape_name.clone());

                    // Update operand_shapes with the output shape before moving shape_values
                    if let Some(output_id) = op.output_operand {
                        let output_shape: Vec<u32> =
                            shape_values.iter().map(|&v| v as u32).collect();
                        operand_shapes.insert(output_id, output_shape);
                    }

                    // Add shape as an initializer (constant tensor)
                    initializers.push(TensorProto {
                        name: shape_name,
                        data_type: ProtoDataType::Int64 as i32,
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
                        name: op_name,
                        op_type: op_type.to_string(),
                        attribute: vec![], // No attributes for Expand or Reshape
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
                        name: format!("{}_pre_cast", op_name),
                        op_type: "Cast".to_string(),
                        attribute: vec![AttributeProto {
                            name: "to".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: ProtoDataType::Float as i32 as i64,
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
                            name: axes_name,
                            data_type: ProtoDataType::Int64 as i32,
                            dims: vec![axes_i64.len() as i64],
                            int64_data: axes_i64,
                            ..Default::default()
                        });
                    } else {
                        // Add axes as an attribute (for operations that don't support axes as input in opset 13)
                        attributes.push(AttributeProto {
                            name: "axes".to_string(),
                            r#type: AttributeType::Ints as i32,
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
                        name: "keepdims".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: if keep_dims { 1 } else { 0 },
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
                    name: op_name.clone(),
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: attributes,
                    ..Default::default()
                });

                // Cast back to original type if needed
                if needs_cast {
                    let original_type = Self::data_type_code(input_operand.descriptor.data_type);
                    nodes.push(NodeProto {
                        input: vec![reduce_output_name],
                        output: vec![final_output_name],
                        name: format!("{}_post_cast", op_name),
                        op_type: "Cast".to_string(),
                        attribute: vec![AttributeProto {
                            name: "to".to_string(),
                            r#type: AttributeType::Int as i32,
                            i: original_type as i32 as i64,
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

                // Special case: 0D tensor (scalar) cannot be sliced
                // Use Identity node instead (ONNX Runtime doesn't support slicing scalars)
                // A scalar has no axes, so any slice operation should just return the scalar unchanged
                if is_0d {
                    nodes.push(NodeProto {
                        input: vec![inputs[0].clone()],
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: op_name,
                        op_type: "Identity".to_string(),
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
                    name: starts_name,
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![starts_len as i64],
                    int64_data: starts,
                    ..Default::default()
                });

                // Add ends as initializer
                let ends_name = format!("{}_ends", op_name);
                inputs.push(ends_name.clone());
                initializers.push(TensorProto {
                    name: ends_name,
                    data_type: ProtoDataType::Int64 as i32,
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
                    name: axes_name,
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![axes_data.len() as i64],
                    int64_data: axes_data,
                    ..Default::default()
                });

                // Add steps as initializer (if provided)
                if let Some(steps_data) = steps {
                    let steps_name = format!("{}_steps", op_name);
                    inputs.push(steps_name.clone());
                    initializers.push(TensorProto {
                        name: steps_name,
                        data_type: ProtoDataType::Int64 as i32,
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
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
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
                        name: "axis".to_string(),
                        r#type: AttributeType::Int as i32,
                        i: axis as i64,
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
                            name: splits_name.clone(),
                            data_type: ProtoDataType::Int64 as i32,
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
                    name: op_name,
                    op_type: "Split".to_string(),
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
                    name: clamp_min_name.clone(),
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![],
                    int64_data: vec![-dim_size],
                    ..Default::default()
                });

                initializers.push(TensorProto {
                    name: clamp_max_name.clone(),
                    data_type: ProtoDataType::Int64 as i32,
                    dims: vec![],
                    int64_data: vec![dim_size - 1],
                    ..Default::default()
                });

                // Insert Clip node (Clamp was deprecated in favor of Clip in opset 11+)
                nodes.push(NodeProto {
                    input: vec![indices_after_cast, clamp_min_name, clamp_max_name],
                    output: vec![clamped_indices_name.clone()],
                    name: format!("{}_clip_indices", op_name),
                    op_type: "Clip".to_string(),
                    ..Default::default()
                });

                // ONNX Gather handles indices shape correctly, no reshape needed
                // The output shape is automatically: data.shape[0:axis] + indices.shape + data.shape[axis+1:]
                let final_indices = clamped_indices_name;

                inputs.push(final_indices);

                // Create Gather node
                let attributes = Self::create_operation_attributes(op);
                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: "Gather".to_string(),
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
                        name: format!("{}_transpose_input", op_name),
                        op_type: "Transpose".to_string(),
                        attribute: vec![AttributeProto {
                            name: "perm".to_string(),
                            r#type: AttributeType::Ints as i32,
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
                        name: format!("{}_transpose_filter", op_name),
                        op_type: "Transpose".to_string(),
                        attribute: vec![AttributeProto {
                            name: "perm".to_string(),
                            r#type: AttributeType::Ints as i32,
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
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
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
                                name: op_name,
                                op_type: "Add".to_string(),
                                ..Default::default()
                            });
                        } else {
                            // output = input - input = 0
                            let input_name = operand_name(graph, input_id);
                            nodes.push(NodeProto {
                                input: vec![input_name.clone(), input_name],
                                output: vec![output_name],
                                name: op_name,
                                op_type: "Sub".to_string(),
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
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
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
                    name: format!("{}_add", op_name),
                    op_type: "Add".to_string(),
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
                    name: format!("{}_clip", op_name),
                    op_type: "Clip".to_string(),
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
                    name: format!("{}_div", op_name),
                    op_type: "Div".to_string(),
                    ..Default::default()
                });

                // Step 4: Multiply by x
                nodes.push(NodeProto {
                    input: vec![input_name, div_output],
                    output: vec![output_name],
                    name: format!("{}_mul", op_name),
                    op_type: "Mul".to_string(),
                    ..Default::default()
                });
            } else if op.op_type == "unsqueeze" || op.op_type == "squeeze" {
                // Unsqueeze/Squeeze operations - in ONNX opset 13+, axes must be provided as input tensor
                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| operand_name(graph, *id))
                    .collect();

                // Get axes from attributes and create as input tensor
                if let Some(axes_i64) = Self::parse_i64_array(op, "axes") {
                    let axes_name = format!("{}_axes", op_name);
                    inputs.push(axes_name.clone());

                    initializers.push(TensorProto {
                        name: axes_name,
                        data_type: ProtoDataType::Int64 as i32,
                        dims: vec![axes_i64.len() as i64],
                        int64_data: axes_i64,
                        ..Default::default()
                    });
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: Self::onnx_op_type(&op.op_type),
                    attribute: vec![], // No attributes for Unsqueeze/Squeeze in opset 13+
                    ..Default::default()
                });
            } else if op.op_type.eq_ignore_ascii_case("scatternd") {
                // ScatterND requires int64 indices - insert Cast if needed
                let mut inputs: Vec<String> = Vec::new();

                // Debug: print shapes of all inputs
                debug_print!("[SCATTERND DEBUG] Operation: {}", op_name);
                for (i, &input_id) in op.input_operands.iter().enumerate() {
                    let shape = operand_shapes.get(&input_id);
                    let desc_shape = graph.operand(input_id).map(|o| &o.descriptor.shape);
                    debug_print!(
                        "  Input {}: operand_id={}, tracked_shape={:?}, descriptor_shape={:?}",
                        i,
                        input_id,
                        shape,
                        desc_shape
                    );
                }

                // Input 0: data
                if let Some(&data_id) = op.input_operands.first() {
                    inputs.push(operand_name(graph, data_id));
                }

                // Input 1: indices - must be int64
                if let Some(&indices_id) = op.input_operands.get(1) {
                    if let Some(indices_operand) = graph.operand(indices_id) {
                        let indices_dtype = type_overrides
                            .get(&indices_id)
                            .copied()
                            .unwrap_or(indices_operand.descriptor.data_type);

                        if !matches!(indices_dtype, DataType::Int64) {
                            // Insert Cast node to convert indices to int64
                            let cast_output = format!("{}_indices_cast", op_name);
                            nodes.push(NodeProto {
                                input: vec![operand_name(graph, indices_id)],
                                output: vec![cast_output.clone()],
                                name: format!("{}_cast_indices", op_name),
                                op_type: "Cast".to_string(),
                                attribute: vec![AttributeProto {
                                    name: "to".to_string(),
                                    r#type: AttributeType::Int as i32,
                                    i: ProtoDataType::Int64 as i32 as i64,
                                    ..Default::default()
                                }],
                                ..Default::default()
                            });
                            inputs.push(cast_output);
                        } else {
                            inputs.push(operand_name(graph, indices_id));
                        }
                    } else {
                        inputs.push(operand_name(graph, indices_id));
                    }
                }

                // Input 2: updates
                if let Some(&updates_id) = op.input_operands.get(2) {
                    inputs.push(operand_name(graph, updates_id));
                }

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: op_name,
                    op_type: "ScatterND".to_string(),
                    attribute: vec![],
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
                        name: op_name.clone(),
                        op_type: Self::onnx_op_type(&op.op_type),
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
                            .map(|id| {
                                // Resolve remapping for skipped concat outputs
                                let resolved_id = operand_remapping.get(id).copied().unwrap_or(*id);
                                operand_name(graph, resolved_id)
                            })
                            .collect(),
                        output: vec![operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: op_name,
                        op_type: Self::onnx_op_type(&op.op_type),
                        attribute: attributes,
                        ..Default::default()
                    });
                }
            }
        }

        // Add value_info ONLY for operands where we have explicit shape/type tracking
        // This provides guidance to ONNX Runtime for operations we explicitly handle
        // (concat, unsqueeze, binary ops) while letting it infer shapes for others
        let mut seen_names = std::collections::HashSet::new();
        for vi in inputs_val.iter().chain(outputs_val.iter()) {
            if !vi.name.is_empty() {
                seen_names.insert(vi.name.clone());
            }
        }

        for (idx, operand) in graph.operands.iter().enumerate() {
            let operand_id = idx as u32;
            let name = operand
                .name
                .clone()
                .unwrap_or_else(|| format!("operand_{}", operand_id));

            if seen_names.contains(&name) {
                continue;
            }

            // Only add value_info if we have explicit shape tracking
            // ONNX Runtime can infer types but struggles with shapes, so shape guidance is critical
            if let Some(tracked_shape) = shape_overrides.get(&operand_id) {
                let mut desc = operand.descriptor.clone();
                desc.shape = tracked_shape.clone();
                // Use tracked type if available, otherwise use descriptor type
                if let Some(&tracked_type) = type_overrides.get(&operand_id) {
                    desc.data_type = tracked_type;
                }
                value_infos.push(value_info(&name, &desc));
            }
        }

        let graph_proto = GraphProto {
            name: "webnn_graph".to_string(),
            node: nodes,
            input: inputs_val,
            output: outputs_val,
            initializer: initializers,
            value_info: value_infos,
            ..Default::default()
        };

        let model = ModelProto {
            ir_version: 8, // IR version 8 = ONNX 1.10+ (supports opset 14-15)
            model_version: 1,
            producer_name: "rustnn".to_string(),
            producer_version: "0.1.0".to_string(),
            graph: Some(graph_proto),
            opset_import: vec![OperatorSetIdProto {
                version: 14,            // Opset 14 adds Trilu support
                domain: "".to_string(), // Empty string = default ONNX domain
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
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(crate::protos::onnx::type_proto::Value::TensorType(
                TensorTypeProto {
                    elem_type: OnnxConverter::data_type_code(desc.data_type) as i32,
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
                                denotation: String::new(),
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
