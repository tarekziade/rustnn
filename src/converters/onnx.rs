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
                    Self::operand_name(
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
                    Self::operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    ),
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

                // Get input operand data type - min/max must match this type
                let input_operand = graph.operand(op.input_operands[0]).ok_or_else(|| {
                    GraphError::InvalidConversionOperand {
                        operand: op.input_operands[0],
                    }
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
                    output: vec![Self::operand_name(
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
                    output: vec![Self::operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
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
                    output: vec![Self::operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: vec![], // No attributes for Expand
                    ..Default::default()
                });
            } else if op.op_type.starts_with("reduce") {
                // Reduction operations - in ONNX opset 13, only ReduceSum supports axes as input
                // In opset 18+, ReduceMean, ReduceProd, ReduceMax, ReduceMin also support axes as input
                // But we're using opset 13, so only ReduceSum gets axes as input
                let supports_axes_as_input = matches!(op.op_type.as_str(), "reduceSum");

                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| Self::operand_name(graph, *id))
                    .collect();

                let mut attributes = Vec::new();

                // Extract axes from attributes
                if let Some(axes) = op.attributes.get("axes").and_then(|v| v.as_array()) {
                    let axes_i64: Vec<i64> = axes
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as i64))
                        .collect();

                    if !axes_i64.is_empty() {
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

                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![Self::operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    )],
                    name: Some(op_name),
                    op_type: Some(Self::onnx_op_type(&op.op_type)),
                    attribute: attributes,
                    ..Default::default()
                });
            } else if op.op_type == "slice" {
                // Slice operation - ONNX requires starts, ends, axes, steps as input tensors
                // Special case: ONNX Runtime doesn't support slicing 0D tensors

                // Check if input is 0D (scalar)
                let input_operand_id = op.input_operands[0];
                let input_operand = graph.operand(input_operand_id).ok_or_else(|| {
                    GraphError::InvalidConversionOperand {
                        operand: input_operand_id,
                    }
                })?;
                let is_0d = input_operand.descriptor.shape.is_empty();

                let mut inputs: Vec<String> = op
                    .input_operands
                    .iter()
                    .map(|id| Self::operand_name(graph, *id))
                    .collect();

                // Extract starts, sizes, axes, and steps from attributes
                let starts = op
                    .attributes
                    .get("starts")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();

                let sizes = op
                    .attributes
                    .get("sizes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                    .unwrap_or_default();

                // Special case: 0D tensor with empty starts/sizes is a no-op
                // Use Identity node instead of Slice (ONNX Runtime doesn't support slicing scalars)
                if is_0d && starts.is_empty() && sizes.is_empty() {
                    nodes.push(NodeProto {
                        input: vec![inputs[0].clone()],
                        output: vec![Self::operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        )],
                        name: Some(op_name),
                        op_type: Some("Identity".to_string()),
                        ..Default::default()
                    });
                    continue; // Skip the rest of the slice handling
                }

                let axes = op
                    .attributes
                    .get("axes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>());

                let steps = op
                    .attributes
                    .get("strides")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>());

                // Convert sizes to ends: ends[i] = starts[i] + sizes[i]
                let ends: Vec<i64> = starts
                    .iter()
                    .zip(sizes.iter())
                    .map(|(start, size)| start + size)
                    .collect();

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
                    output: vec![Self::operand_name(
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
                    .map(|id| Self::operand_name(graph, *id))
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
                    .map(|id| Self::operand_name(graph, *id))
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
                inputs.push(Self::operand_name(graph, data_operand_id));

                // Get axis parameter (default is 0)
                let axis = op
                    .attributes
                    .get("axis")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as usize;

                // Get input shape and dimension size at axis
                let data_operand = graph.operand(data_operand_id).ok_or_else(|| {
                    GraphError::InvalidConversionOperand {
                        operand: data_operand_id,
                    }
                })?;
                let dim_size = data_operand.descriptor.shape[axis] as i64;

                // Second input: indices tensor - may need casting and clamping
                let indices_id = op.input_operands[1];
                let indices_name = Self::operand_name(graph, indices_id);
                let indices_operand = graph.operand(indices_id).ok_or_else(|| {
                    GraphError::InvalidConversionOperand {
                        operand: indices_id,
                    }
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

                inputs.push(clamped_indices_name);

                // Create Gather node
                let attributes = Self::create_operation_attributes(op);
                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![Self::operand_name(
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
                let input_name = Self::operand_name(graph, op.input_operands[0]);
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
                let filter_name = Self::operand_name(graph, op.input_operands[1]);
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
                    conv_inputs.push(Self::operand_name(graph, op.input_operands[2]));
                }

                // Create Conv/ConvTranspose node
                let attributes = Self::create_operation_attributes(op);
                nodes.push(NodeProto {
                    input: conv_inputs,
                    output: vec![Self::operand_name(
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
                let input_operand = graph
                    .operand(input_id)
                    .ok_or_else(|| GraphError::InvalidConversionOperand { operand: input_id })?;
                let input_data_type = Self::data_type_code(input_operand.descriptor.data_type);

                let mut inputs: Vec<String> = vec![Self::operand_name(graph, input_id)];

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
                        .unwrap_or_else(|| vec![]);

                    // Handle empty axes or 0D tensor with axes=[-1] case
                    // When axes are empty or when input is 0D scalar, no normalization occurs
                    // This matches Chromium's implementation
                    let is_0d_with_default_axes =
                        input_operand.descriptor.shape.is_empty() && axes == vec![-1];
                    if axes.is_empty() || is_0d_with_default_axes {
                        let output_name = Self::operand_name(
                            graph,
                            op.output_operand.expect("Single-output operation expected"),
                        );

                        if has_bias && op.input_operands.len() > 2 {
                            // output = bias + 0
                            let bias_name = Self::operand_name(graph, op.input_operands[2]);
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
                            let input_name = Self::operand_name(graph, input_id);
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
                    // For layer norm, scale/bias shape depends on normalized axes
                    let axes = op
                        .attributes
                        .get("axes")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<i64>>())
                        .unwrap_or_else(|| vec![-1]);

                    // Calculate size of normalized dimensions
                    let mut size = 1i64;
                    for &axis in &axes {
                        let actual_axis = if axis < 0 {
                            (input_operand.descriptor.shape.len() as i64 + axis) as usize
                        } else {
                            axis as usize
                        };
                        if actual_axis < input_operand.descriptor.shape.len() {
                            size *= input_operand.descriptor.shape[actual_axis] as i64;
                        }
                    }
                    vec![size]
                } else if op.op_type == "batchNormalization"
                    || op.op_type == "instanceNormalization"
                {
                    // For batch/instance norm, scale/bias shape is [channels]
                    // Channels is typically dimension 1 for NCHW layout
                    let channels = if input_operand.descriptor.shape.len() > 1 {
                        input_operand.descriptor.shape[1] as i64
                    } else {
                        1
                    };
                    vec![channels]
                } else {
                    vec![1]
                };

                // Add scale input (from operand or create default with 1.0)
                if has_scale && op.input_operands.len() > 1 {
                    inputs.push(Self::operand_name(graph, op.input_operands[1]));
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
                    inputs.push(Self::operand_name(graph, op.input_operands[2]));
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

                // For batch normalization, add mean and variance (required inputs 4 and 5)
                if op.op_type == "batchNormalization" {
                    if op.input_operands.len() > 3 {
                        inputs.push(Self::operand_name(graph, op.input_operands[3])); // mean
                    }
                    if op.input_operands.len() > 4 {
                        inputs.push(Self::operand_name(graph, op.input_operands[4])); // variance
                    }
                }

                let attributes = Self::create_operation_attributes(op);
                nodes.push(NodeProto {
                    input: inputs,
                    output: vec![Self::operand_name(
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
                let input_name = Self::operand_name(graph, op.input_operands[0]);
                let output_name = Self::operand_name(
                    graph,
                    op.output_operand.expect("Single-output operation expected"),
                );

                // Get input data type for scalar initializers
                let input_operand = graph.operand(op.input_operands[0]).ok_or_else(|| {
                    GraphError::InvalidConversionOperand {
                        operand: op.input_operands[0],
                    }
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
                        matches!(
                            operand.descriptor.data_type,
                            DataType::Int8 | DataType::Uint8 | DataType::Int32
                        )
                    } else {
                        false
                    }
                });

                if requires_float && has_integer_inputs {
                    // Cast inputs to float32, execute operation, cast output back
                    let mut cast_inputs = Vec::new();
                    let mut original_types = Vec::new();

                    for &input_id in &op.input_operands {
                        let input_name = Self::operand_name(graph, input_id);
                        let input_operand = graph.operand(input_id).ok_or_else(|| {
                            GraphError::InvalidConversionOperand { operand: input_id }
                        })?;

                        original_types.push(input_operand.descriptor.data_type);

                        if matches!(
                            input_operand.descriptor.data_type,
                            DataType::Int8 | DataType::Uint8 | DataType::Int32
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
                    let attributes = Self::create_operation_attributes(op);

                    nodes.push(NodeProto {
                        input: cast_inputs,
                        output: vec![float_output_name.clone()],
                        name: Some(op_name.clone()),
                        op_type: Some(Self::onnx_op_type(&op.op_type)),
                        attribute: attributes,
                        ..Default::default()
                    });

                    // Cast output back to original type (use first input's type as reference)
                    let output_type = original_types[0];
                    let final_output_name = Self::operand_name(
                        graph,
                        op.output_operand.expect("Single-output operation expected"),
                    );

                    nodes.push(Self::create_cast_node(
                        &format!("{}_cast_output", op_name),
                        float_output_name,
                        final_output_name,
                        Self::data_type_code(output_type),
                    ));
                } else {
                    // Regular operation - no Cast nodes needed
                    let attributes = Self::create_operation_attributes(op);

                    nodes.push(NodeProto {
                        input: op
                            .input_operands
                            .iter()
                            .map(|id| Self::operand_name(graph, *id))
                            .collect(),
                        output: vec![Self::operand_name(
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
