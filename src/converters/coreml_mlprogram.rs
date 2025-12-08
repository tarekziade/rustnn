/// CoreML MLProgram (MIL) converter
///
/// This converter generates CoreML MLProgram models using the Model Intermediate Language (MIL).
/// MLProgram is the modern CoreML format (spec v7+, iOS 15+, macOS 12+) that supports:
/// - Flexible MIL operations
/// - Quantization operations
/// - Better optimization
///
/// This replaces the legacy NeuralNetwork format.
use crate::error::GraphError;
use crate::graph::{DataType, GraphInfo, Operation};
use crate::protos::coreml::mil_spec::{
    Argument, Block, Dimension, Function, NamedValueType, Operation as MilOperation, Program,
    TensorType, ValueType, argument::binding::Binding, dimension,
};
use crate::protos::coreml::specification::Model;
use prost::Message;
use std::collections::HashMap;

/// MIL operation type names (matching Chromium's implementation)
mod mil_ops {
    // Binary operations
    pub const ADD: &str = "add";
    pub const SUB: &str = "sub";
    pub const MUL: &str = "mul";
    pub const DIV: &str = "real_div";
    pub const POW: &str = "pow";
    pub const MATMUL: &str = "matmul";

    // Activation functions
    pub const RELU: &str = "relu";
    pub const SIGMOID: &str = "sigmoid";
    pub const TANH: &str = "tanh";
    pub const SOFTMAX: &str = "softmax";

    // Convolution and pooling
    pub const CONV: &str = "conv";
    pub const CONV_TRANSPOSE: &str = "conv_transpose";
    pub const AVG_POOL: &str = "avg_pool";
    pub const MAX_POOL: &str = "max_pool";
    pub const GLOBAL_AVG_POOL: &str = "reduce_mean"; // Global pooling via reduction
    pub const GLOBAL_MAX_POOL: &str = "reduce_max"; // Global pooling via reduction

    // Normalization
    pub const BATCH_NORM: &str = "batch_norm";
    pub const INSTANCE_NORM: &str = "instance_norm";
    pub const LAYER_NORM: &str = "layer_norm";

    // Reduction operations
    pub const REDUCE_SUM: &str = "reduce_sum";
    pub const REDUCE_MEAN: &str = "reduce_mean";
    pub const REDUCE_MAX: &str = "reduce_max";
    pub const REDUCE_MIN: &str = "reduce_min";
    pub const REDUCE_PROD: &str = "reduce_prod";
    pub const REDUCE_L1: &str = "reduce_l1_norm";
    pub const REDUCE_L2: &str = "reduce_l2_norm";
    pub const REDUCE_LOG_SUM: &str = "reduce_log_sum";
    pub const REDUCE_LOG_SUM_EXP: &str = "reduce_log_sum_exp";
    pub const REDUCE_SUM_SQUARE: &str = "reduce_sum_square";

    // Element-wise unary operations
    pub const ABS: &str = "abs";
    pub const CEIL: &str = "ceil";
    pub const FLOOR: &str = "floor";
    pub const ROUND: &str = "round";
    pub const NEG: &str = "mul"; // Multiply by -1
    pub const IDENTITY: &str = "identity";
    pub const EXP: &str = "exp";
    pub const LOG: &str = "log";
    pub const SQRT: &str = "sqrt";
    pub const SIGN: &str = "sign";
    pub const SIN: &str = "sin";
    pub const COS: &str = "cos";
    pub const TAN: &str = "tan";
    pub const ASIN: &str = "asin";
    pub const ACOS: &str = "acos";
    pub const ATAN: &str = "atan";
    pub const SINH: &str = "sinh";
    pub const COSH: &str = "cosh";
    pub const ASINH: &str = "asinh";
    pub const ACOSH: &str = "acosh";
    pub const ATANH: &str = "atanh";
    pub const ERF: &str = "erf";
    pub const RECIPROCAL: &str = "inverse";

    // Logic operations
    pub const EQUAL: &str = "equal";
    pub const GREATER: &str = "greater";
    pub const GREATER_EQUAL: &str = "greater_equal";
    pub const LESS: &str = "less";
    pub const LESS_EQUAL: &str = "less_equal";
    pub const LOGICAL_NOT: &str = "logical_not";
    pub const LOGICAL_AND: &str = "logical_and";
    pub const LOGICAL_OR: &str = "logical_or";
    pub const LOGICAL_XOR: &str = "logical_xor";

    // Quantization
    pub const DEQUANTIZE: &str = "dequantize";
    pub const QUANTIZE: &str = "quantize";

    // Shape operations
    pub const RESHAPE: &str = "reshape";

    // Tensor manipulation operations
    pub const TRANSPOSE: &str = "transpose";
    pub const CONCAT: &str = "concat";
    pub const SLICE: &str = "slice_by_size";
    pub const EXPAND: &str = "tile";
    pub const GATHER: &str = "gather";
    pub const SPLIT: &str = "split";
    pub const WHERE: &str = "select";
    pub const PAD: &str = "pad";
}

#[derive(Default)]
pub struct CoremlMlProgramConverter;

impl CoremlMlProgramConverter {
    /// Get operand name for an operand ID
    fn operand_name(graph: &GraphInfo, id: u32) -> String {
        graph
            .operand(id)
            .and_then(|op| op.name.clone())
            .unwrap_or_else(|| format!("operand_{}", id))
    }

    /// Create a MIL Value for a tensor operand
    fn create_value(
        graph: &GraphInfo,
        operand_id: u32,
    ) -> Result<(String, NamedValueType), GraphError> {
        let operand = graph
            .operand(operand_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("Operand {} not found", operand_id),
            })?;

        let name = Self::operand_name(graph, operand_id);

        // Create ValueType for the operand
        let dtype = Self::mil_data_type(&operand.descriptor.data_type)?;

        // Convert shape to MIL Dimensions
        let dimensions: Vec<Dimension> = operand
            .descriptor
            .shape
            .iter()
            .map(|&d| Dimension {
                dimension: Some(dimension::Dimension::Constant(
                    dimension::ConstantDimension { size: d as u64 },
                )),
            })
            .collect();

        let value_type = ValueType {
            r#type: Some(
                crate::protos::coreml::mil_spec::value_type::Type::TensorType(TensorType {
                    rank: dimensions.len() as i64,
                    data_type: dtype,
                    dimensions,
                    attributes: HashMap::new(), // Empty attributes for now
                }),
            ),
        };

        Ok((
            name.clone(),
            NamedValueType {
                name,
                r#type: Some(value_type),
            },
        ))
    }

    /// Convert WebNN DataType to MIL DataType
    fn mil_data_type(data_type: &DataType) -> Result<i32, GraphError> {
        use crate::protos::coreml::mil_spec::DataType as MilDataType;

        Ok(match data_type {
            DataType::Float32 => MilDataType::Float32 as i32,
            DataType::Float16 => MilDataType::Float16 as i32,
            DataType::Int32 => MilDataType::Int32 as i32,
            DataType::Int8 => MilDataType::Int8 as i32,
            DataType::Uint32 => MilDataType::Uint32 as i32,
            DataType::Uint8 => MilDataType::Uint8 as i32,
        })
    }

    /// Create a MIL operation
    fn create_mil_operation(
        op_type: &str,
        inputs: HashMap<String, Argument>,
        outputs: Vec<NamedValueType>,
    ) -> MilOperation {
        MilOperation {
            r#type: op_type.to_string(),
            inputs,
            outputs,
            ..Default::default()
        }
    }

    /// Create an Argument from an operand name
    fn create_argument(operand_name: &str) -> Argument {
        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Name(operand_name.to_string())),
            }],
        }
    }

    /// Create an Argument from an immediate integer array value
    fn create_immediate_int_array(values: &[u32]) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, dimension,
            tensor_value, value, value_type,
        };

        let int_values: Vec<i64> = values.iter().map(|&v| v as i64).collect();

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::LongInts(
                tensor_value::RepeatedLongInts { values: int_values },
            )),
        };

        let value = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int64 as i32,
                    rank: 1,
                    dimensions: vec![Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension {
                                size: values.len() as u64,
                            },
                        )),
                    }],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(value)),
            }],
        }
    }

    /// Create an Argument from an immediate integer scalar value
    fn create_immediate_int(value: u32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::LongInts(
                tensor_value::RepeatedLongInts {
                    values: vec![value as i64],
                },
            )),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int64 as i32,
                    rank: 0, // Scalar
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Create an Argument from an immediate float scalar value
    fn create_immediate_float(value: f32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
                values: vec![value],
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Float32 as i32,
                    rank: 0, // Scalar
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Map WebNN operation to MIL operation
    fn convert_operation(
        &self,
        graph: &GraphInfo,
        op: &Operation,
    ) -> Result<MilOperation, GraphError> {
        let mil_op_type = self.get_mil_op_type(&op.op_type)?;

        // Get input operand names
        let input_names: Vec<String> = op
            .input_operands
            .iter()
            .map(|&id| Self::operand_name(graph, id))
            .collect();

        // Get output operand info
        let (_output_name, output_type) = Self::create_value(graph, op.output_operand)?;

        // Create inputs map based on operation type
        let inputs = self.create_operation_inputs(graph, op, &input_names)?;

        // Create outputs
        let outputs = vec![output_type];

        Ok(Self::create_mil_operation(mil_op_type, inputs, outputs))
    }

    /// Get MIL operation type for WebNN operation
    fn get_mil_op_type(&self, webnn_op: &str) -> Result<&'static str, GraphError> {
        let mil_type = match webnn_op.to_lowercase().as_str() {
            // Binary operations
            "add" => mil_ops::ADD,
            "sub" => mil_ops::SUB,
            "mul" => mil_ops::MUL,
            "div" => mil_ops::DIV,
            "pow" => mil_ops::POW,
            "matmul" => mil_ops::MATMUL,

            // Activation functions
            "relu" => mil_ops::RELU,
            "sigmoid" => mil_ops::SIGMOID,
            "tanh" => mil_ops::TANH,
            "softmax" => mil_ops::SOFTMAX,

            // Convolution and pooling
            "conv2d" => mil_ops::CONV,
            "convtranspose2d" => mil_ops::CONV_TRANSPOSE,
            "averagepool2d" => mil_ops::AVG_POOL,
            "maxpool2d" => mil_ops::MAX_POOL,
            "globalaveragepool" => mil_ops::GLOBAL_AVG_POOL,
            "globalmaxpool" => mil_ops::GLOBAL_MAX_POOL,

            // Normalization
            "batchnormalization" => mil_ops::BATCH_NORM,
            "instancenormalization" => mil_ops::INSTANCE_NORM,
            "layernormalization" => mil_ops::LAYER_NORM,

            // Reduction operations
            "reducesum" => mil_ops::REDUCE_SUM,
            "reducemean" => mil_ops::REDUCE_MEAN,
            "reducemax" => mil_ops::REDUCE_MAX,
            "reducemin" => mil_ops::REDUCE_MIN,
            "reduceproduct" => mil_ops::REDUCE_PROD,
            "reducel1" => mil_ops::REDUCE_L1,
            "reducel2" => mil_ops::REDUCE_L2,
            "reducelogsum" => mil_ops::REDUCE_LOG_SUM,
            "reducelogsumexp" => mil_ops::REDUCE_LOG_SUM_EXP,
            "reducesumsquare" => mil_ops::REDUCE_SUM_SQUARE,

            // Element-wise unary operations
            "abs" => mil_ops::ABS,
            "ceil" => mil_ops::CEIL,
            "floor" => mil_ops::FLOOR,
            "round" => mil_ops::ROUND,
            "neg" => mil_ops::NEG,
            "identity" => mil_ops::IDENTITY,
            "exp" => mil_ops::EXP,
            "log" => mil_ops::LOG,
            "sqrt" => mil_ops::SQRT,
            "sign" => mil_ops::SIGN,
            "sin" => mil_ops::SIN,
            "cos" => mil_ops::COS,
            "tan" => mil_ops::TAN,
            "asin" => mil_ops::ASIN,
            "acos" => mil_ops::ACOS,
            "atan" => mil_ops::ATAN,
            "sinh" => mil_ops::SINH,
            "cosh" => mil_ops::COSH,
            "asinh" => mil_ops::ASINH,
            "acosh" => mil_ops::ACOSH,
            "atanh" => mil_ops::ATANH,
            "erf" => mil_ops::ERF,
            "reciprocal" => mil_ops::RECIPROCAL,

            // Logic operations
            "equal" => mil_ops::EQUAL,
            "greater" => mil_ops::GREATER,
            "greaterorequal" => mil_ops::GREATER_EQUAL,
            "lesser" => mil_ops::LESS,
            "lesserorequal" => mil_ops::LESS_EQUAL,
            "logicalnot" => mil_ops::LOGICAL_NOT,
            "logicaland" => mil_ops::LOGICAL_AND,
            "logicalor" => mil_ops::LOGICAL_OR,
            "logicalxor" => mil_ops::LOGICAL_XOR,

            // Quantization
            "dequantizelinear" => mil_ops::DEQUANTIZE,
            "quantizelinear" => mil_ops::QUANTIZE,

            // Shape operations
            "reshape" => mil_ops::RESHAPE,

            // Tensor manipulation
            "transpose" => mil_ops::TRANSPOSE,
            "concat" => mil_ops::CONCAT,
            "slice" => mil_ops::SLICE,
            "expand" => mil_ops::EXPAND,
            "gather" => mil_ops::GATHER,
            "split" => mil_ops::SPLIT,
            "where" => mil_ops::WHERE,
            "pad" => mil_ops::PAD,

            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("Unsupported operation: {}", webnn_op),
                });
            }
        };

        Ok(mil_type)
    }

    /// Create inputs map for MIL operation
    fn create_operation_inputs(
        &self,
        _graph: &GraphInfo,
        op: &Operation,
        input_names: &[String],
    ) -> Result<HashMap<String, Argument>, GraphError> {
        let mut inputs = HashMap::new();

        match op.op_type.to_lowercase().as_str() {
            // Binary operations: x, y
            "add" | "sub" | "mul" | "div" | "pow" | "matmul" | "equal" | "greater"
            | "greaterorequal" | "lesser" | "lesserorequal" | "logicaland" | "logicalor"
            | "logicalxor" => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("y".to_string(), Self::create_argument(&input_names[1]));
                }
            }

            // Unary operations: x
            "relu" | "sigmoid" | "tanh" | "softmax" | "abs" | "ceil" | "floor" | "round"
            | "neg" | "sign" | "identity" | "exp" | "log" | "sqrt" | "reciprocal" | "sin"
            | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "asinh" | "acosh"
            | "atanh" | "erf" | "logicalnot" | "globalaveragepool" | "globalmaxpool" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
            }

            // Quantization operations: input, scale, zero_point
            "dequantizelinear" | "quantizelinear" => {
                if input_names.len() >= 3 {
                    inputs.insert("input".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("scale".to_string(), Self::create_argument(&input_names[1]));
                    inputs.insert(
                        "zero_point".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }
            }

            // Reshape: x, shape
            "reshape" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // TODO: Add shape parameter from attributes
            }

            // Convolution operations: input, filter + parameters
            "conv2d" => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("weight".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add optional bias if present (third input)
                if input_names.len() >= 3 {
                    inputs.insert("bias".to_string(), Self::create_argument(&input_names[2]));
                }

                // Add parameters from attributes
                if let Some(strides) = op.attributes.get("strides").and_then(|v| v.as_array()) {
                    let strides_u32: Vec<u32> = strides
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !strides_u32.is_empty() {
                        inputs.insert(
                            "strides".to_string(),
                            Self::create_immediate_int_array(&strides_u32),
                        );
                    }
                }

                if let Some(dilations) = op.attributes.get("dilations").and_then(|v| v.as_array()) {
                    let dilations_u32: Vec<u32> = dilations
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !dilations_u32.is_empty() {
                        inputs.insert(
                            "dilations".to_string(),
                            Self::create_immediate_int_array(&dilations_u32),
                        );
                    }
                }

                if let Some(pads) = op.attributes.get("pads").and_then(|v| v.as_array()) {
                    let pads_u32: Vec<u32> = pads
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !pads_u32.is_empty() {
                        inputs.insert(
                            "pad".to_string(),
                            Self::create_immediate_int_array(&pads_u32),
                        );
                    }
                }

                if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
                    inputs.insert(
                        "groups".to_string(),
                        Self::create_immediate_int(groups as u32),
                    );
                }
            }

            // Transposed convolution: input, filter + parameters
            "convtranspose2d" => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("weight".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add optional bias if present (third input)
                if input_names.len() >= 3 {
                    inputs.insert("bias".to_string(), Self::create_argument(&input_names[2]));
                }

                // Add parameters from attributes
                if let Some(strides) = op.attributes.get("strides").and_then(|v| v.as_array()) {
                    let strides_u32: Vec<u32> = strides
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !strides_u32.is_empty() {
                        inputs.insert(
                            "strides".to_string(),
                            Self::create_immediate_int_array(&strides_u32),
                        );
                    }
                }

                if let Some(dilations) = op.attributes.get("dilations").and_then(|v| v.as_array()) {
                    let dilations_u32: Vec<u32> = dilations
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !dilations_u32.is_empty() {
                        inputs.insert(
                            "dilations".to_string(),
                            Self::create_immediate_int_array(&dilations_u32),
                        );
                    }
                }

                if let Some(pads) = op.attributes.get("pads").and_then(|v| v.as_array()) {
                    let pads_u32: Vec<u32> = pads
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !pads_u32.is_empty() {
                        inputs.insert(
                            "pad".to_string(),
                            Self::create_immediate_int_array(&pads_u32),
                        );
                    }
                }

                if let Some(output_padding) = op
                    .attributes
                    .get("outputPadding")
                    .and_then(|v| v.as_array())
                {
                    let output_padding_u32: Vec<u32> = output_padding
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !output_padding_u32.is_empty() {
                        inputs.insert(
                            "output_shape".to_string(),
                            Self::create_immediate_int_array(&output_padding_u32),
                        );
                    }
                }

                if let Some(groups) = op.attributes.get("groups").and_then(|v| v.as_u64()) {
                    inputs.insert(
                        "groups".to_string(),
                        Self::create_immediate_int(groups as u32),
                    );
                }
            }

            // Pooling operations: input + parameters
            "averagepool2d" | "maxpool2d" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add parameters from attributes
                if let Some(window_dimensions) = op
                    .attributes
                    .get("windowDimensions")
                    .and_then(|v| v.as_array())
                {
                    let window_dimensions_u32: Vec<u32> = window_dimensions
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !window_dimensions_u32.is_empty() {
                        inputs.insert(
                            "kernel_sizes".to_string(),
                            Self::create_immediate_int_array(&window_dimensions_u32),
                        );
                    }
                }

                if let Some(strides) = op.attributes.get("strides").and_then(|v| v.as_array()) {
                    let strides_u32: Vec<u32> = strides
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !strides_u32.is_empty() {
                        inputs.insert(
                            "strides".to_string(),
                            Self::create_immediate_int_array(&strides_u32),
                        );
                    }
                }

                if let Some(dilations) = op.attributes.get("dilations").and_then(|v| v.as_array()) {
                    let dilations_u32: Vec<u32> = dilations
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !dilations_u32.is_empty() {
                        inputs.insert(
                            "dilations".to_string(),
                            Self::create_immediate_int_array(&dilations_u32),
                        );
                    }
                }

                if let Some(pads) = op.attributes.get("pads").and_then(|v| v.as_array()) {
                    let pads_u32: Vec<u32> = pads
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !pads_u32.is_empty() {
                        inputs.insert(
                            "pad".to_string(),
                            Self::create_immediate_int_array(&pads_u32),
                        );
                    }
                }
            }

            // Normalization operations
            "batchnormalization" | "instancenormalization" | "layernormalization" => {
                // Add input operands (input, mean, variance, optional scale, optional bias)
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if input_names.len() >= 2 {
                    inputs.insert("mean".to_string(), Self::create_argument(&input_names[1]));
                }
                if input_names.len() >= 3 {
                    inputs.insert(
                        "variance".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }
                // Scale and bias are optional (4th and 5th inputs)
                if input_names.len() >= 4 {
                    inputs.insert("gamma".to_string(), Self::create_argument(&input_names[3]));
                }
                if input_names.len() >= 5 {
                    inputs.insert("beta".to_string(), Self::create_argument(&input_names[4]));
                }

                // Add epsilon parameter
                if let Some(epsilon) = op.attributes.get("epsilon").and_then(|v| v.as_f64()) {
                    inputs.insert(
                        "epsilon".to_string(),
                        Self::create_immediate_float(epsilon as f32),
                    );
                }
            }

            // Tensor manipulation operations
            "transpose" => {
                // transpose: x, perm (permutation)
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add permutation parameter if present
                if let Some(perm) = op.attributes.get("permutation").and_then(|v| v.as_array()) {
                    let perm_u32: Vec<u32> = perm
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !perm_u32.is_empty() {
                        inputs.insert(
                            "perm".to_string(),
                            Self::create_immediate_int_array(&perm_u32),
                        );
                    }
                }
            }

            "concat" => {
                // concat: values (list of tensors), axis
                // In CoreML, all inputs are listed as separate named inputs
                for (idx, input_name) in input_names.iter().enumerate() {
                    inputs.insert(format!("values_{}", idx), Self::create_argument(input_name));
                }

                // Add axis parameter
                if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
                    inputs.insert("axis".to_string(), Self::create_immediate_int(axis as u32));
                }
            }

            "slice" => {
                // slice_by_size: x, begin, size
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add starts (begin) parameter
                if let Some(starts) = op.attributes.get("starts").and_then(|v| v.as_array()) {
                    let starts_u32: Vec<u32> = starts
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !starts_u32.is_empty() {
                        inputs.insert(
                            "begin".to_string(),
                            Self::create_immediate_int_array(&starts_u32),
                        );
                    }
                }

                // Add sizes parameter
                if let Some(sizes) = op.attributes.get("sizes").and_then(|v| v.as_array()) {
                    let sizes_u32: Vec<u32> = sizes
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !sizes_u32.is_empty() {
                        inputs.insert(
                            "size".to_string(),
                            Self::create_immediate_int_array(&sizes_u32),
                        );
                    }
                }
            }

            "expand" => {
                // tile: x, reps (repetitions)
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add newShape parameter as reps
                if let Some(new_shape) = op.attributes.get("newShape").and_then(|v| v.as_array()) {
                    let new_shape_u32: Vec<u32> = new_shape
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !new_shape_u32.is_empty() {
                        inputs.insert(
                            "reps".to_string(),
                            Self::create_immediate_int_array(&new_shape_u32),
                        );
                    }
                }
            }

            "gather" => {
                // gather: params (data), indices, axis
                if input_names.len() >= 2 {
                    inputs.insert("params".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                }

                // Add axis parameter (defaults to 0)
                if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
                    inputs.insert("axis".to_string(), Self::create_immediate_int(axis as u32));
                }
            }

            "split" => {
                // split: x, num_splits or split_sizes, axis
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add splits parameter (can be int or array)
                if let Some(splits) = op.attributes.get("splits") {
                    if let Some(count) = splits.as_u64() {
                        inputs.insert(
                            "num_splits".to_string(),
                            Self::create_immediate_int(count as u32),
                        );
                    } else if let Some(sizes) = splits.as_array() {
                        let sizes_u32: Vec<u32> = sizes
                            .iter()
                            .filter_map(|v| v.as_u64().map(|u| u as u32))
                            .collect();
                        if !sizes_u32.is_empty() {
                            inputs.insert(
                                "split_sizes".to_string(),
                                Self::create_immediate_int_array(&sizes_u32),
                            );
                        }
                    }
                }

                // Add axis parameter (defaults to 0)
                if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
                    inputs.insert("axis".to_string(), Self::create_immediate_int(axis as u32));
                }
            }

            "where" => {
                // select: cond, a (true_value), b (false_value)
                if input_names.len() >= 3 {
                    inputs.insert("cond".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("a".to_string(), Self::create_argument(&input_names[1]));
                    inputs.insert("b".to_string(), Self::create_argument(&input_names[2]));
                }
            }

            "pad" => {
                // pad: x, pad, mode, constant_val
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add padding parameter
                if let Some(padding) = op.attributes.get("padding").and_then(|v| v.as_array()) {
                    let padding_u32: Vec<u32> = padding
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !padding_u32.is_empty() {
                        inputs.insert(
                            "pad".to_string(),
                            Self::create_immediate_int_array(&padding_u32),
                        );
                    }
                }

                // Add mode parameter (defaults to "constant")
                // CoreML pad modes: "constant", "reflect", "replicate"
                // WebNN modes: "constant", "edge", "reflection", "symmetric"
                // Note: "edge" maps to "replicate", "reflection" and "symmetric" are similar

                // Add constant value if present
                if let Some(value) = op.attributes.get("value").and_then(|v| v.as_f64()) {
                    inputs.insert(
                        "constant_val".to_string(),
                        Self::create_immediate_float(value as f32),
                    );
                }
            }

            _ => {}
        }

        Ok(inputs)
    }
}

impl super::GraphConverter for CoremlMlProgramConverter {
    fn format(&self) -> &'static str {
        "coreml"
    }

    fn convert(&self, graph_info: &GraphInfo) -> Result<super::ConvertedGraph, GraphError> {
        // Create MLProgram
        let mut program = Program {
            version: 1,
            ..Default::default()
        };

        // Create main function
        let mut main_function = Function::default();

        // Add function inputs from graph inputs
        for &input_id in &graph_info.input_operands {
            let (_name, value_type) = Self::create_value(graph_info, input_id)?;
            main_function.inputs.push(value_type);
        }

        // Create main block
        let mut main_block = Block::default();

        // Convert all operations to MIL operations
        for op in &graph_info.operations {
            let mil_op = self.convert_operation(graph_info, op)?;
            main_block.operations.push(mil_op);
        }

        // Add block outputs (output operand names)
        for &output_id in &graph_info.output_operands {
            let output_name = Self::operand_name(graph_info, output_id);
            main_block.outputs.push(output_name);
        }

        // Add block to function
        main_function
            .block_specializations
            .insert("CoreML7".to_string(), main_block);

        // Add function to program
        program.functions.insert("main".to_string(), main_function);

        // Create Model
        let mut model = Model {
            specification_version: 7, // CoreML 7 (iOS 15+, macOS 12+)
            ..Default::default()
        };

        // Set MLProgram
        model.r#type = Some(crate::protos::coreml::specification::model::Type::MlProgram(program));

        // Serialize to bytes
        let mut buffer = Vec::new();
        model
            .encode(&mut buffer)
            .map_err(|e| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("Failed to encode model: {}", e),
            })?;

        Ok(super::ConvertedGraph {
            format: "coreml",
            content_type: "application/x-coreml-model",
            data: buffer,
        })
    }
}
