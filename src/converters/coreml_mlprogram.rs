/// CoreML MLProgram (MIL) converter
///
/// This converter generates CoreML MLProgram models using the Model Intermediate Language (MIL).
/// MLProgram is the modern CoreML format (spec v7+, iOS 15+, macOS 12+) that supports:
/// - Flexible MIL operations
/// - Quantization operations
/// - Better optimization
///
/// This replaces the legacy NeuralNetwork format.
use crate::converters::operand_name;
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

    // Advanced activation operations
    pub const GELU: &str = "gelu";

    // Specialized activation operations
    pub const PRELU: &str = "prelu";
    pub const ELU: &str = "elu";
    pub const LEAKY_RELU: &str = "leaky_relu";
    pub const SOFTPLUS: &str = "softplus";
    pub const SOFTSIGN: &str = "softsign";
    pub const HARD_SIGMOID: &str = "sigmoid_hard";
    pub const HARD_SWISH: &str = "mul"; // TODO: Implement as x * hardSigmoid(x)

    // Dimension manipulation operations
    pub const SQUEEZE: &str = "squeeze";
    pub const UNSQUEEZE: &str = "expand_dims";

    // Arg reduce operations
    pub const ARG_MAX: &str = "reduce_argmax";
    pub const ARG_MIN: &str = "reduce_argmin";

    // Type conversion operations
    pub const CAST: &str = "cast";

    // Scatter operations
    pub const SCATTER_ELEMENTS: &str = "scatter";
    pub const SCATTER_ND: &str = "scatter_nd";

    // Tile operation
    pub const TILE: &str = "tile";

    // Triangular operation
    pub const TRIANGULAR: &str = "band_part";

    // Clamp operation
    pub const CLIP: &str = "clip";
}

#[derive(Default)]
pub struct CoremlMlProgramConverter;

impl CoremlMlProgramConverter {
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

        let name = operand_name(graph, operand_id);

        // Create ValueType for the operand
        let dtype = Self::mil_data_type(&operand.descriptor.data_type)?;

        // Convert shape to MIL Dimensions
        // CoreML requires explicit shape constraints - convert scalars (0D) to 1D [1]
        // Following Chromium's approach: https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/coreml/graph_builder_coreml.cc
        let shape_to_convert = if operand.descriptor.shape.is_empty() {
            // Scalar (0D) tensor -> reshape to [1] for CoreML compatibility
            vec![1u32]
        } else {
            operand.descriptor.shape.clone()
        };

        let dimensions: Vec<Dimension> = shape_to_convert
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
            DataType::Int64 => MilDataType::Int64 as i32,
        })
    }

    /// Create a const operation for a constant operand
    fn create_const_operation(
        graph: &GraphInfo,
        operand_id: u32,
        operand: &crate::graph::Operand,
        constant_data: &crate::graph::ConstantData,
        weight_builder: &mut super::WeightFileBuilder,
    ) -> Result<MilOperation, GraphError> {
        use crate::protos::coreml::mil_spec::{TensorValue, Value, tensor_value, value};

        let (_name, output_type) = Self::create_value(graph, operand_id)?;

        // Create tensor value from constant data
        let tensor_value = match operand.descriptor.data_type {
            crate::graph::DataType::Float32 => {
                // Convert raw bytes to f32 values
                let float_count = constant_data.data.len() / 4;
                let mut floats = Vec::with_capacity(float_count);
                for i in 0..float_count {
                    let bytes = [
                        constant_data.data[i * 4],
                        constant_data.data[i * 4 + 1],
                        constant_data.data[i * 4 + 2],
                        constant_data.data[i * 4 + 3],
                    ];
                    floats.push(f32::from_le_bytes(bytes));
                }
                TensorValue {
                    value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
                        values: floats,
                    })),
                }
            }
            crate::graph::DataType::Int32 => {
                // Convert raw bytes to i32 values
                let int_count = constant_data.data.len() / 4;
                let mut ints = Vec::with_capacity(int_count);
                for i in 0..int_count {
                    let bytes = [
                        constant_data.data[i * 4],
                        constant_data.data[i * 4 + 1],
                        constant_data.data[i * 4 + 2],
                        constant_data.data[i * 4 + 3],
                    ];
                    ints.push(i32::from_le_bytes(bytes));
                }
                TensorValue {
                    value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                        values: ints,
                    })),
                }
            }
            crate::graph::DataType::Float16 => {
                // CoreML MLProgram (MIL) requires non-scalar Float16 constants to be stored
                // in a separate weight file with BlobFileValue references, not as immediate values.
                // Only scalar (0D) Float16 can be stored as immediate bytes.
                //
                // Chromium's implementation uses WeightsFileHandle::Write() which:
                // - For scalars (empty shape): stores as immediate value
                // - For non-scalars: writes to weights.bin with 64-byte alignment
                //
                // Reference: chromium/src/services/webnn/coreml/graph_builder_coreml.cc

                let is_scalar = operand.descriptor.shape.is_empty();

                if !is_scalar {
                    // Non-scalar Float16: add to weight file and return BlobFileValue
                    let element_count = constant_data.data.len() / 2; // 2 bytes per f16
                    let offset = weight_builder.add_weight(
                        operand_id,
                        element_count,
                        &constant_data.data,
                    )?;

                    // Create BlobFileValue reference
                    let blob_file_value = Value {
                        doc_string: String::new(),
                        r#type: output_type.r#type.clone(),
                        value: Some(value::Value::BlobFileValue(value::BlobFileValue {
                            file_name: "@model_path/weights/weights.bin".to_string(),
                            offset,
                        })),
                    };

                    // Create const operation with BlobFileValue
                    let mut attributes = HashMap::new();
                    attributes.insert("val".to_string(), blob_file_value);

                    return Ok(MilOperation {
                        r#type: "const".to_string(),
                        inputs: HashMap::new(),
                        outputs: vec![output_type],
                        attributes,
                        ..Default::default()
                    });
                }

                // Scalar Float16: store as immediate bytes
                TensorValue {
                    value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                        values: constant_data.data.clone().into(),
                    })),
                }
            }
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!(
                        "Unsupported constant data type: {:?}",
                        operand.descriptor.data_type
                    ),
                });
            }
        };

        // Create immediate value
        let immediate_value = Value {
            doc_string: String::new(),
            r#type: output_type.r#type.clone(),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };

        // Create const operation
        // Note: const operations in CoreML MIL use attributes, not inputs, for the value
        let mut attributes = HashMap::new();
        attributes.insert("val".to_string(), immediate_value);

        Ok(MilOperation {
            r#type: "const".to_string(),
            inputs: HashMap::new(),
            outputs: vec![output_type],
            attributes,
            ..Default::default()
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

    /// Create an Argument from multiple operand names (tuple/list of references)
    /// Used for variadic parameters like concat's 'values'
    fn create_argument_tuple(operand_names: &[String]) -> Argument {
        Argument {
            arguments: operand_names
                .iter()
                .map(|name| crate::protos::coreml::mil_spec::argument::Binding {
                    binding: Some(Binding::Name(name.clone())),
                })
                .collect(),
        }
    }

    /// Create an Argument from an immediate integer array value (int32)
    fn create_immediate_int_array(values: &[u32]) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, dimension,
            tensor_value, value, value_type,
        };

        let int_values: Vec<i32> = values.iter().map(|&v| v as i32).collect();

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values: int_values,
            })),
        };

        let value = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int32 as i32,
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

    /// Create an Argument from an immediate integer scalar value (int32)
    fn create_immediate_int(value: u32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values: vec![value as i32],
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int32 as i32,
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

    /// Create an Argument from an immediate float16 value (scalar)
    fn create_immediate_float16(value: f32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        // Convert f32 to f16 bytes
        let f16_bits = half::f16::from_f32(value).to_bits();
        let bytes = f16_bits.to_le_bytes().to_vec();

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                values: bytes.into(),
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Float16 as i32,
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

    /// Create an Argument from an immediate string value
    fn create_immediate_string(value: &str) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Strings(
                tensor_value::RepeatedStrings {
                    values: vec![value.to_string()],
                },
            )),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::String as i32,
                    rank: 0, // Scalar string
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

    /// Create an immediate bool argument
    fn create_immediate_bool(value: bool) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Bools(tensor_value::RepeatedBools {
                values: vec![value],
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Bool as i32,
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

    /// Create an argument referencing a named value
    fn create_name_argument(name: String) -> Argument {
        use crate::protos::coreml::mil_spec::argument::binding::Binding;

        Argument {
            arguments: vec![crate::protos::coreml::mil_spec::argument::Binding {
                binding: Some(Binding::Name(name)),
            }],
        }
    }

    /// Create an immediate int argument
    fn create_int_argument(value: i32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values: vec![value],
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int32 as i32,
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

    /// Create an immediate int array argument
    fn create_int_array_argument(values: Vec<i32>) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };

        let values_len = values.len();

        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values,
            })),
        };

        let val = Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int32 as i32,
                    rank: 1, // 1D array
                    dimensions: vec![Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension {
                                size: values_len as u64,
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
                binding: Some(Binding::Value(val)),
            }],
        }
    }

    /// Map WebNN operation to MIL operation (with optional operand name overrides)
    fn convert_operation_with_overrides(
        &self,
        graph: &GraphInfo,
        op: &Operation,
        operand_name_overrides: &HashMap<u32, String>,
    ) -> Result<MilOperation, GraphError> {
        // Handle multi-output operations separately
        if op.op_type == "split" {
            return self.convert_split_operation(graph, op);
        }

        let mil_op_type = self.get_mil_op_type(&op.op_type)?;

        // Get input operand names, using overrides if available
        let input_names: Vec<String> = op
            .input_operands
            .iter()
            .map(|&id| {
                operand_name_overrides
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| operand_name(graph, id))
            })
            .collect();

        // Get output operand info
        // Check if this is a single-output or multi-output operation
        let output_id = if let Some(id) = op.output_operand {
            // Single-output operation
            id
        } else if !op.output_operands.is_empty() {
            // Multi-output operation not handled yet
            return Err(GraphError::ConversionFailed {
                format: "CoreML MLProgram".to_string(),
                reason: format!(
                    "operation '{}' has multiple outputs but is not implemented as multi-output. \
                     Only 'split' is currently supported as multi-output.",
                    op.op_type
                ),
            });
        } else {
            // No outputs at all - this shouldn't happen but handle gracefully
            return Err(GraphError::ConversionFailed {
                format: "CoreML MLProgram".to_string(),
                reason: format!("operation '{}' has no output operands", op.op_type),
            });
        };

        let (_output_name, output_type) = Self::create_value(graph, output_id)?;

        // Create inputs map based on operation type
        let inputs = self.create_operation_inputs(graph, op, &input_names)?;

        // Create outputs
        let outputs = vec![output_type];

        Ok(Self::create_mil_operation(mil_op_type, inputs, outputs))
    }

    /// Map WebNN operation to MIL operation (convenience wrapper without overrides)
    #[allow(dead_code)]
    fn convert_operation(
        &self,
        graph: &GraphInfo,
        op: &Operation,
    ) -> Result<MilOperation, GraphError> {
        self.convert_operation_with_overrides(graph, op, &HashMap::new())
    }

    /// Convert split operation (multi-output)
    fn convert_split_operation(
        &self,
        graph: &GraphInfo,
        op: &Operation,
    ) -> Result<MilOperation, GraphError> {
        // Get input operand name
        let input_name = operand_name(graph, op.input_operands[0]);

        // Get output types
        let outputs: Vec<NamedValueType> = op
            .output_operands
            .iter()
            .map(|&id| {
                let (_name, value_type) = Self::create_value(graph, id)?;
                Ok(value_type)
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        // Create inputs
        let mut inputs: HashMap<String, Argument> = HashMap::new();

        // Add main input (x)
        inputs.insert("x".to_string(), Self::create_name_argument(input_name));

        // Add num_splits or split_sizes
        if let Some(splits_val) = op.attributes.get("splits") {
            if let Some(count) = splits_val.as_u64() {
                // Equal splits - use num_splits
                inputs.insert(
                    "num_splits".to_string(),
                    Self::create_int_argument(count as i32),
                );
            } else if let Some(sizes) = splits_val.as_array() {
                // Explicit split sizes - use split_sizes
                let split_sizes: Vec<i32> = sizes
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as i32))
                    .collect();
                inputs.insert(
                    "split_sizes".to_string(),
                    Self::create_int_array_argument(split_sizes),
                );
            }
        }

        // Add axis
        if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_i64()) {
            inputs.insert("axis".to_string(), Self::create_int_argument(axis as i32));
        }

        Ok(Self::create_mil_operation("split", inputs, outputs))
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
            "gemm" => mil_ops::MATMUL, // Gemm maps to matmul with transpose handling

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

            // Advanced operations
            "gelu" => mil_ops::GELU,
            "squeeze" => mil_ops::SQUEEZE,
            "unsqueeze" => mil_ops::UNSQUEEZE,
            "argmax" => mil_ops::ARG_MAX,
            "argmin" => mil_ops::ARG_MIN,
            "cast" => mil_ops::CAST,

            // Specialized activation operations
            "prelu" => mil_ops::PRELU,
            "elu" => mil_ops::ELU,
            "leakyrelu" => mil_ops::LEAKY_RELU,
            "softplus" => mil_ops::SOFTPLUS,
            "softsign" => mil_ops::SOFTSIGN,
            "hardsigmoid" => mil_ops::HARD_SIGMOID,
            "hardswish" => mil_ops::HARD_SWISH,

            // Scatter operations
            "scatterelements" => mil_ops::SCATTER_ELEMENTS,
            "scatternd" => mil_ops::SCATTER_ND,

            // Tile operation
            "tile" => mil_ops::TILE,

            // Triangular operation
            "triangular" => mil_ops::TRIANGULAR,

            // Clamp operation
            "clamp" => mil_ops::CLIP,

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
            "add" | "sub" | "mul" | "div" | "pow" | "equal" | "greater" | "greaterorequal"
            | "lesser" | "lesserorequal" | "logicaland" | "logicalor" | "logicalxor" => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("y".to_string(), Self::create_argument(&input_names[1]));
                }
            }

            // MatMul operation: x, y, transpose_x, transpose_y
            // CoreML requires transpose parameters, WebNN doesn't have them so default to false
            "matmul" => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("y".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add transpose_x parameter (required by CoreML, defaults to false)
                inputs.insert(
                    "transpose_x".to_string(),
                    Self::create_immediate_bool(false),
                );

                // Add transpose_y parameter (required by CoreML, defaults to false)
                inputs.insert(
                    "transpose_y".to_string(),
                    Self::create_immediate_bool(false),
                );
            }

            // Gemm operation: General Matrix Multiplication
            // Y = alpha * op(A) * op(B) + beta * C
            // CoreML matmul handles: Y = A * B (with transpose options)
            // For now, we support transpose options and basic matmul
            // TODO: Support alpha, beta, and bias (C) by decomposing into mul and add operations
            "gemm" => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("y".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add transpose parameters if specified
                if let Some(a_transpose) = op
                    .attributes
                    .get("aTranspose")
                    .or_else(|| op.attributes.get("a_transpose"))
                    .and_then(|v| v.as_bool())
                {
                    inputs.insert(
                        "transpose_x".to_string(),
                        Self::create_immediate_bool(a_transpose),
                    );
                }

                if let Some(b_transpose) = op
                    .attributes
                    .get("bTranspose")
                    .or_else(|| op.attributes.get("b_transpose"))
                    .and_then(|v| v.as_bool())
                {
                    inputs.insert(
                        "transpose_y".to_string(),
                        Self::create_immediate_bool(b_transpose),
                    );
                }

                // Note: alpha, beta, and bias (C) are not yet supported
                // These would require decomposing gemm into multiple operations:
                // 1. matmul(op(A), op(B))
                // 2. mul by alpha if != 1.0
                // 3. add beta * C if C is provided
            }

            // Global pooling operations (reduce over spatial dimensions)
            "globalaveragepool" | "globalmaxpool" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Global pooling reduces over spatial dimensions (2, 3) for NCHW format
                inputs.insert(
                    "axes".to_string(),
                    Self::create_immediate_int_array(&[2, 3]),
                );
                // Keep dimensions to maintain rank
                inputs.insert("keep_dims".to_string(), Self::create_immediate_bool(true));
            }

            // Softmax operation (requires axis parameter)
            "softmax" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Default axis is -1 (last dimension) if not specified
                let axis = op
                    .attributes
                    .get("axis")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(-1) as i32;
                inputs.insert("axis".to_string(), Self::create_immediate_int(axis as u32));
            }

            // Neg operation: implemented as mul by -1, requires x and y parameters
            // CoreML neg is actually a mul operation, so we need both operands
            "neg" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Add -1.0 as the multiplier (y parameter required by CoreML mul)
                inputs.insert("y".to_string(), Self::create_immediate_float(-1.0));
            }

            // Unary operations: x
            "relu" | "sigmoid" | "tanh" | "abs" | "ceil" | "floor" | "round" | "sign"
            | "identity" | "exp" | "sqrt" | "reciprocal" | "sin" | "cos" | "tan" | "asin"
            | "acos" | "atan" | "sinh" | "cosh" | "asinh" | "acosh" | "atanh" | "erf"
            | "logicalnot" | "softplus" | "softsign" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
            }

            // Log operation requires epsilon parameter
            "log" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // CoreML log requires epsilon parameter (default to 1e-45 for numerical stability)
                inputs.insert("epsilon".to_string(), Self::create_immediate_float(1e-45));
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

            // Specialized activation: prelu - x, slope (two inputs)
            "prelu" => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("alpha".to_string(), Self::create_argument(&input_names[1]));
                }
            }

            // Specialized activations with alpha parameter: elu, leakyRelu
            "elu" | "leakyrelu" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Add alpha parameter from attributes
                if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
                    inputs.insert(
                        "alpha".to_string(),
                        Self::create_immediate_float(alpha as f32),
                    );
                }
            }

            // HardSwish: decomposed in main loop (hardsigmoid + mul)
            // This case should never be reached due to continue in main loop
            "hardswish" => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "hardswish should be decomposed in main loop, not here".to_string(),
                });
            }

            // HardSigmoid: x, alpha, beta parameters
            "hardsigmoid" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Add alpha parameter from attributes
                if let Some(alpha) = op.attributes.get("alpha").and_then(|v| v.as_f64()) {
                    inputs.insert(
                        "alpha".to_string(),
                        Self::create_immediate_float(alpha as f32),
                    );
                }
                // Add beta parameter from attributes
                if let Some(beta) = op.attributes.get("beta").and_then(|v| v.as_f64()) {
                    inputs.insert(
                        "beta".to_string(),
                        Self::create_immediate_float(beta as f32),
                    );
                }
            }

            // Clamp operation: x, alpha (min), beta (max)
            "clamp" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // CoreML clip operation requires BOTH alpha and beta parameters
                // WebNN clamp defaults: minValue=-Infinity, maxValue=+Infinity
                let min_value = op
                    .attributes
                    .get("minValue")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::NEG_INFINITY) as f32;
                let max_value = op
                    .attributes
                    .get("maxValue")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::INFINITY) as f32;

                // Alpha and beta must match input type (CoreML requirement)
                // Check first input operand type and use appropriate immediate value method
                let use_float16 = if !op.input_operands.is_empty() {
                    if let Some(input_operand) = _graph.operand(op.input_operands[0]) {
                        input_operand.descriptor.data_type == DataType::Float16
                    } else {
                        false
                    }
                } else {
                    false
                };

                if use_float16 {
                    inputs.insert(
                        "alpha".to_string(),
                        Self::create_immediate_float16(min_value),
                    );
                    inputs.insert(
                        "beta".to_string(),
                        Self::create_immediate_float16(max_value),
                    );
                } else {
                    inputs.insert("alpha".to_string(), Self::create_immediate_float(min_value));
                    inputs.insert("beta".to_string(), Self::create_immediate_float(max_value));
                }
            }

            // Transpose operation: x, permutation
            "transpose" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add permutation parameter (required by CoreML)
                // If not specified in WebNN, default is to reverse all dimensions
                // Note: Empty perm array is valid for 0D scalar tensors
                if let Some(permutation) =
                    op.attributes.get("permutation").and_then(|v| v.as_array())
                {
                    let perm_u32: Vec<u32> = permutation
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    // Always add perm parameter, even if empty (for 0D scalars)
                    inputs.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&perm_u32),
                    );
                } else {
                    // Default: reverse all dimensions
                    // Get input operand to determine rank
                    if !op.input_operands.is_empty() {
                        if let Some(input_operand) = _graph.operand(op.input_operands[0]) {
                            let rank = input_operand.descriptor.shape.len();
                            let default_perm: Vec<u32> =
                                (0..rank).rev().map(|i| i as u32).collect();
                            // Always add perm parameter, even if empty (for 0D scalars)
                            inputs.insert(
                                "perm".to_string(),
                                Self::create_immediate_int_array(&default_perm),
                            );
                        }
                    }
                }
            }

            // Reshape: x, shape
            "reshape" => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add shape parameter from attributes (required by CoreML)
                // Note: Empty shape array is valid for 0D scalar tensors
                if let Some(new_shape) = op.attributes.get("newShape").and_then(|v| v.as_array()) {
                    let shape_values: Vec<u32> = new_shape
                        .iter()
                        .filter_map(|v| v.as_i64().map(|i| i as u32))
                        .collect();

                    // Always add shape parameter, even if empty (for 0D scalars)
                    inputs.insert(
                        "shape".to_string(),
                        Self::create_immediate_int_array(&shape_values),
                    );
                }
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

                // Add pad_type - required parameter in CoreML
                // Use "custom" when explicit padding is provided
                inputs.insert(
                    "pad_type".to_string(),
                    Self::create_immediate_string("custom"),
                );
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

                // CoreML requires pad_type parameter (defaults to "custom" for explicit padding)
                inputs.insert(
                    "pad_type".to_string(),
                    Self::create_immediate_string("custom"),
                );

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

                // Handle outputSizes (explicit output spatial dimensions [H, W])
                // Following Chromium: For conv_transpose, CoreML requires output_shape
                // to be the full output tensor dimensions [N, C, H, W] (from output operand),
                // NOT just the spatial dimensions from outputSizes attribute.
                // See: graph_builder_coreml.cc lines 2328-2334
                // When explicit outputSizes is provided, we need to compute full output shape.
                // For now, skip adding output_shape when using padding (custom pad_type).
                // TODO: Compute full output shape from outputSizes + input shape + channels

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

            // Layer normalization (different from batch/instance normalization)
            "layernormalization" => {
                // Check if axes is empty - CoreML doesn't support empty axes
                // Following Chromium (graph_builder_coreml.cc:4000-4019):
                // When axes is empty, mean equals input, so output = bias + (scale * 0)
                // We emulate this by: input - input = 0, then 0 + bias
                let axes_vec: Vec<i32> =
                    if let Some(axes) = op.attributes.get("axes").and_then(|v| v.as_array()) {
                        axes.iter()
                            .filter_map(|v| v.as_i64().map(|i| i as i32))
                            .collect()
                    } else {
                        Vec::new()
                    };

                if axes_vec.is_empty() {
                    // Empty axes case: use sub operation (input - input = 0)
                    // CoreML doesn't support empty axes, so we emulate it
                    // Note: This will be handled by inserting a sub operation in convert_operation
                    // For now, return error as this needs special multi-operation handling
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "CoreML layer_norm with empty axes requires special handling (not yet implemented)".to_string(),
                    });
                }

                // Add input operand (only x)
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Scale (gamma) is optional (2nd input)
                // CoreML requires scale/bias to be constant tensors (not graph inputs)
                // Following Chromium: TODO(crbug.com/338529226) - these params must be constant
                if input_names.len() >= 2 && op.input_operands.len() >= 2 {
                    let scale_operand_id = op.input_operands[1];
                    if let Some(scale_operand) = _graph.operand(scale_operand_id) {
                        if scale_operand.kind != crate::graph::OperandKind::Constant {
                            return Err(GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: "CoreML layer_norm requires scale (gamma) parameter to be a constant tensor, not a graph input".to_string(),
                            });
                        }
                    }
                    inputs.insert("gamma".to_string(), Self::create_argument(&input_names[1]));
                }

                // Bias (beta) is optional (3rd input)
                if input_names.len() >= 3 && op.input_operands.len() >= 3 {
                    let bias_operand_id = op.input_operands[2];
                    if let Some(bias_operand) = _graph.operand(bias_operand_id) {
                        if bias_operand.kind != crate::graph::OperandKind::Constant {
                            return Err(GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: "CoreML layer_norm requires bias (beta) parameter to be a constant tensor, not a graph input".to_string(),
                            });
                        }
                    }
                    inputs.insert("beta".to_string(), Self::create_argument(&input_names[2]));
                }

                // Add axes parameter (REQUIRED by CoreML, must not be empty)
                inputs.insert(
                    "axes".to_string(),
                    Self::create_int_array_argument(axes_vec),
                );

                // Add epsilon parameter
                if let Some(epsilon) = op.attributes.get("epsilon").and_then(|v| v.as_f64()) {
                    inputs.insert(
                        "epsilon".to_string(),
                        Self::create_immediate_float(epsilon as f32),
                    );
                }
            }

            // Batch/instance normalization (have mean, variance inputs)
            "batchnormalization" | "instancenormalization" => {
                // Add input operands (input, mean, variance, optional scale, optional bias)
                // CoreML requires mean and variance to be constant tensors (not graph inputs)
                // Following Chromium pattern (though they don't validate this yet)
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Mean parameter (2nd input) - must be constant
                if input_names.len() >= 2 && op.input_operands.len() >= 2 {
                    let mean_operand_id = op.input_operands[1];
                    if let Some(mean_operand) = _graph.operand(mean_operand_id) {
                        if mean_operand.kind != crate::graph::OperandKind::Constant {
                            return Err(GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!(
                                    "CoreML {} requires mean parameter to be a constant tensor, not a graph input",
                                    op.op_type
                                ),
                            });
                        }
                    }
                    inputs.insert("mean".to_string(), Self::create_argument(&input_names[1]));
                }

                // Variance parameter (3rd input) - must be constant
                if input_names.len() >= 3 && op.input_operands.len() >= 3 {
                    let variance_operand_id = op.input_operands[2];
                    if let Some(variance_operand) = _graph.operand(variance_operand_id) {
                        if variance_operand.kind != crate::graph::OperandKind::Constant {
                            return Err(GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!(
                                    "CoreML {} requires variance parameter to be a constant tensor, not a graph input",
                                    op.op_type
                                ),
                            });
                        }
                    }
                    inputs.insert(
                        "variance".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }

                // Scale and bias are optional (4th and 5th inputs)
                // These can be either constants or graph inputs in CoreML
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

            "concat" => {
                // concat: values (variadic list of tensors), axis
                // CoreML expects a single 'values' parameter containing a tuple of all inputs
                if !input_names.is_empty() {
                    inputs.insert(
                        "values".to_string(),
                        Self::create_argument_tuple(&input_names),
                    );
                }

                // Add axis parameter
                if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
                    inputs.insert("axis".to_string(), Self::create_immediate_int(axis as u32));
                }

                // Add interleave parameter (defaults to false)
                inputs.insert("interleave".to_string(), Self::create_immediate_bool(false));
            }

            "slice" => {
                // slice_by_size: x, begin, size
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add starts (begin) parameter
                // Note: Empty arrays are valid for no-op slices on 0D tensors
                if let Some(starts) = op.attributes.get("starts").and_then(|v| v.as_array()) {
                    let starts_u32: Vec<u32> = starts
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    // Always add begin parameter, even if empty
                    inputs.insert(
                        "begin".to_string(),
                        Self::create_immediate_int_array(&starts_u32),
                    );
                }

                // Add sizes parameter (required by CoreML)
                // Note: Empty arrays are valid for no-op slices on 0D tensors
                if let Some(sizes) = op.attributes.get("sizes").and_then(|v| v.as_array()) {
                    let sizes_u32: Vec<u32> = sizes
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    // Always add size parameter, even if empty (required by CoreML)
                    inputs.insert(
                        "size".to_string(),
                        Self::create_immediate_int_array(&sizes_u32),
                    );
                }
            }

            "expand" => {
                // CoreML tile operation requires input rank to match reps length
                // If reshape was added before this operation, use reshaped input name
                //  Otherwise use original input

                if let Some(new_shape) = op.attributes.get("newShape").and_then(|v| v.as_array()) {
                    let new_shape_u32: Vec<u32> = new_shape
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();

                    // Get input operand shape
                    if !op.input_operands.is_empty() {
                        if let Some(input_operand) = _graph.operand(op.input_operands[0]) {
                            let input_shape = &input_operand.descriptor.shape;
                            let input_rank = input_shape.len();
                            let output_rank = new_shape_u32.len();

                            // Determine input name for tile operation
                            let tile_input_name = if input_rank < output_rank {
                                // A reshape was added, use the reshaped output name
                                // The reshape output name is: {input_name}_expand_reshaped
                                format!("{}_expand_reshaped", input_names[0])
                            } else {
                                // No reshape, use original input
                                input_names[0].clone()
                            };

                            inputs.insert(
                                "x".to_string(),
                                Self::create_name_argument(tile_input_name),
                            );

                            // Create reshaped dimensions (right-aligned, padded with 1s on left)
                            let mut reshaped_dims = vec![1u32; output_rank];
                            for i in 0..input_rank {
                                reshaped_dims[output_rank - i - 1] =
                                    input_shape[input_rank - i - 1];
                            }

                            // Calculate reps: reps[i] = output_shape[i] / reshaped_input_shape[i]
                            let reps: Vec<i32> = new_shape_u32
                                .iter()
                                .zip(reshaped_dims.iter())
                                .map(|(&output_dim, &reshaped_dim)| {
                                    if reshaped_dim == output_dim {
                                        1
                                    } else if reshaped_dim == 1 {
                                        output_dim as i32
                                    } else {
                                        // Should not happen - dimensions must match or input must be 1
                                        1
                                    }
                                })
                                .collect();

                            inputs
                                .insert("reps".to_string(), Self::create_int_array_argument(reps));
                        }
                    }
                }
            }

            "gather" => {
                // gather: x (data), indices, axis, validate_indices
                // CoreML uses 'x' for the data input, not 'params'
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                }

                // Add axis parameter (REQUIRED by CoreML, defaults to 0)
                let axis = op
                    .attributes
                    .get("axis")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                inputs.insert("axis".to_string(), Self::create_immediate_int(axis));

                // Add validate_indices parameter (required by CoreML)
                // Chromium sets this to false to avoid validation issues
                // TODO: Handle negative and out-of-bounds indices properly
                inputs.insert(
                    "validate_indices".to_string(),
                    Self::create_immediate_bool(false),
                );
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

            "gelu" => {
                // gelu: x (mode is optional, defaults to "EXACT")
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // CoreML GELU supports "EXACT" and "TANH_APPROXIMATION" modes
                // WebNN GELU has no mode parameter (uses exact by default)
            }

            "squeeze" => {
                // squeeze: x, axes (optional)
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add axes parameter if present
                if let Some(axes) = op.attributes.get("axes").and_then(|v| v.as_array()) {
                    let axes_u32: Vec<u32> = axes
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !axes_u32.is_empty() {
                        inputs.insert(
                            "axes".to_string(),
                            Self::create_immediate_int_array(&axes_u32),
                        );
                    }
                }
            }

            "unsqueeze" => {
                // expand_dims: x, axes
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add axes parameter (required for unsqueeze)
                if let Some(axes) = op.attributes.get("axes").and_then(|v| v.as_array()) {
                    let axes_u32: Vec<u32> = axes
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !axes_u32.is_empty() {
                        inputs.insert(
                            "axes".to_string(),
                            Self::create_immediate_int_array(&axes_u32),
                        );
                    }
                }
            }

            "argMax" | "argMin" => {
                // reduce_argmax/reduce_argmin: x, axis, keep_dims
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add axis parameter (required)
                if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_u64()) {
                    inputs.insert("axis".to_string(), Self::create_immediate_int(axis as u32));
                }

                // Add keep_dims parameter (defaults to false)
                if let Some(keep_dims) = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                {
                    inputs.insert(
                        "keep_dims".to_string(),
                        Self::create_immediate_bool(keep_dims),
                    );
                }

                // Note: outputDataType is handled by the output tensor's data type
            }

            "cast" => {
                // cast: x, dtype
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add dtype parameter (required)
                // CoreML expects dtype as a string, not an integer
                if let Some(to_type) = op.attributes.get("to").and_then(|v| v.as_str()) {
                    let dtype_string = match to_type {
                        "float32" => "fp32",
                        "float16" => "fp16",
                        "int32" => "int32",
                        "uint32" => "uint32",
                        "int8" => "int8",
                        "uint8" => "uint8",
                        "int64" => "int64",
                        _ => "fp32", // default
                    };
                    inputs.insert(
                        "dtype".to_string(),
                        Self::create_immediate_string(dtype_string),
                    );
                }
            }

            "scatterElements" => {
                // scatter: data, indices, updates, axis
                if input_names.len() >= 3 {
                    inputs.insert("data".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                    inputs.insert(
                        "updates".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }

                // Add axis parameter
                if let Some(axis) = op.attributes.get("axis").and_then(|v| v.as_i64()) {
                    inputs.insert("axis".to_string(), Self::create_immediate_int(axis as u32));
                }
            }

            "scatterND" => {
                // scatter_nd: data, indices, updates
                if input_names.len() >= 3 {
                    inputs.insert("data".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                    inputs.insert(
                        "updates".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }
            }

            "tile" => {
                // tile: x, reps
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add repetitions parameter
                if let Some(reps) = op.attributes.get("repetitions").and_then(|v| v.as_array()) {
                    let reps_u32: Vec<u32> = reps
                        .iter()
                        .filter_map(|v| v.as_u64().map(|u| u as u32))
                        .collect();
                    if !reps_u32.is_empty() {
                        inputs.insert(
                            "reps".to_string(),
                            Self::create_immediate_int_array(&reps_u32),
                        );
                    }
                }
            }

            "triangular" => {
                // band_part: x, lower, upper
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // CoreML band_part uses lower and upper bounds instead of upper/diagonal
                // upper=true, diagonal=0 means: lower=-1 (all below diagonal), upper=0 (main diagonal and above)
                // upper=false, diagonal=0 means: lower=0 (main diagonal and below), upper=-1 (all above diagonal)
                let is_upper = op
                    .attributes
                    .get("upper")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let diagonal = op
                    .attributes
                    .get("diagonal")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);

                // Convert WebNN (upper, diagonal) to CoreML (lower, upper)
                // For upper triangle: keep diagonal and above
                // For lower triangle: keep diagonal and below
                let (lower_bound, upper_bound) = if is_upper {
                    // Upper triangle: remove elements below diagonal+k
                    (diagonal as i32, -1) // keep from diagonal+k upward
                } else {
                    // Lower triangle: remove elements above diagonal+k
                    (-1, diagonal as i32) // keep from diagonal+k downward
                };

                inputs.insert(
                    "lower".to_string(),
                    Self::create_immediate_int(lower_bound as u32),
                );
                inputs.insert(
                    "upper".to_string(),
                    Self::create_immediate_int(upper_bound as u32),
                );
            }

            // Reduction operations: reduceSum, reduceMean, reduceMax, etc.
            "reducesum" | "reducemean" | "reducemax" | "reducemin" | "reduceproduct"
            | "reducel1" | "reducel2" | "reducelogsum" | "reducelogsumexp" | "reducesumsquare" => {
                // All reduce operations: x, axes, keep_dims
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add axes parameter (optional - if not specified, reduces over all dimensions)
                if let Some(axes) = op.attributes.get("axes").and_then(|v| v.as_array()) {
                    let axes_i32: Vec<i32> = axes
                        .iter()
                        .filter_map(|v| v.as_i64().map(|i| i as i32))
                        .collect();
                    if !axes_i32.is_empty() {
                        // CoreML expects signed integers for axes
                        inputs.insert(
                            "axes".to_string(),
                            Self::create_immediate_int_array(
                                &axes_i32.iter().map(|&i| i as u32).collect::<Vec<u32>>(),
                            ),
                        );
                    }
                }

                // Add keep_dims parameter (required by CoreML, defaults to false per WebNN spec)
                let keep_dims = op
                    .attributes
                    .get("keepDimensions")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                inputs.insert(
                    "keep_dims".to_string(),
                    Self::create_immediate_bool(keep_dims),
                );
            }

            _ => {}
        }

        Ok(inputs)
    }

    /// Create a FeatureType for model description from an OperandDescriptor
    fn create_feature_type(
        descriptor: &crate::graph::OperandDescriptor,
    ) -> Result<crate::protos::coreml::specification::FeatureType, GraphError> {
        use crate::protos::coreml::specification::{ArrayFeatureType, FeatureType, feature_type};

        // Map WebNN data type to CoreML array data type
        // CoreML feature descriptions (I/O) ONLY support: DOUBLE, FLOAT32, FLOAT16, INT32
        // Even though Int8 exists in protobuf enum, CoreML runtime rejects it
        let array_data_type = match descriptor.data_type {
            DataType::Float32 => {
                crate::protos::coreml::specification::array_feature_type::ArrayDataType::Float32
            }
            DataType::Float16 => {
                crate::protos::coreml::specification::array_feature_type::ArrayDataType::Float16
            }
            DataType::Int32 => {
                crate::protos::coreml::specification::array_feature_type::ArrayDataType::Int32
            }
            // Unsupported types - CoreML feature descriptions only support DOUBLE, FLOAT32, FLOAT16, INT32
            // These must be skipped in tests
            DataType::Int8 | DataType::Uint8 | DataType::Uint32 | DataType::Int64 => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!(
                        "Unsupported feature data type: {:?}. CoreML feature descriptions only support DOUBLE, FLOAT32, FLOAT16, INT32.",
                        descriptor.data_type
                    ),
                });
            }
        };

        // Create array feature type with shape
        let mut array_feature = ArrayFeatureType {
            data_type: array_data_type as i32,
            ..Default::default()
        };

        // Add shape dimensions
        // CoreML requires explicit shape constraints - convert scalars (0D) to 1D [1]
        // Following Chromium's approach for scalar handling
        let shape_to_use = if descriptor.shape.is_empty() {
            vec![1] // Scalar (0D) tensor -> [1] for CoreML compatibility
        } else {
            descriptor.shape.clone()
        };

        for &dim in &shape_to_use {
            array_feature.shape.push(dim as i64);
        }

        Ok(FeatureType {
            r#type: Some(feature_type::Type::MultiArrayType(array_feature)),
            is_optional: false,
        })
    }
}

impl super::GraphConverter for CoremlMlProgramConverter {
    fn format(&self) -> &'static str {
        "coreml"
    }

    fn convert(&self, graph_info: &GraphInfo) -> Result<super::ConvertedGraph, GraphError> {
        // Create weight file builder for Float16 constants
        let mut weight_builder = super::WeightFileBuilder::new();

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

        // Add constant operands as const operations
        for (operand_id, constant_data) in &graph_info.constant_operand_ids_to_handles {
            let operand =
                graph_info
                    .operand(*operand_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Constant operand {} not found", operand_id),
                    })?;

            let const_op = Self::create_const_operation(
                graph_info,
                *operand_id,
                operand,
                constant_data,
                &mut weight_builder,
            )?;
            main_block.operations.push(const_op);
        }

        // First pass: Handle filter layout transformations for conv operations
        // Create a map of operand IDs to their transposed filter names
        let mut operand_name_overrides: HashMap<u32, String> = HashMap::new();

        for op in &graph_info.operations {
            let op_type_lower = op.op_type.to_lowercase();

            // Check if this is a convolution operation that needs filter transposition
            if (op_type_lower == "conv2d" || op_type_lower == "convtranspose2d")
                && op.input_operands.len() >= 2
            {
                if let Some(filter_layout) =
                    op.attributes.get("filterLayout").and_then(|v| v.as_str())
                {
                    let expected_layout = if op_type_lower == "conv2d" {
                        "oihw"
                    } else {
                        "iohw"
                    };

                    if filter_layout != expected_layout {
                        let filter_operand_id = op.input_operands[1];

                        if let Some(filter_operand) = graph_info.operand(filter_operand_id) {
                            // Calculate transpose permutation
                            let perm = match (op_type_lower.as_str(), filter_layout) {
                                // Conv2d conversions to oihw [O, I, H, W]
                                ("conv2d", "hwio") => vec![3, 2, 0, 1], // [H, W, I, O] -> [O, I, H, W]
                                ("conv2d", "ohwi") => vec![0, 3, 1, 2], // [O, H, W, I] -> [O, I, H, W]
                                ("conv2d", "ihwo") => vec![3, 0, 1, 2], // [I, H, W, O] -> [O, I, H, W]

                                // Conv_transpose2d conversions to iohw [I, O, H, W]
                                ("convtranspose2d", "hwoi") => vec![3, 2, 0, 1], // [H, W, O, I] -> [I, O, H, W]
                                ("convtranspose2d", "ohwi") => vec![3, 0, 1, 2], // [O, H, W, I] -> [I, O, H, W]
                                ("convtranspose2d", "hwio") => vec![2, 3, 0, 1], // [H, W, I, O] -> [I, O, H, W]

                                _ => continue, // Skip unsupported layouts
                            };

                            // Create transpose operation for filter
                            let filter_name = operand_name(graph_info, filter_operand_id);
                            let transposed_filter_name = format!("{}_transposed", filter_name);

                            // Store the override mapping
                            operand_name_overrides
                                .insert(filter_operand_id, transposed_filter_name.clone());

                            let mut transpose_inputs: HashMap<String, Argument> = HashMap::new();
                            transpose_inputs
                                .insert("x".to_string(), Self::create_name_argument(filter_name));
                            transpose_inputs.insert(
                                "perm".to_string(),
                                Self::create_immediate_int_array(
                                    &perm.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                                ),
                            );

                            // Calculate transposed shape
                            let original_shape = &filter_operand.descriptor.shape;
                            let transposed_shape: Vec<u32> =
                                perm.iter().map(|&i| original_shape[i as usize]).collect();

                            // Create tensor type for transposed filter
                            let dtype = Self::mil_data_type(&filter_operand.descriptor.data_type)?;
                            let dimensions: Vec<Dimension> = transposed_shape
                                .iter()
                                .map(|&d| Dimension {
                                    dimension: Some(dimension::Dimension::Constant(
                                        dimension::ConstantDimension { size: d as u64 },
                                    )),
                                })
                                .collect();

                            let value_type = ValueType {
                                r#type: Some(
                                    crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                        TensorType {
                                            rank: dimensions.len() as i64,
                                            data_type: dtype,
                                            dimensions,
                                            attributes: HashMap::new(),
                                        },
                                    ),
                                ),
                            };

                            let transpose_output_type = NamedValueType {
                                name: transposed_filter_name.clone(),
                                r#type: Some(value_type),
                            };

                            let transpose_op = Self::create_mil_operation(
                                "transpose",
                                transpose_inputs,
                                vec![transpose_output_type],
                            );

                            main_block.operations.push(transpose_op);
                        }
                    }
                }

                // Also check for nhwc input layout that needs transposition
                if let Some(input_layout) =
                    op.attributes.get("inputLayout").and_then(|v| v.as_str())
                {
                    if input_layout == "nhwc" && op.input_operands.len() >= 1 {
                        let input_operand_id = op.input_operands[0];

                        // Only transpose if not already transposed
                        if !operand_name_overrides.contains_key(&input_operand_id) {
                            if let Some(input_operand) = graph_info.operand(input_operand_id) {
                                // NHWC -> NCHW transposition: [0, 3, 1, 2]
                                let perm = vec![0, 3, 1, 2];

                                // Create transpose operation for input
                                let input_name = operand_name(graph_info, input_operand_id);
                                let transposed_input_name = format!("{}_nchw", input_name);

                                // Store the override mapping
                                operand_name_overrides
                                    .insert(input_operand_id, transposed_input_name.clone());

                                let mut transpose_inputs: HashMap<String, Argument> =
                                    HashMap::new();
                                transpose_inputs.insert(
                                    "x".to_string(),
                                    Self::create_name_argument(input_name),
                                );
                                transpose_inputs.insert(
                                    "perm".to_string(),
                                    Self::create_immediate_int_array(
                                        &perm.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                                    ),
                                );

                                // Calculate transposed shape
                                let original_shape = &input_operand.descriptor.shape;
                                let transposed_shape: Vec<u32> =
                                    perm.iter().map(|&i| original_shape[i as usize]).collect();

                                // Create tensor type for transposed input
                                let dtype =
                                    Self::mil_data_type(&input_operand.descriptor.data_type)?;
                                let dimensions: Vec<Dimension> = transposed_shape
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
                                            attributes: HashMap::new(),
                                        }),
                                    ),
                                };

                                let transpose_output_type = NamedValueType {
                                    name: transposed_input_name.clone(),
                                    r#type: Some(value_type),
                                };

                                let transpose_op = Self::create_mil_operation(
                                    "transpose",
                                    transpose_inputs,
                                    vec![transpose_output_type],
                                );

                                main_block.operations.push(transpose_op);
                            }
                        }
                    }
                }
            }
        }

        // Convert all operations to MIL operations
        for op in &graph_info.operations {
            // Special handling for expand operation (may need reshape first)
            if op.op_type.to_lowercase() == "expand" {
                // Check if rank-increasing expand (add reshape operation first)
                if !op.input_operands.is_empty() {
                    if let Some(input_operand) = graph_info.operand(op.input_operands[0]) {
                        if let Some(new_shape) =
                            op.attributes.get("newShape").and_then(|v| v.as_array())
                        {
                            let input_rank = input_operand.descriptor.shape.len();
                            let output_rank = new_shape.len();

                            if input_rank < output_rank {
                                // Need to add reshape operation first
                                // Create reshaped dimensions (right-aligned, padded with 1s on left)
                                let mut reshaped_dims = vec![1u32; output_rank];
                                for i in 0..input_rank {
                                    reshaped_dims[output_rank - i - 1] =
                                        input_operand.descriptor.shape[input_rank - i - 1];
                                }

                                //Create reshape operation
                                let input_name = operand_name(graph_info, op.input_operands[0]);
                                // Use input name to create unique intermediate name (don't rely on output_operands)
                                let reshape_output_name = format!("{}_expand_reshaped", input_name);

                                let mut reshape_inputs: HashMap<String, Argument> = HashMap::new();
                                reshape_inputs.insert(
                                    "x".to_string(),
                                    Self::create_name_argument(input_name),
                                );
                                reshape_inputs.insert(
                                    "shape".to_string(),
                                    Self::create_int_array_argument(
                                        reshaped_dims.iter().map(|&v| v as i32).collect(),
                                    ),
                                );

                                // Create tensor type for reshape output
                                let dtype =
                                    Self::mil_data_type(&input_operand.descriptor.data_type)?;
                                let dimensions: Vec<Dimension> = reshaped_dims
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
                                            attributes: HashMap::new(),
                                        }),
                                    ),
                                };

                                let reshape_output_type = NamedValueType {
                                    name: reshape_output_name.clone(),
                                    r#type: Some(value_type),
                                };

                                let reshape_mil_op = Self::create_mil_operation(
                                    "reshape",
                                    reshape_inputs,
                                    vec![reshape_output_type],
                                );

                                main_block.operations.push(reshape_mil_op);
                            }
                        }
                    }
                }
            }

            // Special handling for hardswish (decompose into hardsigmoid + mul)
            // Following Chromium: hardswish = x * hardsigmoid(x, alpha=1/6, beta=0.5)
            // Note: op_type is "hardSwish" but we normalize to lowercase
            let op_type_lower = op.op_type.to_lowercase();
            if op_type_lower == "hardswish" {
                // Validate inputs/outputs exist
                // Note: hardswish uses output_operand (singular), not output_operands
                if op.input_operands.is_empty() || op.output_operand.is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "hardswish requires input and output operand".to_string(),
                    });
                }

                let input_operand = graph_info.operand(op.input_operands[0]).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Input operand {} not found", op.input_operands[0]),
                    }
                })?;
                {
                    let input_name = operand_name(graph_info, op.input_operands[0]);
                    let hardsigmoid_output_name = format!("{}_hardswish_hardsigmoid", input_name);

                    // Create hardsigmoid operation with alpha=1/6, beta=0.5
                    let mut hardsigmoid_inputs: HashMap<String, Argument> = HashMap::new();
                    hardsigmoid_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(input_name.clone()),
                    );
                    hardsigmoid_inputs
                        .insert("alpha".to_string(), Self::create_immediate_float(1.0 / 6.0));
                    hardsigmoid_inputs
                        .insert("beta".to_string(), Self::create_immediate_float(0.5));

                    // Create tensor type for hardsigmoid output
                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                    let dimensions: Vec<Dimension> = input_operand
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
                            crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                TensorType {
                                    rank: dimensions.len() as i64,
                                    data_type: dtype,
                                    dimensions,
                                    attributes: HashMap::new(),
                                },
                            ),
                        ),
                    };

                    let hardsigmoid_output_type = NamedValueType {
                        name: hardsigmoid_output_name.clone(),
                        r#type: Some(value_type),
                    };

                    let hardsigmoid_op = Self::create_mil_operation(
                        "sigmoid_hard",
                        hardsigmoid_inputs,
                        vec![hardsigmoid_output_type],
                    );

                    main_block.operations.push(hardsigmoid_op);

                    // Create mul operation: x * hardsigmoid_output
                    let mut mul_inputs: HashMap<String, Argument> = HashMap::new();
                    mul_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    mul_inputs.insert(
                        "y".to_string(),
                        Self::create_name_argument(hardsigmoid_output_name),
                    );

                    // Get output name (using singular output_operand field)
                    let output_operand_id = op.output_operand.unwrap();
                    let output_name = operand_name(graph_info, output_operand_id);
                    let output_operand =
                        graph_info.operand(output_operand_id).ok_or_else(|| {
                            GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("Output operand {} not found", output_operand_id),
                            }
                        })?;

                    let output_dtype = Self::mil_data_type(&output_operand.descriptor.data_type)?;
                    let output_dimensions: Vec<Dimension> = output_operand
                        .descriptor
                        .shape
                        .iter()
                        .map(|&d| Dimension {
                            dimension: Some(dimension::Dimension::Constant(
                                dimension::ConstantDimension { size: d as u64 },
                            )),
                        })
                        .collect();

                    let output_value_type = ValueType {
                        r#type: Some(
                            crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                TensorType {
                                    rank: output_dimensions.len() as i64,
                                    data_type: output_dtype,
                                    dimensions: output_dimensions,
                                    attributes: HashMap::new(),
                                },
                            ),
                        ),
                    };

                    let mul_output_type = NamedValueType {
                        name: output_name,
                        r#type: Some(output_value_type),
                    };

                    let mul_op =
                        Self::create_mil_operation("mul", mul_inputs, vec![mul_output_type]);

                    main_block.operations.push(mul_op);
                }

                // Skip normal operation conversion for hardswish
                continue;
            }

            // Special handling for neg (decompose into mul(x, -1) with typed constant)
            // Following Chromium: neg = mul(x, -1) with constant matching input dtype
            if op_type_lower == "neg" {
                // Validate inputs/outputs exist
                if op.input_operands.is_empty() || op.output_operand.is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "neg requires input and output operand".to_string(),
                    });
                }

                let input_operand = graph_info.operand(op.input_operands[0]).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Input operand {} not found", op.input_operands[0]),
                    }
                })?;

                let input_name = operand_name(graph_info, op.input_operands[0]);

                // Create typed -1 constant matching input dtype
                let neg_one_immediate = match input_operand.descriptor.data_type {
                    DataType::Float32 => Self::create_immediate_float(-1.0f32),
                    DataType::Float16 => Self::create_immediate_float16(-1.0f32),
                    DataType::Int32 => {
                        // create_immediate_int accepts u32 but converts to i32 internally
                        // We need to reimplement for -1 value
                        use crate::protos::coreml::mil_spec::{
                            DataType as MilDataType, TensorType, TensorValue, Value, ValueType,
                            argument, tensor_value, value, value_type,
                        };

                        let tensor_value = TensorValue {
                            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                                values: vec![-1i32],
                            })),
                        };

                        let val = Value {
                            doc_string: String::new(),
                            r#type: Some(ValueType {
                                r#type: Some(value_type::Type::TensorType(TensorType {
                                    data_type: MilDataType::Int32 as i32,
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
                            arguments: vec![argument::Binding {
                                binding: Some(argument::binding::Binding::Value(val)),
                            }],
                        }
                    }
                    _ => {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "Unsupported data type for neg: {:?}",
                                input_operand.descriptor.data_type
                            ),
                        });
                    }
                };

                // Create mul operation: x * (-1)
                let mut mul_inputs: HashMap<String, Argument> = HashMap::new();
                mul_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                mul_inputs.insert("y".to_string(), neg_one_immediate);

                // Get output name
                let output_operand_id = op.output_operand.unwrap();
                let output_name = operand_name(graph_info, output_operand_id);
                let output_operand = graph_info.operand(output_operand_id).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Output operand {} not found", output_operand_id),
                    }
                })?;

                let output_dtype = Self::mil_data_type(&output_operand.descriptor.data_type)?;
                let output_dimensions: Vec<Dimension> = output_operand
                    .descriptor
                    .shape
                    .iter()
                    .map(|&d| Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension { size: d as u64 },
                        )),
                    })
                    .collect();

                let output_value_type = ValueType {
                    r#type: Some(
                        crate::protos::coreml::mil_spec::value_type::Type::TensorType(TensorType {
                            rank: output_dimensions.len() as i64,
                            data_type: output_dtype,
                            dimensions: output_dimensions,
                            attributes: HashMap::new(),
                        }),
                    ),
                };

                let mul_output_type = NamedValueType {
                    name: output_name,
                    r#type: Some(output_value_type),
                };

                let mul_op = Self::create_mil_operation("mul", mul_inputs, vec![mul_output_type]);

                main_block.operations.push(mul_op);

                // Skip normal operation conversion for neg
                continue;
            }

            let mil_op =
                self.convert_operation_with_overrides(graph_info, op, &operand_name_overrides)?;
            main_block.operations.push(mil_op);
        }

        // Add block outputs (output operand names)
        for &output_id in &graph_info.output_operands {
            let output_name = operand_name(graph_info, output_id);
            main_block.outputs.push(output_name);
        }

        // Add block to function
        main_function.opset = "CoreML7".to_string(); // Specify the active block specialization
        main_function
            .block_specializations
            .insert("CoreML7".to_string(), main_block);

        // Add function to program
        program.functions.insert("main".to_string(), main_function);

        // Create Model
        let mut model = Model {
            specification_version: 9, // CoreML 9 (iOS 18+, macOS 15+) - required for empty inputs
            ..Default::default()
        };

        // Create ModelDescription with function descriptions
        use crate::protos::coreml::specification::{
            FeatureDescription, FunctionDescription, ModelDescription,
        };

        let mut function_desc = FunctionDescription {
            name: "main".to_string(),
            ..Default::default()
        };

        // Add input descriptions
        for &input_id in &graph_info.input_operands {
            if let Some(operand) = graph_info.operand(input_id) {
                let input_name = operand_name(graph_info, input_id);
                function_desc.input.push(FeatureDescription {
                    name: input_name,
                    r#type: Some(Self::create_feature_type(&operand.descriptor)?),
                    ..Default::default()
                });
            }
        }

        // Add output descriptions
        for &output_id in &graph_info.output_operands {
            if let Some(operand) = graph_info.operand(output_id) {
                let output_name = operand_name(graph_info, output_id);
                function_desc.output.push(FeatureDescription {
                    name: output_name,
                    r#type: Some(Self::create_feature_type(&operand.descriptor)?),
                    ..Default::default()
                });
            }
        }

        model.description = Some(ModelDescription {
            functions: vec![function_desc],
            default_function_name: "main".to_string(),
            ..Default::default()
        });

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

        // Finalize weight file if any weights were added
        let weights_data = if weight_builder.has_weights() {
            Some(weight_builder.finalize())
        } else {
            None
        };

        Ok(super::ConvertedGraph {
            format: "coreml",
            content_type: "application/x-coreml-model",
            data: buffer,
            weights_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::GraphConverter;
    use crate::graph::{
        ConstantData, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation,
    };
    use std::collections::HashMap;

    /// Helper to create a simple graph with a Float16 constant
    fn create_graph_with_float16_constant(shape: Vec<u32>, data: Vec<u8>) -> GraphInfo {
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![1], // Output is operand 1
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
        };

        // Operand 0: Float16 constant
        graph.operands.push(Operand {
            name: Some("constant".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: shape.clone(),
                pending_permutation: vec![],
            },
        });

        // Operand 1: Output (relu result)
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape,
                pending_permutation: vec![],
            },
        });

        // Add constant data
        graph
            .constant_operand_ids_to_handles
            .insert(0, ConstantData { data, label: None });

        // Add a simple relu operation
        graph.operations.push(Operation {
            op_type: "relu".to_string(),
            input_operands: vec![0],
            output_operand: Some(1),
            output_operands: vec![],
            attributes: serde_json::Value::Null,
            label: None,
        });

        graph
    }

    #[test]
    fn test_float16_scalar_constant_uses_immediate_value() {
        // Create a scalar Float16 constant (shape = [])
        let f16_val = half::f16::from_f32(1.5);
        let data = f16_val.to_le_bytes().to_vec();

        let graph = create_graph_with_float16_constant(vec![], data.clone());

        // Convert the graph
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify no weights_data (scalar uses immediate value)
        assert!(
            result.weights_data.is_none(),
            "Scalar Float16 should not use weight file"
        );

        // Verify the model data is valid protobuf
        assert!(!result.data.is_empty(), "Model data should not be empty");
    }

    #[test]
    fn test_float16_1d_constant_uses_weight_file() {
        // Create a 1D Float16 constant [3] - non-scalar
        let data = vec![
            0x00, 0x3C, // f16: 1.0
            0x00, 0x40, // f16: 2.0
            0x00, 0x42, // f16: 3.0
        ];

        let graph = create_graph_with_float16_constant(vec![3], data.clone());

        // Convert the graph
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify weights_data is present
        assert!(
            result.weights_data.is_some(),
            "Non-scalar Float16 should use weight file"
        );

        let weights = result.weights_data.unwrap();

        // Verify weight file structure
        // Expected structure:
        // [0-3]: sentinel (0xDEADBEEF)
        // [4-11]: count (3)
        // [12-17]: data (6 bytes)
        // [18-63]: padding (46 bytes)
        assert_eq!(weights.len(), 64, "Weight file should be 64-byte aligned");

        // Verify sentinel
        let sentinel = u32::from_le_bytes([weights[0], weights[1], weights[2], weights[3]]);
        assert_eq!(sentinel, 0xDEADBEEF, "Sentinel should be 0xDEADBEEF");

        // Verify count
        let count = u64::from_le_bytes([
            weights[4],
            weights[5],
            weights[6],
            weights[7],
            weights[8],
            weights[9],
            weights[10],
            weights[11],
        ]);
        assert_eq!(count, 3, "Element count should be 3");

        // Verify data
        assert_eq!(
            &weights[12..18],
            &data[..],
            "Weight data should match input"
        );
    }

    #[test]
    fn test_float16_2d_constant_uses_weight_file() {
        // Create a 2D Float16 constant [2, 2]
        let data = vec![
            0x00, 0x3C, // f16: 1.0
            0x00, 0x40, // f16: 2.0
            0x00, 0x42, // f16: 3.0
            0x00, 0x44, // f16: 4.0
        ];

        let graph = create_graph_with_float16_constant(vec![2, 2], data.clone());

        // Convert the graph
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify weights_data is present
        assert!(
            result.weights_data.is_some(),
            "2D Float16 constant should use weight file"
        );

        let weights = result.weights_data.unwrap();

        // Verify count matches 2x2 = 4 elements
        let count = u64::from_le_bytes([
            weights[4],
            weights[5],
            weights[6],
            weights[7],
            weights[8],
            weights[9],
            weights[10],
            weights[11],
        ]);
        assert_eq!(count, 4, "Element count should be 4");
    }

    #[test]
    fn test_multiple_float16_constants_in_weight_file() {
        // Create a graph with TWO Float16 constants
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![2],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
        };

        // Operand 0: First Float16 constant [2]
        let data1 = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0
        graph.operands.push(Operand {
            name: Some("constant1".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: vec![2],
                pending_permutation: vec![],
            },
        });
        graph.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: data1,
                label: None,
            },
        );

        // Operand 1: Second Float16 constant [2]
        let data2 = vec![0x00, 0x42, 0x00, 0x44]; // 3.0, 4.0
        graph.operands.push(Operand {
            name: Some("constant2".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: vec![2],
                pending_permutation: vec![],
            },
        });
        graph.constant_operand_ids_to_handles.insert(
            1,
            ConstantData {
                data: data2,
                label: None,
            },
        );

        // Operand 2: Output
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: vec![2],
                pending_permutation: vec![],
            },
        });

        // Add operation: output = constant1 + constant2
        graph.operations.push(Operation {
            op_type: "add".to_string(),
            input_operands: vec![0, 1],
            output_operand: Some(2),
            output_operands: vec![],
            attributes: serde_json::Value::Null,
            label: None,
        });

        // Convert
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify weights_data is present
        assert!(
            result.weights_data.is_some(),
            "Multiple Float16 constants should use weight file"
        );

        let weights = result.weights_data.unwrap();

        // Should have two entries:
        // Entry 1: offset 0, 64 bytes
        // Entry 2: offset 64, 64 bytes
        // Total: 128 bytes
        assert_eq!(
            weights.len(),
            128,
            "Two Float16 constants should result in 128-byte weight file"
        );

        // Verify first entry sentinel at offset 0
        let sentinel1 = u32::from_le_bytes([weights[0], weights[1], weights[2], weights[3]]);
        assert_eq!(sentinel1, 0xDEADBEEF, "First entry sentinel");

        // Verify second entry sentinel at offset 64
        let sentinel2 = u32::from_le_bytes([weights[64], weights[65], weights[66], weights[67]]);
        assert_eq!(sentinel2, 0xDEADBEEF, "Second entry sentinel");
    }

    #[test]
    fn test_float32_constant_no_weight_file() {
        // Create a graph with Float32 constant (should NOT use weight file)
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
        };

        // Float32 constant
        let data = vec![0x00, 0x00, 0x80, 0x3F]; // 1.0 as f32
        graph.operands.push(Operand {
            name: Some("constant".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![1],
                pending_permutation: vec![],
            },
        });
        graph
            .constant_operand_ids_to_handles
            .insert(0, ConstantData { data, label: None });

        // Output
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![1],
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(Operation {
            op_type: "relu".to_string(),
            input_operands: vec![0],
            output_operand: Some(1),
            output_operands: vec![],
            attributes: serde_json::Value::Null,
            label: None,
        });

        // Convert
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify NO weights_data (Float32 uses immediate values)
        assert!(
            result.weights_data.is_none(),
            "Float32 constants should not use weight file"
        );
    }
}
