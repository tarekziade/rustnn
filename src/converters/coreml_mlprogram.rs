/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
use crate::graph::{DataType, Dimension as GraphDimension, GraphInfo, OperandKind};
use crate::operator_enums::MLOperandDataType;
use crate::operator_options::MLDimension;
use crate::operators::Operation;
use crate::protos::coreml::mil_spec::{
    Argument, Block, Dimension, Function, NamedValueType, Operation as MilOperation, Program,
    TensorType, ValueType, argument::binding::Binding, dimension,
};
use crate::protos::coreml::specification::Model;
use prost::Message;
use std::collections::HashMap;

/// Convert zero_point byte data from a source dtype to a target dtype.
/// Only Int32 → Uint8 and Int32 → Int8 are supported; all other pairs are returned as-is.
/// Values are clamped to the target range to avoid silent corruption.
fn convert_zp_bytes(src: &[u8], src_dtype: &DataType, tgt_dtype: &DataType) -> Vec<u8> {
    match (src_dtype, tgt_dtype) {
        (DataType::Int32, DataType::Uint8) => {
            let count = src.len() / 4;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let v = i32::from_le_bytes([
                    src[i * 4],
                    src[i * 4 + 1],
                    src[i * 4 + 2],
                    src[i * 4 + 3],
                ]);
                out.push(v.clamp(0, 255) as u8);
            }
            out
        }
        (DataType::Int32, DataType::Int8) => {
            let count = src.len() / 4;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let v = i32::from_le_bytes([
                    src[i * 4],
                    src[i * 4 + 1],
                    src[i * 4 + 2],
                    src[i * 4 + 3],
                ]);
                out.push(v.clamp(-128, 127) as i8 as u8);
            }
            out
        }
        // Uint32 → Int32: same 4 bytes, just reinterpreted as signed.
        (DataType::Uint32, DataType::Int32) => src.to_vec(),
        // Int64 → Int32: take the lower 4 bytes of each 8-byte element.
        (DataType::Int64, DataType::Int32) => {
            let count = src.len() / 8;
            let mut out = Vec::with_capacity(count * 4);
            for i in 0..count {
                out.extend_from_slice(&src[i * 8..i * 8 + 4]);
            }
            out
        }
        // Float32 <-> Float16: quantize/dequantize scale constants whose float type
        // doesn't match the float tensor ("input and scale must have the same data type").
        (DataType::Float32, DataType::Float16) => {
            let count = src.len() / 4;
            let mut out = Vec::with_capacity(count * 2);
            for i in 0..count {
                let v = f32::from_le_bytes([
                    src[i * 4],
                    src[i * 4 + 1],
                    src[i * 4 + 2],
                    src[i * 4 + 3],
                ]);
                out.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
            }
            out
        }
        (DataType::Float16, DataType::Float32) => {
            let count = src.len() / 2;
            let mut out = Vec::with_capacity(count * 4);
            for i in 0..count {
                let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                out.extend_from_slice(&half::f16::from_bits(bits).to_f32().to_le_bytes());
            }
            out
        }
        _ => src.to_vec(),
    }
}

/// MIL operation type names (matching Chromium's implementation)
mod mil_ops {
    // Binary operations
    pub const ADD: &str = "add";
    pub const SUB: &str = "sub";
    pub const MUL: &str = "mul";
    pub const DIV: &str = "real_div";
    pub const POW: &str = "pow";
    /// Element-wise maximum (WebNN max).
    pub const MAXIMUM: &str = "maximum";
    /// Element-wise minimum (WebNN min).
    pub const MINIMUM: &str = "minimum";
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
    pub const L2_POOL: &str = "l2_pool";
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
    pub const ROUND_EVEN: &str = "round"; // WebNN roundEven: round to nearest even (MIL "round")
    pub const NEG: &str = "mul"; // Multiply by -1
    pub const IDENTITY: &str = "identity";
    pub const EXP: &str = "exp";
    pub const LOG: &str = "log";
    pub const SQRT: &str = "sqrt";
    pub const SIGN: &str = "sign";
    pub const SIN: &str = "sin";
    pub const COS: &str = "cos";
    pub const TAN: &str = "tan";
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
    pub const SLICE_BY_INDEX: &str = "slice_by_index";
    pub const EXPAND: &str = "tile";
    pub const GATHER: &str = "gather";
    pub const GATHER_ALONG_AXIS: &str = "gather_along_axis";
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
    // WebNN scatterElements (indices/updates share the data's rank, scattered along
    // one axis) maps to MIL `scatter_along_axis`; MIL `scatter` expects rank-1 indices.
    pub const SCATTER_ELEMENTS: &str = "scatter_along_axis";
    pub const SCATTER_ND: &str = "scatter_nd";

    // Tile operation
    pub const TILE: &str = "tile";
    pub const REVERSE: &str = "reverse";
    pub const CUM_SUM: &str = "cumsum";

    // Triangular operation
    pub const TRIANGULAR: &str = "band_part";

    // Clamp operation
    pub const CLIP: &str = "clip";

    // NaN and infinity detection operations
    pub const IS_NAN: &str = "is_nan";
    pub const IS_INF: &str = "is_inf";

    // Gather N-dimensional
    pub const GATHER_ND: &str = "gather_nd";

    // Upsample/resample operations
    pub const UPSAMPLE_NEAREST_NEIGHBOR: &str = "upsample_nearest_neighbor";
    pub const UPSAMPLE_BILINEAR: &str = "upsample_bilinear";
}

// Default epsilon value used by several CoreML operations for numerical stability.
const DEFAULT_EPSILON: f32 = 1e-45;

#[derive(Default)]
pub struct CoremlMlProgramConverter;

impl CoremlMlProgramConverter {
    /// Parse MLNumber values represented as JSON numbers or strings.
    /// Supports non-finite strings used by WPT/interchange JSON.
    fn parse_mlnumber_f64(value: Option<&serde_json::Value>) -> Option<f64> {
        let v = value?;
        if let Some(n) = v.as_f64() {
            return Some(n);
        }
        let s = v.as_str()?.trim().to_ascii_lowercase();
        match s.as_str() {
            "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
            "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
            "nan" => Some(f64::NAN),
            _ => s.parse::<f64>().ok(),
        }
    }

    /// Parse clamp bounds from MLNumber. NaN is treated as "missing bound".
    fn parse_clamp_bound(value: Option<&serde_json::Value>, default: f64) -> f64 {
        Self::parse_mlnumber_f64(value)
            .filter(|v| !v.is_nan())
            .unwrap_or(default)
    }

    fn mil_dimension_from_graph_dim(dim: &GraphDimension) -> Dimension {
        match dim {
            GraphDimension::Static(v) => Dimension {
                dimension: Some(dimension::Dimension::Constant(
                    dimension::ConstantDimension { size: *v as u64 },
                )),
            },
            GraphDimension::Dynamic(_) => Dimension {
                dimension: Some(dimension::Dimension::Unknown(dimension::UnknownDimension {
                    variadic: false,
                })),
            },
        }
    }

    fn mil_dimensions_from_graph_shape(
        shape: &[GraphDimension],
        scalar_as_one_dim: bool,
    ) -> Vec<Dimension> {
        if shape.is_empty() && scalar_as_one_dim {
            return vec![Dimension {
                dimension: Some(dimension::Dimension::Constant(
                    dimension::ConstantDimension { size: 1 },
                )),
            }];
        }
        shape
            .iter()
            .map(Self::mil_dimension_from_graph_dim)
            .collect()
    }

    fn permute_graph_shape(shape: &[GraphDimension], perm: &[u32]) -> Vec<GraphDimension> {
        perm.iter().map(|&i| shape[i as usize].clone()).collect()
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

        let name = operand_name(graph, operand_id);

        let dtype = Self::mil_data_type(&operand.descriptor.data_type)?;
        let value_type =
            Self::create_named_value_type(name.clone(), dtype, &operand.descriptor.shape, true);

        Ok((name, value_type))
    }

    fn create_named_value_type(
        name: String,
        data_type: i32,
        shape: &[GraphDimension],
        scalar_as_one_dim: bool,
    ) -> NamedValueType {
        let dimensions = Self::mil_dimensions_from_graph_shape(shape, scalar_as_one_dim);

        let value_type = ValueType {
            r#type: Some(
                crate::protos::coreml::mil_spec::value_type::Type::TensorType(TensorType {
                    rank: dimensions.len() as i64,
                    data_type,
                    dimensions,
                    attributes: HashMap::new(),
                }),
            ),
        };

        NamedValueType {
            name,
            r#type: Some(value_type),
        }
    }

    fn create_value_with_mil_type(
        graph: &GraphInfo,
        operand_id: u32,
        name: String,
        data_type: i32,
    ) -> Result<NamedValueType, GraphError> {
        let operand = graph
            .operand(operand_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("Operand {} not found", operand_id),
            })?;

        Ok(Self::create_named_value_type(
            name,
            data_type,
            &operand.descriptor.shape,
            true,
        ))
    }

    fn output_name_for_operand(
        graph: &GraphInfo,
        operand_id: u32,
        operand_name_overrides: &HashMap<u32, String>,
    ) -> String {
        operand_name_overrides
            .get(&operand_id)
            .cloned()
            .unwrap_or_else(|| operand_name(graph, operand_id))
    }

    fn create_output_value(
        graph: &GraphInfo,
        operand_id: u32,
        operand_name_overrides: &HashMap<u32, String>,
    ) -> Result<(String, NamedValueType), GraphError> {
        let name = Self::output_name_for_operand(graph, operand_id, operand_name_overrides);
        let value_type = Self::create_value_with_mil_type(
            graph,
            operand_id,
            name.clone(),
            Self::graph_value_mil_type(
                &graph
                    .operand(operand_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Operand {} not found", operand_id),
                    })?
                    .descriptor
                    .data_type,
            )?,
        )?;
        Ok((name, value_type))
    }

    fn interface_mil_data_type(data_type: &DataType) -> i32 {
        use crate::protos::coreml::mil_spec::DataType as MilDataType;

        match data_type {
            DataType::Float32 => MilDataType::Float32 as i32,
            DataType::Float16 => MilDataType::Float16 as i32,
            DataType::Int32 => MilDataType::Int32 as i32,
            DataType::Int4
            | DataType::Uint4
            | DataType::Int8
            | DataType::Uint8
            | DataType::Uint32
            | DataType::Int64
            | DataType::Uint64 => MilDataType::Float32 as i32,
        }
    }

    fn cast_dtype_string_for_mil_type(data_type: i32) -> Result<&'static str, GraphError> {
        use crate::protos::coreml::mil_spec::DataType as MilDataType;

        match data_type {
            value if value == MilDataType::Float32 as i32 => Ok("fp32"),
            value if value == MilDataType::Float16 as i32 => Ok("fp16"),
            value if value == MilDataType::Int32 as i32 => Ok("int32"),
            value if value == MilDataType::Bool as i32 => Ok("bool"),
            _ => Err(GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("Unsupported MIL cast dtype {}", data_type),
            }),
        }
    }

    fn cast_dtype_string_for_graph_type(data_type: &DataType) -> Result<&'static str, GraphError> {
        match data_type {
            DataType::Float32 => Ok("fp32"),
            DataType::Float16 => Ok("fp16"),
            DataType::Int32 => Ok("int32"),
            DataType::Uint32 => Ok("uint32"),
            DataType::Int8 => Ok("int8"),
            DataType::Uint8 => Ok("uint8"),
            DataType::Int64 => Ok("int64"),
            DataType::Int4 | DataType::Uint4 | DataType::Uint64 => {
                Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("Unsupported graph cast dtype {:?}", data_type),
                })
            }
        }
    }

    /// MIL type used for an operand's value *inside* the graph. CoreML MIL has no
    /// int4/uint4/int64/uint32/uint64 tensor type, so those are represented as int32 (a
    /// proxy); the model interface and executor reconcile width/sign/packing at the boundary.
    fn graph_value_mil_type(data_type: &DataType) -> Result<i32, GraphError> {
        if Self::is_wide_int(data_type) {
            Ok(crate::protos::coreml::mil_spec::DataType::Int32 as i32)
        } else {
            Self::mil_data_type(data_type)
        }
    }

    /// Whether a WebNN type has no native MIL tensor representation and is proxied
    /// through int32 inside the graph (int4/uint4 sub-byte, and int64/uint32/uint64).
    fn is_wide_int(data_type: &DataType) -> bool {
        matches!(
            data_type,
            DataType::Int4
                | DataType::Uint4
                | DataType::Uint32
                | DataType::Int64
                | DataType::Uint64
        )
    }

    /// Whether an integer op that runs through fp32 can round-trip this input type
    /// back to a MIL-representable integer (int8/uint8/int32 natively, or int32 for
    /// the wide-int proxies).
    fn is_castable_int(data_type: &DataType) -> bool {
        matches!(
            data_type,
            DataType::Int8 | DataType::Uint8 | DataType::Int32
        ) || Self::is_wide_int(data_type)
    }

    /// The MIL cast dtype string used to convert an fp32 intermediate back to an
    /// operand's internal integer representation (int32 for the wide-int proxies).
    fn int_back_cast_dtype(data_type: &DataType) -> Result<&'static str, GraphError> {
        if Self::is_wide_int(data_type) {
            Ok("int32")
        } else {
            Self::cast_dtype_string_for_graph_type(data_type)
        }
    }

    /// Convert WebNN DataType to MIL DataType
    fn mil_data_type(data_type: &DataType) -> Result<i32, GraphError> {
        use crate::protos::coreml::mil_spec::DataType as MilDataType;

        Ok(match data_type {
            DataType::Int4 | DataType::Uint4 => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml".to_string(),
                    reason: "int4/uint4 tensors are not supported in CoreML conversion yet"
                        .to_string(),
                });
            }
            DataType::Float32 => MilDataType::Float32 as i32,
            DataType::Float16 => MilDataType::Float16 as i32,
            DataType::Int32 => MilDataType::Int32 as i32,
            DataType::Int8 => MilDataType::Int8 as i32,
            DataType::Uint32 => MilDataType::Uint32 as i32,
            DataType::Uint8 => MilDataType::Uint8 as i32,
            DataType::Int64 => MilDataType::Int64 as i32,
            DataType::Uint64 => MilDataType::Uint64 as i32,
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

        let name = operand_name(graph, operand_id);
        // int64/uint32/uint64 constants have no MIL tensor type; emit them as int32.
        let dtype = Self::graph_value_mil_type(&operand.descriptor.data_type)?;
        // Keep WebNN scalar constants at rank 0. Promoting them to [1] breaks
        // MIL ops such as `quantize` that distinguish scalars from vectors.
        let output_type =
            Self::create_named_value_type(name, dtype, &operand.descriptor.shape, false);

        // Non-scalar weight-carrying constants go into the blob weight file
        // (BlobFileValue), never as immediate proto values: CoreML's on-device
        // compiler re-serializes immediate constants through its textual MIL
        // writer (MIL::Text::BasicSerializer), which takes minutes and
        // gigabytes for large weights. Mirrors Chromium's
        // graph_builder_coreml.cc, which blob-writes all non-scalar weights.
        let blob_type = match operand.descriptor.data_type {
            crate::graph::DataType::Float16 => {
                Some(super::weight_file_builder::blob_data_type::FLOAT16)
            }
            crate::graph::DataType::Float32 => {
                Some(super::weight_file_builder::blob_data_type::FLOAT32)
            }
            crate::graph::DataType::Uint8 => {
                Some(super::weight_file_builder::blob_data_type::UINT8)
            }
            crate::graph::DataType::Int8 => Some(super::weight_file_builder::blob_data_type::INT8),
            _ => None,
        };
        if let Some(mil_data_type) = blob_type.filter(|_| !operand.descriptor.shape.is_empty()) {
            let offset =
                weight_builder.add_weight(operand_id, mil_data_type, &constant_data.data)?;
            let blob_file_value = Value {
                doc_string: String::new(),
                r#type: output_type.r#type.clone(),
                value: Some(value::Value::BlobFileValue(value::BlobFileValue {
                    file_name: "@model_path/weights/weights.bin".to_string(),
                    offset,
                })),
            };
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
            // Non-scalar Float16 went to the weight file above; only scalar
            // (0D) Float16 can be stored as immediate bytes.
            crate::graph::DataType::Float16 => TensorValue {
                value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                    values: constant_data.data.clone().into(),
                })),
            },
            crate::graph::DataType::Int8 | crate::graph::DataType::Uint8 => TensorValue {
                value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                    values: constant_data.data.clone().into(),
                })),
            },
            crate::graph::DataType::Int64 => {
                // int64 has no MIL tensor type; emit as int32 values (narrowing).
                let values: Vec<i32> = constant_data
                    .data
                    .chunks_exact(8)
                    .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()) as i32)
                    .collect();
                TensorValue {
                    value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                        values,
                    })),
                }
            }
            crate::graph::DataType::Uint32 => {
                // uint32 has no MIL tensor type; emit as int32 (bit-preserving).
                let values: Vec<i32> = constant_data
                    .data
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) as i32)
                    .collect();
                TensorValue {
                    value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                        values,
                    })),
                }
            }
            crate::graph::DataType::Uint64 => {
                // uint64 has no MIL tensor type; emit as int32 (narrowing).
                let values: Vec<i32> = constant_data
                    .data
                    .chunks_exact(8)
                    .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as i32)
                    .collect();
                TensorValue {
                    value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                        values,
                    })),
                }
            }
            crate::graph::DataType::Int4 => {
                // int4 is packed two values per byte; unpack (sign-extended) to int32.
                let count: usize = operand
                    .descriptor
                    .static_or_max_shape()
                    .iter()
                    .map(|&d| d as usize)
                    .product();
                let values = crate::graph::unpack_int4(&constant_data.data, count);
                TensorValue {
                    value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                        values,
                    })),
                }
            }
            crate::graph::DataType::Uint4 => {
                // uint4 is packed two values per byte; unpack to int32.
                let count: usize = operand
                    .descriptor
                    .static_or_max_shape()
                    .iter()
                    .map(|&d| d as usize)
                    .product();
                let values: Vec<i32> = crate::graph::unpack_uint4(&constant_data.data, count)
                    .into_iter()
                    .map(|v| v as i32)
                    .collect();
                TensorValue {
                    value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                        values,
                    })),
                }
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

    /// Like `create_immediate_float` but produces a rank-1 `tensor<fp32,[1]>`
    /// rather than a rank-0 scalar. MIL ops like `mul` preserve rank rather
    /// than broadcasting scalars up, so when an operation's declared output
    /// rank is 1 and one of its inputs is the scalar -1 (as in the `neg`
    /// lowering), we need the constant to have matching rank.
    fn create_immediate_float_1d(value: f32) -> Argument {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, dimension,
            tensor_value, value, value_type,
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
                    rank: 1,
                    dimensions: vec![Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension { size: 1 },
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
        if matches!(&op, Operation::Split { .. }) {
            return self.convert_split_operation(graph, op);
        }

        let mil_op_type = self.get_mil_op_type(op.op_type())?;

        // slice with non-unit strides must use slice_by_index instead of slice_by_size.
        let mil_op_type = if let Operation::Slice { options, .. } = op {
            let strides = options
                .as_ref()
                .map(|o| o.strides.as_slice())
                .unwrap_or(&[]);
            if strides.iter().any(|&s| s != 1) && !strides.is_empty() {
                mil_ops::SLICE_BY_INDEX
            } else {
                mil_op_type
            }
        } else {
            mil_op_type
        };

        // Get input operand names, using overrides if available
        let input_names = Self::input_names_for_operation(graph, op, operand_name_overrides);

        // Get output operand info
        // Check if this is a single-output or multi-output operation
        let output_id = if let Some(id) = op.output_operand() {
            // Single-output operation
            id
        } else if !op.output_operands().is_empty() {
            // Multi-output operation not handled yet
            return Err(GraphError::ConversionFailed {
                format: "CoreML MLProgram".to_string(),
                reason: format!(
                    "operation '{}' has multiple outputs but is not implemented as multi-output. \
                     Only 'split' is currently supported as multi-output.",
                    op.op_type()
                ),
            });
        } else {
            // No outputs at all - this shouldn't happen but handle gracefully
            return Err(GraphError::ConversionFailed {
                format: "CoreML MLProgram".to_string(),
                reason: format!("operation '{}' has no output operands", op.op_type()),
            });
        };

        let (_output_name, output_type) =
            Self::create_output_value(graph, output_id, operand_name_overrides)?;

        self.convert_operation_with_input_names_and_outputs(
            graph,
            op,
            &input_names,
            vec![output_type],
            mil_op_type,
        )
    }

    fn input_names_for_operation(
        graph: &GraphInfo,
        op: &Operation,
        operand_name_overrides: &HashMap<u32, String>,
    ) -> Vec<String> {
        op.input_operands()
            .iter()
            .map(|&id| {
                operand_name_overrides
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| operand_name(graph, id))
            })
            .collect()
    }

    fn convert_operation_with_input_names_and_outputs(
        &self,
        graph: &GraphInfo,
        op: &Operation,
        input_names: &[String],
        outputs: Vec<NamedValueType>,
        mil_op_type: &str,
    ) -> Result<MilOperation, GraphError> {
        let mut inputs = self.create_operation_inputs(graph, op, input_names)?;

        // MIL `quantize` declares `output_dtype` as a required const string input.
        if matches!(op, Operation::QuantizeLinear { .. })
            && let Some(output_id) = op.output_operand()
            && let Some(output_operand) = graph.operand(output_id)
        {
            let dtype_str =
                Self::cast_dtype_string_for_graph_type(&output_operand.descriptor.data_type)?;
            inputs.insert(
                "output_dtype".to_string(),
                Self::create_immediate_string(dtype_str),
            );
        }

        Ok(Self::create_mil_operation(mil_op_type, inputs, outputs))
    }

    fn create_cast_operation(
        input_name: String,
        output_type: NamedValueType,
        dtype: &str,
    ) -> MilOperation {
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), Self::create_name_argument(input_name));
        inputs.insert("dtype".to_string(), Self::create_immediate_string(dtype));
        Self::create_mil_operation(mil_ops::CAST, inputs, vec![output_type])
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
        let Operation::Split {
            input,
            splits,
            options,
            ..
        } = &op
        else {
            return Err(GraphError::ConversionFailed {
                format: "CoreML MLProgram".to_string(),
                reason: "expected Split operator".to_string(),
            });
        };
        let input_id = *input;

        // Get input operand name
        let input_name = operand_name(graph, input_id);

        // Get output types
        let outputs: Vec<NamedValueType> = op
            .output_operands()
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

        // num_splits or split_sizes from operation; axis from MLSplitOptions.
        let axis = options.as_ref().map(|o| o.axis).unwrap_or(0);
        if splits.is_empty() {
            inputs.insert(
                "num_splits".to_string(),
                Self::create_int_argument(op.output_operands().len() as i32),
            );
        } else {
            let split_sizes: Vec<i32> = splits.iter().map(|&u| u as i32).collect();
            inputs.insert(
                "split_sizes".to_string(),
                Self::create_int_array_argument(split_sizes),
            );
        }
        inputs.insert("axis".to_string(), Self::create_int_argument(axis as i32));

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
            "max" => mil_ops::MAXIMUM,
            "min" => mil_ops::MINIMUM,
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
            "l2pool2d" => mil_ops::L2_POOL,
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
            "roundeven" => mil_ops::ROUND_EVEN,
            "neg" => mil_ops::NEG,
            "identity" => mil_ops::IDENTITY,
            "exp" => mil_ops::EXP,
            "log" => mil_ops::LOG,
            "sqrt" => mil_ops::SQRT,
            "sign" => mil_ops::SIGN,
            "sin" => mil_ops::SIN,
            "cos" => mil_ops::COS,
            "tan" => mil_ops::TAN,
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
            "gatherelements" => mil_ops::GATHER_ALONG_AXIS,
            "split" => mil_ops::SPLIT,
            "where" => mil_ops::WHERE,
            "pad" => mil_ops::PAD,
            "cumulativesum" => mil_ops::CUM_SUM,
            "cumulative_sum" => mil_ops::CUM_SUM,

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

            // Reverse operation
            "reverse" => mil_ops::REVERSE,

            // Triangular operation
            "triangular" => mil_ops::TRIANGULAR,

            // Clamp operation
            "clamp" => mil_ops::CLIP,

            // NaN and infinity detection
            "isnan" => mil_ops::IS_NAN,
            "isinfinite" => mil_ops::IS_INF,

            // Gather N-dimensional
            "gathernd" => mil_ops::GATHER_ND,

            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("Unsupported operation: {}", webnn_op),
                });
            }
        };

        Ok(mil_type)
    }

    /// Compute the effective (possibly asymmetric) end-padding for a 2D pooling op.
    ///
    /// WebNN's `outputShapeRounding="ceil"` and explicit `outputSizes` both grow the
    /// output beyond what floor rounding of the base padding yields. CoreML forbids
    /// asymmetric padding under `ceil_mode` and does not take `outputSizes`, so instead
    /// we fold the extra size into additional end-padding and pool with plain floor
    /// rounding: for a target output `out`, the input must span `(out-1)*stride + kernel`
    /// after padding, so `end_pad += needed - (input + begin_pad + end_pad)`.
    ///
    /// `kernel`/`strides` are `[H, W]`; `base_pad` is WebNN `[Hbegin, Hend, Wbegin, Wend]`.
    /// Returns the adjusted `[Hbegin, Hend', Wbegin, Wend']`, or `None` if shapes are
    /// unavailable (caller falls back to the base padding).
    fn pool_effective_padding(
        graph: &GraphInfo,
        op: &Operation,
        kernel: &[u32],
        strides: &[u32],
        base_pad: &[u32],
        is_nhwc: bool,
    ) -> Option<Vec<u32>> {
        if kernel.len() < 2 || strides.len() < 2 || base_pad.len() < 4 {
            return None;
        }
        let input_shape = op
            .input_operands()
            .first()
            .and_then(|&id| graph.operand(id))
            .map(|o| o.descriptor.static_or_max_shape())?;
        let output_shape = op
            .output_operand()
            .and_then(|id| graph.operand(id))
            .map(|o| o.descriptor.static_or_max_shape())?;
        if input_shape.len() < 4 || output_shape.len() < 4 {
            return None;
        }
        // Spatial dimension indices for [H, W] under each layout.
        let (h_idx, w_idx) = if is_nhwc { (1, 2) } else { (2, 3) };
        let mut pad = base_pad.to_vec();
        for (axis, &dim_idx) in [h_idx, w_idx].iter().enumerate() {
            let input_dim = input_shape[dim_idx];
            let output_dim = output_shape[dim_idx];
            let k = kernel[axis];
            let s = strides[axis].max(1);
            let begin = base_pad[axis * 2];
            let end = base_pad[axis * 2 + 1];
            let needed = output_dim.saturating_sub(1) * s + k;
            let current = input_dim + begin + end;
            let extra = needed.saturating_sub(current);
            pad[axis * 2 + 1] = end + extra;
        }
        Some(pad)
    }

    /// Derive the per-channel axis for a single-non-unit-dim `scale_shape`
    /// against `input_shape`. A rank-aligned scale (same rank as the input)
    /// names its axis by the position of its non-unit dim — unambiguous even
    /// when several input dims share the channel length (square weights);
    /// only a rank-mismatched scale falls back to the first input dim
    /// matching the channel count. Returns `None` unless the input dim at
    /// the derived axis exactly equals the scale length: a single-axis
    /// BLOCKWISE scale (length that merely divides some input dim) has no
    /// per-channel axis.
    fn qdq_per_channel_axis(input_shape: &[u32], scale_shape: &[u32]) -> Option<usize> {
        let squeezed: Vec<u32> = scale_shape.iter().copied().filter(|&d| d != 1).collect();
        if squeezed.len() != 1 {
            return None;
        }
        let len = squeezed[0];
        let axis = if scale_shape.len() == input_shape.len() {
            scale_shape.iter().position(|&d| d != 1).unwrap_or(0)
        } else {
            input_shape.iter().position(|&d| d == len)?
        };
        (input_shape.get(axis) == Some(&len)).then_some(axis)
    }

    /// Whether a quantize/dequantize with this quantized integer type and scale shape
    /// can use CoreML's native `quantize`/`dequantize`. CoreML supports only int8/uint8
    /// quantized tensors with a scalar (per-tensor) or single-axis (per-channel) scale
    /// that exactly covers its input axis; int32 tensors, block-wise scales, and
    /// multi-axis scales are not supported and must be decomposed into elementwise
    /// arithmetic.
    fn qdq_native_supported(
        quant_dtype: &DataType,
        input_shape: &[u32],
        scale_shape: &[u32],
    ) -> bool {
        if !matches!(quant_dtype, DataType::Int8 | DataType::Uint8) {
            return false;
        }
        // per-tensor (scalar), or per-channel (a valid axis exactly covered by the
        // scale — a blockwise scale that merely divides an input dim yields no axis).
        scale_shape.iter().all(|&d| d == 1)
            || Self::qdq_per_channel_axis(input_shape, scale_shape).is_some()
    }

    /// Whether a quantize/dequantize op must be lowered to elementwise arithmetic because
    /// CoreML's native op can't express it. int4/uint4 tensors are excluded (they cannot
    /// be materialized at all) and scalar tensors keep the native rank-0 fast path.
    fn qdq_should_decompose(graph: &GraphInfo, op: &Operation) -> bool {
        let (quant_id, tensor_shape_id, scale_id, zero_point_id) = match op {
            // dequantize: the quantized tensor is the input; its type/shape drive the check.
            Operation::DequantizeLinear {
                input,
                scale,
                zero_point,
                ..
            } => (*input, *input, *scale, *zero_point),
            // quantize: the quantized tensor is the output; the float input carries the shape.
            Operation::QuantizeLinear {
                input,
                scale,
                zero_point,
                ..
            } => match op.output_operand() {
                Some(out) => (out, *input, *scale, *zero_point),
                None => return false,
            },
            _ => return false,
        };
        let (Some(quant_op), Some(tensor_op), Some(scale_op)) = (
            graph.operand(quant_id),
            graph.operand(tensor_shape_id),
            graph.operand(scale_id),
        ) else {
            return false;
        };
        // CoreML's native quantize/dequantize require const scale/zero_point;
        // runtime-computed ones (e.g. DynamicQuantizeLinear lowering) must be
        // decomposed into elementwise arithmetic.
        if scale_op.kind != crate::graph::OperandKind::Constant {
            return true;
        }
        if let Some(zp_id) = zero_point_id
            && let Some(zp_op) = graph.operand(zp_id)
            && zp_op.kind != crate::graph::OperandKind::Constant
        {
            return true;
        }
        let quant_dt = quant_op.descriptor.data_type;
        if tensor_op.descriptor.shape.is_empty() {
            // Scalars normally use the native rank-0 fast path, but int4/uint4 have no
            // native quantize/dequantize at all, so decompose them even when scalar.
            return matches!(quant_dt, DataType::Int4 | DataType::Uint4);
        }
        let tensor_shape = tensor_op.descriptor.static_or_max_shape();
        let scale_shape = scale_op.descriptor.static_or_max_shape();
        !Self::qdq_native_supported(&quant_dt, &tensor_shape, &scale_shape)
    }

    /// Build a value type for a concrete static `shape` with the given MIL dtype.
    fn value_type_for_static_shape(name: String, dtype: i32, shape: &[u32]) -> NamedValueType {
        let dims: Vec<GraphDimension> = shape.iter().map(|&d| GraphDimension::Static(d)).collect();
        Self::create_named_value_type(name, dtype, &dims, false)
    }

    /// Map a WebNN recurrent-network activation name to its MIL op.
    fn rnn_activation_op(name: &str) -> Result<&'static str, GraphError> {
        Ok(match name.to_lowercase().as_str() {
            "relu" => mil_ops::RELU,
            "tanh" => mil_ops::TANH,
            "sigmoid" => mil_ops::SIGMOID,
            other => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("unsupported RNN activation '{}'", other),
                });
            }
        })
    }

    /// Emit `out = x . y^T` (MIL matmul with transpose_y). Returns `out_name`.
    fn rnn_matmul_ty(
        block: &mut Block,
        x: &str,
        y: &str,
        out_name: String,
        dtype: i32,
        out_shape: &[u32],
    ) -> String {
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), Self::create_name_argument(x.to_string()));
        inputs.insert("y".to_string(), Self::create_name_argument(y.to_string()));
        inputs.insert(
            "transpose_x".to_string(),
            Self::create_immediate_bool(false),
        );
        inputs.insert("transpose_y".to_string(), Self::create_immediate_bool(true));
        let ty = Self::value_type_for_static_shape(out_name.clone(), dtype, out_shape);
        block.operations.push(Self::create_mil_operation(
            mil_ops::MATMUL,
            inputs,
            vec![ty],
        ));
        out_name
    }

    /// Emit an elementwise binary op `out = f(x, y)`. Returns `out_name`.
    fn rnn_binary(
        block: &mut Block,
        mil_op: &str,
        x: &str,
        y: &str,
        out_name: String,
        dtype: i32,
        shape: &[u32],
    ) -> String {
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), Self::create_name_argument(x.to_string()));
        inputs.insert("y".to_string(), Self::create_name_argument(y.to_string()));
        let ty = Self::value_type_for_static_shape(out_name.clone(), dtype, shape);
        block
            .operations
            .push(Self::create_mil_operation(mil_op, inputs, vec![ty]));
        out_name
    }

    /// Emit an elementwise unary op `out = f(x)`. Returns `out_name`.
    fn rnn_unary(
        block: &mut Block,
        mil_op: &str,
        x: &str,
        out_name: String,
        dtype: i32,
        shape: &[u32],
    ) -> String {
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), Self::create_name_argument(x.to_string()));
        let ty = Self::value_type_for_static_shape(out_name.clone(), dtype, shape);
        block
            .operations
            .push(Self::create_mil_operation(mil_op, inputs, vec![ty]));
        out_name
    }

    /// Emit `slice_by_size(x, begin, size)`. Returns `out_name`.
    fn rnn_slice(
        block: &mut Block,
        x: &str,
        begin: &[u32],
        size: &[u32],
        out_name: String,
        dtype: i32,
    ) -> String {
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), Self::create_name_argument(x.to_string()));
        inputs.insert("begin".to_string(), Self::create_immediate_int_array(begin));
        inputs.insert("size".to_string(), Self::create_immediate_int_array(size));
        let ty = Self::value_type_for_static_shape(out_name.clone(), dtype, size);
        block
            .operations
            .push(Self::create_mil_operation(mil_ops::SLICE, inputs, vec![ty]));
        out_name
    }

    /// Emit an int32 constant tensor with the given shape (scalar when `shape` is empty).
    /// Returns `out_name`.
    fn emit_int32_const(
        block: &mut Block,
        values: &[i32],
        shape: &[u32],
        out_name: String,
    ) -> String {
        use crate::protos::coreml::mil_spec::{TensorValue, Value, tensor_value, value};
        let int32 = crate::protos::coreml::mil_spec::DataType::Int32 as i32;
        let ct = Self::value_type_for_static_shape(out_name.clone(), int32, shape);
        let tv = TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                values: values.to_vec(),
            })),
        };
        let imm = Value {
            doc_string: String::new(),
            r#type: ct.r#type.clone(),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tv)),
            })),
        };
        let mut attrs = HashMap::new();
        attrs.insert("val".to_string(), imm);
        block.operations.push(MilOperation {
            r#type: "const".to_string(),
            inputs: HashMap::new(),
            outputs: vec![ct],
            attributes: attrs,
            ..Default::default()
        });
        out_name
    }

    /// Normalize gather-style indices to WebNN semantics: wrap negatives (`idx + size`)
    /// then clamp out-of-bounds to `[0, size-1]`. `sizes` is the axis size per index
    /// component (length 1 for gather/gatherElements, or `K` for gatherND's last axis),
    /// broadcast against `idx_shape`. `idx_name` must already be int32.
    fn emit_gather_index_norm(
        block: &mut Block,
        idx_name: &str,
        idx_shape: &[u32],
        sizes: &[u32],
        prefix: &str,
    ) -> String {
        let int32 = crate::protos::coreml::mil_spec::DataType::Int32 as i32;
        let bool_t = crate::protos::coreml::mil_spec::DataType::Bool as i32;
        let p = |s: &str| format!("{prefix}_{s}");
        let sizes_i32: Vec<i32> = sizes.iter().map(|&s| s as i32).collect();
        let sizes_m1: Vec<i32> = sizes.iter().map(|&s| s as i32 - 1).collect();
        // Single-axis sizes are scalars (broadcast against any index shape); gatherND's
        // per-component sizes are a rank-1 [K] vector broadcast over the last index axis.
        let cshape: &[u32] = if sizes.len() == 1 {
            &[]
        } else {
            &[sizes.len() as u32]
        };
        let size_c = Self::emit_int32_const(block, &sizes_i32, cshape, p("gsz"));
        let sizem1_c = Self::emit_int32_const(block, &sizes_m1, cshape, p("gszm1"));
        let zero_c = Self::emit_int32_const(block, &[0], &[], p("gz"));
        let is_neg = Self::rnn_binary(
            block,
            mil_ops::LESS,
            idx_name,
            &zero_c,
            p("gneg"),
            bool_t,
            idx_shape,
        );
        let is_neg_i = Self::rnn_unary_cast(block, &is_neg, p("gnegi"), int32, idx_shape);
        let offset = Self::rnn_binary(
            block,
            mil_ops::MUL,
            &is_neg_i,
            &size_c,
            p("goff"),
            int32,
            idx_shape,
        );
        let wrapped = Self::rnn_binary(
            block,
            mil_ops::ADD,
            idx_name,
            &offset,
            p("gwrap"),
            int32,
            idx_shape,
        );
        let mx = Self::rnn_binary(
            block,
            mil_ops::MAXIMUM,
            &wrapped,
            &zero_c,
            p("gmx"),
            int32,
            idx_shape,
        );
        Self::rnn_binary(
            block,
            mil_ops::MINIMUM,
            &mx,
            &sizem1_c,
            p("gcl"),
            int32,
            idx_shape,
        )
    }

    /// Emit a `cast(x, dtype)` producing `out_name` with the given shape/dtype.
    fn rnn_unary_cast(
        block: &mut Block,
        x: &str,
        out_name: String,
        dtype: i32,
        shape: &[u32],
    ) -> String {
        let out_type = Self::value_type_for_static_shape(out_name.clone(), dtype, shape);
        let dtype_str = Self::cast_dtype_string_for_mil_type(dtype).unwrap_or("int32");
        block.operations.push(Self::create_cast_operation(
            x.to_string(),
            out_type,
            dtype_str,
        ));
        out_name
    }

    /// Emit `reshape(x, shape)`. Returns `out_name`.
    fn rnn_reshape(
        block: &mut Block,
        x: &str,
        shape: &[u32],
        out_name: String,
        dtype: i32,
    ) -> String {
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), Self::create_name_argument(x.to_string()));
        inputs.insert("shape".to_string(), Self::create_immediate_int_array(shape));
        let ty = Self::value_type_for_static_shape(out_name.clone(), dtype, shape);
        block
            .operations
            .push(Self::create_mil_operation("reshape", inputs, vec![ty]));
        out_name
    }

    /// Emit `concat(values, axis)`. Returns `out_name`.
    fn rnn_concat(
        block: &mut Block,
        names: &[String],
        axis: u32,
        out_name: String,
        dtype: i32,
        out_shape: &[u32],
    ) -> String {
        let mut inputs = HashMap::new();
        inputs.insert("values".to_string(), Self::create_argument_tuple(names));
        inputs.insert("axis".to_string(), Self::create_immediate_int(axis));
        inputs.insert("interleave".to_string(), Self::create_immediate_bool(false));
        let ty = Self::value_type_for_static_shape(out_name.clone(), dtype, out_shape);
        block.operations.push(Self::create_mil_operation(
            mil_ops::CONCAT,
            inputs,
            vec![ty],
        ));
        out_name
    }

    /// Emit a constant zero tensor of the given shape/dtype. Returns `out_name`.
    fn rnn_zeros(block: &mut Block, shape: &[u32], out_name: String, dtype: i32) -> String {
        use crate::protos::coreml::mil_spec::{TensorValue, Value, tensor_value, value};
        let count: usize = shape.iter().map(|&d| d as usize).product();
        let f32_dtype = crate::protos::coreml::mil_spec::DataType::Float32 as i32;
        let f32_name = if dtype == f32_dtype {
            out_name.clone()
        } else {
            format!("{}_zf32", out_name)
        };
        let const_type = Self::value_type_for_static_shape(f32_name.clone(), f32_dtype, shape);
        let tensor_value = TensorValue {
            value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
                values: vec![0.0f32; count],
            })),
        };
        let immediate = Value {
            doc_string: String::new(),
            r#type: const_type.r#type.clone(),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })),
        };
        let mut attributes = HashMap::new();
        attributes.insert("val".to_string(), immediate);
        block.operations.push(MilOperation {
            r#type: "const".to_string(),
            inputs: HashMap::new(),
            outputs: vec![const_type],
            attributes,
            ..Default::default()
        });
        if dtype == f32_dtype {
            return out_name;
        }
        let casted = Self::value_type_for_static_shape(out_name.clone(), dtype, shape);
        block.operations.push(Self::create_cast_operation(
            f32_name,
            casted,
            Self::cast_dtype_string_for_mil_type(dtype).unwrap_or("fp16"),
        ));
        out_name
    }

    /// Emit one GRU time step and return the new hidden-state tensor name (shape [b, h]).
    /// Same gate math as `emit_gru_cell_decomposition`, parameterized on tensor names so
    /// the sequence `gru` can unroll it over time steps and directions.
    #[allow(clippy::too_many_arguments)]
    fn emit_gru_step(
        block: &mut Block,
        x_name: &str,
        w_name: &str,
        r_name: &str,
        bias_name: Option<&str>,
        rbias_name: Option<&str>,
        hid_name: &str,
        b: u32,
        i: u32,
        h: u32,
        layout: &str,
        act0: &str,
        act1: &str,
        reset_after: bool,
        prefix: &str,
        dtype: i32,
    ) -> String {
        let bh = [b, h];
        let (z_off, r_off, n_off) = match layout {
            "rzn" => (h, 0, 2 * h),
            _ => (0, h, 2 * h),
        };
        let p = |s: &str| format!("{}_{}", prefix, s);
        let gate = |block: &mut Block, off: u32, act: &str, tag: &str| -> String {
            let wg = Self::rnn_slice(
                block,
                w_name,
                &[off, 0],
                &[h, i],
                p(&format!("w{tag}")),
                dtype,
            );
            let mut xw =
                Self::rnn_matmul_ty(block, x_name, &wg, p(&format!("xw{tag}")), dtype, &bh);
            if let Some(bn) = bias_name {
                let bg = Self::rnn_slice(block, bn, &[off], &[h], p(&format!("b{tag}")), dtype);
                xw = Self::rnn_binary(block, "add", &xw, &bg, p(&format!("xwb{tag}")), dtype, &bh);
            }
            let rg = Self::rnn_slice(
                block,
                r_name,
                &[off, 0],
                &[h, h],
                p(&format!("r{tag}")),
                dtype,
            );
            let mut hr =
                Self::rnn_matmul_ty(block, hid_name, &rg, p(&format!("hr{tag}")), dtype, &bh);
            if let Some(rbn) = rbias_name {
                let rbg = Self::rnn_slice(block, rbn, &[off], &[h], p(&format!("rb{tag}")), dtype);
                hr = Self::rnn_binary(block, "add", &hr, &rbg, p(&format!("hrb{tag}")), dtype, &bh);
            }
            let pre = Self::rnn_binary(block, "add", &xw, &hr, p(&format!("pre{tag}")), dtype, &bh);
            Self::rnn_unary(block, act, &pre, p(&format!("g{tag}")), dtype, &bh)
        };
        let z = gate(block, z_off, act0, "z");
        let r = gate(block, r_off, act0, "r");

        let wn = Self::rnn_slice(block, w_name, &[n_off, 0], &[h, i], p("wn"), dtype);
        let mut xwn = Self::rnn_matmul_ty(block, x_name, &wn, p("xwn"), dtype, &bh);
        if let Some(bn) = bias_name {
            let bg = Self::rnn_slice(block, bn, &[n_off], &[h], p("bn"), dtype);
            xwn = Self::rnn_binary(block, "add", &xwn, &bg, p("xwbn"), dtype, &bh);
        }
        let rn = Self::rnn_slice(block, r_name, &[n_off, 0], &[h, h], p("rn"), dtype);
        let hrn = if reset_after {
            let mut hr = Self::rnn_matmul_ty(block, hid_name, &rn, p("hrn"), dtype, &bh);
            if let Some(rbn) = rbias_name {
                let rbg = Self::rnn_slice(block, rbn, &[n_off], &[h], p("rbn"), dtype);
                hr = Self::rnn_binary(block, "add", &hr, &rbg, p("hrbn"), dtype, &bh);
            }
            Self::rnn_binary(block, "mul", &r, &hr, p("rhn"), dtype, &bh)
        } else {
            let rh = Self::rnn_binary(block, "mul", &r, hid_name, p("rh"), dtype, &bh);
            let mut hr = Self::rnn_matmul_ty(block, &rh, &rn, p("hrn"), dtype, &bh);
            if let Some(rbn) = rbias_name {
                let rbg = Self::rnn_slice(block, rbn, &[n_off], &[h], p("rbn"), dtype);
                hr = Self::rnn_binary(block, "add", &hr, &rbg, p("hrbn"), dtype, &bh);
            }
            hr
        };
        let npre = Self::rnn_binary(block, "add", &xwn, &hrn, p("npre"), dtype, &bh);
        let n = Self::rnn_unary(block, act1, &npre, p("n"), dtype, &bh);
        let hsubn = Self::rnn_binary(block, "sub", hid_name, &n, p("hsubn"), dtype, &bh);
        let zmul = Self::rnn_binary(block, "mul", &z, &hsubn, p("zmul"), dtype, &bh);
        Self::rnn_binary(block, "add", &n, &zmul, p("h"), dtype, &bh)
    }

    /// Emit one LSTM time step; returns the new (hidden, cell) tensor names (shape [b, h]).
    #[allow(clippy::too_many_arguments)]
    fn emit_lstm_step(
        block: &mut Block,
        x_name: &str,
        w_name: &str,
        r_name: &str,
        bias_name: Option<&str>,
        rbias_name: Option<&str>,
        peephole_name: Option<&str>,
        hid_name: &str,
        cell_name: &str,
        b: u32,
        i: u32,
        h: u32,
        layout: &str,
        f0: &str,
        f1: &str,
        f2: &str,
        prefix: &str,
        dtype: i32,
    ) -> (String, String) {
        let bh = [b, h];
        let (i_off, o_off, f_off, g_off) = match layout {
            "ifgo" => (0, 3 * h, h, 2 * h),
            _ => (0, h, 2 * h, 3 * h),
        };
        let (pi_off, po_off, pf_off) = (0u32, h, 2 * h);
        let p = |s: &str| format!("{}_{}", prefix, s);
        let gate = |block: &mut Block,
                    off: u32,
                    act: &str,
                    tag: &str,
                    peep: Option<(u32, &str)>|
         -> String {
            let wg = Self::rnn_slice(
                block,
                w_name,
                &[off, 0],
                &[h, i],
                p(&format!("w{tag}")),
                dtype,
            );
            let mut xw =
                Self::rnn_matmul_ty(block, x_name, &wg, p(&format!("xw{tag}")), dtype, &bh);
            if let Some(bn) = bias_name {
                let bg = Self::rnn_slice(block, bn, &[off], &[h], p(&format!("b{tag}")), dtype);
                xw = Self::rnn_binary(block, "add", &xw, &bg, p(&format!("xwb{tag}")), dtype, &bh);
            }
            let rg = Self::rnn_slice(
                block,
                r_name,
                &[off, 0],
                &[h, h],
                p(&format!("r{tag}")),
                dtype,
            );
            let mut hr =
                Self::rnn_matmul_ty(block, hid_name, &rg, p(&format!("hr{tag}")), dtype, &bh);
            if let Some(rbn) = rbias_name {
                let rbg = Self::rnn_slice(block, rbn, &[off], &[h], p(&format!("rb{tag}")), dtype);
                hr = Self::rnn_binary(block, "add", &hr, &rbg, p(&format!("hrb{tag}")), dtype, &bh);
            }
            let mut pre =
                Self::rnn_binary(block, "add", &xw, &hr, p(&format!("pre{tag}")), dtype, &bh);
            if let (Some((poff, cname)), Some(pw)) = (peep, peephole_name) {
                let pg = Self::rnn_slice(block, pw, &[poff], &[h], p(&format!("p{tag}")), dtype);
                let pc =
                    Self::rnn_binary(block, "mul", &pg, cname, p(&format!("pc{tag}")), dtype, &bh);
                pre = Self::rnn_binary(
                    block,
                    "add",
                    &pre,
                    &pc,
                    p(&format!("pre2{tag}")),
                    dtype,
                    &bh,
                );
            }
            Self::rnn_unary(block, act, &pre, p(&format!("g{tag}")), dtype, &bh)
        };
        let gi = gate(block, i_off, f0, "i", Some((pi_off, cell_name)));
        let gf = gate(block, f_off, f0, "f", Some((pf_off, cell_name)));
        let gg = gate(block, g_off, f1, "g", None);
        let fc = Self::rnn_binary(block, "mul", &gf, cell_name, p("fc"), dtype, &bh);
        let ig = Self::rnn_binary(block, "mul", &gi, &gg, p("ig"), dtype, &bh);
        let cnew = Self::rnn_binary(block, "add", &fc, &ig, p("c"), dtype, &bh);
        let go = gate(block, o_off, f0, "o", Some((po_off, &cnew)));
        let tanh_c = Self::rnn_unary(block, f2, &cnew, p("tanhc"), dtype, &bh);
        let hnew = Self::rnn_binary(block, "mul", &go, &tanh_c, p("h"), dtype, &bh);
        (hnew, cnew)
    }

    /// Compute reshape targets that make a block-wise quantization scale broadcast against
    /// the tensor. Each axis `i` where `1 < scale[i] < input[i]` is split into
    /// `[scale[i], input[i]/scale[i]]` for the tensor and `[scale[i], 1]` for the scale;
    /// per-tensor/per-element axes keep their single dim (ordinary broadcasting applies).
    /// Splitting only genuine block axes keeps the rank low (CoreML reshape allows rank <= 5).
    /// Returns `(interleaved_input_shape, interleaved_scale_shape)`.
    fn qdq_interleave_shapes(
        input_shape: &[u32],
        scale_shape: &[u32],
    ) -> Result<(Vec<u32>, Vec<u32>), GraphError> {
        let rank = input_shape.len();
        // WebNN block dims right-align with the input.
        let mut aligned = vec![1u32; rank];
        if scale_shape.len() <= rank {
            let off = rank - scale_shape.len();
            for (i, &d) in scale_shape.iter().enumerate() {
                aligned[off + i] = d;
            }
        }
        let mut interleaved_input: Vec<u32> = Vec::with_capacity(rank * 2);
        let mut interleaved_scale: Vec<u32> = Vec::with_capacity(rank * 2);
        for i in 0..rank {
            let nb = aligned[i].max(1);
            if !input_shape[i].is_multiple_of(nb) {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!(
                        "quantize/dequantize: input dim {} not divisible by scale dim {}",
                        input_shape[i], nb
                    ),
                });
            }
            let block = input_shape[i] / nb;
            if nb > 1 && block > 1 {
                interleaved_input.push(nb);
                interleaved_input.push(block);
                interleaved_scale.push(nb);
                interleaved_scale.push(1);
            } else {
                interleaved_input.push(input_shape[i]);
                interleaved_scale.push(nb);
            }
        }
        Ok((interleaved_input, interleaved_scale))
    }

    /// Whether a `dequantizeLinear` should be emitted as MIL's
    /// `constexpr_affine_dequantize`: quantized data, scale and zeroPoint are
    /// all graph constants (a compressed weight), the data is non-scalar
    /// int8/uint8, and the scale layout is one the native op could express
    /// (per-tensor scalar or single-axis per-channel).
    ///
    /// Espresso constant-folds a regular `dequantize` (or the elementwise
    /// decomposition) over a constant into a dense float tensor at compile
    /// time — minutes of CPU and GBs of RAM for transformer-sized weights.
    /// The constexpr form is CoreML's weight-compression representation and
    /// keeps the weight packed through compilation.
    fn constexpr_dequantize_supported(graph: &GraphInfo, op: &Operation) -> bool {
        let Operation::DequantizeLinear {
            input,
            scale,
            zero_point,
            ..
        } = op
        else {
            return false;
        };
        // qdq_should_decompose == false already guarantees const scale/zp and
        // a scalar or single-axis per-channel scale.
        if Self::qdq_should_decompose(graph, op) {
            return false;
        }
        let (Some(input_op), Some(scale_op)) = (graph.operand(*input), graph.operand(*scale))
        else {
            return false;
        };
        if input_op.kind != crate::graph::OperandKind::Constant {
            return false;
        }
        if !matches!(
            input_op.descriptor.data_type,
            DataType::Int8 | DataType::Uint8
        ) {
            return false;
        }
        if input_op.descriptor.shape.is_empty() {
            return false;
        }
        // constexpr_affine_dequantize requires scale and zero_point to be both
        // scalar or both per-channel vectors; a synthesized zero point is
        // always scalar, so a missing zeroPoint pairs only with a scalar scale.
        match zero_point {
            None => {
                let per_channel = scale_op
                    .descriptor
                    .static_or_max_shape()
                    .iter()
                    .any(|&d| d != 1);
                if per_channel {
                    return false;
                }
            }
            Some(zp) => {
                // The blob payload was written with the zero point's own dtype;
                // a dtype-coerced zero point (e.g. int32) would mismatch it.
                let Some(zp_op) = graph.operand(*zp) else {
                    return false;
                };
                if zp_op.descriptor.data_type != input_op.descriptor.data_type {
                    return false;
                }
            }
        }
        // A per-channel scale must exactly cover the derived axis:
        // constexpr_affine_dequantize cannot express a single-axis BLOCKWISE
        // scale. qdq_native_supported enforces the same rule, but re-check via
        // the shared helper so this path never depends on that gate's internals.
        let scale_shape = scale_op.descriptor.static_or_max_shape();
        let input_shape = input_op.descriptor.static_or_max_shape();
        if scale_shape.iter().any(|&d| d != 1)
            && Self::qdq_per_channel_axis(&input_shape, &scale_shape).is_none()
        {
            return false;
        }
        true
    }

    /// A compile-time `Value` for one `constexpr_affine_dequantize` parameter:
    /// a `BlobFileValue` when the constant's payload is already in the weight
    /// file (all non-scalar weight dtypes are), otherwise an immediate tensor
    /// built from the graph's constant bytes.
    fn constexpr_param_value(
        graph: &GraphInfo,
        weight_builder: &super::WeightFileBuilder,
        operand_id: u32,
        dims: &[u32],
    ) -> Result<crate::protos::coreml::mil_spec::Value, GraphError> {
        use crate::protos::coreml::mil_spec::{
            Dimension, TensorType, TensorValue, Value, ValueType, dimension, tensor_value, value,
            value_type,
        };

        let operand = graph
            .operand(operand_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("constexpr operand {} not found", operand_id),
            })?;
        let mil_type = Self::mil_data_type(&operand.descriptor.data_type)?;
        let dimensions: Vec<Dimension> = dims
            .iter()
            .map(|&d| Dimension {
                dimension: Some(dimension::Dimension::Constant(
                    dimension::ConstantDimension { size: d as u64 },
                )),
            })
            .collect();
        let value_type = ValueType {
            r#type: Some(value_type::Type::TensorType(TensorType {
                rank: dimensions.len() as i64,
                data_type: mil_type,
                dimensions,
                attributes: HashMap::new(),
            })),
        };

        let inner = if let Some(offset) = weight_builder.get_offset(operand_id) {
            value::Value::BlobFileValue(value::BlobFileValue {
                file_name: "@model_path/weights/weights.bin".to_string(),
                offset,
            })
        } else {
            let constant = graph
                .constant_operand_ids_to_handles
                .get(&operand_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("constexpr operand {} has no constant data", operand_id),
                })?;
            let tensor_value = match operand.descriptor.data_type {
                DataType::Float32 => TensorValue {
                    value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
                        values: constant
                            .data
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                    })),
                },
                DataType::Float16 | DataType::Int8 | DataType::Uint8 => TensorValue {
                    value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                        values: constant.data.clone().into(),
                    })),
                },
                other => {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("unsupported constexpr parameter dtype {other:?}"),
                    });
                }
            };
            value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tensor_value)),
            })
        };
        Ok(Value {
            doc_string: String::new(),
            r#type: Some(value_type),
            value: Some(inner),
        })
    }

    /// Immediate rank-0 zero of the given quantized dtype (int8/uint8), the
    /// synthesized `zero_point` for `constexpr_affine_dequantize`.
    fn constexpr_zero_value(
        data_type: &DataType,
    ) -> Result<crate::protos::coreml::mil_spec::Value, GraphError> {
        use crate::protos::coreml::mil_spec::{
            TensorType, TensorValue, Value, ValueType, tensor_value, value, value_type,
        };

        let mil_type = match data_type {
            DataType::Int8 | DataType::Uint8 => Self::mil_data_type(data_type)?,
            other => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("no quantized zero-point immediate for {other:?}"),
                });
            }
        };
        Ok(Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: mil_type,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(TensorValue {
                    value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                        values: vec![0u8].into(),
                    })),
                })),
            })),
        })
    }

    /// Emit `constexpr_affine_dequantize` (dequantized = scale * (data - zp))
    /// for a dequantizeLinear over constant weights. CoreML requires constexpr
    /// parameters as compile-time value ATTRIBUTES (immediate or blob-file),
    /// not name-bound inputs; large payloads reuse the blob offsets the const
    /// emission pass already wrote.
    fn emit_constexpr_affine_dequantize(
        graph: &GraphInfo,
        op: &Operation,
        overrides: &HashMap<u32, String>,
        weight_builder: &super::WeightFileBuilder,
        main_block: &mut Block,
    ) -> Result<(), GraphError> {
        let Operation::DequantizeLinear {
            input,
            scale,
            zero_point,
            ..
        } = op
        else {
            return Err(GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "emit_constexpr_affine_dequantize called on non-dequantize op".to_string(),
            });
        };
        let output_id = op
            .output_operand()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "dequantizeLinear has no output operand".to_string(),
            })?;
        let (_output_name, output_type) = Self::create_output_value(graph, output_id, overrides)?;

        let input_op = graph
            .operand(*input)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("dequantize input operand {} not found", input),
            })?;
        let scale_op = graph
            .operand(*scale)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("dequantize scale operand {} not found", scale),
            })?;

        // The constant pre-scan squeezes scale/zeroPoint const ops; use the
        // same squeezed layout here (scalar, or 1-D of the channel length).
        let scale_shape = scale_op.descriptor.static_or_max_shape();
        let input_shape = input_op.descriptor.static_or_max_shape();
        let squeezed: Vec<u32> = scale_shape.iter().copied().filter(|&d| d != 1).collect();
        // Per-channel: derive the axis from the scale layout (rank-aligned
        // scales name their axis by position — unambiguous even for square
        // weights); 0 for per-tensor. constexpr_dequantize_supported already
        // validated the axis, so None here only means per-tensor.
        let axis = Self::qdq_per_channel_axis(&input_shape, &scale_shape).unwrap_or(0) as u32;
        let param_shape: &[u32] = if squeezed.len() == 1 && squeezed[0] > 1 {
            &squeezed
        } else {
            &[]
        };

        let mut attributes = HashMap::new();
        attributes.insert(
            "quantized_data".to_string(),
            Self::constexpr_param_value(graph, weight_builder, *input, &input_shape)?,
        );
        attributes.insert(
            "scale".to_string(),
            Self::constexpr_param_value(graph, weight_builder, *scale, param_shape)?,
        );
        let zp_value = match zero_point {
            Some(zp) => Self::constexpr_param_value(graph, weight_builder, *zp, param_shape)?,
            None => Self::constexpr_zero_value(&input_op.descriptor.data_type)?,
        };
        attributes.insert("zero_point".to_string(), zp_value);
        attributes.insert("axis".to_string(), Self::constexpr_axis_value(axis));

        main_block.operations.push(MilOperation {
            r#type: "constexpr_affine_dequantize".to_string(),
            inputs: HashMap::new(),
            outputs: vec![output_type],
            attributes,
            ..Default::default()
        });
        Ok(())
    }

    /// Immediate int32 scalar `Value` (attribute form of `create_immediate_int`).
    fn constexpr_axis_value(axis: u32) -> crate::protos::coreml::mil_spec::Value {
        use crate::protos::coreml::mil_spec::{
            DataType as MilDataType, TensorType, TensorValue, Value, ValueType, tensor_value,
            value, value_type,
        };
        Value {
            doc_string: String::new(),
            r#type: Some(ValueType {
                r#type: Some(value_type::Type::TensorType(TensorType {
                    data_type: MilDataType::Int32 as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                })),
            }),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(TensorValue {
                    value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
                        values: vec![axis as i32],
                    })),
                })),
            })),
        }
    }

    /// Lower `reduceLogSumExp` using the numerically stable max-shift identity:
    ///
    /// `log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))`.
    ///
    /// CoreML's native `reduce_log_sum_exp` overflows and underflows for the
    /// large-magnitude WebNN conformance inputs. Keep the reduced axes until
    /// the final step so `max(x)` broadcasts correctly for arbitrary axes.
    fn emit_stable_reduce_log_sum_exp(
        graph: &GraphInfo,
        op: &Operation,
        overrides: &HashMap<u32, String>,
        main_block: &mut Block,
    ) -> Result<(), GraphError> {
        let (input_id, options) = match op {
            Operation::ReduceLogSumExp { input, options, .. } => (*input, options.as_ref()),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "stable reduceLogSumExp lowering called on another operation"
                        .to_string(),
                });
            }
        };
        let output_id = op
            .output_operand()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "reduceLogSumExp has no output operand".to_string(),
            })?;
        let input_operand =
            graph
                .operand(input_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("reduceLogSumExp input operand {} not found", input_id),
                })?;
        let output_operand =
            graph
                .operand(output_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("reduceLogSumExp output operand {} not found", output_id),
                })?;

        let input_name = Self::output_name_for_operand(graph, input_id, overrides);
        let (output_name, output_type) = Self::create_output_value(graph, output_id, overrides)?;

        // CoreML represents WebNN scalars as one-element rank-1 tensors at the
        // model boundary, so default scalar reduction uses the synthetic axis 0.
        let input_rank = input_operand.descriptor.shape.len();
        let working_rank = input_rank.max(1);
        let axes: Vec<u32> = match options.and_then(|opts| opts.axes.as_ref()) {
            Some(axes) => axes.clone(),
            None => (0..working_rank as u32).collect(),
        };

        // An explicitly empty axes sequence is an identity for reduceLogSumExp.
        if axes.is_empty() {
            let mut identity_inputs = HashMap::new();
            identity_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
            main_block.operations.push(Self::create_mil_operation(
                mil_ops::IDENTITY,
                identity_inputs,
                vec![output_type],
            ));
            return Ok(());
        }

        if let Some(&axis) = axes.iter().find(|&&axis| axis as usize >= working_rank) {
            return Err(GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!(
                    "reduceLogSumExp axis {} is out of range for rank {}",
                    axis, working_rank
                ),
            });
        }

        let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
        let working_shape = if input_rank == 0 {
            vec![GraphDimension::Static(1)]
        } else {
            input_operand.descriptor.shape.clone()
        };
        let working_input_name = if input_rank == 0 {
            let reshaped_name = format!("{}_lse_input1d", output_name);
            let reshaped_type =
                Self::create_named_value_type(reshaped_name.clone(), dtype, &working_shape, false);
            let mut reshape_inputs = HashMap::new();
            reshape_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
            reshape_inputs.insert("shape".to_string(), Self::create_immediate_int_array(&[1]));
            main_block.operations.push(Self::create_mil_operation(
                mil_ops::RESHAPE,
                reshape_inputs,
                vec![reshaped_type],
            ));
            reshaped_name
        } else {
            input_name
        };
        let mut keep_dims_shape = working_shape.clone();
        for &axis in &axes {
            keep_dims_shape[axis as usize] = GraphDimension::Static(1);
        }

        let max_name = format!("{}_lse_max", output_name);
        let max_type =
            Self::create_named_value_type(max_name.clone(), dtype, &keep_dims_shape, false);
        let mut max_inputs = HashMap::new();
        max_inputs.insert(
            "x".to_string(),
            Self::create_name_argument(working_input_name.clone()),
        );
        max_inputs.insert("axes".to_string(), Self::create_immediate_int_array(&axes));
        max_inputs.insert("keep_dims".to_string(), Self::create_immediate_bool(true));
        main_block.operations.push(Self::create_mil_operation(
            mil_ops::REDUCE_MAX,
            max_inputs,
            vec![max_type],
        ));

        let shifted_name = format!("{}_lse_shifted", output_name);
        let shifted_type =
            Self::create_named_value_type(shifted_name.clone(), dtype, &working_shape, false);
        let mut sub_inputs = HashMap::new();
        sub_inputs.insert(
            "x".to_string(),
            Self::create_name_argument(working_input_name),
        );
        sub_inputs.insert(
            "y".to_string(),
            Self::create_name_argument(max_name.clone()),
        );
        main_block.operations.push(Self::create_mil_operation(
            mil_ops::SUB,
            sub_inputs,
            vec![shifted_type],
        ));

        let exp_name = format!("{}_lse_exp", output_name);
        let exp_type =
            Self::create_named_value_type(exp_name.clone(), dtype, &working_shape, false);
        let mut exp_inputs = HashMap::new();
        exp_inputs.insert("x".to_string(), Self::create_name_argument(shifted_name));
        main_block.operations.push(Self::create_mil_operation(
            mil_ops::EXP,
            exp_inputs,
            vec![exp_type],
        ));

        let sum_name = format!("{}_lse_sum", output_name);
        let sum_type =
            Self::create_named_value_type(sum_name.clone(), dtype, &keep_dims_shape, false);
        let mut sum_inputs = HashMap::new();
        sum_inputs.insert("x".to_string(), Self::create_name_argument(exp_name));
        sum_inputs.insert("axes".to_string(), Self::create_immediate_int_array(&axes));
        sum_inputs.insert("keep_dims".to_string(), Self::create_immediate_bool(true));
        main_block.operations.push(Self::create_mil_operation(
            mil_ops::REDUCE_SUM,
            sum_inputs,
            vec![sum_type],
        ));

        let log_name = format!("{}_lse_log", output_name);
        let log_type =
            Self::create_named_value_type(log_name.clone(), dtype, &keep_dims_shape, false);
        let mut log_inputs = HashMap::new();
        log_inputs.insert("x".to_string(), Self::create_name_argument(sum_name));
        log_inputs.insert(
            "epsilon".to_string(),
            Self::create_immediate_float(DEFAULT_EPSILON),
        );
        main_block.operations.push(Self::create_mil_operation(
            mil_ops::LOG,
            log_inputs,
            vec![log_type],
        ));

        let keep_dimensions = options.map(|opts| opts.keep_dimensions).unwrap_or(false);
        if keep_dimensions || input_rank == 0 {
            let mut add_inputs = HashMap::new();
            add_inputs.insert("x".to_string(), Self::create_name_argument(max_name));
            add_inputs.insert("y".to_string(), Self::create_name_argument(log_name));
            main_block.operations.push(Self::create_mil_operation(
                mil_ops::ADD,
                add_inputs,
                vec![output_type],
            ));
            return Ok(());
        }

        let keep_dims_name = format!("{}_lse_keepdims", output_name);
        let keep_dims_type =
            Self::create_named_value_type(keep_dims_name.clone(), dtype, &keep_dims_shape, false);
        let mut add_inputs = HashMap::new();
        add_inputs.insert("x".to_string(), Self::create_name_argument(max_name));
        add_inputs.insert("y".to_string(), Self::create_name_argument(log_name));
        main_block.operations.push(Self::create_mil_operation(
            mil_ops::ADD,
            add_inputs,
            vec![keep_dims_type],
        ));

        if output_operand.descriptor.shape.is_empty() {
            // WebNN scalar outputs use CoreML's one-element boundary representation.
            let mut reshape_inputs = HashMap::new();
            reshape_inputs.insert("x".to_string(), Self::create_name_argument(keep_dims_name));
            reshape_inputs.insert("shape".to_string(), Self::create_immediate_int_array(&[1]));
            main_block.operations.push(Self::create_mil_operation(
                mil_ops::RESHAPE,
                reshape_inputs,
                vec![output_type],
            ));
        } else {
            let mut squeeze_inputs = HashMap::new();
            squeeze_inputs.insert("x".to_string(), Self::create_name_argument(keep_dims_name));
            squeeze_inputs.insert("axes".to_string(), Self::create_immediate_int_array(&axes));
            main_block.operations.push(Self::create_mil_operation(
                mil_ops::SQUEEZE,
                squeeze_inputs,
                vec![output_type],
            ));
        }

        Ok(())
    }

    /// Lower `dequantizeLinear` as `(input - zeroPoint) * scale` in elementwise form.
    ///
    /// Handles quantized types and scale shapes CoreML's native `dequantize` cannot:
    /// int32 tensors, block-wise scales, and multi-axis scales. Block quantization is
    /// expressed by reshaping each axis `i` of length `input[i]` into `[scale[i],
    /// block[i]]` (with `block[i] = input[i]/scale[i]`) and the scale/zeroPoint into
    /// `[scale[i], 1]`, so ordinary broadcasting applies; the result is reshaped back.
    fn emit_dequantize_decomposition(
        graph: &GraphInfo,
        op: &Operation,
        overrides: &HashMap<u32, String>,
        main_block: &mut Block,
    ) -> Result<(), GraphError> {
        let (input_id, scale_id, zp_id) = match op {
            Operation::DequantizeLinear {
                input,
                scale,
                zero_point,
                ..
            } => (*input, *scale, *zero_point),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "emit_dequantize_decomposition called on non-dequantize op".to_string(),
                });
            }
        };
        let output_id = op
            .output_operand()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "dequantizeLinear has no output operand".to_string(),
            })?;
        let input_op = graph
            .operand(input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("dequantize input operand {} not found", input_id),
            })?;
        let scale_op = graph
            .operand(scale_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("dequantize scale operand {} not found", scale_id),
            })?;
        let input_shape: Vec<u32> = if input_op.descriptor.shape.is_empty() {
            vec![1]
        } else {
            input_op.descriptor.static_or_max_shape()
        };
        let scale_shape = scale_op.descriptor.static_or_max_shape();

        // Output MIL dtype is the (float) scale/output type.
        let out_dtype = Self::mil_data_type(
            &graph
                .operand(output_id)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: format!("dequantize output operand {} not found", output_id),
                })?
                .descriptor
                .data_type,
        )?;
        let float_str = Self::cast_dtype_string_for_mil_type(out_dtype)?;
        let (output_name, output_type) = Self::create_output_value(graph, output_id, overrides)?;

        let (interleaved_input, interleaved_scale) =
            Self::qdq_interleave_shapes(&input_shape, &scale_shape)?;

        // 1. cast input -> float, reshape to interleaved.
        let input_name = Self::output_name_for_operand(graph, input_id, overrides);
        let in_f_name = format!("{}_dq_in_f", output_name);
        let in_f_type =
            Self::value_type_for_static_shape(in_f_name.clone(), out_dtype, &input_shape);
        main_block.operations.push(Self::create_cast_operation(
            input_name, in_f_type, float_str,
        ));
        let in_r_name = format!("{}_dq_in_r", output_name);
        let in_r_type =
            Self::value_type_for_static_shape(in_r_name.clone(), out_dtype, &interleaved_input);
        let mut in_reshape = HashMap::new();
        in_reshape.insert("x".to_string(), Self::create_name_argument(in_f_name));
        in_reshape.insert(
            "shape".to_string(),
            Self::create_immediate_int_array(&interleaved_input),
        );
        main_block.operations.push(Self::create_mil_operation(
            "reshape",
            in_reshape,
            vec![in_r_type],
        ));

        // 2. scale -> reshape to interleaved.
        let scale_name = Self::output_name_for_operand(graph, scale_id, overrides);
        let scale_r_name = format!("{}_dq_scale_r", output_name);
        let scale_r_type =
            Self::value_type_for_static_shape(scale_r_name.clone(), out_dtype, &interleaved_scale);
        let mut scale_reshape = HashMap::new();
        scale_reshape.insert("x".to_string(), Self::create_name_argument(scale_name));
        scale_reshape.insert(
            "shape".to_string(),
            Self::create_immediate_int_array(&interleaved_scale),
        );
        main_block.operations.push(Self::create_mil_operation(
            "reshape",
            scale_reshape,
            vec![scale_r_type],
        ));

        // 3. (input - zeroPoint), if a zero_point is present.
        let minus_zp_name = if let Some(zp_id) = zp_id {
            let zp_name = Self::output_name_for_operand(graph, zp_id, overrides);
            let zp_f_name = format!("{}_dq_zp_f", output_name);
            let zp_f_type =
                Self::value_type_for_static_shape(zp_f_name.clone(), out_dtype, &scale_shape);
            main_block
                .operations
                .push(Self::create_cast_operation(zp_name, zp_f_type, float_str));
            let zp_r_name = format!("{}_dq_zp_r", output_name);
            let zp_r_type =
                Self::value_type_for_static_shape(zp_r_name.clone(), out_dtype, &interleaved_scale);
            let mut zp_reshape = HashMap::new();
            zp_reshape.insert("x".to_string(), Self::create_name_argument(zp_f_name));
            zp_reshape.insert(
                "shape".to_string(),
                Self::create_immediate_int_array(&interleaved_scale),
            );
            main_block.operations.push(Self::create_mil_operation(
                "reshape",
                zp_reshape,
                vec![zp_r_type],
            ));
            let sub_name = format!("{}_dq_sub", output_name);
            let sub_type =
                Self::value_type_for_static_shape(sub_name.clone(), out_dtype, &interleaved_input);
            let mut sub_inputs = HashMap::new();
            sub_inputs.insert("x".to_string(), Self::create_name_argument(in_r_name));
            sub_inputs.insert("y".to_string(), Self::create_name_argument(zp_r_name));
            main_block.operations.push(Self::create_mil_operation(
                mil_ops::SUB,
                sub_inputs,
                vec![sub_type],
            ));
            sub_name
        } else {
            in_r_name
        };

        // 4. multiply by scale, reshape back to the input shape.
        let mul_name = format!("{}_dq_mul", output_name);
        let mul_type =
            Self::value_type_for_static_shape(mul_name.clone(), out_dtype, &interleaved_input);
        let mut mul_inputs = HashMap::new();
        mul_inputs.insert("x".to_string(), Self::create_name_argument(minus_zp_name));
        mul_inputs.insert("y".to_string(), Self::create_name_argument(scale_r_name));
        main_block.operations.push(Self::create_mil_operation(
            mil_ops::MUL,
            mul_inputs,
            vec![mul_type],
        ));

        let mut out_reshape = HashMap::new();
        out_reshape.insert("x".to_string(), Self::create_name_argument(mul_name));
        out_reshape.insert(
            "shape".to_string(),
            Self::create_immediate_int_array(&input_shape),
        );
        main_block.operations.push(Self::create_mil_operation(
            "reshape",
            out_reshape,
            vec![output_type],
        ));
        Ok(())
    }

    /// Lower `quantizeLinear` as `cast(clamp(round(input / scale) + zeroPoint, qmin, qmax))`
    /// in elementwise form. Handles quantized types and scale shapes CoreML's native
    /// `quantize` cannot: int32 outputs, block-wise scales, and multi-axis scales. Block
    /// quantization uses the same reshape trick as the dequantize decomposition.
    fn emit_quantize_decomposition(
        graph: &GraphInfo,
        op: &Operation,
        overrides: &HashMap<u32, String>,
        main_block: &mut Block,
    ) -> Result<(), GraphError> {
        let (input_id, scale_id, zp_id) = match op {
            Operation::QuantizeLinear {
                input,
                scale,
                zero_point,
                ..
            } => (*input, *scale, *zero_point),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "emit_quantize_decomposition called on non-quantize op".to_string(),
                });
            }
        };
        let output_id = op
            .output_operand()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "quantizeLinear has no output operand".to_string(),
            })?;
        let input_op = graph
            .operand(input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("quantize input operand {} not found", input_id),
            })?;
        let scale_op = graph
            .operand(scale_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("quantize scale operand {} not found", scale_id),
            })?;
        let output_op = graph
            .operand(output_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("quantize output operand {} not found", output_id),
            })?;
        let input_shape: Vec<u32> = if input_op.descriptor.shape.is_empty() {
            vec![1]
        } else {
            input_op.descriptor.static_or_max_shape()
        };
        let scale_shape = scale_op.descriptor.static_or_max_shape();
        let quant_dt = output_op.descriptor.data_type;

        // Compute in fp32 regardless of the input's float type: the output is an integer,
        // so upcasting is safe and avoids fp16 rounding (e.g. an exact 12347 that fp16
        // can't represent) and clip dtype-mismatch errors.
        let float_dtype = crate::protos::coreml::mil_spec::DataType::Float32 as i32;
        let (output_name, output_type) = Self::create_output_value(graph, output_id, overrides)?;
        // The quantized output is represented in the graph as its native MIL int type
        // (int8/uint8) or, for the sub-byte/wide proxies (int4/uint4/...), as int32; the
        // executor packs/narrows it to the true width on readback.
        let out_int_str = Self::int_back_cast_dtype(&quant_dt)?;

        let (interleaved_input, interleaved_scale) =
            Self::qdq_interleave_shapes(&input_shape, &scale_shape)?;

        // 1. cast input -> fp32, reshape to interleaved form.
        let input_name = Self::output_name_for_operand(graph, input_id, overrides);
        let in_f_name = format!("{}_q_in_f", output_name);
        let in_f_type =
            Self::value_type_for_static_shape(in_f_name.clone(), float_dtype, &input_shape);
        main_block
            .operations
            .push(Self::create_cast_operation(input_name, in_f_type, "fp32"));
        let in_r_name = format!("{}_q_in_r", output_name);
        let in_r_type =
            Self::value_type_for_static_shape(in_r_name.clone(), float_dtype, &interleaved_input);
        let mut in_reshape = HashMap::new();
        in_reshape.insert("x".to_string(), Self::create_name_argument(in_f_name));
        in_reshape.insert(
            "shape".to_string(),
            Self::create_immediate_int_array(&interleaved_input),
        );
        main_block.operations.push(Self::create_mil_operation(
            "reshape",
            in_reshape,
            vec![in_r_type],
        ));

        // 2. cast scale -> fp32, reshape to interleaved form.
        let scale_name = Self::output_name_for_operand(graph, scale_id, overrides);
        let scale_f_name = format!("{}_q_scale_f", output_name);
        let scale_f_type =
            Self::value_type_for_static_shape(scale_f_name.clone(), float_dtype, &scale_shape);
        main_block.operations.push(Self::create_cast_operation(
            scale_name,
            scale_f_type,
            "fp32",
        ));
        let scale_r_name = format!("{}_q_scale_r", output_name);
        let scale_r_type = Self::value_type_for_static_shape(
            scale_r_name.clone(),
            float_dtype,
            &interleaved_scale,
        );
        let mut scale_reshape = HashMap::new();
        scale_reshape.insert("x".to_string(), Self::create_name_argument(scale_f_name));
        scale_reshape.insert(
            "shape".to_string(),
            Self::create_immediate_int_array(&interleaved_scale),
        );
        main_block.operations.push(Self::create_mil_operation(
            "reshape",
            scale_reshape,
            vec![scale_r_type],
        ));

        // 3. div = input / scale, then round to nearest even.
        let div_name = format!("{}_q_div", output_name);
        let div_type =
            Self::value_type_for_static_shape(div_name.clone(), float_dtype, &interleaved_input);
        let mut div_inputs = HashMap::new();
        div_inputs.insert("x".to_string(), Self::create_name_argument(in_r_name));
        div_inputs.insert("y".to_string(), Self::create_name_argument(scale_r_name));
        main_block.operations.push(Self::create_mil_operation(
            mil_ops::DIV,
            div_inputs,
            vec![div_type],
        ));
        let round_name = format!("{}_q_round", output_name);
        let round_type =
            Self::value_type_for_static_shape(round_name.clone(), float_dtype, &interleaved_input);
        let mut round_inputs = HashMap::new();
        round_inputs.insert("x".to_string(), Self::create_name_argument(div_name));
        main_block.operations.push(Self::create_mil_operation(
            mil_ops::ROUND_EVEN,
            round_inputs,
            vec![round_type],
        ));

        // 4. add the zero_point (cast to float, reshaped), if present.
        let biased_name = if let Some(zp_id) = zp_id {
            let zp_name = Self::output_name_for_operand(graph, zp_id, overrides);
            let zp_f_name = format!("{}_q_zp_f", output_name);
            let zp_f_type =
                Self::value_type_for_static_shape(zp_f_name.clone(), float_dtype, &scale_shape);
            main_block.operations.push(Self::create_cast_operation(
                zp_name,
                zp_f_type,
                Self::cast_dtype_string_for_mil_type(float_dtype)?,
            ));
            let zp_r_name = format!("{}_q_zp_r", output_name);
            let zp_r_type = Self::value_type_for_static_shape(
                zp_r_name.clone(),
                float_dtype,
                &interleaved_scale,
            );
            let mut zp_reshape = HashMap::new();
            zp_reshape.insert("x".to_string(), Self::create_name_argument(zp_f_name));
            zp_reshape.insert(
                "shape".to_string(),
                Self::create_immediate_int_array(&interleaved_scale),
            );
            main_block.operations.push(Self::create_mil_operation(
                "reshape",
                zp_reshape,
                vec![zp_r_type],
            ));
            let add_name = format!("{}_q_add", output_name);
            let add_type = Self::value_type_for_static_shape(
                add_name.clone(),
                float_dtype,
                &interleaved_input,
            );
            let mut add_inputs = HashMap::new();
            add_inputs.insert("x".to_string(), Self::create_name_argument(round_name));
            add_inputs.insert("y".to_string(), Self::create_name_argument(zp_r_name));
            main_block.operations.push(Self::create_mil_operation(
                "add",
                add_inputs,
                vec![add_type],
            ));
            add_name
        } else {
            round_name
        };

        // 5. clamp to the quantized type's range (int32 spans the proxy and needs no clamp).
        let clamped_name = match quant_dt {
            DataType::Int8 | DataType::Uint8 | DataType::Int4 | DataType::Uint4 => {
                let (qmin, qmax) = match quant_dt {
                    DataType::Uint8 => (0.0f32, 255.0f32),
                    DataType::Int4 => (-8.0f32, 7.0f32),
                    DataType::Uint4 => (0.0f32, 15.0f32),
                    _ => (-128.0f32, 127.0f32),
                };
                let clip_name = format!("{}_q_clip", output_name);
                let clip_type = Self::value_type_for_static_shape(
                    clip_name.clone(),
                    float_dtype,
                    &interleaved_input,
                );
                let mut clip_inputs = HashMap::new();
                clip_inputs.insert("x".to_string(), Self::create_name_argument(biased_name));
                clip_inputs.insert("alpha".to_string(), Self::create_immediate_float(qmin));
                clip_inputs.insert("beta".to_string(), Self::create_immediate_float(qmax));
                main_block.operations.push(Self::create_mil_operation(
                    mil_ops::CLIP,
                    clip_inputs,
                    vec![clip_type],
                ));
                clip_name
            }
            _ => biased_name,
        };

        // 6. reshape back to the output shape (still float), then cast to the int type.
        let out_f_name = format!("{}_q_out_f", output_name);
        let out_f_type =
            Self::value_type_for_static_shape(out_f_name.clone(), float_dtype, &input_shape);
        let mut out_reshape = HashMap::new();
        out_reshape.insert("x".to_string(), Self::create_name_argument(clamped_name));
        out_reshape.insert(
            "shape".to_string(),
            Self::create_immediate_int_array(&input_shape),
        );
        main_block.operations.push(Self::create_mil_operation(
            "reshape",
            out_reshape,
            vec![out_f_type],
        ));
        main_block.operations.push(Self::create_cast_operation(
            out_f_name,
            output_type,
            out_int_str,
        ));
        Ok(())
    }

    /// Lower a WebNN `gruCell` (single time step) into primitive MIL ops.
    ///
    /// For gate order z (update), r (reset), n (new) with per-gate weight/recurrence rows:
    ///   z = f0(X·Wz^T + bz + H·Rz^T + rbz)
    ///   r = f0(X·Wr^T + br + H·Rr^T + rbr)
    ///   n = f1(X·Wn^T + bn + (r ⊙ H)·Rn^T + rbn)            (reset_after = false)
    ///   n = f1(X·Wn^T + bn + r ⊙ (H·Rn^T + rbn))            (reset_after = true)
    ///   Hnew = (1 - z) ⊙ n + z ⊙ H = n + z ⊙ (H - n)
    /// Default activations f0=sigmoid, f1=tanh; default layout "zrn".
    fn emit_gru_cell_decomposition(
        graph: &GraphInfo,
        op: &Operation,
        overrides: &HashMap<u32, String>,
        block: &mut Block,
    ) -> Result<(), GraphError> {
        let (input_id, weight_id, rec_id, hidden_id, hidden_size, opts) = match op {
            Operation::GruCell {
                input,
                weight,
                recurrence,
                hidden_state,
                hidden_size,
                options,
                ..
            } => (
                *input,
                *weight,
                *recurrence,
                *hidden_state,
                *hidden_size,
                options.as_ref(),
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "emit_gru_cell_decomposition called on non-gruCell op".to_string(),
                });
            }
        };
        let output_id = op
            .output_operand()
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "gruCell has no output operand".to_string(),
            })?;
        let input_op = graph
            .operand(input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "gruCell input operand not found".to_string(),
            })?;
        let dtype = Self::mil_data_type(&input_op.descriptor.data_type)?;
        let in_shape = input_op.descriptor.static_or_max_shape();
        let b = in_shape[0];
        let i = in_shape[1];
        let h = hidden_size;
        let bh = [b, h];

        let x_name = Self::output_name_for_operand(graph, input_id, overrides);
        let w_name = Self::output_name_for_operand(graph, weight_id, overrides);
        let r_name = Self::output_name_for_operand(graph, rec_id, overrides);
        let hid_name = Self::output_name_for_operand(graph, hidden_id, overrides);
        let bias_name = opts
            .and_then(|o| o.bias)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let rbias_name = opts
            .and_then(|o| o.recurrent_bias)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let reset_after = opts.map(|o| o.reset_after).unwrap_or(false);
        let layout = opts
            .map(|o| o.layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("zrn");
        let acts: Vec<String> = opts
            .and_then(|o| o.activations.clone())
            .unwrap_or_else(|| vec!["sigmoid".to_string(), "tanh".to_string()]);
        let act0 = Self::rnn_activation_op(&acts[0])?;
        let act1 = Self::rnn_activation_op(acts.get(1).map(|s| s.as_str()).unwrap_or("tanh"))?;

        // Row offsets of each gate within the 3*hidden weight/recurrence tensors.
        let (z_off, r_off, n_off) = match layout {
            "rzn" => (h, 0, 2 * h),
            _ => (0, h, 2 * h),
        };

        let (output_name, output_type) = Self::create_output_value(graph, output_id, overrides)?;
        let p = |s: &str| format!("{}_{}", output_name, s);

        // Compute a "reset/update"-style gate: activation(X·Wg^T + bg + H·Rg^T + rbg).
        let gate = |block: &mut Block, off: u32, act: &str, tag: &str| -> String {
            let wg = Self::rnn_slice(
                block,
                &w_name,
                &[off, 0],
                &[h, i],
                p(&format!("w{tag}")),
                dtype,
            );
            let mut xw =
                Self::rnn_matmul_ty(block, &x_name, &wg, p(&format!("xw{tag}")), dtype, &bh);
            if let Some(bn) = &bias_name {
                let bg = Self::rnn_slice(block, bn, &[off], &[h], p(&format!("b{tag}")), dtype);
                xw = Self::rnn_binary(block, "add", &xw, &bg, p(&format!("xwb{tag}")), dtype, &bh);
            }
            let rg = Self::rnn_slice(
                block,
                &r_name,
                &[off, 0],
                &[h, h],
                p(&format!("r{tag}")),
                dtype,
            );
            let mut hr =
                Self::rnn_matmul_ty(block, &hid_name, &rg, p(&format!("hr{tag}")), dtype, &bh);
            if let Some(rbn) = &rbias_name {
                let rbg = Self::rnn_slice(block, rbn, &[off], &[h], p(&format!("rb{tag}")), dtype);
                hr = Self::rnn_binary(block, "add", &hr, &rbg, p(&format!("hrb{tag}")), dtype, &bh);
            }
            let pre = Self::rnn_binary(block, "add", &xw, &hr, p(&format!("pre{tag}")), dtype, &bh);
            Self::rnn_unary(block, act, &pre, p(&format!("g{tag}")), dtype, &bh)
        };

        let z = gate(block, z_off, act0, "z");
        let r = gate(block, r_off, act0, "r");

        // New gate n, whose recurrent term depends on reset_after.
        let wn = Self::rnn_slice(block, &w_name, &[n_off, 0], &[h, i], p("wn"), dtype);
        let mut xwn = Self::rnn_matmul_ty(block, &x_name, &wn, p("xwn"), dtype, &bh);
        if let Some(bn) = &bias_name {
            let bg = Self::rnn_slice(block, bn, &[n_off], &[h], p("bn"), dtype);
            xwn = Self::rnn_binary(block, "add", &xwn, &bg, p("xwbn"), dtype, &bh);
        }
        let rn = Self::rnn_slice(block, &r_name, &[n_off, 0], &[h, h], p("rn"), dtype);
        let hrn = if reset_after {
            // r ⊙ (H·Rn^T + rbn)
            let mut hr = Self::rnn_matmul_ty(block, &hid_name, &rn, p("hrn"), dtype, &bh);
            if let Some(rbn) = &rbias_name {
                let rbg = Self::rnn_slice(block, rbn, &[n_off], &[h], p("rbn"), dtype);
                hr = Self::rnn_binary(block, "add", &hr, &rbg, p("hrbn"), dtype, &bh);
            }
            Self::rnn_binary(block, "mul", &r, &hr, p("rhn"), dtype, &bh)
        } else {
            // (r ⊙ H)·Rn^T + rbn
            let rh = Self::rnn_binary(block, "mul", &r, &hid_name, p("rh"), dtype, &bh);
            let mut hr = Self::rnn_matmul_ty(block, &rh, &rn, p("hrn"), dtype, &bh);
            if let Some(rbn) = &rbias_name {
                let rbg = Self::rnn_slice(block, rbn, &[n_off], &[h], p("rbn"), dtype);
                hr = Self::rnn_binary(block, "add", &hr, &rbg, p("hrbn"), dtype, &bh);
            }
            hr
        };
        let npre = Self::rnn_binary(block, "add", &xwn, &hrn, p("npre"), dtype, &bh);
        let n = Self::rnn_unary(block, act1, &npre, p("n"), dtype, &bh);

        // Hnew = n + z ⊙ (H - n)
        let hsubn = Self::rnn_binary(block, "sub", &hid_name, &n, p("hsubn"), dtype, &bh);
        let zmul = Self::rnn_binary(block, "mul", &z, &hsubn, p("zmul"), dtype, &bh);
        let mut add_inputs = HashMap::new();
        add_inputs.insert("x".to_string(), Self::create_name_argument(n));
        add_inputs.insert("y".to_string(), Self::create_name_argument(zmul));
        block.operations.push(Self::create_mil_operation(
            "add",
            add_inputs,
            vec![output_type],
        ));
        Ok(())
    }

    /// Lower a WebNN `lstmCell` (single time step) into primitive MIL ops.
    ///
    /// Gates i (input), o (output), f (forget), g (cell); optional peephole weights
    /// [pi, po, pf]:
    ///   i = f0(X·Wi^T + bi + H·Ri^T + rbi + pi ⊙ C)
    ///   f = f0(X·Wf^T + bf + H·Rf^T + rbf + pf ⊙ C)
    ///   g = f1(X·Wg^T + bg + H·Rg^T + rbg)
    ///   Cnew = f ⊙ C + i ⊙ g
    ///   o = f0(X·Wo^T + bo + H·Ro^T + rbo + po ⊙ Cnew)
    ///   Hnew = o ⊙ f2(Cnew)
    /// Default activations f0=sigmoid, f1=f2=tanh; default gate layout "iofg".
    /// Outputs are [Hnew, Cnew].
    fn emit_lstm_cell_decomposition(
        graph: &GraphInfo,
        op: &Operation,
        overrides: &HashMap<u32, String>,
        block: &mut Block,
    ) -> Result<(), GraphError> {
        let (input_id, weight_id, rec_id, hidden_id, cell_id, hidden_size, opts) = match op {
            Operation::LstmCell {
                input,
                weight,
                recurrence,
                hidden_state,
                cell_state,
                hidden_size,
                options,
                ..
            } => (
                *input,
                *weight,
                *recurrence,
                *hidden_state,
                *cell_state,
                *hidden_size,
                options.as_ref(),
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "emit_lstm_cell_decomposition called on non-lstmCell op".to_string(),
                });
            }
        };
        let out_ids = op.output_operands();
        if out_ids.len() < 2 {
            return Err(GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "lstmCell requires two outputs (hidden, cell)".to_string(),
            });
        }
        let (out_h_id, out_c_id) = (out_ids[0], out_ids[1]);
        let input_op = graph
            .operand(input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "lstmCell input operand not found".to_string(),
            })?;
        let dtype = Self::mil_data_type(&input_op.descriptor.data_type)?;
        let in_shape = input_op.descriptor.static_or_max_shape();
        let i = in_shape[1];
        let h = hidden_size;
        let bh = [in_shape[0], h];

        let x_name = Self::output_name_for_operand(graph, input_id, overrides);
        let w_name = Self::output_name_for_operand(graph, weight_id, overrides);
        let r_name = Self::output_name_for_operand(graph, rec_id, overrides);
        let hid_name = Self::output_name_for_operand(graph, hidden_id, overrides);
        let cell_name = Self::output_name_for_operand(graph, cell_id, overrides);
        let bias_name = opts
            .and_then(|o| o.bias)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let rbias_name = opts
            .and_then(|o| o.recurrent_bias)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let peephole_name = opts
            .and_then(|o| o.peephole_weight)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let layout = opts
            .map(|o| o.layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("iofg");
        let acts: Vec<String> = opts.and_then(|o| o.activations.clone()).unwrap_or_else(|| {
            vec![
                "sigmoid".to_string(),
                "tanh".to_string(),
                "tanh".to_string(),
            ]
        });
        let f0 = Self::rnn_activation_op(&acts[0])?;
        let f1 = Self::rnn_activation_op(acts.get(1).map(|s| s.as_str()).unwrap_or("tanh"))?;
        let f2 = Self::rnn_activation_op(acts.get(2).map(|s| s.as_str()).unwrap_or("tanh"))?;

        // Gate row offsets within the 4*hidden weight/recurrence tensors.
        let (i_off, o_off, f_off, g_off) = match layout {
            "ifgo" => (0, 3 * h, h, 2 * h),
            _ => (0, h, 2 * h, 3 * h),
        };
        // Peephole weights are ordered [i, o, f].
        let (pi_off, po_off, pf_off) = (0u32, h, 2 * h);

        let (h_name, h_type) = Self::create_output_value(graph, out_h_id, overrides)?;
        let (c_name, c_type) = Self::create_output_value(graph, out_c_id, overrides)?;
        let p = |s: &str| format!("{}_{}", h_name, s);

        // gate = activation(X·Wg^T + bg + H·Rg^T + rbg [+ pg ⊙ cstate])
        let gate = |block: &mut Block,
                    off: u32,
                    act: &str,
                    tag: &str,
                    peep: Option<(u32, &str)>|
         -> String {
            let wg = Self::rnn_slice(
                block,
                &w_name,
                &[off, 0],
                &[h, i],
                p(&format!("w{tag}")),
                dtype,
            );
            let mut xw =
                Self::rnn_matmul_ty(block, &x_name, &wg, p(&format!("xw{tag}")), dtype, &bh);
            if let Some(bn) = &bias_name {
                let bg = Self::rnn_slice(block, bn, &[off], &[h], p(&format!("b{tag}")), dtype);
                xw = Self::rnn_binary(block, "add", &xw, &bg, p(&format!("xwb{tag}")), dtype, &bh);
            }
            let rg = Self::rnn_slice(
                block,
                &r_name,
                &[off, 0],
                &[h, h],
                p(&format!("r{tag}")),
                dtype,
            );
            let mut hr =
                Self::rnn_matmul_ty(block, &hid_name, &rg, p(&format!("hr{tag}")), dtype, &bh);
            if let Some(rbn) = &rbias_name {
                let rbg = Self::rnn_slice(block, rbn, &[off], &[h], p(&format!("rb{tag}")), dtype);
                hr = Self::rnn_binary(block, "add", &hr, &rbg, p(&format!("hrb{tag}")), dtype, &bh);
            }
            let mut pre =
                Self::rnn_binary(block, "add", &xw, &hr, p(&format!("pre{tag}")), dtype, &bh);
            if let (Some((poff, cname)), Some(pw)) = (peep, &peephole_name) {
                let pg = Self::rnn_slice(block, pw, &[poff], &[h], p(&format!("p{tag}")), dtype);
                let pc =
                    Self::rnn_binary(block, "mul", &pg, cname, p(&format!("pc{tag}")), dtype, &bh);
                pre = Self::rnn_binary(
                    block,
                    "add",
                    &pre,
                    &pc,
                    p(&format!("pre2{tag}")),
                    dtype,
                    &bh,
                );
            }
            Self::rnn_unary(block, act, &pre, p(&format!("g{tag}")), dtype, &bh)
        };

        let cell_owned = cell_name.clone();
        let gate_i = gate(block, i_off, f0, "i", Some((pi_off, &cell_owned)));
        let gate_f = gate(block, f_off, f0, "f", Some((pf_off, &cell_owned)));
        let gate_g = gate(block, g_off, f1, "g", None);

        // Cnew = f ⊙ C + i ⊙ g  (emitted as the cell output).
        let fc = Self::rnn_binary(block, "mul", &gate_f, &cell_name, p("fc"), dtype, &bh);
        let ig = Self::rnn_binary(block, "mul", &gate_i, &gate_g, p("ig"), dtype, &bh);
        let mut cnew_inputs = HashMap::new();
        cnew_inputs.insert("x".to_string(), Self::create_name_argument(fc));
        cnew_inputs.insert("y".to_string(), Self::create_name_argument(ig));
        block
            .operations
            .push(Self::create_mil_operation("add", cnew_inputs, vec![c_type]));

        // o depends on Cnew; Hnew = o ⊙ f2(Cnew).
        let gate_o = gate(block, o_off, f0, "o", Some((po_off, &c_name)));
        let tanh_c = Self::rnn_unary(block, f2, &c_name, p("tanhc"), dtype, &bh);
        let mut h_inputs = HashMap::new();
        h_inputs.insert("x".to_string(), Self::create_name_argument(gate_o));
        h_inputs.insert("y".to_string(), Self::create_name_argument(tanh_c));
        block
            .operations
            .push(Self::create_mil_operation("mul", h_inputs, vec![h_type]));
        Ok(())
    }

    /// Stack per-direction final states [b,h] into the WebNN output[0] shape
    /// [num_dir, b, h], writing it to `out_name`.
    fn rnn_pack_final(
        block: &mut Block,
        per_dir: &[String],
        b: u32,
        h: u32,
        base: &str,
        out_name: String,
        dtype: i32,
    ) {
        let nd = per_dir.len() as u32;
        if nd == 1 {
            Self::rnn_reshape(block, &per_dir[0], &[1, b, h], out_name, dtype);
            return;
        }
        let parts: Vec<String> = per_dir
            .iter()
            .enumerate()
            .map(|(d, name)| {
                Self::rnn_reshape(block, name, &[1, b, h], format!("{base}_f3d{d}"), dtype)
            })
            .collect();
        Self::rnn_concat(block, &parts, 0, out_name, dtype, &[nd, b, h]);
    }

    /// Stack per-direction, per-time states into the WebNN sequence output shape
    /// [steps, num_dir, b, h], writing it to `out_name`. `seq[d][t]` is the [b,h] hidden.
    fn rnn_pack_sequence(
        block: &mut Block,
        seq: &[Vec<String>],
        steps: u32,
        hidden_shape: [u32; 2],
        base: &str,
        out_name: String,
        dtype: i32,
    ) {
        let [b, h] = hidden_shape;
        let nd = seq.len() as u32;
        // For each time step build a [1, nd, b, h] slab.
        let mut time_slabs: Vec<String> = Vec::with_capacity(steps as usize);
        for t in 0..steps {
            let dir_parts: Vec<String> = (0..nd as usize)
                .map(|d| {
                    Self::rnn_reshape(
                        block,
                        &seq[d][t as usize],
                        &[1, 1, b, h],
                        format!("{base}_s{t}_d{d}"),
                        dtype,
                    )
                })
                .collect();
            let slab = if nd == 1 {
                dir_parts.into_iter().next().unwrap()
            } else {
                Self::rnn_concat(
                    block,
                    &dir_parts,
                    1,
                    format!("{base}_s{t}"),
                    dtype,
                    &[1, nd, b, h],
                )
            };
            time_slabs.push(slab);
        }
        if steps == 1 {
            // Rename the single slab to the output by an identity reshape.
            Self::rnn_reshape(block, &time_slabs[0], &[1, nd, b, h], out_name, dtype);
            return;
        }
        Self::rnn_concat(block, &time_slabs, 0, out_name, dtype, &[steps, nd, b, h]);
    }

    /// Lower a WebNN sequence `gru` by unrolling `emit_gru_step` over time steps and
    /// directions (forward/backward/bidirectional), assembling output[0] = last hidden
    /// [num_dir, b, h] and, when requested, output[1] = all steps [steps, num_dir, b, h].
    fn emit_gru_decomposition(
        graph: &GraphInfo,
        op: &Operation,
        overrides: &HashMap<u32, String>,
        block: &mut Block,
    ) -> Result<(), GraphError> {
        let (input_id, weight_id, rec_id, steps, hidden_size, opts) = match op {
            Operation::Gru {
                input,
                weight,
                recurrence,
                steps,
                hidden_size,
                options,
                ..
            } => (
                *input,
                *weight,
                *recurrence,
                *steps,
                *hidden_size,
                options.as_ref(),
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "emit_gru_decomposition called on non-gru op".to_string(),
                });
            }
        };
        let out_ids = op.output_operands().to_vec();
        let input_op = graph
            .operand(input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "gru input operand not found".to_string(),
            })?;
        let dtype = Self::mil_data_type(&input_op.descriptor.data_type)?;
        let in_shape = input_op.descriptor.static_or_max_shape();
        let batch = in_shape[1];
        let i = in_shape[2];
        let h = hidden_size;

        let direction = opts.map(|o| o.direction.as_str()).unwrap_or("forward");
        let nd: u32 = if direction.eq_ignore_ascii_case("both") {
            2
        } else {
            1
        };
        let reset_after = opts.map(|o| o.reset_after).unwrap_or(false);
        let layout = opts
            .map(|o| o.layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("zrn");
        let acts: Vec<String> = opts
            .and_then(|o| o.activations.clone())
            .unwrap_or_else(|| vec!["sigmoid".to_string(), "tanh".to_string()]);
        let act0 = Self::rnn_activation_op(&acts[0])?;
        let act1 = Self::rnn_activation_op(acts.get(1).map(|s| s.as_str()).unwrap_or("tanh"))?;

        let input_name = Self::output_name_for_operand(graph, input_id, overrides);
        let weight_name = Self::output_name_for_operand(graph, weight_id, overrides);
        let rec_name = Self::output_name_for_operand(graph, rec_id, overrides);
        let bias_name = opts
            .and_then(|o| o.bias)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let rbias_name = opts
            .and_then(|o| o.recurrent_bias)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let init_name = opts
            .and_then(|o| o.initial_hidden_state)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let base = operand_name(graph, out_ids[0]);

        let mut per_dir_final: Vec<String> = Vec::with_capacity(nd as usize);
        let mut per_dir_seq: Vec<Vec<String>> = Vec::with_capacity(nd as usize);
        for d in 0..nd {
            let backward = direction.eq_ignore_ascii_case("backward")
                || (direction.eq_ignore_ascii_case("both") && d == 1);
            let dp = format!("{base}_d{d}");
            let wsl = Self::rnn_slice(
                block,
                &weight_name,
                &[d, 0, 0],
                &[1, 3 * h, i],
                format!("{dp}_wsl"),
                dtype,
            );
            let wd = Self::rnn_reshape(block, &wsl, &[3 * h, i], format!("{dp}_w"), dtype);
            let rsl = Self::rnn_slice(
                block,
                &rec_name,
                &[d, 0, 0],
                &[1, 3 * h, h],
                format!("{dp}_rsl"),
                dtype,
            );
            let rd = Self::rnn_reshape(block, &rsl, &[3 * h, h], format!("{dp}_r"), dtype);
            let bd = bias_name.as_ref().map(|bn| {
                let s =
                    Self::rnn_slice(block, bn, &[d, 0], &[1, 3 * h], format!("{dp}_bsl"), dtype);
                Self::rnn_reshape(block, &s, &[3 * h], format!("{dp}_b"), dtype)
            });
            let rbd = rbias_name.as_ref().map(|bn| {
                let s =
                    Self::rnn_slice(block, bn, &[d, 0], &[1, 3 * h], format!("{dp}_rbsl"), dtype);
                Self::rnn_reshape(block, &s, &[3 * h], format!("{dp}_rb"), dtype)
            });
            let mut hid = if let Some(inm) = &init_name {
                let s = Self::rnn_slice(
                    block,
                    inm,
                    &[d, 0, 0],
                    &[1, batch, h],
                    format!("{dp}_h0sl"),
                    dtype,
                );
                Self::rnn_reshape(block, &s, &[batch, h], format!("{dp}_h0"), dtype)
            } else {
                Self::rnn_zeros(block, &[batch, h], format!("{dp}_h0"), dtype)
            };
            let mut seq_by_time: Vec<String> = vec![String::new(); steps as usize];
            let order: Vec<u32> = if backward {
                (0..steps).rev().collect()
            } else {
                (0..steps).collect()
            };
            for t in order {
                let xsl = Self::rnn_slice(
                    block,
                    &input_name,
                    &[t, 0, 0],
                    &[1, batch, i],
                    format!("{dp}_x{t}sl"),
                    dtype,
                );
                let xt = Self::rnn_reshape(block, &xsl, &[batch, i], format!("{dp}_x{t}"), dtype);
                hid = Self::emit_gru_step(
                    block,
                    &xt,
                    &wd,
                    &rd,
                    bd.as_deref(),
                    rbd.as_deref(),
                    &hid,
                    batch,
                    i,
                    h,
                    layout,
                    act0,
                    act1,
                    reset_after,
                    &format!("{dp}_t{t}"),
                    dtype,
                );
                seq_by_time[t as usize] = hid.clone();
            }
            per_dir_final.push(hid);
            per_dir_seq.push(seq_by_time);
        }

        let out0 = operand_name(graph, out_ids[0]);
        Self::rnn_pack_final(block, &per_dir_final, batch, h, &base, out0, dtype);
        if out_ids.len() > 1 {
            let out1 = operand_name(graph, out_ids[1]);
            Self::rnn_pack_sequence(block, &per_dir_seq, steps, [batch, h], &base, out1, dtype);
        }
        Ok(())
    }

    /// Lower a WebNN sequence `lstm` by unrolling `emit_lstm_step` over time steps and
    /// directions. Outputs: [0] last hidden [num_dir, b, h], [1] last cell [num_dir, b, h],
    /// and (when requested) [2] all hidden steps [steps, num_dir, b, h].
    fn emit_lstm_decomposition(
        graph: &GraphInfo,
        op: &Operation,
        overrides: &HashMap<u32, String>,
        block: &mut Block,
    ) -> Result<(), GraphError> {
        let (input_id, weight_id, rec_id, steps, hidden_size, opts) = match op {
            Operation::Lstm {
                input,
                weight,
                recurrence,
                steps,
                hidden_size,
                options,
                ..
            } => (
                *input,
                *weight,
                *recurrence,
                *steps,
                *hidden_size,
                options.as_ref(),
            ),
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "emit_lstm_decomposition called on non-lstm op".to_string(),
                });
            }
        };
        let out_ids = op.output_operands().to_vec();
        if out_ids.len() < 2 {
            return Err(GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "lstm requires hidden and cell outputs".to_string(),
            });
        }
        let input_op = graph
            .operand(input_id)
            .ok_or_else(|| GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: "lstm input operand not found".to_string(),
            })?;
        let dtype = Self::mil_data_type(&input_op.descriptor.data_type)?;
        let in_shape = input_op.descriptor.static_or_max_shape();
        let batch = in_shape[1];
        let i = in_shape[2];
        let h = hidden_size;

        let direction = opts.map(|o| o.direction.as_str()).unwrap_or("forward");
        let nd: u32 = if direction.eq_ignore_ascii_case("both") {
            2
        } else {
            1
        };
        let layout = opts
            .map(|o| o.layout.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("iofg");
        let acts: Vec<String> = opts.and_then(|o| o.activations.clone()).unwrap_or_else(|| {
            vec![
                "sigmoid".to_string(),
                "tanh".to_string(),
                "tanh".to_string(),
            ]
        });
        let f0 = Self::rnn_activation_op(&acts[0])?;
        let f1 = Self::rnn_activation_op(acts.get(1).map(|s| s.as_str()).unwrap_or("tanh"))?;
        let f2 = Self::rnn_activation_op(acts.get(2).map(|s| s.as_str()).unwrap_or("tanh"))?;

        let input_name = Self::output_name_for_operand(graph, input_id, overrides);
        let weight_name = Self::output_name_for_operand(graph, weight_id, overrides);
        let rec_name = Self::output_name_for_operand(graph, rec_id, overrides);
        let bias_name = opts
            .and_then(|o| o.bias)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let rbias_name = opts
            .and_then(|o| o.recurrent_bias)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let peephole_name = opts
            .and_then(|o| o.peephole_weight)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let init_h_name = opts
            .and_then(|o| o.initial_hidden_state)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let init_c_name = opts
            .and_then(|o| o.initial_cell_state)
            .map(|id| Self::output_name_for_operand(graph, id, overrides));
        let base = operand_name(graph, out_ids[0]);

        let mut per_dir_h: Vec<String> = Vec::with_capacity(nd as usize);
        let mut per_dir_c: Vec<String> = Vec::with_capacity(nd as usize);
        let mut per_dir_seq: Vec<Vec<String>> = Vec::with_capacity(nd as usize);
        for d in 0..nd {
            let backward = direction.eq_ignore_ascii_case("backward")
                || (direction.eq_ignore_ascii_case("both") && d == 1);
            let dp = format!("{base}_d{d}");
            let wsl = Self::rnn_slice(
                block,
                &weight_name,
                &[d, 0, 0],
                &[1, 4 * h, i],
                format!("{dp}_wsl"),
                dtype,
            );
            let wd = Self::rnn_reshape(block, &wsl, &[4 * h, i], format!("{dp}_w"), dtype);
            let rsl = Self::rnn_slice(
                block,
                &rec_name,
                &[d, 0, 0],
                &[1, 4 * h, h],
                format!("{dp}_rsl"),
                dtype,
            );
            let rd = Self::rnn_reshape(block, &rsl, &[4 * h, h], format!("{dp}_r"), dtype);
            let bd = bias_name.as_ref().map(|bn| {
                let s =
                    Self::rnn_slice(block, bn, &[d, 0], &[1, 4 * h], format!("{dp}_bsl"), dtype);
                Self::rnn_reshape(block, &s, &[4 * h], format!("{dp}_b"), dtype)
            });
            let rbd = rbias_name.as_ref().map(|bn| {
                let s =
                    Self::rnn_slice(block, bn, &[d, 0], &[1, 4 * h], format!("{dp}_rbsl"), dtype);
                Self::rnn_reshape(block, &s, &[4 * h], format!("{dp}_rb"), dtype)
            });
            let pd = peephole_name.as_ref().map(|pn| {
                let s =
                    Self::rnn_slice(block, pn, &[d, 0], &[1, 3 * h], format!("{dp}_psl"), dtype);
                Self::rnn_reshape(block, &s, &[3 * h], format!("{dp}_p"), dtype)
            });
            let mut hid = if let Some(inm) = &init_h_name {
                let s = Self::rnn_slice(
                    block,
                    inm,
                    &[d, 0, 0],
                    &[1, batch, h],
                    format!("{dp}_h0sl"),
                    dtype,
                );
                Self::rnn_reshape(block, &s, &[batch, h], format!("{dp}_h0"), dtype)
            } else {
                Self::rnn_zeros(block, &[batch, h], format!("{dp}_h0"), dtype)
            };
            let mut cell = if let Some(inm) = &init_c_name {
                let s = Self::rnn_slice(
                    block,
                    inm,
                    &[d, 0, 0],
                    &[1, batch, h],
                    format!("{dp}_c0sl"),
                    dtype,
                );
                Self::rnn_reshape(block, &s, &[batch, h], format!("{dp}_c0"), dtype)
            } else {
                Self::rnn_zeros(block, &[batch, h], format!("{dp}_c0"), dtype)
            };
            let mut seq_by_time: Vec<String> = vec![String::new(); steps as usize];
            let order: Vec<u32> = if backward {
                (0..steps).rev().collect()
            } else {
                (0..steps).collect()
            };
            for t in order {
                let xsl = Self::rnn_slice(
                    block,
                    &input_name,
                    &[t, 0, 0],
                    &[1, batch, i],
                    format!("{dp}_x{t}sl"),
                    dtype,
                );
                let xt = Self::rnn_reshape(block, &xsl, &[batch, i], format!("{dp}_x{t}"), dtype);
                let (nh, nc) = Self::emit_lstm_step(
                    block,
                    &xt,
                    &wd,
                    &rd,
                    bd.as_deref(),
                    rbd.as_deref(),
                    pd.as_deref(),
                    &hid,
                    &cell,
                    batch,
                    i,
                    h,
                    layout,
                    f0,
                    f1,
                    f2,
                    &format!("{dp}_t{t}"),
                    dtype,
                );
                hid = nh;
                cell = nc;
                seq_by_time[t as usize] = hid.clone();
            }
            per_dir_h.push(hid);
            per_dir_c.push(cell);
            per_dir_seq.push(seq_by_time);
        }

        let out_h = operand_name(graph, out_ids[0]);
        Self::rnn_pack_final(block, &per_dir_h, batch, h, &base, out_h, dtype);
        let out_c = operand_name(graph, out_ids[1]);
        Self::rnn_pack_final(
            block,
            &per_dir_c,
            batch,
            h,
            &format!("{base}_c"),
            out_c,
            dtype,
        );
        if out_ids.len() > 2 {
            let out_seq = operand_name(graph, out_ids[2]);
            Self::rnn_pack_sequence(
                block,
                &per_dir_seq,
                steps,
                [batch, h],
                &base,
                out_seq,
                dtype,
            );
        }
        Ok(())
    }

    /// Create inputs map for MIL operation
    fn create_operation_inputs(
        &self,
        graph: &GraphInfo,
        op: &Operation,
        input_names: &[String],
    ) -> Result<HashMap<String, Argument>, GraphError> {
        let mut inputs = HashMap::new();

        match &op {
            // Binary operations: x, y
            Operation::Add { .. }
            | Operation::Sub { .. }
            | Operation::Mul { .. }
            | Operation::Div { .. }
            | Operation::Pow { .. }
            | Operation::Max { .. }
            | Operation::Min { .. }
            | Operation::Equal { .. }
            | Operation::Greater { .. }
            | Operation::GreaterOrEqual { .. }
            | Operation::Lesser { .. }
            | Operation::LesserOrEqual { .. }
            | Operation::LogicalAnd { .. }
            | Operation::LogicalOr { .. }
            | Operation::LogicalXor { .. }
                if input_names.len() >= 2 => {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("y".to_string(), Self::create_argument(&input_names[1]));
                }

            // MatMul operation: x, y, transpose_x, transpose_y
            // CoreML requires transpose parameters, WebNN doesn't have them so default to false
            Operation::Matmul { .. } => {
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
            Operation::Gemm { options, .. } => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("y".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add transpose parameters from operator options
                if let Some(opts) = options {
                    inputs.insert(
                        "transpose_x".to_string(),
                        Self::create_immediate_bool(opts.a_transpose),
                    );
                    inputs.insert(
                        "transpose_y".to_string(),
                        Self::create_immediate_bool(opts.b_transpose),
                    );
                }

                // Note: alpha, beta, and bias (C) are not yet supported
                // These would require decomposing gemm into multiple operations:
                // 1. matmul(op(A), op(B))
                // 2. mul by alpha if != 1.0
                // 3. add beta * C if C is provided
            }

            // Global pooling operations (reduce over spatial dimensions)
            Operation::GlobalAveragePool { .. } | Operation::GlobalMaxPool { .. } => {
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

            // Softmax operation (axis is required by WebNN spec)
            Operation::Softmax { axis, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                inputs.insert("axis".to_string(), Self::create_immediate_int(*axis));
            }

            // Neg operation: implemented as mul by -1, requires x and y parameters
            // CoreML neg is actually a mul operation, so we need both operands
            Operation::Neg { .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Add -1.0 as the multiplier (y parameter required by CoreML mul)
                inputs.insert("y".to_string(), Self::create_immediate_float(-1.0));
            }

            // Unary operations: x
            Operation::Relu { .. }
            | Operation::Sigmoid { .. }
            | Operation::Tanh { .. }
            | Operation::Abs { .. }
            | Operation::Ceil { .. }
            | Operation::Floor { .. }
            | Operation::RoundEven { .. }
            | Operation::Sign { .. }
            | Operation::Identity { .. }
            | Operation::Exp { .. }
            | Operation::Sqrt { .. }
            | Operation::Sin { .. }
            | Operation::Cos { .. }
            | Operation::Tan { .. }
            | Operation::Erf { .. }
            | Operation::LogicalNot { .. }
            | Operation::Softplus { .. }
            | Operation::Softsign { .. }
                if !input_names.is_empty() => {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

            Operation::Reciprocal { .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // Reciprocal requires epsilon parameter (default to 1e-45 for numerical stability)
                inputs.insert(
                    "epsilon".to_string(),
                    Self::create_immediate_float(DEFAULT_EPSILON),
                );
            }

            // Log operation requires epsilon parameter
            Operation::Log { .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // CoreML log requires epsilon parameter (default to 1e-45 for numerical stability)
                inputs.insert(
                    "epsilon".to_string(),
                    Self::create_immediate_float(DEFAULT_EPSILON),
                );
            }

            // Quantization operations: input, scale, zero_point[, axis for per-channel]
            Operation::DequantizeLinear { input: inp_id, scale: scale_id, .. }
            | Operation::QuantizeLinear { input: inp_id, scale: scale_id, .. }
                if input_names.len() >= 2 => {
                    inputs.insert("input".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("scale".to_string(), Self::create_argument(&input_names[1]));
                    if input_names.len() >= 3 {
                        inputs.insert(
                            "zero_point".to_string(),
                            Self::create_argument(&input_names[2]),
                        );
                    }
                    // When scale is truly per-channel (one non-unit dim), CoreML requires
                    // an explicit axis. For per-tensor (scalar or single-element), omit axis.
                    // Note: single-element scales [1] are squeezed to scalar at constant emit
                    // time; emitting axis alongside a scalar scale causes a CoreML compile error.
                    // Multi-dimensional scales are squeezed to 1D at emission time (all size-1
                    // dimensions removed). qdq_native_supported gated this op, so every
                    // per-channel scale reaching here has a valid derived axis (rank-aligned
                    // derivation disambiguates square/coincident input dims).
                    if let Some(scale_op) = graph.operand(*scale_id) {
                        let scale_shape = scale_op.descriptor.static_or_max_shape();
                        let axis = graph.operand(*inp_id).and_then(|inp| {
                            Self::qdq_per_channel_axis(
                                &inp.descriptor.static_or_max_shape(),
                                &scale_shape,
                            )
                        });
                        if let Some(axis) = axis {
                            inputs.insert(
                                "axis".to_string(),
                                Self::create_immediate_int(axis as u32),
                            );
                        }
                    }
                }

            // Specialized activation: prelu - x, slope (two inputs)
            Operation::Prelu { .. }
                if input_names.len() >= 2 => {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("alpha".to_string(), Self::create_argument(&input_names[1]));
                }

            // Specialized activations with alpha parameter: elu, leakyRelu
            Operation::Elu { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                let alpha = options.as_ref().map(|o| o.alpha as f32).unwrap_or(1.0);
                inputs.insert("alpha".to_string(), Self::create_immediate_float(alpha));
            }
            Operation::LeakyRelu { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                let alpha = options.as_ref().map(|o| o.alpha as f32).unwrap_or(0.01);
                inputs.insert("alpha".to_string(), Self::create_immediate_float(alpha));
            }

            // HardSwish: decomposed in main loop (hardsigmoid + mul)
            // This case should never be reached due to continue in main loop
            Operation::HardSwish { .. } => {
                return Err(GraphError::ConversionFailed {
                    format: "coreml_mlprogram".to_string(),
                    reason: "hardswish should be decomposed in main loop, not here".to_string(),
                });
            }

            // HardSigmoid: x, alpha, beta parameters
            Operation::HardSigmoid { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if let Some(opts) = options {
                    inputs.insert(
                        "alpha".to_string(),
                        Self::create_immediate_float(opts.alpha as f32),
                    );
                    inputs.insert(
                        "beta".to_string(),
                        Self::create_immediate_float(opts.beta as f32),
                    );
                }
            }

            // Clamp operation: x, alpha (min), beta (max)
            Operation::Clamp { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // CoreML clip operation requires BOTH alpha and beta parameters
                // WebNN clamp defaults: minValue=-Infinity, maxValue=+Infinity
                let (min_value, max_value) = options
                    .as_ref()
                    .map(|o| {
                        let min =
                            Self::parse_clamp_bound(o.min_value.as_ref(), f64::NEG_INFINITY) as f32;
                        let max =
                            Self::parse_clamp_bound(o.max_value.as_ref(), f64::INFINITY) as f32;
                        (min, max)
                    })
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY));

                // Alpha and beta must match input type (CoreML requirement)
                // Check first input operand type and use appropriate immediate value method
                let use_float16 = if !op.input_operands().is_empty() {
                    if let Some(input_operand) = graph.operand(op.input_operands()[0]) {
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
            Operation::Transpose { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add permutation parameter (required by CoreML)
                // If not specified in WebNN, default is to reverse all dimensions
                if let Some(opts) = options
                    && !opts.permutation.is_empty()
                {
                    inputs.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&opts.permutation),
                    );
                } else if !op.input_operands().is_empty()
                    && let Some(input_operand) = graph.operand(op.input_operands()[0])
                {
                    let rank = input_operand.descriptor.shape.len();
                    let default_perm: Vec<u32> = (0..rank).rev().map(|i| i as u32).collect();
                    inputs.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&default_perm),
                    );
                }
            }

            // Reshape: x, shape
            Operation::Reshape { new_shape, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // MIL `reshape` declares `shape` as a required input. Always emit
                // it, including the valid WebNN scalar-output case (`new_shape = []`)
                let shape_values = if new_shape.is_empty() {
                    Vec::new()
                } else {
                    crate::operator_options::mldimensions_static_or_max(new_shape)
                };
                inputs.insert(
                    "shape".to_string(),
                    Self::create_immediate_int_array(&shape_values),
                );
            }

            // Convolution operations: input, filter + parameters
            Operation::Conv2d { options, .. } => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("weight".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add optional bias if present (third input)
                if input_names.len() >= 3 {
                    inputs.insert("bias".to_string(), Self::create_argument(&input_names[2]));
                }

                // MIL `conv` requires `strides`, `pad`, `dilations`, `groups` — all
                // four are declared as required inputs in the MIL op schema, so
                // Apple's CoreML loader rejects the model with
                // "Required param '...' is missing" when any is omitted. Emit the
                // WebNN defaults when the WebNN graph left them unset.
                let (strides, dilations, padding, groups) = match options {
                    Some(o) => (
                        if o.strides.is_empty() {
                            vec![1, 1]
                        } else {
                            o.strides.clone()
                        },
                        if o.dilations.is_empty() {
                            vec![1, 1]
                        } else {
                            o.dilations.clone()
                        },
                        if o.padding.is_empty() {
                            vec![0, 0, 0, 0]
                        } else {
                            o.padding.clone()
                        },
                        o.groups,
                    ),
                    None => (vec![1, 1], vec![1, 1], vec![0, 0, 0, 0], 1),
                };
                inputs.insert(
                    "strides".to_string(),
                    Self::create_immediate_int_array(&strides),
                );
                inputs.insert(
                    "dilations".to_string(),
                    Self::create_immediate_int_array(&dilations),
                );
                inputs.insert(
                    "pad".to_string(),
                    Self::create_immediate_int_array(&padding),
                );
                inputs.insert("groups".to_string(), Self::create_immediate_int(groups));

                // Add pad_type - required parameter in CoreML
                // Use "custom" when explicit padding is provided
                inputs.insert(
                    "pad_type".to_string(),
                    Self::create_immediate_string("custom"),
                );
            }

            // Transposed convolution: input, filter + parameters
            Operation::ConvTranspose2d { options, .. } => {
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("weight".to_string(), Self::create_argument(&input_names[1]));
                }

                // Add optional bias if present (from options, not input_names)
                if let Some(opts) = options
                    && let Some(bias_id) = opts.bias {
                        inputs.insert(
                            "bias".to_string(),
                            Self::create_argument(&operand_name(graph, bias_id)),
                        );
                    }

                // CoreML requires pad_type parameter (defaults to "custom" for explicit padding)
                inputs.insert(
                    "pad_type".to_string(),
                    Self::create_immediate_string("custom"),
                );

                // MIL `conv_transpose` requires `strides`, `pad`, `dilations`,
                // `groups` — same as `conv` above. Apple's loader emits
                // "Required param '...' is missing" when any is dropped, even
                // when the WebNN graph left the attribute at its default.
                let (strides, dilations, padding, groups) = match options {
                    Some(o) => (
                        if o.strides.is_empty() {
                            vec![1, 1]
                        } else {
                            o.strides.clone()
                        },
                        if o.dilations.is_empty() {
                            vec![1, 1]
                        } else {
                            o.dilations.clone()
                        },
                        if o.padding.is_empty() {
                            vec![0, 0, 0, 0]
                        } else {
                            o.padding.clone()
                        },
                        o.groups,
                    ),
                    None => (vec![1, 1], vec![1, 1], vec![0, 0, 0, 0], 1),
                };
                inputs.insert(
                    "strides".to_string(),
                    Self::create_immediate_int_array(&strides),
                );
                inputs.insert(
                    "dilations".to_string(),
                    Self::create_immediate_int_array(&dilations),
                );
                inputs.insert(
                    "pad".to_string(),
                    Self::create_immediate_int_array(&padding),
                );
                inputs.insert("groups".to_string(), Self::create_immediate_int(groups));

                // MIL conv_transpose takes the complete output tensor shape,
                // not only WebNN's two spatial outputSizes values. Chromium
                // supplies this for every transposed convolution as well.
                if let Some(output_id) = op.output_operand()
                    && let Some(output) = graph.operand(output_id)
                {
                    let mut output_shape = output.descriptor.static_or_max_shape();
                    let input_layout = options
                        .as_ref()
                        .map(|o| o.input_layout.as_str())
                        .unwrap_or("");
                    if input_layout.eq_ignore_ascii_case("nhwc") && output_shape.len() == 4 {
                        output_shape = vec![
                            output_shape[0],
                            output_shape[3],
                            output_shape[1],
                            output_shape[2],
                        ];
                    }
                    inputs.insert(
                        "output_shape".to_string(),
                        Self::create_immediate_int_array(&output_shape),
                    );
                }
            }

            // Pooling operations: input + parameters
            Operation::AveragePool2d {
                options: pool_opts, ..
            }
            | Operation::MaxPool2d {
                options: pool_opts, ..
            }
            | Operation::L2Pool2d {
                options: pool_opts, ..
            } => {
                // NHWC pooling is handled by emitting NHWC→NCHW transpose wrappers in the
                // main convert loop before reaching here. By the time we get here the input
                // is already in NCHW form, but the graph operands remain in WebNN layout.
                let is_nhwc = pool_opts
                    .as_ref()
                    .map(|o| o.layout.eq_ignore_ascii_case("nhwc"))
                    .unwrap_or(false);

                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // CoreML pooling has no dilation parameter.
                if let Some(dil) = pool_opts.as_ref().map(|o| &o.dilations)
                    && !dil.is_empty()
                    && dil.iter().any(|&d| d != 1)
                {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!(
                            "CoreML pooling does not support non-default dilations; got {:?} for {}",
                            dil,
                            op.op_type()
                        ),
                    });
                }

                // Kernel sizes: explicit windowDimensions, else the full spatial extent.
                let kernel: Vec<u32> = pool_opts
                    .as_ref()
                    .and_then(|o| o.window_dimensions.clone())
                    .filter(|w| !w.is_empty())
                    .or_else(|| {
                        op.input_operands()
                            .first()
                            .and_then(|&id| graph.operand(id))
                            .map(|o| o.descriptor.static_or_max_shape())
                            .filter(|s| s.len() >= 4)
                            .map(|s| {
                                if is_nhwc {
                                    vec![s[1], s[2]]
                                } else {
                                    vec![s[2], s[3]]
                                }
                            })
                    })
                    .unwrap_or_else(|| vec![1, 1]);
                inputs.insert(
                    "kernel_sizes".to_string(),
                    Self::create_immediate_int_array(&kernel),
                );

                let strides = pool_opts
                    .as_ref()
                    .map(|o| o.strides.clone())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| vec![1, 1]);
                inputs.insert(
                    "strides".to_string(),
                    Self::create_immediate_int_array(&strides),
                );

                // Fold outputShapeRounding="ceil"/outputSizes into extra end-padding and
                // pool with floor rounding + custom padding (see pool_effective_padding).
                let base_pad = pool_opts
                    .as_ref()
                    .map(|o| o.padding.clone())
                    .filter(|p| !p.is_empty())
                    .unwrap_or_else(|| vec![0, 0, 0, 0]);
                let pad = Self::pool_effective_padding(
                    graph, op, &kernel, &strides, &base_pad, is_nhwc,
                )
                .unwrap_or(base_pad);
                inputs.insert("pad".to_string(), Self::create_immediate_int_array(&pad));
                inputs.insert(
                    "pad_type".to_string(),
                    Self::create_immediate_string("custom"),
                );
                inputs.insert("ceil_mode".to_string(), Self::create_immediate_bool(false));

                // WebNN averagePool2d excludes padded elements from the divisor (a
                // boundary window covering N real elements divides by N, not the kernel
                // area). Only average pooling accepts this parameter. Do NOT
                // switch this to false: CoreML then miscounts the divisor for
                // asymmetric custom padding (end-side excluded, begin included).
                if matches!(&op, Operation::AveragePool2d { .. }) {
                    inputs.insert(
                        "exclude_padding_from_average".to_string(),
                        Self::create_immediate_bool(true),
                    );
                }
            }

            // Layer normalization (different from batch/instance normalization)
            Operation::LayerNormalization { options, .. } => {
                // Build sorted axes vector (CoreML requires sorted axes; empty axes are handled
                // as a special case in the main convert loop before reaching here).
                // axes=None means "last axis" per WebNN spec.
                let mut axes_vec: Vec<i32> =
                    if let Some(ax) = options.as_ref().and_then(|o| o.axes.as_ref()) {
                        ax.iter().map(|&u| u as i32).collect()
                    } else {
                        // axes=None: WebNN defaults to all axes except the batch
                        // dimension, i.e. the sequence [1, 2, ..., N-1].
                        let input_rank = op
                            .input_operands()
                            .first()
                            .and_then(|&id| graph.operand(id))
                            .map(|o| o.descriptor.shape.len())
                            .unwrap_or(1);
                        if input_rank > 1 {
                            (1..input_rank as i32).collect()
                        } else {
                            vec![0]
                        }
                    };
                axes_vec.sort_unstable();

                // Add input operand (only x; scale/bias come from options, not input_operands)
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Scale (gamma) and bias (beta) are stored in options, not in input_operands.
                if let Some(opts) = options {
                    if let Some(scale_id) = opts.scale {
                        inputs.insert(
                            "gamma".to_string(),
                            Self::create_argument(&operand_name(graph, scale_id)),
                        );
                    }
                    if let Some(bias_id) = opts.bias {
                        inputs.insert(
                            "beta".to_string(),
                            Self::create_argument(&operand_name(graph, bias_id)),
                        );
                    }
                    let use_f16 = op
                        .input_operands()
                        .first()
                        .and_then(|&id| graph.operand(id))
                        .map(|o| o.descriptor.data_type == DataType::Float16)
                        .unwrap_or(false);
                    let eps_arg = if use_f16 {
                        Self::create_immediate_float16(opts.epsilon as f32)
                    } else {
                        Self::create_immediate_float(opts.epsilon as f32)
                    };
                    inputs.insert("epsilon".to_string(), eps_arg);
                }

                // Add axes parameter (REQUIRED by CoreML, must not be empty;
                // empty axes are caught before this point).
                inputs.insert(
                    "axes".to_string(),
                    Self::create_int_array_argument(axes_vec),
                );
            }

            // Batch/instance normalization (have mean, variance inputs)
            Operation::BatchNormalization { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if input_names.len() >= 2 && op.input_operands().len() >= 2 {
                    let mean_operand_id = op.input_operands()[1];
                    if let Some(mean_operand) = graph.operand(mean_operand_id)
                        && mean_operand.kind != crate::graph::OperandKind::Constant
                    {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "CoreML {} requires mean parameter to be a constant tensor, not a graph input",
                                op.op_type()
                            ),
                        });
                    }
                    inputs.insert("mean".to_string(), Self::create_argument(&input_names[1]));
                }
                if input_names.len() >= 3 && op.input_operands().len() >= 3 {
                    let variance_operand_id = op.input_operands()[2];
                    if let Some(variance_operand) = graph.operand(variance_operand_id)
                        && variance_operand.kind != crate::graph::OperandKind::Constant
                    {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "CoreML {} requires variance parameter to be a constant tensor, not a graph input",
                                op.op_type()
                            ),
                        });
                    }
                    inputs.insert(
                        "variance".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                }
                if let Some(opts) = options {
                    if let Some(sid) = opts.scale {
                        inputs.insert(
                            "gamma".to_string(),
                            Self::create_argument(&operand_name(graph, sid)),
                        );
                    } else if input_names.len() >= 4 {
                        inputs.insert("gamma".to_string(), Self::create_argument(&input_names[3]));
                    }
                    if let Some(bid) = opts.bias {
                        inputs.insert(
                            "beta".to_string(),
                            Self::create_argument(&operand_name(graph, bid)),
                        );
                    } else if input_names.len() >= 5 {
                        inputs.insert("beta".to_string(), Self::create_argument(&input_names[4]));
                    }
                    let use_f16_bn = op
                        .input_operands()
                        .first()
                        .and_then(|&id| graph.operand(id))
                        .map(|o| o.descriptor.data_type == DataType::Float16)
                        .unwrap_or(false);
                    let eps_bn = if use_f16_bn {
                        Self::create_immediate_float16(opts.epsilon as f32)
                    } else {
                        Self::create_immediate_float(opts.epsilon as f32)
                    };
                    inputs.insert("epsilon".to_string(), eps_bn);
                }
            }
            Operation::InstanceNormalization { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if let Some(opts) = options {
                    if let Some(scale_id) = opts.scale {
                        inputs.insert(
                            "gamma".to_string(),
                            Self::create_argument(&operand_name(graph, scale_id)),
                        );
                    }
                    if let Some(bias_id) = opts.bias {
                        inputs.insert(
                            "beta".to_string(),
                            Self::create_argument(&operand_name(graph, bias_id)),
                        );
                    }
                    let use_f16_in = op
                        .input_operands()
                        .first()
                        .and_then(|&id| graph.operand(id))
                        .map(|o| o.descriptor.data_type == DataType::Float16)
                        .unwrap_or(false);
                    let eps_in = if use_f16_in {
                        Self::create_immediate_float16(opts.epsilon as f32)
                    } else {
                        Self::create_immediate_float(opts.epsilon as f32)
                    };
                    inputs.insert("epsilon".to_string(), eps_in);
                }
            }

            Operation::Concat { axis, .. } => {
                // concat: values (variadic list of tensors), axis
                // CoreML expects a single 'values' parameter containing a tuple of all inputs
                if !input_names.is_empty() {
                    inputs.insert(
                        "values".to_string(),
                        Self::create_argument_tuple(input_names),
                    );
                }

                inputs.insert("axis".to_string(), Self::create_immediate_int(*axis));
                inputs.insert("interleave".to_string(), Self::create_immediate_bool(false));
            }

            Operation::Slice {
                starts,
                sizes,
                options,
                ..
            } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                let strides = options
                    .as_ref()
                    .map(|o| o.strides.as_slice())
                    .unwrap_or(&[]);
                let has_nontrivial_strides =
                    strides.iter().any(|&s| s != 1) && !strides.is_empty();

                if has_nontrivial_strides {
                    // slice_by_index: begin, end (exclusive), stride.
                    // CoreML produces ceil((end - begin) / stride) per axis.
                    // WebNN sizes[i] = window width (input elements spanned), so:
                    //   end[i] = starts[i] + sizes[i]
                    //   stride[i] = strides[i]
                    inputs.insert(
                        "begin".to_string(),
                        Self::create_immediate_int_array(starts),
                    );
                    let ends: Vec<u32> = starts
                        .iter()
                        .zip(sizes.iter())
                        .map(|(&s, d)| s + d.static_or_max())
                        .collect();
                    inputs.insert("end".to_string(), Self::create_immediate_int_array(&ends));
                    inputs.insert(
                        "stride".to_string(),
                        Self::create_immediate_int_array(strides),
                    );
                } else {
                    // slice_by_size: x, begin, size. Both `begin` and `size` are
                    // declared required inputs in MIL's slice_by_size schema. Apple
                    // rejects the model with "Required param 'size' is missing"
                    // when an empty-shape no-op slice (0D scalar, WPT surfaces this)
                    // is emitted without the fields. Emit them as empty int32 arrays
                    // in that case so the MIL loader sees the param even though the
                    // tensor is rank-0.
                    inputs.insert(
                        "begin".to_string(),
                        Self::create_immediate_int_array(starts),
                    );
                    let sizes_u32: Vec<u32> = sizes.iter().map(|d| d.static_or_max()).collect();
                    inputs.insert(
                        "size".to_string(),
                        Self::create_immediate_int_array(&sizes_u32),
                    );
                }
            }

            Operation::Expand { new_shape, .. } => {
                // CoreML tile operation requires input rank to match reps length
                // If reshape was added before this operation, use reshaped input name
                //  Otherwise use original input

                if let Some(new_shape_u32) = (!new_shape.is_empty()).then(|| {
                    new_shape
                        .iter()
                        .map(MLDimension::static_or_max)
                        .collect::<Vec<u32>>()
                }) {
                    // Get input operand shape
                    if !op.input_operands().is_empty()
                        && let Some(input_operand) = graph.operand(op.input_operands()[0])
                    {
                        let input_shape = input_operand.descriptor.static_or_max_shape();
                        let input_rank = input_shape.len();
                        let output_rank = new_shape_u32.len();

                        // Determine input name for tile operation
                        let tile_input_name = if input_rank < output_rank {
                            // Matches the producer site's output-derived name.
                            format!(
                                "{}_expand_reshaped",
                                operand_name(
                                    graph,
                                    op.output_operand().unwrap_or(op.input_operands()[0])
                                )
                            )
                        } else {
                            // No reshape, use original input
                            input_names[0].clone()
                        };

                        inputs.insert("x".to_string(), Self::create_name_argument(tile_input_name));

                        // Create reshaped dimensions (right-aligned, padded with 1s on left)
                        let mut reshaped_dims = vec![1u32; output_rank];
                        for i in 0..input_rank {
                            reshaped_dims[output_rank - i - 1] = input_shape[input_rank - i - 1];
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

                        inputs.insert("reps".to_string(), Self::create_int_array_argument(reps));
                    }
                }
            }

            Operation::Gather { options, .. } => {
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
                let axis = options.as_ref().map(|o| o.axis).unwrap_or(0);
                inputs.insert("axis".to_string(), Self::create_immediate_int(axis));

                // Add validate_indices parameter (required by CoreML)
                // Chromium sets this to false to avoid validation issues
                // TODO: Handle negative and out-of-bounds indices properly
                inputs.insert(
                    "validate_indices".to_string(),
                    Self::create_immediate_bool(false),
                );
            }

            Operation::GatherElements { options, .. } => {
                // gather_along_axis: x, indices, axis, validate_indices
                if input_names.len() >= 2 {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                }

                let axis = options.as_ref().map(|o| o.axis).unwrap_or(0);
                inputs.insert("axis".to_string(), Self::create_immediate_int(axis));

                inputs.insert(
                    "validate_indices".to_string(),
                    Self::create_immediate_bool(false),
                );
            }

            Operation::Split {
                splits, options, ..
            } => {
                // split: x, num_splits or split_sizes, axis
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if let Some(opts) = options {
                    if splits.is_empty() {
                        inputs.insert(
                            "num_splits".to_string(),
                            Self::create_immediate_int(op.output_operands().len() as u32),
                        );
                    } else {
                        inputs.insert(
                            "split_sizes".to_string(),
                            Self::create_immediate_int_array(splits),
                        );
                    }
                    inputs.insert("axis".to_string(), Self::create_immediate_int(opts.axis));
                }
            }

            Operation::Where { .. }
                // select: cond, a (true_value), b (false_value)
                if input_names.len() >= 3 => {
                    inputs.insert("cond".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert("a".to_string(), Self::create_argument(&input_names[1]));
                    inputs.insert("b".to_string(), Self::create_argument(&input_names[2]));
                }

            Operation::Pad {
                beginning_padding,
                ending_padding,
                options,
                ..
            } => {
                // pad: x, pad, mode, constant_val
                // All four params are required by CoreML (even when using defaults).
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // CoreML expects pad as [begin_0, end_0, begin_1, end_1, ...]
                let pad: Vec<u32> = beginning_padding
                    .iter()
                    .zip(ending_padding.iter())
                    .flat_map(|(a, b)| [*a, *b])
                    .collect();
                inputs.insert("pad".to_string(), Self::create_immediate_int_array(&pad));

                // Mode: WebNN → CoreML mapping
                // WebNN: "constant", "edge", "reflection", "symmetric"
                // CoreML: "constant", "replicate", "reflect"
                let webnn_mode = options
                    .as_ref()
                    .map(|o| o.mode.as_str())
                    .unwrap_or("constant");
                let coreml_mode = match webnn_mode {
                    "edge" => "replicate",
                    "reflection" | "symmetric" => "reflect",
                    _ => "constant",
                };
                inputs.insert(
                    "mode".to_string(),
                    Self::create_immediate_string(coreml_mode),
                );

                // constant_val must match the dtype of the input tensor.
                let constant_val = options
                    .as_ref()
                    .and_then(|o| Self::parse_mlnumber_f64(o.value.as_ref()))
                    .unwrap_or(0.0);
                let input_dtype = op
                    .input_operands()
                    .first()
                    .and_then(|&id| graph.operand(id))
                    .map(|o| &o.descriptor.data_type)
                    .cloned()
                    .unwrap_or(DataType::Float32);
                let cval_arg = match input_dtype {
                    DataType::Float16 => Self::create_immediate_float16(constant_val as f32),
                    DataType::Int32 => Self::create_immediate_int(constant_val as u32),
                    DataType::Int8 | DataType::Uint8 => {
                        Self::create_immediate_int(constant_val as u32)
                    }
                    _ => Self::create_immediate_float(constant_val as f32),
                };
                inputs.insert("constant_val".to_string(), cval_arg);
            }

            Operation::Gelu { .. } => {
                // gelu: x, mode
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                // watchOS MIL loader rejects gelu without an explicit mode ("Required
                // param 'mode' is missing"); iOS/macOS loaders accept the default.
                // WebNN spec has no mode parameter — exact (erf) is the implicit default.
                inputs.insert("mode".to_string(), Self::create_immediate_string("EXACT"));
            }

            Operation::Squeeze { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if let Some(opts) = options
                    && !opts.axes.is_empty()
                {
                    inputs.insert(
                        "axes".to_string(),
                        Self::create_immediate_int_array(&opts.axes),
                    );
                }
            }

            Operation::Unsqueeze { options, .. } => {
                // expand_dims: x, axes
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                if let Some(opts) = options
                    && !opts.axes.is_empty()
                {
                    inputs.insert(
                        "axes".to_string(),
                        Self::create_immediate_int_array(&opts.axes),
                    );
                }
            }

            Operation::ArgMax { axis, options, .. } | Operation::ArgMin { axis, options, .. } => {
                // reduce_argmax/reduce_argmin: x, axis, keep_dims
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                inputs.insert("axis".to_string(), Self::create_immediate_int(*axis));
                if let Some(opts) = options {
                    inputs.insert(
                        "keep_dims".to_string(),
                        Self::create_immediate_bool(opts.keep_dimensions),
                    );
                }
                // Note: outputDataType is handled by the output tensor's data type
            }

            Operation::Cast { data_type: to, .. } => {
                // cast: x, dtype
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Add dtype parameter (required)
                let to_type = to;
                let dtype_string = match to_type {
                    MLOperandDataType::Float32 => "fp32",
                    MLOperandDataType::Float16 => "fp16",
                    MLOperandDataType::Int32 => "int32",
                    MLOperandDataType::Uint32 => "uint32",
                    MLOperandDataType::Int8 => "int8",
                    MLOperandDataType::Uint8 => "uint8",
                    MLOperandDataType::Int64 => "int64",
                    MLOperandDataType::Uint64 => {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: "Unsupported graph cast dtype Uint64".to_string(),
                        });
                    }
                    MLOperandDataType::Int4 | MLOperandDataType::Uint4 => {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: "int4/uint4 cast targets are not supported in CoreML conversion"
                                .to_string(),
                        });
                    }
                };
                inputs.insert(
                    "dtype".to_string(),
                    Self::create_immediate_string(dtype_string),
                );
            }

            Operation::ScatterElements { options, .. } => {
                // scatter: data, indices, updates, axis, mode
                // mode is required by CoreML (default "update").
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

                if let Some(opts) = options {
                    inputs.insert("axis".to_string(), Self::create_immediate_int(opts.axis));
                }

                // CoreML requires explicit mode and validate_indices.
                inputs.insert("mode".to_string(), Self::create_immediate_string("update"));
                inputs.insert(
                    "validate_indices".to_string(),
                    Self::create_immediate_bool(false),
                );
            }

            Operation::ScatterND { .. }
                // scatter_nd: data, indices, updates, mode (required by CoreML)
                if input_names.len() >= 3 => {
                    inputs.insert("data".to_string(), Self::create_argument(&input_names[0]));
                    inputs.insert(
                        "indices".to_string(),
                        Self::create_argument(&input_names[1]),
                    );
                    inputs.insert(
                        "updates".to_string(),
                        Self::create_argument(&input_names[2]),
                    );
                    inputs.insert("mode".to_string(), Self::create_immediate_string("update"));
                    inputs.insert(
                        "validate_indices".to_string(),
                        Self::create_immediate_bool(false),
                    );
                }

            Operation::Tile { repetitions, .. } => {
                // tile: x, reps
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // reps is required by CoreML even when all values are 1
                inputs.insert(
                    "reps".to_string(),
                    Self::create_immediate_int_array(repetitions),
                );
            }

            Operation::CumulativeSum { axis, options, .. } => {
                // cumsum: x, axis, exclusive, reverse
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                inputs.insert("axis".to_string(), Self::create_int_argument(*axis as i32));
                if let Some(opts) = options {
                    inputs.insert(
                        "exclusive".to_string(),
                        Self::create_immediate_bool(opts.exclusive),
                    );
                    inputs.insert(
                        "reverse".to_string(),
                        Self::create_immediate_bool(opts.reversed),
                    );
                }
            }

            Operation::Reverse { options, .. } => {
                // reverse: x, axes
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // Default behavior: reverse all axes when options.axes is omitted.
                let axes_u32: Vec<u32> = match options.as_ref() {
                    Some(opts) => match opts.axes.as_ref() {
                        Some(axes) => axes.clone(),
                        None => {
                            if let Some(input_id) = op.input_operands().first() {
                                if let Some(input_operand) = graph.operand(*input_id) {
                                    (0..input_operand.descriptor.shape.len())
                                        .map(|axis| axis as u32)
                                        .collect()
                                } else {
                                    Vec::new()
                                }
                            } else {
                                Vec::new()
                            }
                        }
                    },
                    None => {
                        if let Some(input_id) = op.input_operands().first() {
                            if let Some(input_operand) = graph.operand(*input_id) {
                                (0..input_operand.descriptor.shape.len())
                                    .map(|axis| axis as u32)
                                    .collect()
                            } else {
                                Vec::new()
                            }
                        } else {
                            Vec::new()
                        }
                    }
                };

                // Always provide axes, including empty arrays (explicit no-op).
                inputs.insert(
                    "axes".to_string(),
                    Self::create_immediate_int_array(&axes_u32),
                );
            }

            Operation::Triangular { options, .. } => {
                // band_part: x, lower, upper
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                // CoreML band_part uses lower and upper bounds instead of upper/diagonal
                let is_upper = options.as_ref().and_then(|o| o.upper).unwrap_or(true);
                let diagonal = options.as_ref().map(|o| o.diagonal as i64).unwrap_or(0);

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

            Operation::ReduceLogSumExp { options, .. } => {
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }
                if let Some(opts) = options {
                    if let Some(axes) = opts.axes.as_ref()
                        && !axes.is_empty()
                    {
                        inputs.insert("axes".to_string(), Self::create_immediate_int_array(axes));
                    }
                    inputs.insert(
                        "keep_dims".to_string(),
                        Self::create_immediate_bool(opts.keep_dimensions),
                    );
                }
            }

            // Reduction operations: reduceSum, reduceMean, reduceMax, etc.
            Operation::ReduceSum { options, .. }
            | Operation::ReduceMean { options, .. }
            | Operation::ReduceMax { options, .. }
            | Operation::ReduceMin { options, .. }
            | Operation::ReduceProduct { options, .. }
            | Operation::ReduceL1 { options, .. }
            | Operation::ReduceL2 { options, .. }
            | Operation::ReduceLogSum { options, .. }
            | Operation::ReduceSumSquare { options, .. } => {
                // All reduce operations: x, axes, keep_dims
                if !input_names.is_empty() {
                    inputs.insert("x".to_string(), Self::create_argument(&input_names[0]));
                }

                if let Some(opts) = options {
                    if let Some(axes) = opts.axes.as_ref()
                        && !axes.is_empty()
                    {
                        inputs.insert("axes".to_string(), Self::create_immediate_int_array(axes));
                    }
                    inputs.insert(
                        "keep_dims".to_string(),
                        Self::create_immediate_bool(opts.keep_dimensions),
                    );
                }
            }

            Operation::GatherND { input, indices, .. } => {
                // CoreML gather_nd crashes (SIGABRT) on rank-5+ inputs; guard against it.
                if let Some(input_op) = graph.operand(*input)
                    && input_op.descriptor.shape.len() > 4 {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "CoreML gather_nd does not support input rank > 4; got rank {}",
                                input_op.descriptor.shape.len()
                            ),
                        });
                    }
                // gather_nd: x (data), indices, validate_indices (required by CoreML)
                inputs.insert(
                    "x".to_string(),
                    Self::create_argument(&operand_name(graph, *input)),
                );
                inputs.insert(
                    "indices".to_string(),
                    Self::create_argument(&operand_name(graph, *indices)),
                );
                inputs.insert(
                    "validate_indices".to_string(),
                    Self::create_immediate_bool(false),
                );
            }

            // isNaN and isInfinite: single input 'x'
            Operation::IsNaN { input, .. } | Operation::IsInfinite { input, .. } => {
                inputs.insert(
                    "x".to_string(),
                    Self::create_argument(&operand_name(graph, *input)),
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
            // Unsupported types - assume they have been converted to FLOAT32.
            DataType::Int4
            | DataType::Uint4
            | DataType::Int8
            | DataType::Uint8
            | DataType::Uint32
            | DataType::Int64
            | DataType::Uint64 => {
                crate::protos::coreml::specification::array_feature_type::ArrayDataType::Float32
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
            descriptor.static_or_max_shape()
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
        if !crate::graph::dynamic_inputs_enabled() && graph_info.has_dynamic_dimensions() {
            return Err(GraphError::DynamicInputsFeatureDisabled);
        }

        // Create weight file builder for Float16 constants
        let mut weight_builder = super::WeightFileBuilder::new();
        // Upper bound: every constant blob-written, plus a 64-byte metadata
        // block and up-to-64-byte alignment padding each. Reserved capacity is
        // virtual until written, so overshooting for immediate-value constants
        // costs nothing.
        weight_builder.reserve(
            graph_info
                .constant_operand_ids_to_handles
                .values()
                .map(|c| c.data.len() + 128)
                .sum(),
        );

        // Create MLProgram
        let mut program = Program {
            version: 1,
            ..Default::default()
        };

        // Create main function
        let mut main_function = Function::default();

        // Keep MLProgram boundary types aligned with CoreML feature-description
        // restrictions. Unsupported WebNN I/O types (such as uint8) are exposed
        // as float32 at the function boundary and cast to/from the internal
        // graph representation inside the main block.
        let mut operand_name_overrides: HashMap<u32, String> = HashMap::new();

        for &output_id in &graph_info.output_operands {
            if let Some(operand) = graph_info.operand(output_id) {
                let graph_mil_type = Self::graph_value_mil_type(&operand.descriptor.data_type)?;
                // Wide ints (int64/uint32/uint64) are int32 both inside the graph and
                // at the interface (an int32 proxy), so they need no boundary cast.
                let interface_mil_type = if Self::is_wide_int(&operand.descriptor.data_type) {
                    crate::protos::coreml::mil_spec::DataType::Int32 as i32
                } else {
                    Self::interface_mil_data_type(&operand.descriptor.data_type)
                };
                if graph_mil_type != interface_mil_type {
                    let output_name = operand_name(graph_info, output_id);
                    operand_name_overrides.insert(output_id, format!("{}_graph", output_name));
                }
            }
        }

        // Add function inputs from graph inputs
        for &input_id in &graph_info.input_operands {
            let operand =
                graph_info
                    .operand(input_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Input operand {} not found", input_id),
                    })?;
            let input_name = operand_name(graph_info, input_id);
            let value_type = Self::create_value_with_mil_type(
                graph_info,
                input_id,
                input_name,
                Self::interface_mil_data_type(&operand.descriptor.data_type),
            )?;
            main_function.inputs.push(value_type);
        }

        // Create main block
        let mut main_block = Block::default();
        // Tracks outputs that are proxied as int32 (e.g. argmin/argmax with int64 WebNN type).
        // The final interface-cast loop uses this to cast to int32 rather than float32.
        let mut int32_proxy_output_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for &input_id in &graph_info.input_operands {
            let operand =
                graph_info
                    .operand(input_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Input operand {} not found", input_id),
                    })?;
            // int4/uint4/int64/uint32/uint64 have no MIL tensor type; represent them
            // internally as int32 (a proxy). The fp32 model input is cast to int32 here;
            // the executor delivers the original integer values widened to float32.
            let is_wide_int = Self::is_wide_int(&operand.descriptor.data_type);
            let graph_mil_type = if is_wide_int {
                crate::protos::coreml::mil_spec::DataType::Int32 as i32
            } else {
                Self::mil_data_type(&operand.descriptor.data_type)?
            };
            let interface_mil_type = Self::interface_mil_data_type(&operand.descriptor.data_type);
            if graph_mil_type != interface_mil_type {
                let input_name = operand_name(graph_info, input_id);
                let graph_input_name = format!("{}_graph", input_name);
                operand_name_overrides.insert(input_id, graph_input_name.clone());
                let graph_input_type = Self::create_value_with_mil_type(
                    graph_info,
                    input_id,
                    graph_input_name,
                    graph_mil_type,
                )?;
                let cast_str = if is_wide_int {
                    "int32"
                } else {
                    Self::cast_dtype_string_for_graph_type(&operand.descriptor.data_type)?
                };
                main_block.operations.push(Self::create_cast_operation(
                    input_name,
                    graph_input_type,
                    cast_str,
                ));
            }
        }

        // MIL `quantize` / `dequantize` require:
        //  • `scale` (and `zero_point`) to be rank 0 (scalar) or rank 1 constants.
        //  • `zero_point` type to match the quantized tensor type (int8 or uint8).
        //    Some WebNN graphs store zero_point as Int32, which CoreML rejects.
        // Pre-scan all quantize/dequantize operations to collect:
        //   scale_ids_to_squeeze  — scale/zp operand IDs with rank > 1 (need shape squeezing)
        //   zp_id_to_dtype        — zero_point operand IDs mapped to their required data type
        let mut scale_ids_to_squeeze: std::collections::HashSet<u32> =
            std::collections::HashSet::new();
        let mut zp_id_to_dtype: HashMap<u32, DataType> = HashMap::new();
        for op in &graph_info.operations {
            let op_lower = op.op_type().to_lowercase();
            if matches!(op_lower.as_str(), "quantizelinear" | "dequantizelinear") {
                let input_ids = op.input_operands();
                // Skip decomposed quantize/dequantize ops: they consume scale/zero_point in
                // their native dtype and shape (see emit_*_decomposition), so the per-channel
                // squeeze and int8 zero_point coercion below must not apply.
                if Self::qdq_should_decompose(graph_info, op) {
                    continue;
                }
                // Squeeze shape for both scale (index 1) and zero_point (index 2).
                // CoreML interprets a rank-1 scale as per-channel; for per-tensor semantics,
                // scale/zp must be scalar. Squeeze rank > 1 AND rank-1 with single element [1].
                for &param_idx in &[1usize, 2usize] {
                    if let Some(&param_id) = input_ids.get(param_idx)
                        && let Some(param_operand) = graph_info.operand(param_id)
                    {
                        let shape = &param_operand.descriptor.shape;
                        let needs_sq = shape.len() > 1
                            || (shape.len() == 1 && matches!(shape[0], GraphDimension::Static(1)));
                        if needs_sq {
                            scale_ids_to_squeeze.insert(param_id);
                        }
                    }
                }
                // Determine the expected zero_point data type.
                // For quantize: zp type = output type (the quantized representation).
                // For dequantize: zp type = input type (the quantized representation).
                // CoreML only accepts int8 or uint8 for zero_point; coerce anything else
                // (e.g. Int32 from some WebNN graphs) to the nearest compatible type.
                // convert_zp_bytes handles Int32→Int8 and Int32→Uint8 byte reinterpretation.
                let zp_webnn_type = if op_lower == "quantizelinear" {
                    op.output_operand()
                        .and_then(|id| graph_info.operand(id))
                        .map(|o| o.descriptor.data_type)
                } else {
                    // dequantize: first input is the quantized tensor
                    input_ids
                        .first()
                        .and_then(|&id| graph_info.operand(id))
                        .map(|o| o.descriptor.data_type)
                };
                let zp_expected_type = zp_webnn_type.map(|dt| match dt {
                    DataType::Uint8 => DataType::Uint8,
                    _ => DataType::Int8,
                });
                if let (Some(&zp_id), Some(expected_dt)) = (input_ids.get(2), zp_expected_type) {
                    zp_id_to_dtype.insert(zp_id, expected_dt);
                }
                // CoreML requires scale's float type to match the float tensor
                // (quantize: the input; dequantize: the output). Coerce constant
                // scales that disagree (e.g. an fp16 tensor with an fp32 scale).
                let float_dt = if op_lower == "quantizelinear" {
                    input_ids
                        .first()
                        .and_then(|&id| graph_info.operand(id))
                        .map(|o| o.descriptor.data_type)
                } else {
                    op.output_operand()
                        .and_then(|id| graph_info.operand(id))
                        .map(|o| o.descriptor.data_type)
                };
                if let (Some(&scale_id), Some(float_dt)) = (input_ids.get(1), float_dt)
                    && matches!(float_dt, DataType::Float32 | DataType::Float16)
                    && let Some(scale_op) = graph_info.operand(scale_id)
                    && matches!(
                        scale_op.descriptor.data_type,
                        DataType::Float32 | DataType::Float16
                    )
                    && scale_op.descriptor.data_type != float_dt
                    && scale_op.kind == crate::graph::OperandKind::Constant
                {
                    zp_id_to_dtype.insert(scale_id, float_dt);
                }
                // For dequantize, CoreML also requires the INPUT (index 0) to be int8 or uint8.
                // If the input is a constant Int32 tensor, coerce its bytes to int8/uint8 so
                // CoreML accepts it. Non-constant Int32 inputs are handled with a cast op below.
                if op_lower == "dequantizelinear"
                    && let Some(&input_id) = input_ids.first()
                    && let Some(input_op) = graph_info.operand(input_id)
                    && matches!(input_op.descriptor.data_type, DataType::Int32)
                {
                    let target = DataType::Int8;
                    zp_id_to_dtype.insert(input_id, target);
                }
            }
        }

        // Gather/GatherElements: CoreML only accepts int32 (or smaller) for indices.
        // If indices are a constant with uint32 or int64 type, coerce them to int32.
        for op in &graph_info.operations {
            let op_lower = op.op_type().to_lowercase();
            if (op_lower == "gather" || op_lower == "gatherelements")
                && let Some(&idx_id) = op.input_operands().get(1)
                && let Some(idx_op) = graph_info.operand(idx_id)
                && matches!(
                    idx_op.descriptor.data_type,
                    DataType::Uint32 | DataType::Int64
                )
                && idx_op.kind == crate::graph::OperandKind::Constant
            {
                zp_id_to_dtype.insert(idx_id, DataType::Int32);
            }
        }

        // Add constant operands as const operations
        for (operand_id, constant_data) in &graph_info.constant_operand_ids_to_handles {
            let operand =
                graph_info
                    .operand(*operand_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Constant operand {} not found", operand_id),
                    })?;

            // For quantize/dequantize scale and zero_point params, we may need to:
            //  (a) squeeze the shape to rank ≤ 1 (both scale and zp)
            //  (b) coerce the dtype to match the quantized tensor (zp only, e.g. Int32 → Uint8)
            // Both may apply simultaneously; apply them together here.
            let needs_squeeze = scale_ids_to_squeeze.contains(operand_id);
            let needs_type_coerce = zp_id_to_dtype
                .get(operand_id)
                .filter(|expected| **expected != operand.descriptor.data_type)
                .cloned();

            if needs_squeeze || needs_type_coerce.is_some() {
                use crate::graph::OperandDescriptor;

                // Compute the squeezed shape (or keep original if no squeezing needed).
                let final_shape: Vec<GraphDimension> = if needs_squeeze {
                    let squeezed: Vec<u32> = operand
                        .descriptor
                        .shape
                        .iter()
                        .filter_map(|d| match d {
                            GraphDimension::Static(v) if *v != 1 => Some(*v),
                            _ => None,
                        })
                        .collect();
                    squeezed
                        .iter()
                        .map(|&v| GraphDimension::Static(v))
                        .collect()
                } else {
                    operand.descriptor.shape.clone()
                };

                // Compute the final dtype (and convert bytes if needed).
                let (final_dtype, final_data_ref, _coerced_storage);
                if let Some(expected_dtype) = needs_type_coerce {
                    let converted = convert_zp_bytes(
                        &constant_data.data,
                        &operand.descriptor.data_type,
                        &expected_dtype,
                    );
                    _coerced_storage = crate::graph::ConstantData {
                        data: converted,
                        label: None,
                    };
                    final_data_ref = &_coerced_storage;
                    final_dtype = expected_dtype;
                } else {
                    _coerced_storage = crate::graph::ConstantData {
                        data: vec![],
                        label: None,
                    };
                    final_data_ref = constant_data;
                    final_dtype = operand.descriptor.data_type;
                };

                let mut modified_operand = operand.clone();
                modified_operand.descriptor = OperandDescriptor {
                    data_type: final_dtype,
                    shape: final_shape,
                    pending_permutation: Vec::new(),
                };
                let const_op = Self::create_const_operation(
                    graph_info,
                    *operand_id,
                    &modified_operand,
                    final_data_ref,
                    &mut weight_builder,
                )?;
                main_block.operations.push(const_op);
            } else {
                let const_op = Self::create_const_operation(
                    graph_info,
                    *operand_id,
                    operand,
                    constant_data,
                    &mut weight_builder,
                )?;
                main_block.operations.push(const_op);
            }
        }

        // Operations that must be inserted right after the op that produces their input operand.
        // Keyed by the source operand_id.  Value = (transpose ops, transposed_name).
        // We intentionally do NOT set operand_name_overrides[id] until the deferred ops are
        // emitted, so that the operation that *produces* the id still writes the original name.
        let mut deferred_transposes: HashMap<u32, (Vec<MilOperation>, String)> = HashMap::new();

        // First pass: Handle filter layout transformations for conv operations

        for op in &graph_info.operations {
            let op_type_lower = op.op_type().to_lowercase();

            if (op_type_lower == "conv2d" || op_type_lower == "convtranspose2d")
                && op.input_operands().len() >= 2
            {
                let filter_layout = match &op {
                    Operation::Conv2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.filter_layout.as_str())
                        .unwrap_or(""),
                    Operation::ConvTranspose2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.filter_layout.as_str())
                        .unwrap_or(""),
                    _ => "",
                };
                if !filter_layout.is_empty() {
                    let expected_layout = if op_type_lower == "conv2d" {
                        "oihw"
                    } else {
                        "iohw"
                    };

                    if filter_layout != expected_layout {
                        let filter_operand_id = op.input_operands()[1];

                        // Dedup: two convs sharing one filter (tied weights) must
                        // not both define {filter}_transposed. This reuses the FIRST
                        // consumer's permutation; sharing a filter under different
                        // layouts would need per-consumer names.
                        if !operand_name_overrides.contains_key(&filter_operand_id)
                            && !deferred_transposes.contains_key(&filter_operand_id)
                            && let Some(filter_operand) = graph_info.operand(filter_operand_id)
                        {
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

                            let mut transpose_inputs: HashMap<String, Argument> = HashMap::new();
                            transpose_inputs
                                .insert("x".to_string(), Self::create_name_argument(filter_name));
                            transpose_inputs.insert(
                                "perm".to_string(),
                                Self::create_immediate_int_array(&perm),
                            );

                            // Create tensor type for transposed filter
                            let dtype = Self::mil_data_type(&filter_operand.descriptor.data_type)?;
                            let transposed_shape =
                                Self::permute_graph_shape(&filter_operand.descriptor.shape, &perm);
                            let dimensions =
                                Self::mil_dimensions_from_graph_shape(&transposed_shape, false);

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

                            // If the filter operand is a constant or graph input it has already
                            // been emitted, so the transpose and the name override can go here.
                            // If it is an intermediate (e.g. output of dequantizeLinear in a QDQ
                            // graph) the producing operation hasn't been emitted yet — defer the
                            // transpose until right after that operation, and defer the override
                            // too (so the producing op writes the *original* name, not the
                            // transposed name).
                            if matches!(
                                filter_operand.kind,
                                OperandKind::Constant | OperandKind::Input
                            ) {
                                // Override can be set now; the filter has already been emitted.
                                operand_name_overrides
                                    .insert(filter_operand_id, transposed_filter_name.clone());
                                main_block.operations.push(transpose_op);
                            } else {
                                // Do NOT set the override yet; set it after the deferred op fires.
                                deferred_transposes
                                    .entry(filter_operand_id)
                                    .or_insert_with(|| (Vec::new(), transposed_filter_name.clone()))
                                    .0
                                    .push(transpose_op);
                            }
                        }
                    }
                }

                let input_layout = match &op {
                    Operation::Conv2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.input_layout.as_str())
                        .unwrap_or(""),
                    Operation::ConvTranspose2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.input_layout.as_str())
                        .unwrap_or(""),
                    _ => "",
                };
                if input_layout == "nhwc" && !op.input_operands().is_empty() {
                    let input_operand_id = op.input_operands()[0];

                    // Only transpose if not already transposed (deferred entries
                    // only insert their override at flush time, so check both).
                    if !operand_name_overrides.contains_key(&input_operand_id)
                        && !deferred_transposes.contains_key(&input_operand_id)
                        && let Some(input_operand) = graph_info.operand(input_operand_id)
                    {
                        // NHWC -> NCHW transposition: [0, 3, 1, 2]
                        let perm = [0, 3, 1, 2];

                        // Create transpose operation for input
                        let input_name = operand_name(graph_info, input_operand_id);
                        let transposed_input_name = format!("{}_nchw", input_name);

                        let mut transpose_inputs: HashMap<String, Argument> = HashMap::new();
                        transpose_inputs
                            .insert("x".to_string(), Self::create_name_argument(input_name));
                        transpose_inputs.insert(
                            "perm".to_string(),
                            Self::create_immediate_int_array(perm.as_ref()),
                        );

                        // Create tensor type for transposed input
                        let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                        let transposed_shape =
                            Self::permute_graph_shape(&input_operand.descriptor.shape, &perm);
                        let dimensions =
                            Self::mil_dimensions_from_graph_shape(&transposed_shape, false);

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
                            name: transposed_input_name.clone(),
                            r#type: Some(value_type),
                        };

                        let transpose_op = Self::create_mil_operation(
                            "transpose",
                            transpose_inputs,
                            vec![transpose_output_type],
                        );

                        // Same defer logic as for the filter: for constants/inputs the
                        // source has already been emitted so the override and transpose go now.
                        // For intermediates, defer both until the producing op is emitted.
                        if matches!(
                            input_operand.kind,
                            OperandKind::Constant | OperandKind::Input
                        ) {
                            operand_name_overrides
                                .insert(input_operand_id, transposed_input_name.clone());
                            main_block.operations.push(transpose_op);
                        } else {
                            deferred_transposes
                                .entry(input_operand_id)
                                .or_insert_with(|| (Vec::new(), transposed_input_name.clone()))
                                .0
                                .push(transpose_op);
                        }
                    }
                }
            }
        }

        // CoreML's model compiler fuses a `pad` op that directly feeds a pool into
        // the pool's own padding. The fused pool drops the pad's constant value
        // and, for asymmetric padding, miscounts the average divisor (end-side
        // padding is dropped from the denominator regardless of
        // exclude_padding_from_average). Insert a mul-by-1.0 between the pad and
        // the pool to defeat the fusion (a MIL identity op is itself elided by
        // the compiler and does not help).
        let pad_output_ids: std::collections::HashSet<u32> = graph_info
            .operations
            .iter()
            .filter(|p| matches!(p, Operation::Pad { .. }))
            .filter_map(|p| p.output_operand())
            .collect();
        let mut pad_unfused: std::collections::HashSet<u32> = std::collections::HashSet::new();

        // Convert all operations to MIL operations
        for op in &graph_info.operations {
            let op_type_lower = op.op_type().to_lowercase();

            if matches!(
                op,
                Operation::AveragePool2d { .. }
                    | Operation::MaxPool2d { .. }
                    | Operation::L2Pool2d { .. }
            ) && let Some(&pool_in_id) = op.input_operands().first()
                && pad_output_ids.contains(&pool_in_id)
                && !pad_unfused.contains(&pool_in_id)
                && let Some(pool_in_operand) = graph_info.operand(pool_in_id)
                && matches!(
                    pool_in_operand.descriptor.data_type,
                    DataType::Float32 | DataType::Float16
                )
            {
                let in_name =
                    Self::output_name_for_operand(graph_info, pool_in_id, &operand_name_overrides);
                let unfused_name = format!("{in_name}_pad_unfused");
                let out_type = Self::create_value_with_mil_type(
                    graph_info,
                    pool_in_id,
                    unfused_name.clone(),
                    Self::mil_data_type(&pool_in_operand.descriptor.data_type)?,
                )?;
                let mut mul_inputs = HashMap::new();
                mul_inputs.insert("x".to_string(), Self::create_name_argument(in_name));
                mul_inputs.insert(
                    "y".to_string(),
                    if matches!(pool_in_operand.descriptor.data_type, DataType::Float16) {
                        Self::create_immediate_float16(1.0)
                    } else {
                        Self::create_immediate_float(1.0)
                    },
                );
                main_block.operations.push(Self::create_mil_operation(
                    mil_ops::MUL,
                    mul_inputs,
                    vec![out_type],
                ));
                operand_name_overrides.insert(pool_in_id, unfused_name);
                pad_unfused.insert(pool_in_id);
                // Fall through: the pool below consumes the un-fusable mul output.
            }

            // Rank-0 (scalar) no-ops: transpose/tile/slice/expand/pad/reshape that map a
            // 0D scalar to a 0D scalar. CoreML rejects those ops on rank-0 tensors, so emit
            // an identity (input * 1) instead.
            if matches!(
                op_type_lower.as_str(),
                "transpose" | "tile" | "slice" | "expand" | "pad" | "reshape"
            ) && let (Some(&in_id), Some(out_id)) =
                (op.input_operands().first(), op.output_operand())
            {
                let in_scalar = graph_info
                    .operand(in_id)
                    .map(|o| o.descriptor.shape.is_empty())
                    .unwrap_or(false);
                let out_op = graph_info.operand(out_id);
                let out_scalar = out_op
                    .map(|o| o.descriptor.shape.is_empty())
                    .unwrap_or(false);
                let is_float = out_op
                    .map(|o| {
                        matches!(
                            o.descriptor.data_type,
                            DataType::Float32 | DataType::Float16
                        )
                    })
                    .unwrap_or(false);
                if in_scalar && out_scalar && is_float {
                    let in_name =
                        Self::output_name_for_operand(graph_info, in_id, &operand_name_overrides);
                    let (_out_name, out_type) =
                        Self::create_output_value(graph_info, out_id, &operand_name_overrides)?;
                    // Reshape to [1] (create_output_value promotes 0D scalars to [1],
                    // matching the model's rank-1 boundary), which is a no-op copy.
                    let mut ri = HashMap::new();
                    ri.insert("x".to_string(), Self::create_name_argument(in_name));
                    ri.insert(
                        "shape".to_string(),
                        Self::create_immediate_int_array(&[1u32]),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "reshape",
                        ri,
                        vec![out_type],
                    ));
                    continue;
                }
            }

            if matches!(
                op_type_lower.as_str(),
                "equal"
                    | "greater"
                    | "greaterorequal"
                    | "lesser"
                    | "lesserorequal"
                    | "logicalnot"
                    | "logicaland"
                    | "logicalor"
                    | "logicalxor"
                    | "notequal"
                    | "isnan"
                    | "isinfinite"
            ) {
                use crate::protos::coreml::mil_spec::DataType as MilDataType;

                let output_id =
                    op.output_operand()
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "CoreML MLProgram".to_string(),
                            reason: format!("operation '{}' has no output operand", op.op_type()),
                        })?;
                let output_operand =
                    graph_info
                        .operand(output_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Output operand {} not found", output_id),
                        })?;
                if output_operand.descriptor.data_type != DataType::Uint8 {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!(
                            "CoreML logical op '{}' expects uint8 graph output, got {:?}",
                            op.op_type(),
                            output_operand.descriptor.data_type
                        ),
                    });
                }

                let (output_name, output_type) =
                    Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                let bool_output_name = format!("{}_bool", output_name);
                let bool_output_type = Self::create_value_with_mil_type(
                    graph_info,
                    output_id,
                    bool_output_name.clone(),
                    MilDataType::Bool as i32,
                )?;

                let mut input_names =
                    Self::input_names_for_operation(graph_info, op, &operand_name_overrides);

                if matches!(
                    op_type_lower.as_str(),
                    "logicalnot" | "logicaland" | "logicalor" | "logicalxor"
                ) {
                    for (index, &input_id) in op.input_operands().iter().enumerate() {
                        let input_operand = graph_info.operand(input_id).ok_or_else(|| {
                            GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("Input operand {} not found", input_id),
                            }
                        })?;
                        if input_operand.descriptor.data_type == DataType::Uint8 {
                            // Suffix with the consuming op's output id: a bare
                            // `{input}_bool` collides with the producer's own
                            // `{output}_bool` raw result when the input comes
                            // from another comparison/logical op ("Block
                            // redefines I/O name").
                            let bool_input_name =
                                format!("{}_bool_{}_{}", input_names[index], output_id, index);
                            let bool_input_type = Self::create_value_with_mil_type(
                                graph_info,
                                input_id,
                                bool_input_name.clone(),
                                MilDataType::Bool as i32,
                            )?;
                            main_block.operations.push(Self::create_cast_operation(
                                input_names[index].clone(),
                                bool_input_type,
                                "bool",
                            ));
                            input_names[index] = bool_input_name;
                        }
                    }
                }

                if op_type_lower == "notequal" {
                    let equal_output_name = format!("{}_equal", output_name);
                    let equal_output_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        equal_output_name.clone(),
                        MilDataType::Bool as i32,
                    )?;

                    let mut equal_inputs = HashMap::new();
                    equal_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(input_names[0].clone()),
                    );
                    equal_inputs.insert(
                        "y".to_string(),
                        Self::create_name_argument(input_names[1].clone()),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::EQUAL,
                        equal_inputs,
                        vec![equal_output_type],
                    ));

                    let mut not_inputs = HashMap::new();
                    not_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(equal_output_name),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::LOGICAL_NOT,
                        not_inputs,
                        vec![bool_output_type],
                    ));
                } else if op_type_lower == "isnan" {
                    // isNaN(x): NaN is the only value where x != x. Use equal(x,x) then not().
                    let input_name = input_names.first().cloned().unwrap_or_default();
                    let eq_name = format!("{}_isnan_eq", output_name);
                    let eq_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        eq_name.clone(),
                        MilDataType::Bool as i32,
                    )?;
                    let mut eq_inputs = HashMap::new();
                    eq_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(input_name.clone()),
                    );
                    eq_inputs.insert("y".to_string(), Self::create_name_argument(input_name));
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::EQUAL,
                        eq_inputs,
                        vec![eq_type],
                    ));
                    let mut not_inputs = HashMap::new();
                    not_inputs.insert("x".to_string(), Self::create_name_argument(eq_name));
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::LOGICAL_NOT,
                        not_inputs,
                        vec![bool_output_type],
                    ));
                } else if op_type_lower == "isinfinite" {
                    // isInfinite(x): abs(x) > f32::MAX is true only for ±inf; NaN > anything = false.
                    let input_name = input_names.first().cloned().unwrap_or_default();
                    let abs_name = format!("{}_isinf_abs", output_name);
                    // Determine input dtype for abs output type
                    let input_dtype = op
                        .input_operands()
                        .first()
                        .and_then(|&id| graph_info.operand(id))
                        .map(|o| &o.descriptor.data_type)
                        .cloned()
                        .unwrap_or(DataType::Float32);
                    let abs_mil_dtype = Self::mil_data_type(&input_dtype)?;
                    let abs_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        abs_name.clone(),
                        abs_mil_dtype,
                    )?;
                    let mut abs_inputs = HashMap::new();
                    abs_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::ABS,
                        abs_inputs,
                        vec![abs_type],
                    ));
                    // f32::MAX is the largest finite float32; the only float32 > f32::MAX is +inf.
                    let max_val = if input_dtype == DataType::Float16 {
                        65504.0_f32 // f16::MAX
                    } else {
                        f32::MAX
                    };
                    let max_val_arg = if input_dtype == DataType::Float16 {
                        Self::create_immediate_float16(max_val)
                    } else {
                        Self::create_immediate_float(max_val)
                    };
                    let mut gt_inputs = HashMap::new();
                    gt_inputs.insert("x".to_string(), Self::create_name_argument(abs_name));
                    gt_inputs.insert("y".to_string(), max_val_arg);
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::GREATER,
                        gt_inputs,
                        vec![bool_output_type],
                    ));
                } else {
                    let mil_op = self.convert_operation_with_input_names_and_outputs(
                        graph_info,
                        op,
                        &input_names,
                        vec![bool_output_type],
                        self.get_mil_op_type(op.op_type())?,
                    )?;
                    main_block.operations.push(mil_op);
                }

                main_block.operations.push(Self::create_cast_operation(
                    bool_output_name,
                    output_type,
                    "uint8",
                ));
                continue;
            }

            // Prelu decomposition: select(x >= 0, x, x * alpha).
            // CoreML native prelu has strict constraints: alpha must be 1D const, x rank >= 2.
            // The element-wise decomposition handles all shapes, broadcasting, and non-constant
            // alpha without any of those constraints.
            if op_type_lower == "prelu"
                && let (Some(&x_id), Some(&alpha_id)) =
                    (op.input_operands().first(), op.input_operands().get(1))
                && let Some(output_id) = op.output_operand()
            {
                let x_operand =
                    graph_info
                        .operand(x_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("prelu input operand {} not found", x_id),
                        })?;
                let input_dtype = &x_operand.descriptor.data_type;
                let is_float16 = *input_dtype == DataType::Float16;
                // Use the graph proxy type (int32 for wide/int4 ints) so intermediate
                // tensors and the zero comparand share a MIL-representable type.
                let mil_dtype = Self::graph_value_mil_type(input_dtype)?;
                let (output_name, output_type) =
                    Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                let x_name =
                    Self::output_name_for_operand(graph_info, x_id, &operand_name_overrides);
                let alpha_name =
                    Self::output_name_for_operand(graph_info, alpha_id, &operand_name_overrides);

                // cond = greater_equal(x, 0): bool, same shape as x (not the broadcast output)
                let cond_name = format!("{}_prelu_cond", output_name);
                let cond_type = Self::create_value_with_mil_type(
                    graph_info,
                    x_id,
                    cond_name.clone(),
                    crate::protos::coreml::mil_spec::DataType::Bool as i32,
                )?;
                // The comparand must share x's MIL type (float vs int proxy).
                use crate::protos::coreml::mil_spec::DataType as MilDt;
                let zero_arg = if mil_dtype == MilDt::Float16 as i32 {
                    Self::create_immediate_float16(0.0)
                } else if mil_dtype == MilDt::Float32 as i32 {
                    Self::create_immediate_float(0.0)
                } else {
                    Self::create_immediate_int(0)
                };
                let _ = is_float16;
                let mut cond_inputs = HashMap::new();
                cond_inputs.insert("x".to_string(), Self::create_name_argument(x_name.clone()));
                cond_inputs.insert("y".to_string(), zero_arg);
                main_block.operations.push(Self::create_mil_operation(
                    mil_ops::GREATER_EQUAL,
                    cond_inputs,
                    vec![cond_type],
                ));

                // neg_branch = mul(x, alpha): same dtype as x
                let neg_name = format!("{}_prelu_neg", output_name);
                let neg_type = Self::create_value_with_mil_type(
                    graph_info,
                    output_id,
                    neg_name.clone(),
                    mil_dtype,
                )?;
                let mut mul_inputs = HashMap::new();
                mul_inputs.insert("x".to_string(), Self::create_name_argument(x_name.clone()));
                mul_inputs.insert("y".to_string(), Self::create_name_argument(alpha_name));
                main_block.operations.push(Self::create_mil_operation(
                    mil_ops::MUL,
                    mul_inputs,
                    vec![neg_type],
                ));

                // output = select(cond, x, neg_branch)
                let mut sel_inputs = HashMap::new();
                sel_inputs.insert("cond".to_string(), Self::create_name_argument(cond_name));
                sel_inputs.insert("a".to_string(), Self::create_name_argument(x_name));
                sel_inputs.insert("b".to_string(), Self::create_name_argument(neg_name));
                main_block.operations.push(Self::create_mil_operation(
                    mil_ops::WHERE,
                    sel_inputs,
                    vec![output_type],
                ));

                continue;
            }

            // Special handling for clamp with equal bounds.
            // CoreML clip rejects alpha == beta, while WebNN clamp(min==max) is valid and
            // should produce a constant tensor. Lower as: output = input * 0 + bound.
            if op_type_lower == "clamp" {
                let (min_value, max_value) = match &op {
                    Operation::Clamp { options, .. } => options
                        .as_ref()
                        .map(|o| {
                            (
                                Self::parse_clamp_bound(o.min_value.as_ref(), f64::NEG_INFINITY),
                                Self::parse_clamp_bound(o.max_value.as_ref(), f64::INFINITY),
                            )
                        })
                        .unwrap_or((f64::NEG_INFINITY, f64::INFINITY)),
                    _ => (f64::NEG_INFINITY, f64::INFINITY),
                };

                // For integer inputs (int8, uint8, int32), CoreML clip only accepts float.
                // Cast: int → fp32, clip, cast back to int.
                // Only int8/uint8/int32 are supported because CoreML cast cannot produce uint32/int64.
                {
                    let input_id = op.input_operands().first().copied();
                    let output_id_opt = op.output_operand();
                    if let (Some(input_id), Some(output_id)) = (input_id, output_id_opt)
                        && let Some(input_op) = graph_info.operand(input_id)
                    {
                        let int_dtype = &input_op.descriptor.data_type;
                        // Skip the fp32 clip path for equal bounds; that degenerate
                        // case is lowered separately (CoreML clip rejects alpha == beta).
                        let is_castable_int =
                            Self::is_castable_int(int_dtype) && min_value != max_value;
                        if is_castable_int {
                            let int_name = Self::output_name_for_operand(
                                graph_info,
                                input_id,
                                &operand_name_overrides,
                            );
                            let (output_name, output_type) = Self::create_output_value(
                                graph_info,
                                output_id,
                                &operand_name_overrides,
                            )?;

                            // Cast int input to float32
                            let float_name = format!("{}_clamp_float", output_name);
                            let float_type = Self::create_value_with_mil_type(
                                graph_info,
                                output_id,
                                float_name.clone(),
                                crate::protos::coreml::mil_spec::DataType::Float32 as i32,
                            )?;
                            main_block
                                .operations
                                .push(Self::create_cast_operation(int_name, float_type, "fp32"));

                            // Clip in float32
                            let clipped_name = format!("{}_clamp_clipped", output_name);
                            let clipped_type = Self::create_value_with_mil_type(
                                graph_info,
                                output_id,
                                clipped_name.clone(),
                                crate::protos::coreml::mil_spec::DataType::Float32 as i32,
                            )?;
                            let mut clip_inputs = HashMap::new();
                            clip_inputs
                                .insert("x".to_string(), Self::create_name_argument(float_name));
                            clip_inputs.insert(
                                "alpha".to_string(),
                                Self::create_immediate_float(min_value as f32),
                            );
                            clip_inputs.insert(
                                "beta".to_string(),
                                Self::create_immediate_float(max_value as f32),
                            );
                            main_block.operations.push(Self::create_mil_operation(
                                mil_ops::CLIP,
                                clip_inputs,
                                vec![clipped_type],
                            ));

                            // WebNN saturates the clamp result to the output type's
                            // range (a min/max bound outside the type range still yields
                            // an in-range value); CoreML's int cast would wrap instead.
                            // Clip to [type_min, type_max] before casting. (No-op when
                            // the bounds already lie inside the type range.)
                            let (sat_min, sat_max) = match int_dtype {
                                DataType::Int8 => (-128.0f32, 127.0f32),
                                DataType::Uint8 => (0.0f32, 255.0f32),
                                // int32 and the int32 proxies span the full int32 range.
                                _ => (i32::MIN as f32, i32::MAX as f32),
                            };
                            let sat_name = format!("{}_clamp_sat", output_name);
                            let sat_type = Self::create_value_with_mil_type(
                                graph_info,
                                output_id,
                                sat_name.clone(),
                                crate::protos::coreml::mil_spec::DataType::Float32 as i32,
                            )?;
                            let mut sat_inputs = HashMap::new();
                            sat_inputs
                                .insert("x".to_string(), Self::create_name_argument(clipped_name));
                            sat_inputs
                                .insert("alpha".to_string(), Self::create_immediate_float(sat_min));
                            sat_inputs
                                .insert("beta".to_string(), Self::create_immediate_float(sat_max));
                            main_block.operations.push(Self::create_mil_operation(
                                mil_ops::CLIP,
                                sat_inputs,
                                vec![sat_type],
                            ));

                            // Cast back to the operand's internal int representation.
                            let back_dtype = Self::int_back_cast_dtype(int_dtype)?;
                            main_block.operations.push(Self::create_cast_operation(
                                sat_name,
                                output_type,
                                back_dtype,
                            ));
                            continue;
                        }
                    }
                }

                if min_value == max_value {
                    if op.input_operands().is_empty() || op.output_operand().is_none() {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: "clamp requires input and output operand".to_string(),
                        });
                    }

                    let input_id = op.input_operands()[0];
                    let input_operand = graph_info.operand(input_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", input_id),
                        }
                    })?;
                    let output_id = op.output_operand().expect("checked above");
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let input_name = operand_name(graph_info, input_id);
                    let use_float16 = input_operand.descriptor.data_type == DataType::Float16;
                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                    let dimensions = Self::mil_dimensions_from_graph_shape(
                        &input_operand.descriptor.shape,
                        false,
                    );
                    let make_type = |name: String| NamedValueType {
                        name,
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: dimensions.len() as i64,
                                        data_type: dtype,
                                        dimensions: dimensions.clone(),
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };
                    let bound_arg = |v: f64| {
                        if use_float16 {
                            Self::create_immediate_float16(v as f32)
                        } else {
                            Self::create_immediate_float(v as f32)
                        }
                    };
                    // Equal bounds (incl. +/-Infinity): CoreML `clip` rejects alpha == beta,
                    // and `input*0 + bound` yields NaN for non-finite inputs. Use
                    // minimum(maximum(input, min), max), which is exact for infinities.
                    let mx_name = format!("{}_clamp_max", output_name);
                    let mut mx_inputs: HashMap<String, Argument> = HashMap::new();
                    mx_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    mx_inputs.insert("y".to_string(), bound_arg(min_value));
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::MAXIMUM,
                        mx_inputs,
                        vec![make_type(mx_name.clone())],
                    ));
                    let mut mn_inputs: HashMap<String, Argument> = HashMap::new();
                    mn_inputs.insert("x".to_string(), Self::create_name_argument(mx_name));
                    mn_inputs.insert("y".to_string(), bound_arg(max_value));
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::MINIMUM,
                        mn_inputs,
                        vec![output_type],
                    ));

                    continue;
                }
            }

            // Special handling for expand operation (may need reshape first)
            if let Operation::Expand {
                new_shape: expand_shape,
                ..
            } = &op
                && !op.input_operands().is_empty()
                && !expand_shape.is_empty()
                && let Some(input_operand) = graph_info.operand(op.input_operands()[0])
            {
                let new_shape_u32: Vec<u32> = expand_shape
                    .iter()
                    .map(MLDimension::static_or_max)
                    .collect();
                let input_shape = input_operand.descriptor.static_or_max_shape();
                let input_rank = input_shape.len();
                let output_rank = new_shape_u32.len();

                #[allow(clippy::collapsible_if)]
                if input_rank < output_rank {
                    let mut reshaped_dims = vec![1u32; output_rank];
                    for i in 0..input_rank {
                        reshaped_dims[output_rank - i - 1] = input_shape[input_rank - i - 1];
                    }

                    //Create reshape operation
                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        op.input_operands()[0],
                        &operand_name_overrides,
                    );
                    // Named after this expand's own output (unique per op; an
                    // input-derived name collides when two expands share an input).
                    // The consumer site derives the same name.
                    let reshape_output_name = format!(
                        "{}_expand_reshaped",
                        operand_name(
                            graph_info,
                            op.output_operand().unwrap_or(op.input_operands()[0])
                        )
                    );

                    let mut reshape_inputs: HashMap<String, Argument> = HashMap::new();
                    reshape_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    reshape_inputs.insert(
                        "shape".to_string(),
                        Self::create_int_array_argument(
                            reshaped_dims.iter().map(|&v| v as i32).collect(),
                        ),
                    );

                    // graph_value_mil_type: wide ints travel as int32 proxies.
                    let dtype = Self::graph_value_mil_type(&input_operand.descriptor.data_type)?;
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

            // Relu with integer input: CoreML relu only accepts float (fp32/fp16).
            // Cast int → fp32, apply relu, cast back.
            // sub / sign / abs / max / min with int8/uint8: CoreML only supports
            // int32/fp32/fp16. Cast inputs to int32, apply op, cast back to the int type.
            if matches!(
                op_type_lower.as_str(),
                "sub" | "sign" | "abs" | "max" | "min"
            ) {
                let first_input_id = op.input_operands().first().copied();
                let output_id = op.output_operand();
                if let (Some(input_id), Some(output_id)) = (first_input_id, output_id)
                    && let Some(input_op) = graph_info.operand(input_id)
                {
                    let int_dtype = input_op.descriptor.data_type;
                    if matches!(int_dtype, DataType::Int8 | DataType::Uint8) {
                        let (output_name, output_type) = Self::create_output_value(
                            graph_info,
                            output_id,
                            &operand_name_overrides,
                        )?;
                        let back_dtype = Self::cast_dtype_string_for_graph_type(&int_dtype)?;

                        // Cast all inputs to int32
                        let mut int32_names: Vec<String> = Vec::new();
                        for (idx, &in_id) in op.input_operands().iter().enumerate() {
                            let src_name = Self::output_name_for_operand(
                                graph_info,
                                in_id,
                                &operand_name_overrides,
                            );
                            let cast32_name = format!("{}_op_cast32_{}", output_name, idx);
                            let cast32_type = Self::create_value_with_mil_type(
                                graph_info,
                                in_id,
                                cast32_name.clone(),
                                crate::protos::coreml::mil_spec::DataType::Int32 as i32,
                            )?;
                            main_block.operations.push(Self::create_cast_operation(
                                src_name,
                                cast32_type,
                                "int32",
                            ));
                            int32_names.push(cast32_name);
                        }

                        // Apply the op on int32 values
                        let mil_type = self.get_mil_op_type(op.op_type())?;
                        let int32_out_name = format!("{}_op_int32", output_name);
                        let int32_out_type = Self::create_value_with_mil_type(
                            graph_info,
                            output_id,
                            int32_out_name.clone(),
                            crate::protos::coreml::mil_spec::DataType::Int32 as i32,
                        )?;
                        let op_mil = self.convert_operation_with_input_names_and_outputs(
                            graph_info,
                            op,
                            &int32_names,
                            vec![int32_out_type],
                            mil_type,
                        )?;
                        main_block.operations.push(op_mil);

                        // Cast result back to original int type
                        main_block.operations.push(Self::create_cast_operation(
                            int32_out_name,
                            output_type,
                            back_dtype,
                        ));
                        continue;
                    }
                }
            }

            // neg with int8/uint8: emitted as mul-by-(-1), but the -1 multiplier is a
            // float immediate, so route through fp32 (int8 values are exact in fp32),
            // then cast back to the original int type.
            if op_type_lower == "neg"
                && let (Some(&input_id), Some(output_id)) =
                    (op.input_operands().first(), op.output_operand())
                && let Some(input_op) = graph_info.operand(input_id)
            {
                let int_dtype = input_op.descriptor.data_type;
                if matches!(int_dtype, DataType::Int8 | DataType::Uint8) {
                    let int_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let float_name = format!("{}_neg_float", output_name);
                    let float_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        float_name.clone(),
                        crate::protos::coreml::mil_spec::DataType::Float32 as i32,
                    )?;
                    main_block
                        .operations
                        .push(Self::create_cast_operation(int_name, float_type, "fp32"));
                    let neg_name = format!("{}_neg_result", output_name);
                    let neg_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        neg_name.clone(),
                        crate::protos::coreml::mil_spec::DataType::Float32 as i32,
                    )?;
                    let mut neg_inputs = HashMap::new();
                    neg_inputs.insert("x".to_string(), Self::create_name_argument(float_name));
                    neg_inputs.insert("y".to_string(), Self::create_immediate_float(-1.0));
                    main_block.operations.push(Self::create_mil_operation(
                        self.get_mil_op_type("neg")?,
                        neg_inputs,
                        vec![neg_type],
                    ));
                    let back_dtype = Self::cast_dtype_string_for_graph_type(&int_dtype)?;
                    main_block.operations.push(Self::create_cast_operation(
                        neg_name,
                        output_type,
                        back_dtype,
                    ));
                    continue;
                }
            }

            if op_type_lower == "relu"
                && let (Some(&input_id), Some(output_id)) =
                    (op.input_operands().first(), op.output_operand())
                && let Some(input_op) = graph_info.operand(input_id)
            {
                let int_dtype = input_op.descriptor.data_type;
                let is_castable_int = Self::is_castable_int(&int_dtype);
                if is_castable_int {
                    let int_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let float_name = format!("{}_relu_float", output_name);
                    let float_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        float_name.clone(),
                        crate::protos::coreml::mil_spec::DataType::Float32 as i32,
                    )?;
                    main_block
                        .operations
                        .push(Self::create_cast_operation(int_name, float_type, "fp32"));
                    let relu_name = format!("{}_relu_result", output_name);
                    let relu_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        relu_name.clone(),
                        crate::protos::coreml::mil_spec::DataType::Float32 as i32,
                    )?;
                    let mut relu_inputs = HashMap::new();
                    relu_inputs.insert("x".to_string(), Self::create_name_argument(float_name));
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::RELU,
                        relu_inputs,
                        vec![relu_type],
                    ));
                    let back_dtype = Self::int_back_cast_dtype(&int_dtype)?;
                    main_block.operations.push(Self::create_cast_operation(
                        relu_name,
                        output_type,
                        back_dtype,
                    ));
                    continue;
                }
            }

            // Pad with integer input: CoreML pad only accepts float (fp32/fp16).
            // Cast int → fp32, apply pad with float constant_val, cast back.
            if op_type_lower == "pad"
                && let (Some(&input_id), Some(output_id)) =
                    (op.input_operands().first(), op.output_operand())
                && let Some(input_op) = graph_info.operand(input_id)
            {
                let int_dtype = input_op.descriptor.data_type;
                let is_castable_int = Self::is_castable_int(&int_dtype);
                if is_castable_int {
                    let (beginning_padding, ending_padding, options) = match op {
                        Operation::Pad {
                            beginning_padding,
                            ending_padding,
                            options,
                            ..
                        } => (beginning_padding, ending_padding, options.as_ref()),
                        _ => {
                            continue;
                        }
                    };
                    let int_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    // Cast input to float32
                    let float_name = format!("{}_pad_float", output_name);
                    let float_type = Self::create_value_with_mil_type(
                        graph_info,
                        input_id,
                        float_name.clone(),
                        crate::protos::coreml::mil_spec::DataType::Float32 as i32,
                    )?;
                    main_block
                        .operations
                        .push(Self::create_cast_operation(int_name, float_type, "fp32"));
                    // Build pad inputs with float32 constant_val
                    let pad_vec: Vec<u32> = beginning_padding
                        .iter()
                        .zip(ending_padding.iter())
                        .flat_map(|(a, b)| [*a, *b])
                        .collect();
                    let webnn_mode = options.map(|o| o.mode.as_str()).unwrap_or("constant");
                    let coreml_mode = match webnn_mode {
                        "edge" => "replicate",
                        "reflection" | "symmetric" => "reflect",
                        _ => "constant",
                    };
                    let constant_val = options
                        .and_then(|o| Self::parse_mlnumber_f64(o.value.as_ref()))
                        .unwrap_or(0.0);
                    let padded_name = format!("{}_pad_padded", output_name);
                    let padded_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        padded_name.clone(),
                        crate::protos::coreml::mil_spec::DataType::Float32 as i32,
                    )?;
                    let mut pad_inputs = HashMap::new();
                    pad_inputs.insert("x".to_string(), Self::create_name_argument(float_name));
                    pad_inputs.insert(
                        "pad".to_string(),
                        Self::create_immediate_int_array(&pad_vec),
                    );
                    pad_inputs.insert(
                        "mode".to_string(),
                        Self::create_immediate_string(coreml_mode),
                    );
                    pad_inputs.insert(
                        "constant_val".to_string(),
                        Self::create_immediate_float(constant_val as f32),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::PAD,
                        pad_inputs,
                        vec![padded_type],
                    ));
                    // Cast back to the operand's internal int representation.
                    let back_dtype = Self::int_back_cast_dtype(&int_dtype)?;
                    main_block.operations.push(Self::create_cast_operation(
                        padded_name,
                        output_type,
                        back_dtype,
                    ));
                    continue;
                }
            }

            // Special handling for hardswish (decompose into hardsigmoid + mul)
            // Following Chromium: hardswish = x * hardsigmoid(x, alpha=1/6, beta=0.5)
            // Note: op_type is "hardSwish" but we normalize to lowercase
            if op_type_lower == "hardswish" {
                // Validate inputs/outputs exist
                // Note: hardswish uses output_operand (singular), not output_operands
                if op.input_operands().is_empty() || op.output_operand().is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "hardswish requires input and output operand".to_string(),
                    });
                }

                let input_operand =
                    graph_info.operand(op.input_operands()[0]).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", op.input_operands()[0]),
                        }
                    })?;
                {
                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        op.input_operands()[0],
                        &operand_name_overrides,
                    );
                    // Named after this op's output: shared inputs must not collide.
                    let hardsigmoid_output_name = format!(
                        "{}_hardswish_hardsigmoid",
                        Self::output_name_for_operand(
                            graph_info,
                            op.output_operand().unwrap_or(op.input_operands()[0]),
                            &operand_name_overrides,
                        )
                    );

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

                    // Create tensor type for hardsigmoid output.
                    // Use `scalar_as_one_dim: true` to match the promotion the
                    // rest of the converter applies to rank-0 operands — with
                    // `false`, a rank-0 input gives hardsigmoid an rank-0
                    // intermediate while the graph's declared output is
                    // `tensor<fp32, [1]>`, and Apple's loader rejects the
                    // following `mul` with
                    //   "Output '0' has unexpected type 'ios17.mul'.
                    //    Expected tensor<fp32, [1]>; got fp32."
                    // (surfaced by WPT "hardSwish float32 0D scalar default options").
                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                    let dimensions = Self::mil_dimensions_from_graph_shape(
                        &input_operand.descriptor.shape,
                        true,
                    );

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
                    let output_operand_id = op.output_operand().unwrap();
                    let output_name = operand_name(graph_info, output_operand_id);
                    let output_operand =
                        graph_info.operand(output_operand_id).ok_or_else(|| {
                            GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("Output operand {} not found", output_operand_id),
                            }
                        })?;

                    let output_dtype = Self::mil_data_type(&output_operand.descriptor.data_type)?;
                    // Same promotion as the hardsigmoid intermediate above: the
                    // mul output must have the same rank as its inputs, and
                    // the graph's input/output operands already went through
                    // the `scalar_as_one_dim: true` pass when they were
                    // declared in `main` block — so rank-0 becomes [1] here too.
                    let output_dimensions = Self::mil_dimensions_from_graph_shape(
                        &output_operand.descriptor.shape,
                        true,
                    );

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

            // Special handling for gemm: y = alpha * op(a) * op(b) + beta * c
            // Lower to matmul + optional mul(alpha) + optional mul(beta, c) + add.
            if op_type_lower == "gemm" {
                if op.input_operands().len() < 2 || op.output_operand().is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "gemm requires at least 2 input operands and 1 output".to_string(),
                    });
                }

                let output_operand_id = op.output_operand().unwrap();
                let output_operand = graph_info.operand(output_operand_id).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Output operand {} not found", output_operand_id),
                    }
                })?;

                let (output_name, output_type) = Self::create_value(graph_info, output_operand_id)?;

                let (alpha, beta) = match &op {
                    Operation::Gemm { options, .. } => (
                        options.as_ref().map(|o| o.alpha as f32).unwrap_or(1.0),
                        options.as_ref().map(|o| o.beta as f32).unwrap_or(1.0),
                    ),
                    _ => (1.0, 1.0),
                };

                let c_operand_id_opt = match &op {
                    Operation::Gemm { options, .. } => options.as_ref().and_then(|o| o.c),
                    _ => None,
                };
                let has_bias = c_operand_id_opt.is_some();
                let needs_alpha_mul = (alpha - 1.0).abs() > f32::EPSILON;
                let needs_beta_mul = has_bias && (beta - 1.0).abs() > f32::EPSILON;

                let (alpha_arg, beta_arg) = match output_operand.descriptor.data_type {
                    DataType::Float16 => (
                        Self::create_immediate_float16(alpha),
                        Self::create_immediate_float16(beta),
                    ),
                    DataType::Float32 => (
                        Self::create_immediate_float(alpha),
                        Self::create_immediate_float(beta),
                    ),
                    _ => {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "gemm currently supports float32/float16 output, got {:?}",
                                output_operand.descriptor.data_type
                            ),
                        });
                    }
                };

                let mut current_name: String;
                let matmul_output_name = if needs_alpha_mul || has_bias {
                    format!("{}_gemm_matmul", output_name)
                } else {
                    output_name.clone()
                };

                let mut matmul_inputs: HashMap<String, Argument> = HashMap::new();
                matmul_inputs.insert(
                    "x".to_string(),
                    Self::create_name_argument(operand_name(graph_info, op.input_operands()[0])),
                );
                matmul_inputs.insert(
                    "y".to_string(),
                    Self::create_name_argument(operand_name(graph_info, op.input_operands()[1])),
                );
                let (a_transpose, b_transpose) = match &op {
                    Operation::Gemm { options, .. } => (
                        options.as_ref().map(|o| o.a_transpose).unwrap_or(false),
                        options.as_ref().map(|o| o.b_transpose).unwrap_or(false),
                    ),
                    _ => (false, false),
                };
                matmul_inputs.insert(
                    "transpose_x".to_string(),
                    Self::create_immediate_bool(a_transpose),
                );
                matmul_inputs.insert(
                    "transpose_y".to_string(),
                    Self::create_immediate_bool(b_transpose),
                );

                main_block.operations.push(Self::create_mil_operation(
                    "matmul",
                    matmul_inputs,
                    vec![NamedValueType {
                        name: matmul_output_name.clone(),
                        r#type: output_type.r#type.clone(),
                    }],
                ));
                current_name = matmul_output_name;

                if needs_alpha_mul {
                    let alpha_output_name = if has_bias {
                        format!("{}_gemm_alpha", output_name)
                    } else {
                        output_name.clone()
                    };

                    let mut alpha_mul_inputs: HashMap<String, Argument> = HashMap::new();
                    alpha_mul_inputs
                        .insert("x".to_string(), Self::create_name_argument(current_name));
                    alpha_mul_inputs.insert("y".to_string(), alpha_arg);

                    main_block.operations.push(Self::create_mil_operation(
                        "mul",
                        alpha_mul_inputs,
                        vec![NamedValueType {
                            name: alpha_output_name.clone(),
                            r#type: output_type.r#type.clone(),
                        }],
                    ));

                    current_name = alpha_output_name;
                }

                if has_bias {
                    let c_operand_id = c_operand_id_opt.unwrap();
                    let (c_name, c_type) = Self::create_value(graph_info, c_operand_id)?;
                    let scaled_c_name = if needs_beta_mul {
                        format!("{}_gemm_bias", output_name)
                    } else {
                        c_name.clone()
                    };

                    if needs_beta_mul {
                        let mut beta_mul_inputs: HashMap<String, Argument> = HashMap::new();
                        beta_mul_inputs.insert("x".to_string(), Self::create_name_argument(c_name));
                        beta_mul_inputs.insert("y".to_string(), beta_arg);

                        main_block.operations.push(Self::create_mil_operation(
                            "mul",
                            beta_mul_inputs,
                            vec![NamedValueType {
                                name: scaled_c_name.clone(),
                                r#type: c_type.r#type,
                            }],
                        ));
                    }

                    let mut add_inputs: HashMap<String, Argument> = HashMap::new();
                    add_inputs.insert("x".to_string(), Self::create_name_argument(current_name));
                    add_inputs.insert("y".to_string(), Self::create_name_argument(scaled_c_name));

                    main_block.operations.push(Self::create_mil_operation(
                        "add",
                        add_inputs,
                        vec![NamedValueType {
                            name: output_name,
                            r#type: output_type.r#type,
                        }],
                    ));
                }

                continue;
            }

            // Special handling for linear: y = alpha * x + beta
            // Lower to mul + add primitives for backend parity.
            if op_type_lower == "linear" {
                if op.input_operands().is_empty() || op.output_operand().is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "linear requires input and output operand".to_string(),
                    });
                }

                let input_operand =
                    graph_info.operand(op.input_operands()[0]).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", op.input_operands()[0]),
                        }
                    })?;
                let output_operand_id = op.output_operand().unwrap();
                let output_operand = graph_info.operand(output_operand_id).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Output operand {} not found", output_operand_id),
                    }
                })?;

                let (alpha, beta) = match &op {
                    Operation::Linear { options, .. } => options
                        .as_ref()
                        .map(|o| (o.alpha as f32, o.beta as f32))
                        .unwrap_or((1.0, 0.0)),
                    _ => (1.0, 0.0),
                };

                let (alpha_arg, beta_arg) = match input_operand.descriptor.data_type {
                    DataType::Float16 => (
                        Self::create_immediate_float16(alpha),
                        Self::create_immediate_float16(beta),
                    ),
                    DataType::Float32 => (
                        Self::create_immediate_float(alpha),
                        Self::create_immediate_float(beta),
                    ),
                    _ => {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "linear currently supports float32/float16, got {:?}",
                                input_operand.descriptor.data_type
                            ),
                        });
                    }
                };

                let input_name = operand_name(graph_info, op.input_operands()[0]);
                let output_name = operand_name(graph_info, output_operand_id);
                let mul_output_name = format!("{}_linear_mul", output_name);

                let output_dtype = Self::mil_data_type(&output_operand.descriptor.data_type)?;
                let output_shape = if output_operand.descriptor.shape.is_empty() {
                    vec![1u32]
                } else {
                    output_operand.descriptor.static_or_max_shape()
                };
                let output_dimensions: Vec<Dimension> = output_shape
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
                            rank: output_dimensions.len() as i64,
                            data_type: output_dtype,
                            dimensions: output_dimensions.clone(),
                            attributes: HashMap::new(),
                        }),
                    ),
                };

                let mut mul_inputs: HashMap<String, Argument> = HashMap::new();
                mul_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                mul_inputs.insert("y".to_string(), alpha_arg);
                main_block.operations.push(Self::create_mil_operation(
                    "mul",
                    mul_inputs,
                    vec![NamedValueType {
                        name: mul_output_name.clone(),
                        r#type: Some(value_type.clone()),
                    }],
                ));

                let mut add_inputs: HashMap<String, Argument> = HashMap::new();
                add_inputs.insert("x".to_string(), Self::create_name_argument(mul_output_name));
                add_inputs.insert("y".to_string(), beta_arg);
                main_block.operations.push(Self::create_mil_operation(
                    "add",
                    add_inputs,
                    vec![NamedValueType {
                        name: output_name,
                        r#type: Some(value_type),
                    }],
                ));

                continue;
            }

            // Special handling for neg (decompose into mul(x, -1) with typed constant)
            // Following Chromium: neg = mul(x, -1) with constant matching input dtype
            if op_type_lower == "neg" {
                // Validate inputs/outputs exist
                if op.input_operands().is_empty() || op.output_operand().is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "neg requires input and output operand".to_string(),
                    });
                }

                let input_operand =
                    graph_info.operand(op.input_operands()[0]).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", op.input_operands()[0]),
                        }
                    })?;

                let input_name = Self::output_name_for_operand(
                    graph_info,
                    op.input_operands()[0],
                    &operand_name_overrides,
                );

                // Create typed -1 constant matching input dtype. MIL `mul`
                // preserves rank rather than broadcasting scalars, so when
                // the input operand was promoted from rank-0 to rank-1 `[1]`
                // (the `scalar_as_one_dim: true` rule applied to all graph
                // inputs/outputs), the multiplier must be rank-1 `[1]` too.
                // Otherwise Apple's loader rejects with
                //   "Output '0' has unexpected type 'ios17.mul'.
                //    Expected tensor<fp32, [1]>; got fp32."
                // (WPT "neg float32 positive 0D scalar" surfaces this.)
                let neg_one_immediate = match input_operand.descriptor.data_type {
                    DataType::Float32 => Self::create_immediate_float_1d(-1.0f32),
                    DataType::Float16 => Self::create_immediate_float16(-1.0f32),
                    // Int32 and the int32-proxied wide ints all multiply by an int32 -1.
                    DataType::Int32 | DataType::Uint32 | DataType::Int64 | DataType::Uint64 => {
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
                let output_operand_id = op.output_operand().unwrap();
                let output_name = operand_name(graph_info, output_operand_id);
                let output_operand = graph_info.operand(output_operand_id).ok_or_else(|| {
                    GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Output operand {} not found", output_operand_id),
                    }
                })?;

                let output_dtype =
                    Self::graph_value_mil_type(&output_operand.descriptor.data_type)?;
                // Use `scalar_as_one_dim: true` so a rank-0 output is promoted
                // to `tensor<fp32, [1]>`, matching the rank of the mul's
                // (promoted) input/multiplier. See comment on the mul input
                // above — this keeps the full op consistent.
                let output_dimensions =
                    Self::mil_dimensions_from_graph_shape(&output_operand.descriptor.shape, true);

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

            // Special handling for layerNormalization with empty axes.
            // When axes is empty (or not provided), the mean equals the input, so variance is 0
            // and the normalized output is 0 for every element. Following Chromium:
            //   out = sub(x, x)           -> zeros shaped like input
            //   out = add(zeros, bias)    -> if bias is present
            if op_type_lower == "layernormalization" {
                let axes_empty = match &op {
                    Operation::LayerNormalization { options, .. } => {
                        // None means "default to last axis" (not empty)
                        options
                            .as_ref()
                            .and_then(|o| o.axes.as_ref())
                            .map(|ax| ax.is_empty())
                            .unwrap_or(false)
                    }
                    _ => false,
                };

                if axes_empty {
                    if op.input_operands().is_empty() || op.output_operand().is_none() {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: "layerNormalization requires input and output operand"
                                .to_string(),
                        });
                    }

                    let input_id = op.input_operands()[0];
                    let input_operand = graph_info.operand(input_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", input_id),
                        }
                    })?;
                    let output_id = op.output_operand().unwrap();
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;

                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                    let dimensions = Self::mil_dimensions_from_graph_shape(
                        &input_operand.descriptor.shape,
                        false,
                    );
                    let zeros_value_type = ValueType {
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

                    let bias_id_opt = match &op {
                        Operation::LayerNormalization { options, .. } => {
                            options.as_ref().and_then(|o| o.bias)
                        }
                        _ => None,
                    };

                    let zeros_name = if bias_id_opt.is_some() {
                        format!("{}_ln_zeros", output_name)
                    } else {
                        output_name.clone()
                    };

                    // zeros = x - x
                    let mut sub_inputs: HashMap<String, Argument> = HashMap::new();
                    sub_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(input_name.clone()),
                    );
                    sub_inputs.insert("y".to_string(), Self::create_name_argument(input_name));
                    main_block.operations.push(Self::create_mil_operation(
                        "sub",
                        sub_inputs,
                        vec![NamedValueType {
                            name: zeros_name.clone(),
                            r#type: Some(zeros_value_type),
                        }],
                    ));

                    if let Some(bias_id) = bias_id_opt {
                        let bias_name = operand_name(graph_info, bias_id);
                        let mut add_inputs: HashMap<String, Argument> = HashMap::new();
                        add_inputs.insert("x".to_string(), Self::create_name_argument(zeros_name));
                        add_inputs.insert("y".to_string(), Self::create_name_argument(bias_name));
                        main_block.operations.push(Self::create_mil_operation(
                            "add",
                            add_inputs,
                            vec![output_type],
                        ));
                    }

                    continue;
                }
            }

            // Special handling for triangular with non-zero diagonal k.
            // CoreML band_part(x, lower, upper) keeps the band -lower <= row-col <= upper.
            // For upper=true, k>0: the result is input minus the band up to diagonal k-1.
            //   result = input - band_part(input, -1, k-1)
            // For upper=false, k<0: the result is input minus the band from diagonal -(|k|-1).
            //   result = input - band_part(input, |k|-1, -1)
            if op_type_lower == "triangular" {
                let (is_upper, diagonal) = match &op {
                    Operation::Triangular { options, .. } => (
                        options.as_ref().and_then(|o| o.upper).unwrap_or(true),
                        options.as_ref().map(|o| o.diagonal as i64).unwrap_or(0),
                    ),
                    _ => (true, 0),
                };

                let needs_decompose = (is_upper && diagonal > 0) || (!is_upper && diagonal < 0);

                if needs_decompose {
                    if op.input_operands().is_empty() || op.output_operand().is_none() {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: "triangular requires input and output operand".to_string(),
                        });
                    }

                    let input_id = op.input_operands()[0];
                    let input_operand = graph_info.operand(input_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", input_id),
                        }
                    })?;
                    let output_id = op.output_operand().unwrap();
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;

                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;
                    let dimensions = Self::mil_dimensions_from_graph_shape(
                        &input_operand.descriptor.shape,
                        false,
                    );
                    let band_value_type = ValueType {
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
                    let band_name = format!("{}_triangular_band", output_name);

                    // Compute band_part bounds for the subtracted region.
                    // upper=true, k>0: subtract band_part(input, -1, k-1)
                    // upper=false, k<0: subtract band_part(input, |k|-1, -1)
                    let (band_lower, band_upper): (i64, i64) = if is_upper {
                        (-1, diagonal - 1)
                    } else {
                        ((-diagonal) - 1, -1)
                    };

                    let mut band_inputs: HashMap<String, Argument> = HashMap::new();
                    band_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(input_name.clone()),
                    );
                    band_inputs.insert(
                        "lower".to_string(),
                        Self::create_immediate_int(band_lower as i32 as u32),
                    );
                    band_inputs.insert(
                        "upper".to_string(),
                        Self::create_immediate_int(band_upper as i32 as u32),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "band_part",
                        band_inputs,
                        vec![NamedValueType {
                            name: band_name.clone(),
                            r#type: Some(band_value_type),
                        }],
                    ));

                    // result = input - band
                    let mut sub_inputs: HashMap<String, Argument> = HashMap::new();
                    sub_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    sub_inputs.insert("y".to_string(), Self::create_name_argument(band_name));
                    main_block.operations.push(Self::create_mil_operation(
                        "sub",
                        sub_inputs,
                        vec![output_type],
                    ));

                    continue;
                }
            }

            // layer/instance normalization with a runtime (non-constant) scale or
            // bias: CoreML's native layer_norm/instance_norm require const
            // gamma/beta ("Param 'gamma' must be const"), so run the native op
            // without them and apply `y = norm(x) * scale + bias` with explicit
            // mul/add, reshaping scale/bias to a broadcast-compatible shape.
            // Both params are stripped together: the native op computes
            // `norm * gamma + beta`, so applying one after the fact while the
            // other stays inside would change the result.
            if matches!(
                op,
                Operation::LayerNormalization { .. } | Operation::InstanceNormalization { .. }
            ) {
                let (scale_id, bias_id) = match op {
                    Operation::LayerNormalization { options, .. } => (
                        options.as_ref().and_then(|o| o.scale),
                        options.as_ref().and_then(|o| o.bias),
                    ),
                    Operation::InstanceNormalization { options, .. } => (
                        options.as_ref().and_then(|o| o.scale),
                        options.as_ref().and_then(|o| o.bias),
                    ),
                    _ => (None, None),
                };
                let is_runtime = |id: Option<u32>| {
                    id.and_then(|id| graph_info.operand(id))
                        .map(|o| o.kind != crate::graph::OperandKind::Constant)
                        .unwrap_or(false)
                };
                if is_runtime(scale_id) || is_runtime(bias_id) {
                    let x_id = *op.input_operands().first().ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("{} has no input operand", op.op_type()),
                        }
                    })?;
                    let x_op =
                        graph_info
                            .operand(x_id)
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("input operand {x_id} not found"),
                            })?;
                    let x_shape = x_op.descriptor.static_or_max_shape();
                    let rank = x_shape.len();
                    let mil_dtype = Self::graph_value_mil_type(&x_op.descriptor.data_type)?;
                    let out_id =
                        op.output_operand()
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("{} has no output operand", op.op_type()),
                            })?;
                    let (out_name, out_type) =
                        Self::create_output_value(graph_info, out_id, &operand_name_overrides)?;

                    // Broadcast target: scale/bias expanded to the input rank with
                    // 1s on non-normalized (layer) / non-channel (instance) dims.
                    let target: Vec<u32> = match op {
                        Operation::LayerNormalization { options, .. } => {
                            let axes: Vec<usize> = options
                                .as_ref()
                                .and_then(|o| o.axes.as_ref())
                                .map(|ax| ax.iter().map(|&a| a as usize).collect())
                                .unwrap_or_else(|| {
                                    if rank > 1 {
                                        (1..rank).collect()
                                    } else {
                                        vec![0]
                                    }
                                });
                            (0..rank)
                                .map(|i| if axes.contains(&i) { x_shape[i] } else { 1 })
                                .collect()
                        }
                        _ => {
                            let nhwc = match op {
                                Operation::InstanceNormalization { options, .. } => options
                                    .as_ref()
                                    .map(|o| o.layout.eq_ignore_ascii_case("nhwc"))
                                    .unwrap_or(false),
                                _ => false,
                            };
                            let c_idx = if nhwc { rank.saturating_sub(1) } else { 1 };
                            (0..rank)
                                .map(|i| if i == c_idx { x_shape[i] } else { 1 })
                                .collect()
                        }
                    };

                    let mut stripped = op.clone();
                    match &mut stripped {
                        Operation::LayerNormalization {
                            options: Some(o), ..
                        } => {
                            o.scale = None;
                            o.bias = None;
                        }
                        Operation::InstanceNormalization {
                            options: Some(o), ..
                        } => {
                            o.scale = None;
                            o.bias = None;
                        }
                        _ => {}
                    }
                    let norm_name = format!("{out_name}_nogb_{out_id}");
                    let mut norm_overrides = operand_name_overrides.clone();
                    norm_overrides.insert(out_id, norm_name.clone());
                    let mil = self.convert_operation_with_overrides(
                        graph_info,
                        &stripped,
                        &norm_overrides,
                    )?;
                    main_block.operations.push(mil);

                    let mut cur = norm_name;
                    let steps: Vec<(&str, u32, &str)> = scale_id
                        .map(|id| ("mul", id, "gamma"))
                        .into_iter()
                        .chain(bias_id.map(|id| ("add", id, "beta")))
                        .collect();
                    let last = steps.len().saturating_sub(1);
                    for (i, (mil_op, param_id, tag)) in steps.into_iter().enumerate() {
                        let param_name = Self::output_name_for_operand(
                            graph_info,
                            param_id,
                            &operand_name_overrides,
                        );
                        let bcast = Self::rnn_reshape(
                            &mut main_block,
                            &param_name,
                            &target,
                            format!("{out_name}_{tag}_bcast_{out_id}"),
                            mil_dtype,
                        );
                        let (step_name, step_type) = if i == last {
                            (out_name.clone(), out_type.clone())
                        } else {
                            let name = format!("{out_name}_{tag}_applied_{out_id}");
                            let ty = Self::create_value_with_mil_type(
                                graph_info,
                                out_id,
                                name.clone(),
                                mil_dtype,
                            )?;
                            (name, ty)
                        };
                        let mut io = HashMap::new();
                        io.insert("x".to_string(), Self::create_name_argument(cur.clone()));
                        io.insert("y".to_string(), Self::create_name_argument(bcast));
                        main_block.operations.push(Self::create_mil_operation(
                            mil_op,
                            io,
                            vec![step_type],
                        ));
                        cur = step_name;
                    }
                    let _ = cur;
                    continue;
                }
            }

            // Integer division: MIL `real_div` on integer operands has no
            // reliable integer semantics — several CoreML compiler builds
            // constant-fold it with float division (200/16 stays 12.5 through
            // subsequent folded ops), corrupting e.g. packed-nibble unpack
            // chains. Emit `floor_div`, which is integer-defined everywhere.
            // (floor differs from ORT's truncation only for negative
            // quotients, which no supported lowering produces.)
            if matches!(op, Operation::Div { .. })
                && let Some(&div_in) = op.input_operands().first()
                && let Some(div_in_op) = graph_info.operand(div_in)
                && !matches!(
                    div_in_op.descriptor.data_type,
                    DataType::Float32 | DataType::Float16
                )
                && let Some(out_id) = op.output_operand()
            {
                let (_, out_type) =
                    Self::create_output_value(graph_info, out_id, &operand_name_overrides)?;
                let names =
                    Self::input_names_for_operation(graph_info, op, &operand_name_overrides);
                let mut div_in_args = HashMap::new();
                div_in_args.insert(
                    "x".to_string(),
                    Self::create_name_argument(names[0].clone()),
                );
                div_in_args.insert(
                    "y".to_string(),
                    Self::create_name_argument(names[1].clone()),
                );
                main_block.operations.push(Self::create_mil_operation(
                    "floor_div",
                    div_in_args,
                    vec![out_type],
                ));
                continue;
            }

            // identity over a constant: CoreML's compiler elides `identity`
            // ops, and for a const input the plan builder then fails with
            // "Variable is not associated with a name" (error -5) because the
            // alias name was dropped. Emit an exact `mul(x, 1)` instead, which
            // survives compilation. Identity of non-const values is unaffected.
            if matches!(op, Operation::Identity { .. })
                && let Some(&id_in) = op.input_operands().first()
                && let Some(id_in_op) = graph_info.operand(id_in)
                && id_in_op.kind == crate::graph::OperandKind::Constant
                // Scalars keep the plain identity: they pass CoreML's lax
                // identity type-check, while mul's strict inference rejects
                // the rank-1 [1] type the graph declares for them.
                && !id_in_op.descriptor.shape.is_empty()
                && matches!(
                    id_in_op.descriptor.data_type,
                    DataType::Float32 | DataType::Float16 | DataType::Int32
                )
                && let Some(out_id) = op.output_operand()
            {
                let (_, out_type) =
                    Self::create_output_value(graph_info, out_id, &operand_name_overrides)?;
                let one = match id_in_op.descriptor.data_type {
                    DataType::Float16 => Self::create_immediate_float16(1.0),
                    DataType::Int32 => Self::create_immediate_int(1),
                    _ => Self::create_immediate_float(1.0),
                };
                let mut mul_in = HashMap::new();
                mul_in.insert(
                    "x".to_string(),
                    Self::create_name_argument(Self::output_name_for_operand(
                        graph_info,
                        id_in,
                        &operand_name_overrides,
                    )),
                );
                mul_in.insert("y".to_string(), one);
                main_block.operations.push(Self::create_mil_operation(
                    mil_ops::MUL,
                    mul_in,
                    vec![out_type],
                ));
                continue;
            }

            // tile / expand over int8/uint8: MIL `tile` only accepts
            // bool/int32/fp16/fp32 (WebNN expand lowers to tile). Cast to int32,
            // run the op, cast back — exact for all 8-bit values.
            if matches!(op, Operation::Tile { .. } | Operation::Expand { .. })
                && let Some(&tile_in_id) = op.input_operands().first()
                && let Some(tile_in_op) = graph_info.operand(tile_in_id)
                && matches!(
                    tile_in_op.descriptor.data_type,
                    DataType::Int8 | DataType::Uint8
                )
                && let Some(out_id) = op.output_operand()
                && !tile_in_op.descriptor.shape.is_empty()
            {
                let (out_name, out_type) =
                    Self::create_output_value(graph_info, out_id, &operand_name_overrides)?;
                let int32 = crate::protos::coreml::mil_spec::DataType::Int32 as i32;
                let in_shape = tile_in_op.descriptor.static_or_max_shape();

                let cast_in_name = format!("{out_name}_int_in");
                let cast_in_type =
                    Self::value_type_for_static_shape(cast_in_name.clone(), int32, &in_shape);
                main_block.operations.push(Self::create_cast_operation(
                    Self::output_name_for_operand(graph_info, tile_in_id, &operand_name_overrides),
                    cast_in_type,
                    "int32",
                ));

                // Expand with a rank-raising input consumes a helper reshape named
                // after this op's output (see the expand emission); emit it here
                // with the int32 dtype so the reference resolves.
                if matches!(op, Operation::Expand { .. }) {
                    let out_shape = graph_info
                        .operand(out_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .unwrap_or_default();
                    if in_shape.len() < out_shape.len() {
                        let output_rank = out_shape.len();
                        let mut reshaped_dims = vec![1u32; output_rank];
                        for i in 0..in_shape.len() {
                            reshaped_dims[output_rank - i - 1] = in_shape[in_shape.len() - i - 1];
                        }
                        let rs_name =
                            format!("{}_expand_reshaped", operand_name(graph_info, out_id));
                        let rs_type = Self::value_type_for_static_shape(
                            rs_name.clone(),
                            int32,
                            &reshaped_dims,
                        );
                        let mut rs_in: HashMap<String, Argument> = HashMap::new();
                        rs_in.insert(
                            "x".to_string(),
                            Self::create_name_argument(cast_in_name.clone()),
                        );
                        rs_in.insert(
                            "shape".to_string(),
                            Self::create_int_array_argument(
                                reshaped_dims.iter().map(|&v| v as i32).collect(),
                            ),
                        );
                        main_block.operations.push(Self::create_mil_operation(
                            "reshape",
                            rs_in,
                            vec![rs_type],
                        ));
                    }
                }

                let int_out_name = format!("{out_name}_int_out");
                let mut int_overrides = operand_name_overrides.clone();
                int_overrides.insert(tile_in_id, cast_in_name);
                int_overrides.insert(out_id, int_out_name.clone());
                let mut mil =
                    self.convert_operation_with_overrides(graph_info, op, &int_overrides)?;
                // The generic emission types the output after the operand (int8/
                // uint8); the tile actually produces int32 here.
                for nv in &mut mil.outputs {
                    if let Some(vt) = nv.r#type.as_mut()
                        && let Some(crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                            tt,
                        )) = vt.r#type.as_mut()
                    {
                        tt.data_type = int32;
                    }
                }
                main_block.operations.push(mil);

                main_block.operations.push(Self::create_cast_operation(
                    int_out_name,
                    out_type,
                    Self::cast_dtype_string_for_graph_type(&tile_in_op.descriptor.data_type)?,
                ));
                continue;
            }

            // conv2d / convTranspose2d with a runtime (non-constant) bias: MIL
            // declares conv bias const, and CoreML silently miscompiles instead of
            // rejecting (conv2d consumes the bias buffer as weights; convTranspose2d
            // drops the bias). Emit the conv without bias and add it explicitly,
            // reshaped to [1, C_out, 1, 1]. NCHW-only: NHWC convs take a dedicated
            // transpose-wrapped path below that this pre-pass would bypass.
            if matches!(
                op,
                Operation::Conv2d { .. } | Operation::ConvTranspose2d { .. }
            ) {
                let (conv_bias_id, layout_default) = match op {
                    Operation::Conv2d { options, .. } => (
                        options.as_ref().and_then(|o| o.bias),
                        options
                            .as_ref()
                            .map(|o| {
                                o.input_layout.is_empty()
                                    || o.input_layout.eq_ignore_ascii_case("nchw")
                            })
                            .unwrap_or(true),
                    ),
                    Operation::ConvTranspose2d { options, .. } => (
                        options.as_ref().and_then(|o| o.bias),
                        options
                            .as_ref()
                            .map(|o| {
                                o.input_layout.is_empty()
                                    || o.input_layout.eq_ignore_ascii_case("nchw")
                            })
                            .unwrap_or(true),
                    ),
                    _ => (None, true),
                };
                let bias_runtime = conv_bias_id
                    .and_then(|id| graph_info.operand(id))
                    .map(|o| o.kind != crate::graph::OperandKind::Constant)
                    .unwrap_or(false);
                if bias_runtime && layout_default {
                    let bias_id = conv_bias_id.unwrap();
                    let out_id =
                        op.output_operand()
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("{} has no output operand", op.op_type()),
                            })?;
                    let out_op =
                        graph_info
                            .operand(out_id)
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("output operand {out_id} not found"),
                            })?;
                    let out_rank = out_op.descriptor.shape.len();
                    let mil_dtype = Self::graph_value_mil_type(&out_op.descriptor.data_type)?;
                    let (out_name, out_type) =
                        Self::create_output_value(graph_info, out_id, &operand_name_overrides)?;

                    let bias_name =
                        Self::output_name_for_operand(graph_info, bias_id, &operand_name_overrides);
                    let c_out = graph_info
                        .operand(bias_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .and_then(|sh| sh.first().copied())
                        .unwrap_or(1);
                    let mut bias_shape = vec![1u32; out_rank.max(1)];
                    let c_idx = if out_rank >= 2 { 1 } else { 0 };
                    bias_shape[c_idx] = c_out;

                    let mut stripped = op.clone();
                    match &mut stripped {
                        Operation::Conv2d {
                            options: Some(o), ..
                        } => {
                            o.bias = None;
                        }
                        Operation::ConvTranspose2d {
                            options: Some(o), ..
                        } => {
                            o.bias = None;
                        }
                        _ => {}
                    }
                    let nobias_name = format!("{out_name}_nobias_{out_id}");
                    let mut nobias_overrides = operand_name_overrides.clone();
                    nobias_overrides.insert(out_id, nobias_name.clone());
                    let mil = self.convert_operation_with_overrides(
                        graph_info,
                        &stripped,
                        &nobias_overrides,
                    )?;
                    main_block.operations.push(mil);

                    let bias_bcast = Self::rnn_reshape(
                        &mut main_block,
                        &bias_name,
                        &bias_shape,
                        format!("{out_name}_bias_bcast_{out_id}"),
                        mil_dtype,
                    );
                    let mut add_in = HashMap::new();
                    add_in.insert("x".to_string(), Self::create_name_argument(nobias_name));
                    add_in.insert("y".to_string(), Self::create_name_argument(bias_bcast));
                    main_block.operations.push(Self::create_mil_operation(
                        "add",
                        add_in,
                        vec![out_type],
                    ));
                    continue;
                }
            }

            // `where` (MIL: select) — CoreML requires the condition to be bool,
            // but WebNN encodes booleans as uint8. Insert a cast when needed.
            if op_type_lower == "where"
                && let Operation::Where { condition, .. } = op
            {
                let cond_id = *condition;
                let cond_operand =
                    graph_info
                        .operand(cond_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("where condition operand {cond_id} not found"),
                        })?;
                if cond_operand.descriptor.data_type == DataType::Uint8 {
                    let cond_name =
                        Self::output_name_for_operand(graph_info, cond_id, &operand_name_overrides);
                    // Suffix with this op's output id: a bare `{cond}_bool`
                    // collides with the producing comparison's own
                    // `{output}_bool` raw result ("Block redefines I/O name").
                    let where_out = op.output_operand().unwrap_or(cond_id);
                    let bool_cond_name = format!("{cond_name}_bool_{where_out}");
                    let bool_cond_type = Self::create_value_with_mil_type(
                        graph_info,
                        cond_id,
                        bool_cond_name.clone(),
                        crate::protos::coreml::mil_spec::DataType::Bool as i32,
                    )?;
                    main_block.operations.push(Self::create_cast_operation(
                        cond_name,
                        bool_cond_type,
                        "bool",
                    ));

                    let mut overrides = operand_name_overrides.clone();
                    overrides.insert(cond_id, bool_cond_name);
                    let mil_op =
                        self.convert_operation_with_overrides(graph_info, op, &overrides)?;
                    main_block.operations.push(mil_op);
                    continue;
                }
            }

            // Special handling for resample2d: lower to CoreML upsample ops.
            // MIL only upsamples the final two dimensions, so transpose arbitrary
            // WebNN axes into those positions before upsampling and transpose back.
            if op_type_lower == "resample2d" {
                if op.input_operands().is_empty() || op.output_operand().is_none() {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: "resample2d requires input and output operand".to_string(),
                    });
                }

                let input_id = op.input_operands()[0];
                let output_id = op.output_operand().unwrap();

                let input_operand =
                    graph_info
                        .operand(input_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("resample2d input operand {input_id} not found"),
                        })?;

                let opts = match op {
                    Operation::Resample2d { options, .. } => options.clone().unwrap_or_default(),
                    _ => Default::default(),
                };

                let input_shape = input_operand.descriptor.static_or_max_shape();
                if input_shape.len() != 4 {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!(
                            "resample2d requires a 4D input, got {}D",
                            input_shape.len()
                        ),
                    });
                }

                // Resolve axes: default to [2, 3]. Preserve the WebNN axes order:
                // axes[0] maps to MIL height and axes[1] maps to MIL width.
                let axes: Vec<u32> = if opts.axes.is_empty() {
                    vec![2, 3]
                } else {
                    opts.axes.clone()
                };
                if axes.len() != 2
                    || axes[0] >= input_shape.len() as u32
                    || axes[1] >= input_shape.len() as u32
                    || axes[0] == axes[1]
                {
                    return Err(GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!(
                            "resample2d requires two distinct axes for rank {}, got {:?}",
                            input_shape.len(),
                            axes
                        ),
                    });
                }

                let spatial_permutation: Vec<u32> = (0..input_shape.len() as u32)
                    .filter(|axis| !axes.contains(axis))
                    .chain(axes.iter().copied())
                    .collect();
                let inverse_permutation: Vec<u32> = (0..input_shape.len() as u32)
                    .map(|axis| {
                        spatial_permutation
                            .iter()
                            .position(|&candidate| candidate == axis)
                            .expect("spatial permutation contains every input axis")
                            as u32
                    })
                    .collect();
                let identity_permutation: Vec<u32> = (0..input_shape.len() as u32).collect();
                let needs_spatial_transpose = spatial_permutation != identity_permutation;

                let input_name_raw =
                    Self::output_name_for_operand(graph_info, input_id, &operand_name_overrides);
                let (output_name, output_type) =
                    Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;

                let output_operand =
                    graph_info
                        .operand(output_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("resample2d output operand {output_id} not found"),
                        })?;
                let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;

                let (upsample_input_name, upsample_output_name, upsample_output_type) =
                    if needs_spatial_transpose {
                        let spatial_input_shape = Self::permute_graph_shape(
                            &input_operand.descriptor.shape,
                            &spatial_permutation,
                        );
                        let spatial_input_dims =
                            Self::mil_dimensions_from_graph_shape(&spatial_input_shape, false);
                        let spatial_input_name = format!("{}_rs_in_spatial", output_name);
                        let spatial_input_type = NamedValueType {
                            name: spatial_input_name.clone(),
                            r#type: Some(ValueType {
                                r#type: Some(
                                    crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                        TensorType {
                                            rank: spatial_input_dims.len() as i64,
                                            data_type: dtype,
                                            dimensions: spatial_input_dims,
                                            attributes: HashMap::new(),
                                        },
                                    ),
                                ),
                            }),
                        };
                        let mut pre_tp: HashMap<String, Argument> = HashMap::new();
                        pre_tp.insert("x".to_string(), Self::create_name_argument(input_name_raw));
                        pre_tp.insert(
                            "perm".to_string(),
                            Self::create_immediate_int_array(&spatial_permutation),
                        );
                        main_block.operations.push(Self::create_mil_operation(
                            "transpose",
                            pre_tp,
                            vec![spatial_input_type],
                        ));

                        let spatial_output_shape = Self::permute_graph_shape(
                            &output_operand.descriptor.shape,
                            &spatial_permutation,
                        );
                        let spatial_output_dims =
                            Self::mil_dimensions_from_graph_shape(&spatial_output_shape, false);
                        let spatial_output_name = format!("{}_rs_out_spatial", output_name);
                        let spatial_output_type = NamedValueType {
                            name: spatial_output_name.clone(),
                            r#type: Some(ValueType {
                                r#type: Some(
                                    crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                        TensorType {
                                            rank: spatial_output_dims.len() as i64,
                                            data_type: dtype,
                                            dimensions: spatial_output_dims,
                                            attributes: HashMap::new(),
                                        },
                                    ),
                                ),
                            }),
                        };
                        (spatial_input_name, spatial_output_name, spatial_output_type)
                    } else {
                        (input_name_raw, output_name.clone(), output_type.clone())
                    };

                let raw_input_h = input_shape[axes[0] as usize] as f32;
                let raw_input_w = input_shape[axes[1] as usize] as f32;

                // WebNN: when `sizes` is provided it takes precedence and `scales` is ignored.
                let sizes_valid = opts.sizes.as_ref().map(|s| s.len() >= 2).unwrap_or(false);
                let (scale_h, scale_w) = if sizes_valid {
                    if raw_input_h == 0.0 || raw_input_w == 0.0 {
                        return Err(GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: "resample2d: input spatial dimensions are zero".to_string(),
                        });
                    }
                    let sizes = opts.sizes.as_ref().unwrap();
                    let sh = sizes[0] as f32 / raw_input_h;
                    let sw = sizes[1] as f32 / raw_input_w;
                    (sh, sw)
                } else if !opts.scales.is_empty() {
                    let sh = opts.scales.first().copied().unwrap_or(1.0);
                    let sw = opts.scales.get(1).copied().unwrap_or(1.0);
                    (sh, sw)
                } else {
                    (1.0, 1.0)
                };

                let mode = if opts.mode.is_empty() {
                    "nearest-neighbor"
                } else {
                    opts.mode.as_str()
                };

                let mil_op_name = if mode.eq_ignore_ascii_case("linear") {
                    mil_ops::UPSAMPLE_BILINEAR
                } else {
                    mil_ops::UPSAMPLE_NEAREST_NEIGHBOR
                };

                let mut resample_inputs: HashMap<String, Argument> = HashMap::new();
                resample_inputs.insert(
                    "x".to_string(),
                    Self::create_name_argument(upsample_input_name),
                );
                resample_inputs.insert(
                    "scale_factor_height".to_string(),
                    Self::create_immediate_float(scale_h),
                );
                resample_inputs.insert(
                    "scale_factor_width".to_string(),
                    Self::create_immediate_float(scale_w),
                );
                if mode.eq_ignore_ascii_case("linear") {
                    // WebNN linear resample uses half-pixel coordinate centers
                    // (align_corners = false).
                    resample_inputs.insert(
                        "align_corners".to_string(),
                        Self::create_immediate_bool(false),
                    );
                    resample_inputs.insert(
                        "half_pixel_centers".to_string(),
                        Self::create_immediate_bool(true),
                    );
                }

                main_block.operations.push(Self::create_mil_operation(
                    mil_op_name,
                    resample_inputs,
                    vec![upsample_output_type],
                ));

                if needs_spatial_transpose {
                    let mut post_tp: HashMap<String, Argument> = HashMap::new();
                    post_tp.insert(
                        "x".to_string(),
                        Self::create_name_argument(upsample_output_name),
                    );
                    post_tp.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&inverse_permutation),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "transpose",
                        post_tp,
                        vec![output_type],
                    ));
                }

                // Flush deferred transposes for this output
                if let Some((pending_ops, transposed_name)) = deferred_transposes.remove(&output_id)
                {
                    main_block.operations.extend(pending_ops);
                    operand_name_overrides.insert(output_id, transposed_name);
                }
                continue;
            }

            // dequantize over constant weights becomes constexpr_affine_dequantize:
            // Espresso keeps the packed representation through compilation instead of
            // constant-folding into a dense float tensor.
            if op_type_lower == "dequantizelinear"
                && Self::constexpr_dequantize_supported(graph_info, op)
            {
                Self::emit_constexpr_affine_dequantize(
                    graph_info,
                    op,
                    &operand_name_overrides,
                    &weight_builder,
                    &mut main_block,
                )?;
                // A dequantized conv filter carries a deferred layout transpose
                // keyed on this output; flush it like every other emission path.
                if let Some(output_id) = op.output_operand()
                    && let Some((pending_ops, transposed_name)) =
                        deferred_transposes.remove(&output_id)
                {
                    main_block.operations.extend(pending_ops);
                    operand_name_overrides.insert(output_id, transposed_name);
                }
                continue;
            }
            // quantize/dequantize that CoreML's native op can't express (int32 tensors,
            // block-wise or multi-axis scales) is lowered to elementwise arithmetic.
            // int4/uint4 tensors can't be materialized at all, so leave those to the native
            // path (which errors) rather than decomposing.
            if op_type_lower == "dequantizelinear" && Self::qdq_should_decompose(graph_info, op) {
                Self::emit_dequantize_decomposition(
                    graph_info,
                    op,
                    &operand_name_overrides,
                    &mut main_block,
                )?;
                if let Some(output_id) = op.output_operand()
                    && let Some((pending_ops, transposed_name)) =
                        deferred_transposes.remove(&output_id)
                {
                    main_block.operations.extend(pending_ops);
                    operand_name_overrides.insert(output_id, transposed_name);
                }
                continue;
            }
            if op_type_lower == "quantizelinear" && Self::qdq_should_decompose(graph_info, op) {
                Self::emit_quantize_decomposition(
                    graph_info,
                    op,
                    &operand_name_overrides,
                    &mut main_block,
                )?;
                continue;
            }

            // Recurrent networks are decomposed into primitive MIL ops (matmul, add, mul,
            // sub, activations) because CoreML's native lstm/gru don't match WebNN's option
            // set (custom activations, layouts, peephole, reset_after, ...).
            if op_type_lower == "grucell" {
                Self::emit_gru_cell_decomposition(
                    graph_info,
                    op,
                    &operand_name_overrides,
                    &mut main_block,
                )?;
                continue;
            }
            if op_type_lower == "lstmcell" {
                Self::emit_lstm_cell_decomposition(
                    graph_info,
                    op,
                    &operand_name_overrides,
                    &mut main_block,
                )?;
                continue;
            }
            if op_type_lower == "gru" {
                Self::emit_gru_decomposition(
                    graph_info,
                    op,
                    &operand_name_overrides,
                    &mut main_block,
                )?;
                continue;
            }
            if op_type_lower == "lstm" {
                Self::emit_lstm_decomposition(
                    graph_info,
                    op,
                    &operand_name_overrides,
                    &mut main_block,
                )?;
                continue;
            }

            if op_type_lower == "reducelogsumexp" {
                Self::emit_stable_reduce_log_sum_exp(
                    graph_info,
                    op,
                    &operand_name_overrides,
                    &mut main_block,
                )?;
                continue;
            }

            // Dequantize/quantize with rank-0 (scalar) input: CoreML requires rank >= 1.
            // Reshape [] → [1] before the op. The output type from create_output_value already
            // promotes 0D to [1] (scalar_as_one_dim=true), so no reshape back is needed.
            if (op_type_lower == "dequantizelinear" || op_type_lower == "quantizelinear")
                && let Some(&input_id) = op.input_operands().first()
                && let Some(input_op) = graph_info.operand(input_id)
                && input_op.descriptor.shape.is_empty()
            {
                let input_mil_type = Self::mil_data_type(&input_op.descriptor.data_type)?;
                let input_name =
                    Self::output_name_for_operand(graph_info, input_id, &operand_name_overrides);
                let output_id =
                    op.output_operand()
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!(
                                "dequantize/quantize op '{}' has no output",
                                op.op_type()
                            ),
                        })?;
                let output_name =
                    Self::output_name_for_operand(graph_info, output_id, &operand_name_overrides);
                // Reshape [] → [1]
                let reshaped_name = format!("{}_dq_in_1d", output_name);
                let reshaped_type = NamedValueType {
                    name: reshaped_name.clone(),
                    r#type: Some(ValueType {
                        r#type: Some(
                            crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                TensorType {
                                    rank: 1,
                                    data_type: input_mil_type,
                                    dimensions: vec![Dimension {
                                        dimension: Some(dimension::Dimension::Constant(
                                            dimension::ConstantDimension { size: 1 },
                                        )),
                                    }],
                                    attributes: HashMap::new(),
                                },
                            ),
                        ),
                    }),
                };
                let mut reshape_inputs = HashMap::new();
                reshape_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                reshape_inputs.insert(
                    "shape".to_string(),
                    Self::create_immediate_int_array(&[1u32]),
                );
                main_block.operations.push(Self::create_mil_operation(
                    "reshape",
                    reshape_inputs,
                    vec![reshaped_type],
                ));
                // Apply dequantize/quantize with the [1] input override
                let mut overrides_with_1d = operand_name_overrides.clone();
                overrides_with_1d.insert(input_id, reshaped_name);
                let mil_op =
                    self.convert_operation_with_overrides(graph_info, op, &overrides_with_1d)?;
                main_block.operations.push(mil_op);
                continue;
            }

            // BatchNormalization with rank < 3: CoreML batch_norm requires rank >= 3.
            // Reshape input to 3D based on the axis parameter, apply batch_norm, reshape back.
            // Skip when mean is non-constant — the decomposition path below handles that case.
            if op_type_lower == "batchnormalization"
                && let Some(&input_id) = op.input_operands().first()
                && let Some(input_op) = graph_info.operand(input_id)
            {
                let input_rank = input_op.descriptor.shape.len();
                // Runtime params route to the decomposition below; this
                // path's native batch_norm requires them all const.
                let nonconst_param = |id: Option<u32>| {
                    id.and_then(|pid| graph_info.operand(pid))
                        .map(|o| o.kind != crate::graph::OperandKind::Constant)
                        .unwrap_or(false)
                };
                let (bn_scale_id, bn_bias_id) = match op {
                    Operation::BatchNormalization { options, .. } => (
                        options.as_ref().and_then(|o| o.scale),
                        options.as_ref().and_then(|o| o.bias),
                    ),
                    _ => (None, None),
                };
                let any_param_nonconstant = nonconst_param(op.input_operands().get(1).copied())
                    || nonconst_param(op.input_operands().get(2).copied())
                    || nonconst_param(bn_scale_id)
                    || nonconst_param(bn_bias_id);
                if input_rank < 3 && !any_param_nonconstant {
                    let axis = match op {
                        Operation::BatchNormalization { options, .. } => {
                            options.as_ref().map(|o| o.axis as usize).unwrap_or(1)
                        }
                        _ => 1,
                    };
                    let output_id =
                        op.output_operand()
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: "batchNorm op has no output operand".to_string(),
                            })?;
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let input_dtype = Self::mil_data_type(&input_op.descriptor.data_type)?;

                    // Compute 3D shape: [batch_dims_product, C, spatial_dims_product]
                    let shape = &input_op.descriptor.shape;
                    let batch_size: u32 = shape[..axis]
                        .iter()
                        .map(|d| match d {
                            GraphDimension::Static(v) => *v,
                            GraphDimension::Dynamic(d) => d.max_size,
                        })
                        .product::<u32>()
                        .max(1);
                    let channel_size: u32 = if axis < shape.len() {
                        match &shape[axis] {
                            GraphDimension::Static(v) => *v,
                            GraphDimension::Dynamic(d) => d.max_size,
                        }
                    } else {
                        1
                    };
                    let spatial_size: u32 = shape[axis + 1..]
                        .iter()
                        .map(|d| match d {
                            GraphDimension::Static(v) => *v,
                            GraphDimension::Dynamic(d) => d.max_size,
                        })
                        .product::<u32>()
                        .max(1);
                    let shape_3d = [batch_size, channel_size, spatial_size];

                    // Reshape input to 3D
                    let reshaped_input_name = format!("{}_bn_in_3d", output_name);
                    let reshaped_input_type = NamedValueType {
                        name: reshaped_input_name.clone(),
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: 3,
                                        data_type: input_dtype,
                                        dimensions: shape_3d
                                            .iter()
                                            .map(|&d| Dimension {
                                                dimension: Some(dimension::Dimension::Constant(
                                                    dimension::ConstantDimension { size: d as u64 },
                                                )),
                                            })
                                            .collect(),
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };
                    let mut reshape_in_inputs = HashMap::new();
                    reshape_in_inputs
                        .insert("x".to_string(), Self::create_name_argument(input_name));
                    reshape_in_inputs.insert(
                        "shape".to_string(),
                        Self::create_immediate_int_array(&shape_3d),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "reshape",
                        reshape_in_inputs,
                        vec![reshaped_input_type],
                    ));

                    // Apply batch_norm on the 3D tensor
                    let bn_result_name = format!("{}_bn_out_3d", output_name);
                    let bn_result_type = NamedValueType {
                        name: bn_result_name.clone(),
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: 3,
                                        data_type: input_dtype,
                                        dimensions: shape_3d
                                            .iter()
                                            .map(|&d| Dimension {
                                                dimension: Some(dimension::Dimension::Constant(
                                                    dimension::ConstantDimension { size: d as u64 },
                                                )),
                                            })
                                            .collect(),
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };
                    let mut overrides_with_3d = operand_name_overrides.clone();
                    overrides_with_3d.insert(input_id, reshaped_input_name);
                    let bn_op = self.convert_operation_with_input_names_and_outputs(
                        graph_info,
                        op,
                        &Self::input_names_for_operation(graph_info, op, &overrides_with_3d),
                        vec![bn_result_type],
                        self.get_mil_op_type(op.op_type())?,
                    )?;
                    main_block.operations.push(bn_op);

                    // Reshape 3D result back to original output shape
                    let orig_output_shape: Vec<u32> = graph_info
                        .operand(output_id)
                        .map(|o| {
                            o.descriptor
                                .shape
                                .iter()
                                .map(|d| match d {
                                    GraphDimension::Static(v) => *v,
                                    GraphDimension::Dynamic(d) => d.max_size,
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let mut reshape_out_inputs = HashMap::new();
                    reshape_out_inputs
                        .insert("x".to_string(), Self::create_name_argument(bn_result_name));
                    reshape_out_inputs.insert(
                        "shape".to_string(),
                        Self::create_immediate_int_array(&orig_output_shape),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "reshape",
                        reshape_out_inputs,
                        vec![output_type],
                    ));
                    continue;
                }
            }

            // cast targeting int64/uint32/uint64: CoreML MIL has no such tensor type.
            // Emit the cast to int32 and expose the output as an int32 proxy at the
            // model interface (the executor widens int32 -> int64/uint64 on readback,
            // or reinterprets int32 -> uint32 for same-width types).
            if op_type_lower == "cast" {
                use crate::protos::coreml::mil_spec::DataType as MilDataType;
                let output_id =
                    op.output_operand()
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: "cast operation has no output operand".to_string(),
                        })?;
                let out_dt = graph_info
                    .operand(output_id)
                    .map(|o| o.descriptor.data_type);
                if matches!(
                    out_dt,
                    Some(DataType::Uint32 | DataType::Int64 | DataType::Uint64)
                ) {
                    let input_names =
                        Self::input_names_for_operation(graph_info, op, &operand_name_overrides);
                    let (output_name, _) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    // Track the INTERFACE output name so the final cast loop and the model
                    // feature description expose this output as int32 rather than float32.
                    int32_proxy_output_names.insert(operand_name(graph_info, output_id));
                    let out_type = Self::create_value_with_mil_type(
                        graph_info,
                        output_id,
                        output_name,
                        MilDataType::Int32 as i32,
                    )?;
                    let mut inputs: HashMap<String, Argument> = HashMap::new();
                    if let Some(first) = input_names.first() {
                        inputs.insert("x".to_string(), Self::create_argument(first));
                    }
                    inputs.insert("dtype".to_string(), Self::create_immediate_string("int32"));
                    main_block.operations.push(Self::create_mil_operation(
                        mil_ops::CAST,
                        inputs,
                        vec![out_type],
                    ));
                    continue;
                }
            }

            // ArgMax/ArgMin: handle both int input types and int64 output type.
            // CoreML reduce_argmax/reduce_argmin only accepts float input and produces int32.
            // For int8/uint8/int32 inputs: cast to float32 first.
            // For int64 output: cast the int32 result to int64.
            if op_type_lower == "argmax" || op_type_lower == "argmin" {
                use crate::protos::coreml::mil_spec::DataType as MilDataType;
                let output_id =
                    op.output_operand()
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("operation '{}' has no output operand", op.op_type()),
                        })?;
                let output_operand =
                    graph_info
                        .operand(output_id)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Output operand {} not found", output_id),
                        })?;

                let int_input_id = op.input_operands().first().copied();
                let input_needs_float_cast = int_input_id
                    .and_then(|id| graph_info.operand(id))
                    .map(|inp| {
                        matches!(
                            inp.descriptor.data_type,
                            DataType::Int8
                                | DataType::Uint8
                                | DataType::Int32
                                | DataType::Uint32
                                | DataType::Int64
                        )
                    })
                    .unwrap_or(false);
                // CoreML argMax/argMin always outputs int32. When WebNN declares the output as
                // int64 or uint32 (unsupported by CoreML cast), proxy through int32.
                let output_needs_int32_proxy = matches!(
                    output_operand.descriptor.data_type,
                    DataType::Int64 | DataType::Uint32
                );

                if input_needs_float_cast || output_needs_int32_proxy {
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;

                    // When the WebNN output type is unsupported by CoreML, produce int32 output
                    // directly instead of casting to an unsupported type.
                    let final_output_type = if output_needs_int32_proxy {
                        Self::create_value_with_mil_type(
                            graph_info,
                            output_id,
                            output_name.clone(),
                            MilDataType::Int32 as i32,
                        )?
                    } else {
                        output_type.clone()
                    };

                    // Optional: cast integer/unsupported input type to float32
                    let mut input_names =
                        Self::input_names_for_operation(graph_info, op, &operand_name_overrides);
                    if input_needs_float_cast && let Some(input_id) = int_input_id {
                        let orig_name = Self::output_name_for_operand(
                            graph_info,
                            input_id,
                            &operand_name_overrides,
                        );
                        let float_name = format!("{}_arg_float", output_name);
                        let float_type = Self::create_value_with_mil_type(
                            graph_info,
                            input_id,
                            float_name.clone(),
                            MilDataType::Float32 as i32,
                        )?;
                        main_block
                            .operations
                            .push(Self::create_cast_operation(orig_name, float_type, "fp32"));
                        if !input_names.is_empty() {
                            input_names[0] = float_name;
                        }
                    }

                    if output_needs_int32_proxy {
                        // Track the INTERFACE output name (without _graph suffix) for the
                        // final cast loop, which uses int32 instead of the default float32.
                        int32_proxy_output_names.insert(operand_name(graph_info, output_id));
                        // Emit argmax/argmin directly with int32 output (no cast needed).
                        let mil_op_type = self.get_mil_op_type(op.op_type())?;
                        let arg_op = self.convert_operation_with_input_names_and_outputs(
                            graph_info,
                            op,
                            &input_names,
                            vec![final_output_type],
                            mil_op_type,
                        )?;
                        main_block.operations.push(arg_op);
                    } else {
                        // Apply argmax/argmin (CoreML always produces int32), then cast to int32.
                        let int32_name = format!("{}_int32", output_name);
                        let int32_type = Self::create_value_with_mil_type(
                            graph_info,
                            output_id,
                            int32_name.clone(),
                            MilDataType::Int32 as i32,
                        )?;
                        let mil_op_type = self.get_mil_op_type(op.op_type())?;
                        let arg_op = self.convert_operation_with_input_names_and_outputs(
                            graph_info,
                            op,
                            &input_names,
                            vec![int32_type],
                            mil_op_type,
                        )?;
                        main_block.operations.push(arg_op);
                        main_block.operations.push(Self::create_cast_operation(
                            int32_name,
                            output_type,
                            "int32",
                        ));
                    }
                    continue;
                }
            }

            // Gather: cast uint32/int64 indices to int32 (CoreML gather only accepts int32 and smaller).
            // gather / gatherElements: cast indices to int32 and normalize them to WebNN
            // semantics (wrap negatives, clamp out-of-bounds), which CoreML gather does not
            // do on its own (it would read out-of-range memory).
            if matches!(op_type_lower.as_str(), "gather" | "gatherelements") {
                use crate::protos::coreml::mil_spec::DataType as MilDataType;
                let data_id = op.input_operands().first().copied();
                let idx_id = op.input_operands().get(1).copied();
                // CoreML promotes 0D scalars to rank-1, which changes the gather output
                // rank; leave scalar-index gathers to the generic path.
                let idx_is_scalar = idx_id
                    .and_then(|id| graph_info.operand(id))
                    .map(|o| o.descriptor.shape.is_empty())
                    .unwrap_or(true);
                if let (Some(data_id), Some(idx_id), Some(output_id), false) =
                    (data_id, idx_id, op.output_operand(), idx_is_scalar)
                {
                    let axis = match op {
                        Operation::Gather { options, .. } => {
                            options.as_ref().map(|o| o.axis).unwrap_or(0)
                        }
                        Operation::GatherElements { options, .. } => {
                            options.as_ref().map(|o| o.axis).unwrap_or(0)
                        }
                        _ => 0,
                    };
                    let data_shape = graph_info
                        .operand(data_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .unwrap_or_default();
                    let idx_shape = graph_info
                        .operand(idx_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .unwrap_or_default();
                    let axis_size = data_shape.get(axis as usize).copied().unwrap_or(1);

                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let data_name =
                        Self::output_name_for_operand(graph_info, data_id, &operand_name_overrides);
                    let idx_name =
                        Self::output_name_for_operand(graph_info, idx_id, &operand_name_overrides);

                    // Cast indices to int32 (idempotent when already int32).
                    let cast_idx = format!("{}_idx_i32", output_name);
                    let cast_idx_type = Self::create_value_with_mil_type(
                        graph_info,
                        idx_id,
                        cast_idx.clone(),
                        MilDataType::Int32 as i32,
                    )?;
                    main_block.operations.push(Self::create_cast_operation(
                        idx_name,
                        cast_idx_type,
                        "int32",
                    ));
                    let norm_idx = Self::emit_gather_index_norm(
                        &mut main_block,
                        &cast_idx,
                        &idx_shape,
                        &[axis_size],
                        &output_name,
                    );

                    let mil_op = if op_type_lower == "gather" {
                        mil_ops::GATHER
                    } else {
                        mil_ops::GATHER_ALONG_AXIS
                    };
                    let mut gather_inputs: HashMap<String, Argument> = HashMap::new();
                    gather_inputs.insert("x".to_string(), Self::create_name_argument(data_name));
                    gather_inputs
                        .insert("indices".to_string(), Self::create_name_argument(norm_idx));
                    gather_inputs.insert("axis".to_string(), Self::create_immediate_int(axis));
                    gather_inputs.insert(
                        "validate_indices".to_string(),
                        Self::create_immediate_bool(false),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        mil_op,
                        gather_inputs,
                        vec![output_type],
                    ));
                    continue;
                }
            }

            // gatherND: normalize the multi-component indices (wrap negatives / clamp OOB
            // against each indexed input dimension) before CoreML's gather_nd.
            if op_type_lower == "gathernd" {
                use crate::protos::coreml::mil_spec::DataType as MilDataType;
                let data_id = op.input_operands().first().copied();
                let idx_id = op.input_operands().get(1).copied();
                if let (Some(data_id), Some(idx_id), Some(output_id)) =
                    (data_id, idx_id, op.output_operand())
                {
                    let data_shape = graph_info
                        .operand(data_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .unwrap_or_default();
                    let idx_shape = graph_info
                        .operand(idx_id)
                        .map(|o| o.descriptor.static_or_max_shape())
                        .unwrap_or_default();
                    // CoreML gather_nd crashes on rank-5+ data; leave those to the
                    // (guarded) generic path which reports the limitation.
                    let k = idx_shape.last().copied().unwrap_or(0) as usize;
                    if data_shape.len() <= 4 && k >= 1 && k <= data_shape.len() {
                        let sizes: Vec<u32> = data_shape[..k].to_vec();
                        let (output_name, output_type) = Self::create_output_value(
                            graph_info,
                            output_id,
                            &operand_name_overrides,
                        )?;
                        let data_name = Self::output_name_for_operand(
                            graph_info,
                            data_id,
                            &operand_name_overrides,
                        );
                        let idx_name = Self::output_name_for_operand(
                            graph_info,
                            idx_id,
                            &operand_name_overrides,
                        );
                        let cast_idx = format!("{}_idx_i32", output_name);
                        let cast_idx_type = Self::create_value_with_mil_type(
                            graph_info,
                            idx_id,
                            cast_idx.clone(),
                            MilDataType::Int32 as i32,
                        )?;
                        main_block.operations.push(Self::create_cast_operation(
                            idx_name,
                            cast_idx_type,
                            "int32",
                        ));
                        let norm_idx = Self::emit_gather_index_norm(
                            &mut main_block,
                            &cast_idx,
                            &idx_shape,
                            &sizes,
                            &output_name,
                        );
                        let mut gnd_inputs: HashMap<String, Argument> = HashMap::new();
                        gnd_inputs.insert("x".to_string(), Self::create_name_argument(data_name));
                        gnd_inputs
                            .insert("indices".to_string(), Self::create_name_argument(norm_idx));
                        gnd_inputs.insert(
                            "validate_indices".to_string(),
                            Self::create_immediate_bool(false),
                        );
                        main_block.operations.push(Self::create_mil_operation(
                            mil_ops::GATHER_ND,
                            gnd_inputs,
                            vec![output_type],
                        ));
                        continue;
                    }
                }
            }

            // BatchNormalization decomposition: when mean/variance are runtime inputs (non-constant)
            // or when axis != 1 (NHWC or other non-standard layouts), CoreML's native batch_norm
            // cannot be used. Decompose into element-wise ops:
            //   normalized = (x - mean) / sqrt(variance + epsilon)
            //   output = scale * normalized + bias  (if scale/bias present)
            // Mean/variance may need reshaping for broadcasting when axis is not the last dimension.
            if op_type_lower == "batchnormalization"
                && let Operation::BatchNormalization { options, .. } = op
            {
                let mean_id = op.input_operands().get(1).copied();
                let variance_id = op.input_operands().get(2).copied();
                let scale_id = options.as_ref().and_then(|o| o.scale);
                let bias_id = options.as_ref().and_then(|o| o.bias);
                let epsilon = options.as_ref().map(|o| o.epsilon as f32).unwrap_or(1e-5);
                let axis = options.as_ref().map(|o| o.axis as usize).unwrap_or(1);

                let is_runtime_param = |id: Option<u32>| {
                    id.and_then(|id| graph_info.operand(id))
                        .map(|o| o.kind != crate::graph::OperandKind::Constant)
                        .unwrap_or(false)
                };
                let mean_is_nonconstant = is_runtime_param(mean_id);

                let input_id = op.input_operands().first().copied();
                let input_operand = input_id.and_then(|id| graph_info.operand(id));
                let input_rank = input_operand.map(|o| o.descriptor.shape.len()).unwrap_or(0);

                // Decompose when any of mean/variance/gamma/beta is a runtime
                // value (native batch_norm requires them all const) or when
                // axis is non-standard (not 1).
                let use_decomposition = mean_is_nonconstant
                    || is_runtime_param(variance_id)
                    || is_runtime_param(scale_id)
                    || is_runtime_param(bias_id)
                    || (axis != 1 && input_rank >= 2);

                if use_decomposition
                    && let (Some(input_id), Some(mean_id_v), Some(variance_id_v)) =
                        (input_id, mean_id, variance_id)
                {
                    let output_id =
                        op.output_operand()
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: "batchNorm has no output".to_string(),
                            })?;
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let mean_name = Self::output_name_for_operand(
                        graph_info,
                        mean_id_v,
                        &operand_name_overrides,
                    );
                    let var_name = Self::output_name_for_operand(
                        graph_info,
                        variance_id_v,
                        &operand_name_overrides,
                    );

                    let inp_op = graph_info.operand(input_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("batchNorm input {} not found", input_id),
                        }
                    })?;
                    let dtype = Self::mil_data_type(&inp_op.descriptor.data_type)?;
                    let is_f16 = inp_op.descriptor.data_type == DataType::Float16;

                    // When axis is not the last dimension, [C]-shaped params
                    // (mean/variance/gamma/beta) need an explicit reshape to
                    // [1,..,C,..,1] — MIL broadcasting aligns trailing dims.
                    let bcast: Option<(Vec<u32>, Vec<Dimension>)> =
                        if axis != input_rank.saturating_sub(1) && input_rank > 1 {
                            let c_size = inp_op
                                .descriptor
                                .shape
                                .get(axis)
                                .map(|d| match d {
                                    GraphDimension::Static(v) => *v,
                                    GraphDimension::Dynamic(d) => d.max_size,
                                })
                                .unwrap_or(1);
                            let mut bcast_shape = vec![1u32; input_rank];
                            bcast_shape[axis] = c_size;
                            let bcast_dims: Vec<Dimension> = bcast_shape
                                .iter()
                                .map(|&d| Dimension {
                                    dimension: Some(dimension::Dimension::Constant(
                                        dimension::ConstantDimension { size: d as u64 },
                                    )),
                                })
                                .collect();
                            Some((bcast_shape, bcast_dims))
                        } else {
                            None
                        };

                    // Reshape mean/variance to the broadcast shape when needed.
                    // Also return the effective var shape so downstream add/sqrt types match.
                    let (mean_for_sub, var_for_div, effective_var_shape) =
                        if let Some((bcast_shape, bcast_dims)) = &bcast {
                            let bcast_shape = bcast_shape.clone();
                            let bcast_dims = bcast_dims.clone();
                            let effective_var_shape: Vec<GraphDimension> = bcast_shape
                                .iter()
                                .map(|&d| GraphDimension::Static(d))
                                .collect();

                            let mean_rs_name = format!("{}_bn_mean_rs", output_name);
                            let mean_rs_type = NamedValueType {
                            name: mean_rs_name.clone(),
                            r#type: Some(ValueType {
                                r#type: Some(
                                    crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                        TensorType {
                                            rank: input_rank as i64,
                                            data_type: dtype,
                                            dimensions: bcast_dims.clone(),
                                            attributes: HashMap::new(),
                                        },
                                    ),
                                ),
                            }),
                        };
                            let mut rs_in: HashMap<String, Argument> = HashMap::new();
                            rs_in.insert(
                                "x".to_string(),
                                Self::create_name_argument(mean_name.clone()),
                            );
                            rs_in.insert(
                                "shape".to_string(),
                                Self::create_immediate_int_array(&bcast_shape),
                            );
                            main_block.operations.push(Self::create_mil_operation(
                                "reshape",
                                rs_in,
                                vec![mean_rs_type],
                            ));

                            let var_rs_name = format!("{}_bn_var_rs", output_name);
                            let var_rs_type = NamedValueType {
                            name: var_rs_name.clone(),
                            r#type: Some(ValueType {
                                r#type: Some(
                                    crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                        TensorType {
                                            rank: input_rank as i64,
                                            data_type: dtype,
                                            dimensions: bcast_dims,
                                            attributes: HashMap::new(),
                                        },
                                    ),
                                ),
                            }),
                        };
                            let mut rs_var: HashMap<String, Argument> = HashMap::new();
                            rs_var.insert(
                                "x".to_string(),
                                Self::create_name_argument(var_name.clone()),
                            );
                            rs_var.insert(
                                "shape".to_string(),
                                Self::create_immediate_int_array(&bcast_shape),
                            );
                            main_block.operations.push(Self::create_mil_operation(
                                "reshape",
                                rs_var,
                                vec![var_rs_type],
                            ));
                            (mean_rs_name, var_rs_name, effective_var_shape)
                        } else {
                            let orig_var_shape = graph_info
                                .operand(variance_id_v)
                                .map(|o| o.descriptor.shape.clone())
                                .unwrap_or_default();
                            (mean_name, var_name, orig_var_shape)
                        };

                    // Build an intermediate value type matching the output shape.
                    let out_shape = &inp_op.descriptor.shape;
                    let out_dims = Self::mil_dimensions_from_graph_shape(out_shape, false);
                    let make_intermediate = |name: String| NamedValueType {
                        name,
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: out_dims.len() as i64,
                                        data_type: dtype,
                                        dimensions: out_dims.clone(),
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };

                    // x_minus_mean = x - mean
                    let x_minus_mean = format!("{}_bn_xmm", output_name);
                    let mut sub_in: HashMap<String, Argument> = HashMap::new();
                    sub_in.insert("x".to_string(), Self::create_name_argument(input_name));
                    sub_in.insert("y".to_string(), Self::create_name_argument(mean_for_sub));
                    main_block.operations.push(Self::create_mil_operation(
                        "sub",
                        sub_in,
                        vec![make_intermediate(x_minus_mean.clone())],
                    ));

                    // var_plus_eps = variance + epsilon
                    // Use effective_var_shape (may be reshaped to bcast_shape) for type.
                    let var_dims =
                        Self::mil_dimensions_from_graph_shape(&effective_var_shape, false);
                    let make_var_type = |name: String| NamedValueType {
                        name,
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: var_dims.len() as i64,
                                        data_type: dtype,
                                        dimensions: var_dims.clone(),
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };
                    let var_eps_name = format!("{}_bn_veps", output_name);
                    let eps_arg = if is_f16 {
                        Self::create_immediate_float16(epsilon)
                    } else {
                        Self::create_immediate_float(epsilon)
                    };
                    let mut veps_in: HashMap<String, Argument> = HashMap::new();
                    veps_in.insert(
                        "x".to_string(),
                        Self::create_name_argument(var_for_div.clone()),
                    );
                    veps_in.insert("y".to_string(), eps_arg);
                    main_block.operations.push(Self::create_mil_operation(
                        "add",
                        veps_in,
                        vec![make_var_type(var_eps_name.clone())],
                    ));

                    // std = sqrt(var_plus_eps)
                    let std_name = format!("{}_bn_std", output_name);
                    let mut sqrt_in: HashMap<String, Argument> = HashMap::new();
                    sqrt_in.insert("x".to_string(), Self::create_name_argument(var_eps_name));
                    main_block.operations.push(Self::create_mil_operation(
                        "sqrt",
                        sqrt_in,
                        vec![make_var_type(std_name.clone())],
                    ));

                    // normalized = x_minus_mean / std
                    let normed_name = format!("{}_bn_normed", output_name);
                    let mut div_in: HashMap<String, Argument> = HashMap::new();
                    div_in.insert("x".to_string(), Self::create_name_argument(x_minus_mean));
                    div_in.insert("y".to_string(), Self::create_name_argument(std_name));
                    main_block.operations.push(Self::create_mil_operation(
                        "real_div",
                        div_in,
                        vec![make_intermediate(normed_name.clone())],
                    ));

                    // Reshape a [C] param to `bcast` (same treatment as mean/var).
                    let bcast_param =
                        |src_name: String, tag: &str, ops: &mut Vec<MilOperation>| -> String {
                            let Some((bcast_shape, bcast_dims)) = &bcast else {
                                return src_name;
                            };
                            let rs_name = format!("{}_bn_{}_rs", output_name, tag);
                            let rs_type = NamedValueType {
                            name: rs_name.clone(),
                            r#type: Some(ValueType {
                                r#type: Some(
                                    crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                        TensorType {
                                            rank: input_rank as i64,
                                            data_type: dtype,
                                            dimensions: bcast_dims.clone(),
                                            attributes: HashMap::new(),
                                        },
                                    ),
                                ),
                            }),
                        };
                            let mut rs_in: HashMap<String, Argument> = HashMap::new();
                            rs_in.insert("x".to_string(), Self::create_name_argument(src_name));
                            rs_in.insert(
                                "shape".to_string(),
                                Self::create_immediate_int_array(bcast_shape),
                            );
                            ops.push(Self::create_mil_operation("reshape", rs_in, vec![rs_type]));
                            rs_name
                        };

                    // Apply scale (gamma) and bias (beta) if present
                    let after_scale = if let Some(sc_id) = scale_id {
                        let sc_name = Self::output_name_for_operand(
                            graph_info,
                            sc_id,
                            &operand_name_overrides,
                        );
                        let sc_name = bcast_param(sc_name, "gamma", &mut main_block.operations);
                        let scaled_name = format!("{}_bn_scaled", output_name);
                        let mut mul_in: HashMap<String, Argument> = HashMap::new();
                        mul_in.insert("x".to_string(), Self::create_name_argument(normed_name));
                        mul_in.insert("y".to_string(), Self::create_name_argument(sc_name));
                        main_block.operations.push(Self::create_mil_operation(
                            "mul",
                            mul_in,
                            vec![make_intermediate(scaled_name.clone())],
                        ));
                        scaled_name
                    } else {
                        normed_name
                    };

                    let final_name = if let Some(bi_id) = bias_id {
                        let bi_name = Self::output_name_for_operand(
                            graph_info,
                            bi_id,
                            &operand_name_overrides,
                        );
                        let bi_name = bcast_param(bi_name, "beta", &mut main_block.operations);
                        let biased_name = output_name.clone();
                        let mut add_in: HashMap<String, Argument> = HashMap::new();
                        add_in.insert("x".to_string(), Self::create_name_argument(after_scale));
                        add_in.insert("y".to_string(), Self::create_name_argument(bi_name));
                        main_block.operations.push(Self::create_mil_operation(
                            "add",
                            add_in,
                            vec![output_type],
                        ));
                        biased_name
                    } else {
                        // No bias: rename final intermediate to output name via identity
                        let mut id_in: HashMap<String, Argument> = HashMap::new();
                        id_in.insert("x".to_string(), Self::create_name_argument(after_scale));
                        main_block.operations.push(Self::create_mil_operation(
                            "identity",
                            id_in,
                            vec![output_type],
                        ));
                        output_name.clone()
                    };
                    let _ = final_name;
                    continue;
                }
            }

            // Special handling for instanceNormalization with NHWC layout.
            // CoreML instance_norm requires NCHW [N,C,H,W]. For NHWC inputs:
            //   transpose NHWC→NCHW, instance_norm(NCHW), transpose NCHW→NHWC.
            if op_type_lower == "instancenormalization" {
                let inst_layout = match op {
                    Operation::InstanceNormalization { options, .. } => {
                        options.as_ref().map(|o| o.layout.as_str()).unwrap_or("")
                    }
                    _ => "",
                };
                if inst_layout.eq_ignore_ascii_case("nhwc") {
                    let input_id = op.input_operands().first().copied().ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: "instanceNorm op has no input operand".to_string(),
                        }
                    })?;
                    let output_id =
                        op.output_operand()
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: "instanceNorm op has no output operand".to_string(),
                            })?;
                    let input_operand = graph_info.operand(input_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("instanceNorm input operand {} not found", input_id),
                        }
                    })?;
                    let output_operand = graph_info.operand(output_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("instanceNorm output operand {} not found", output_id),
                        }
                    })?;

                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;

                    // Pre-transpose: NHWC [N,H,W,C] -> NCHW [N,C,H,W], perm=[0,3,1,2]
                    let nchw_in_name = format!("{}_in_nchw", output_name);
                    let nchw_perm = [0u32, 3, 1, 2];
                    let nchw_in_shape =
                        Self::permute_graph_shape(&input_operand.descriptor.shape, &nchw_perm);
                    let nchw_in_dims = Self::mil_dimensions_from_graph_shape(&nchw_in_shape, false);
                    let nchw_in_type = NamedValueType {
                        name: nchw_in_name.clone(),
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: nchw_in_dims.len() as i64,
                                        data_type: dtype,
                                        dimensions: nchw_in_dims,
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };
                    let mut pre_tp: HashMap<String, Argument> = HashMap::new();
                    pre_tp.insert("x".to_string(), Self::create_name_argument(input_name));
                    pre_tp.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&nchw_perm),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "transpose",
                        pre_tp,
                        vec![nchw_in_type],
                    ));

                    // instance_norm on NCHW intermediate
                    let nchw_out_name = format!("{}_out_nchw", output_name);
                    let nchw_out_perm = [0u32, 3, 1, 2];
                    let nchw_out_shape =
                        Self::permute_graph_shape(&output_operand.descriptor.shape, &nchw_out_perm);
                    let nchw_out_dims =
                        Self::mil_dimensions_from_graph_shape(&nchw_out_shape, false);
                    let nchw_out_type = NamedValueType {
                        name: nchw_out_name.clone(),
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: nchw_out_dims.len() as i64,
                                        data_type: dtype,
                                        dimensions: nchw_out_dims,
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };
                    let mut overrides_nchw = operand_name_overrides.clone();
                    overrides_nchw.insert(input_id, nchw_in_name);
                    let in_names = Self::input_names_for_operation(graph_info, op, &overrides_nchw);
                    let norm_op = self.convert_operation_with_input_names_and_outputs(
                        graph_info,
                        op,
                        &in_names,
                        vec![nchw_out_type],
                        mil_ops::INSTANCE_NORM,
                    )?;
                    main_block.operations.push(norm_op);

                    // Post-transpose: NCHW [N,C,H,W] -> NHWC [N,H,W,C], perm=[0,2,3,1]
                    let post_perm = [0u32, 2, 3, 1];
                    let mut post_tp: HashMap<String, Argument> = HashMap::new();
                    post_tp.insert("x".to_string(), Self::create_name_argument(nchw_out_name));
                    post_tp.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&post_perm),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "transpose",
                        post_tp,
                        vec![output_type],
                    ));

                    if let Some((pending_ops, transposed_name)) =
                        deferred_transposes.remove(&output_id)
                    {
                        main_block.operations.extend(pending_ops);
                        operand_name_overrides.insert(output_id, transposed_name);
                    }
                    continue;
                }
            }

            // Special handling for conv2d / convTranspose2d with NHWC layout.
            // CoreML conv requires NCHW. The pre-scan (above) has already transposed input and
            // filter operands to NCHW and recorded the overrides. Here we run the conv op with
            // an intermediate NCHW output name, then post-transpose to restore NHWC.
            if op_type_lower == "conv2d" || op_type_lower == "convtranspose2d" {
                let conv_layout = match op {
                    Operation::Conv2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.input_layout.as_str())
                        .unwrap_or(""),
                    Operation::ConvTranspose2d { options, .. } => options
                        .as_ref()
                        .map(|o| o.input_layout.as_str())
                        .unwrap_or(""),
                    _ => "",
                };
                if conv_layout.eq_ignore_ascii_case("nhwc") {
                    let output_id =
                        op.output_operand()
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("{} op has no output operand", op.op_type()),
                            })?;
                    let output_operand = graph_info.operand(output_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("conv2d output operand {} not found", output_id),
                        }
                    })?;

                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let dtype = Self::mil_data_type(&output_operand.descriptor.data_type)?;

                    // Intermediate NCHW output: permute NHWC [N,H',W',C'] → [N,C',H',W']
                    let nchw_perm = [0u32, 3, 1, 2];
                    let nchw_out_shape =
                        Self::permute_graph_shape(&output_operand.descriptor.shape, &nchw_perm);
                    let nchw_out_dims =
                        Self::mil_dimensions_from_graph_shape(&nchw_out_shape, false);
                    let nchw_out_name = format!("{}_nchw_out", output_name);
                    let nchw_out_type = NamedValueType {
                        name: nchw_out_name.clone(),
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: nchw_out_dims.len() as i64,
                                        data_type: dtype,
                                        dimensions: nchw_out_dims,
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };

                    // Run conv with NCHW-transposed inputs (set up by pre-scan) and NCHW output
                    let input_names =
                        Self::input_names_for_operation(graph_info, op, &operand_name_overrides);
                    let mil_op_type = self.get_mil_op_type(op.op_type())?;
                    let conv_op = self.convert_operation_with_input_names_and_outputs(
                        graph_info,
                        op,
                        &input_names,
                        vec![nchw_out_type],
                        mil_op_type,
                    )?;
                    main_block.operations.push(conv_op);

                    // Post-transpose: NCHW [N,C',H',W'] → NHWC [N,H',W',C'], perm=[0,2,3,1]
                    let post_perm = [0u32, 2, 3, 1];
                    let mut post_tp_inputs: HashMap<String, Argument> = HashMap::new();
                    post_tp_inputs
                        .insert("x".to_string(), Self::create_name_argument(nchw_out_name));
                    post_tp_inputs.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&post_perm),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "transpose",
                        post_tp_inputs,
                        vec![output_type],
                    ));

                    // Flush deferred transposes for this output
                    if let Some((pending_ops, transposed_name)) =
                        deferred_transposes.remove(&output_id)
                    {
                        main_block.operations.extend(pending_ops);
                        operand_name_overrides.insert(output_id, transposed_name);
                    }
                    continue;
                }
            }

            // Special handling for averagePool2d / maxPool2d with NHWC layout.
            // CoreML pooling only supports NCHW, so we wrap with transpose ops:
            //   transpose(NHWC→NCHW), pool(NCHW), transpose(NCHW→NHWC).
            if op_type_lower == "averagepool2d"
                || op_type_lower == "maxpool2d"
                || op_type_lower == "l2pool2d"
            {
                let pool_layout = match op {
                    Operation::AveragePool2d { options, .. }
                    | Operation::MaxPool2d { options, .. }
                    | Operation::L2Pool2d { options, .. } => {
                        options.as_ref().map(|o| o.layout.as_str()).unwrap_or("")
                    }
                    _ => "",
                };
                if pool_layout.eq_ignore_ascii_case("nhwc") {
                    let input_id = op.input_operands().first().copied().ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("pool op '{}' has no input operand", op.op_type()),
                        }
                    })?;
                    let output_id =
                        op.output_operand()
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!("pool op '{}' has no output operand", op.op_type()),
                            })?;

                    let input_operand = graph_info.operand(input_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("pool input operand {} not found", input_id),
                        }
                    })?;
                    let output_operand = graph_info.operand(output_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("pool output operand {} not found", output_id),
                        }
                    })?;

                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;
                    let dtype = Self::mil_data_type(&input_operand.descriptor.data_type)?;

                    // Pre-transpose: NHWC [N,H,W,C] -> NCHW [N,C,H,W], perm=[0,3,1,2]
                    let nchw_input_name = format!("{}_pool_nchw_in", output_name);
                    let nchw_input_perm = [0u32, 3, 1, 2];
                    let nchw_input_shape = Self::permute_graph_shape(
                        &input_operand.descriptor.shape,
                        &nchw_input_perm,
                    );
                    let nchw_input_dims =
                        Self::mil_dimensions_from_graph_shape(&nchw_input_shape, false);

                    let nchw_input_type = NamedValueType {
                        name: nchw_input_name.clone(),
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: nchw_input_dims.len() as i64,
                                        data_type: dtype,
                                        dimensions: nchw_input_dims,
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };
                    let mut pre_tp_inputs: HashMap<String, Argument> = HashMap::new();
                    pre_tp_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    pre_tp_inputs.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&nchw_input_perm),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "transpose",
                        pre_tp_inputs,
                        vec![nchw_input_type],
                    ));

                    // Pool (NCHW): intermediate output shape is [N,C,H',W']
                    let nchw_pool_output_name = format!("{}_pool_nchw_out", output_name);
                    // Compute NCHW output shape: permute NHWC output [N,H',W',C] -> [N,C,H',W']
                    let nchw_out_perm = [0u32, 3, 1, 2];
                    let nchw_pool_shape =
                        Self::permute_graph_shape(&output_operand.descriptor.shape, &nchw_out_perm);
                    let nchw_pool_dims =
                        Self::mil_dimensions_from_graph_shape(&nchw_pool_shape, false);
                    let nchw_pool_output_type = NamedValueType {
                        name: nchw_pool_output_name.clone(),
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: nchw_pool_dims.len() as i64,
                                        data_type: dtype,
                                        dimensions: nchw_pool_dims,
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };
                    // Build pool inputs using the NCHW-transposed input name
                    let mut overrides_for_pool = operand_name_overrides.clone();
                    overrides_for_pool.insert(input_id, nchw_input_name);
                    let pool_input_names =
                        Self::input_names_for_operation(graph_info, op, &overrides_for_pool);
                    let mil_op_type = self.get_mil_op_type(op.op_type())?;
                    let pool_op = self.convert_operation_with_input_names_and_outputs(
                        graph_info,
                        op,
                        &pool_input_names,
                        vec![nchw_pool_output_type],
                        mil_op_type,
                    )?;
                    main_block.operations.push(pool_op);

                    // Post-transpose: NCHW [N,C,H',W'] -> NHWC [N,H',W',C], perm=[0,2,3,1]
                    let post_tp_perm = [0u32, 2, 3, 1];
                    let mut post_tp_inputs: HashMap<String, Argument> = HashMap::new();
                    post_tp_inputs.insert(
                        "x".to_string(),
                        Self::create_name_argument(nchw_pool_output_name),
                    );
                    post_tp_inputs.insert(
                        "perm".to_string(),
                        Self::create_immediate_int_array(&post_tp_perm),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "transpose",
                        post_tp_inputs,
                        vec![output_type],
                    ));

                    // Flush deferred transposes for this output
                    if let Some((pending_ops, transposed_name)) =
                        deferred_transposes.remove(&output_id)
                    {
                        main_block.operations.extend(pending_ops);
                        operand_name_overrides.insert(output_id, transposed_name);
                    }
                    continue;
                }
            }

            // Special handling for reduce ops on 0D (scalar) inputs.
            // CoreML requires at least rank-1 inputs for reduce operations.
            // Wrap: reshape([] -> [1]), reduce(axes=[0], keep_dims=False), reshape([1] -> []) if needed.
            let is_reduce_op = matches!(
                op_type_lower.as_str(),
                "reducesum"
                    | "reducemean"
                    | "reducemax"
                    | "reducemin"
                    | "reduceproduct"
                    | "reducel1"
                    | "reducel2"
                    | "reducelogsum"
                    | "reducelogsumexp"
                    | "reducesumsquare"
            );
            // WebNN reduce with an explicit empty `axes` list reduces over NO
            // dimensions: the output shape equals the input shape, but the
            // per-op element transform still applies (reduceL1 -> abs,
            // reduceL2 -> abs, reduceLogSum -> log, reduceSumSquare -> square,
            // reduceLogSumExp -> identity, etc.). CoreML's reduce with an
            // omitted `axes` reduces ALL dimensions, so we instead append a
            // singleton axis and reduce over it: reducing a one-element window
            // yields exactly the WebNN empty-axes semantics for every reduce.
            if is_reduce_op {
                let axes_explicitly_empty = match op {
                    Operation::ReduceSum { options, .. }
                    | Operation::ReduceMean { options, .. }
                    | Operation::ReduceMax { options, .. }
                    | Operation::ReduceMin { options, .. }
                    | Operation::ReduceProduct { options, .. }
                    | Operation::ReduceL1 { options, .. }
                    | Operation::ReduceL2 { options, .. }
                    | Operation::ReduceLogSum { options, .. }
                    | Operation::ReduceLogSumExp { options, .. }
                    | Operation::ReduceSumSquare { options, .. } => options
                        .as_ref()
                        .and_then(|o| o.axes.as_ref())
                        .map(|a| a.is_empty())
                        .unwrap_or(false),
                    _ => false,
                };
                if axes_explicitly_empty {
                    let input_id = op.input_operands().first().copied().ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("reduce op '{}' has no input operand", op.op_type()),
                        }
                    })?;
                    let output_id =
                        op.output_operand()
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!(
                                    "reduce op '{}' has no output operand",
                                    op.op_type()
                                ),
                            })?;
                    let input_op = graph_info.operand(input_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Input operand {} not found", input_id),
                        }
                    })?;
                    let mil_dtype = Self::mil_data_type(&input_op.descriptor.data_type)?;

                    // Reshape input to [input_shape..., 1] and reduce over the new axis.
                    let mut reshaped_shape_vals = input_op.descriptor.static_or_max_shape();
                    reshaped_shape_vals.push(1);
                    let reduce_axis = (reshaped_shape_vals.len() - 1) as u32;
                    let mut reshaped_dims = input_op.descriptor.shape.clone();
                    reshaped_dims.push(GraphDimension::Static(1));

                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    let (output_name, output_type) =
                        Self::create_output_value(graph_info, output_id, &operand_name_overrides)?;

                    let reshaped_name = format!("{}_emptyaxes_rs", output_name);
                    let reshaped_type = Self::create_named_value_type(
                        reshaped_name.clone(),
                        mil_dtype,
                        &reshaped_dims,
                        false,
                    );
                    let mut reshape_inputs: HashMap<String, Argument> = HashMap::new();
                    reshape_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    reshape_inputs.insert(
                        "shape".to_string(),
                        Self::create_immediate_int_array(&reshaped_shape_vals),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "reshape",
                        reshape_inputs,
                        vec![reshaped_type],
                    ));

                    let mil_op_type = self.get_mil_op_type(op.op_type())?;
                    let mut reduce_inputs: HashMap<String, Argument> = HashMap::new();
                    reduce_inputs
                        .insert("x".to_string(), Self::create_name_argument(reshaped_name));
                    reduce_inputs.insert(
                        "axes".to_string(),
                        Self::create_immediate_int_array(&[reduce_axis]),
                    );
                    reduce_inputs
                        .insert("keep_dims".to_string(), Self::create_immediate_bool(false));
                    main_block.operations.push(Self::create_mil_operation(
                        mil_op_type,
                        reduce_inputs,
                        vec![output_type],
                    ));

                    // Flush deferred transposes for this output.
                    if let Some((pending_ops, transposed_name)) =
                        deferred_transposes.remove(&output_id)
                    {
                        main_block.operations.extend(pending_ops);
                        operand_name_overrides.insert(output_id, transposed_name);
                    }
                    continue;
                }
            }
            if is_reduce_op {
                let maybe_0d_input = op.input_operands().first().and_then(|&id| {
                    graph_info
                        .operand(id)
                        .filter(|o| o.descriptor.shape.is_empty())
                        .map(|o| (id, o.descriptor.data_type))
                });
                if let Some((input_id, input_dtype)) = maybe_0d_input {
                    let input_name = Self::output_name_for_operand(
                        graph_info,
                        input_id,
                        &operand_name_overrides,
                    );
                    // Output-id suffix: two reduces sharing one 0-D operand must not collide.
                    let reshaped_input_name = format!(
                        "{}_reduce1d_{}",
                        input_name,
                        op.output_operand().unwrap_or(input_id)
                    );
                    let mil_dtype = Self::mil_data_type(&input_dtype)?;

                    // Build the [1] dimension for the reshaped input
                    let one_dim = Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension { size: 1 },
                        )),
                    };
                    let reshaped_input_type = NamedValueType {
                        name: reshaped_input_name.clone(),
                        r#type: Some(ValueType {
                            r#type: Some(
                                crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                    TensorType {
                                        rank: 1,
                                        data_type: mil_dtype,
                                        dimensions: vec![one_dim],
                                        attributes: HashMap::new(),
                                    },
                                ),
                            ),
                        }),
                    };

                    // Emit: reshape(x=input, shape=[1]) -> reshaped_input_name
                    let mut reshape_inputs: HashMap<String, Argument> = HashMap::new();
                    reshape_inputs.insert("x".to_string(), Self::create_name_argument(input_name));
                    reshape_inputs.insert(
                        "shape".to_string(),
                        Self::create_immediate_int_array(&[1u32]),
                    );
                    main_block.operations.push(Self::create_mil_operation(
                        "reshape",
                        reshape_inputs,
                        vec![reshaped_input_type],
                    ));

                    // Override the input operand name so create_operation_inputs uses the 1D version
                    let mut overrides_with_reshape = operand_name_overrides.clone();
                    overrides_with_reshape.insert(input_id, reshaped_input_name.clone());

                    // Get the output operand
                    let output_id =
                        op.output_operand()
                            .ok_or_else(|| GraphError::ConversionFailed {
                                format: "coreml_mlprogram".to_string(),
                                reason: format!(
                                    "reduce op '{}' has no output operand",
                                    op.op_type()
                                ),
                            })?;
                    let output_operand = graph_info.operand(output_id).ok_or_else(|| {
                        GraphError::ConversionFailed {
                            format: "coreml_mlprogram".to_string(),
                            reason: format!("Output operand {} not found", output_id),
                        }
                    })?;
                    let webnn_output_is_0d = output_operand.descriptor.shape.is_empty();

                    let mil_op_type = self.get_mil_op_type(op.op_type())?;
                    let input_names =
                        Self::input_names_for_operation(graph_info, op, &overrides_with_reshape);

                    if webnn_output_is_0d {
                        // The reduce will output [1] (axes=[0], keep_dims would give [1])
                        // but we need [] — use a reduce-to-scalar intermediate then reshape.
                        let (output_name, output_type) = Self::create_output_value(
                            graph_info,
                            output_id,
                            &operand_name_overrides,
                        )?;
                        let reduce_intermediate_name = format!("{}_reduce_1d", output_name);
                        let reduce_intermediate_type = NamedValueType {
                            name: reduce_intermediate_name.clone(),
                            r#type: Some(ValueType {
                                r#type: Some(
                                    crate::protos::coreml::mil_spec::value_type::Type::TensorType(
                                        TensorType {
                                            rank: 1,
                                            data_type: mil_dtype,
                                            dimensions: vec![Dimension {
                                                dimension: Some(dimension::Dimension::Constant(
                                                    dimension::ConstantDimension { size: 1 },
                                                )),
                                            }],
                                            attributes: HashMap::new(),
                                        },
                                    ),
                                ),
                            }),
                        };

                        // Build reduce inputs manually: x=reshaped, axes=[0], keep_dims=True
                        // so the output stays [1] (which we can reshape to [])
                        let mut reduce_inputs: HashMap<String, Argument> = HashMap::new();
                        if let Some(first) = input_names.first() {
                            reduce_inputs.insert("x".to_string(), Self::create_argument(first));
                        }
                        reduce_inputs.insert(
                            "axes".to_string(),
                            Self::create_immediate_int_array(&[0u32]),
                        );
                        reduce_inputs
                            .insert("keep_dims".to_string(), Self::create_immediate_bool(true));
                        main_block.operations.push(Self::create_mil_operation(
                            mil_op_type,
                            reduce_inputs,
                            vec![reduce_intermediate_type],
                        ));

                        // Reshape [1] -> []
                        let mut reshape_back: HashMap<String, Argument> = HashMap::new();
                        reshape_back.insert(
                            "x".to_string(),
                            Self::create_name_argument(reduce_intermediate_name),
                        );
                        reshape_back
                            .insert("shape".to_string(), Self::create_immediate_int_array(&[]));
                        main_block.operations.push(Self::create_mil_operation(
                            "reshape",
                            reshape_back,
                            vec![output_type],
                        ));
                    } else {
                        // Output is not 0D — just emit normally with the 1D input override
                        let mil_op = self.convert_operation_with_overrides(
                            graph_info,
                            op,
                            &overrides_with_reshape,
                        )?;
                        main_block.operations.push(mil_op);
                    }
                    continue;
                }
            }

            let mil_op =
                self.convert_operation_with_overrides(graph_info, op, &operand_name_overrides)?;
            main_block.operations.push(mil_op);

            // Flush any transpose ops that were waiting for this operation's output, and
            // activate the corresponding operand-name override so that later operations
            // that consume this operand use the transposed name.
            if let Some(output_id) = op.output_operand()
                && let Some((pending_ops, transposed_name)) = deferred_transposes.remove(&output_id)
            {
                main_block.operations.extend(pending_ops);
                operand_name_overrides.insert(output_id, transposed_name);
            }
        }

        // Add block outputs (output operand names)
        for &output_id in &graph_info.output_operands {
            let operand =
                graph_info
                    .operand(output_id)
                    .ok_or_else(|| GraphError::ConversionFailed {
                        format: "coreml_mlprogram".to_string(),
                        reason: format!("Output operand {} not found", output_id),
                    })?;
            let output_name = operand_name(graph_info, output_id);
            let graph_output_name =
                Self::output_name_for_operand(graph_info, output_id, &operand_name_overrides);
            let graph_mil_type = Self::graph_value_mil_type(&operand.descriptor.data_type)?;
            let interface_mil_type = Self::interface_mil_data_type(&operand.descriptor.data_type);
            // Wide ints and argmin/argmax proxy outputs use int32 at the interface
            // (not float32); the executor widens int32 -> int64/uint64 on readback.
            let effective_interface_type = if int32_proxy_output_names.contains(&output_name)
                || Self::is_wide_int(&operand.descriptor.data_type)
            {
                use crate::protos::coreml::mil_spec::DataType as MilDt;
                MilDt::Int32 as i32
            } else {
                interface_mil_type
            };
            if graph_mil_type != effective_interface_type {
                let output_type = Self::create_value_with_mil_type(
                    graph_info,
                    output_id,
                    output_name.clone(),
                    effective_interface_type,
                )?;
                main_block.operations.push(Self::create_cast_operation(
                    graph_output_name,
                    output_type,
                    Self::cast_dtype_string_for_mil_type(effective_interface_type)?,
                ));
            }
            main_block.outputs.push(output_name);
        }

        // CoreML rejects, at MLModel load time ("Error in declaring input X with
        // error -1."), a function input that no operation consumes — e.g.
        // castLike's target_type (only its dtype matters, never its values) or
        // an input whose sole consumer was constant-folded away. Drop such
        // inputs from the function signature and, below, from the model
        // description. Dispatch may still bind a tensor for them: CoreML
        // ignores extra features at prediction time.
        let mut referenced_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        {
            fn walk_block_names(block: &Block, names: &mut std::collections::HashSet<String>) {
                use crate::protos::coreml::mil_spec::argument::binding::Binding;
                for op in &block.operations {
                    for arg in op.inputs.values() {
                        for b in &arg.arguments {
                            if let Some(Binding::Name(n)) = &b.binding {
                                names.insert(n.clone());
                            }
                        }
                    }
                    for nested in &op.blocks {
                        walk_block_names(nested, names);
                    }
                }
            }
            walk_block_names(&main_block, &mut referenced_names);
        }
        main_function
            .inputs
            .retain(|nv| referenced_names.contains(&nv.name));
        let declared_input_names: std::collections::HashSet<String> = main_function
            .inputs
            .iter()
            .map(|nv| nv.name.clone())
            .collect();

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

        // Create ModelDescription with model-level input/output feature descriptions.
        // Single-function MLProgram models must use model-level I/O (not the
        // `functions` field), otherwise CoreML rejects them with
        // "multi-function description syntax" at load time.
        use crate::protos::coreml::specification::{FeatureDescription, ModelDescription};

        let mut input_descriptions = Vec::new();
        for &input_id in &graph_info.input_operands {
            if let Some(operand) = graph_info.operand(input_id) {
                let input_name = operand_name(graph_info, input_id);
                // Unused inputs were dropped from the function signature above;
                // the model description must match it.
                if !declared_input_names.contains(&input_name) {
                    continue;
                }
                input_descriptions.push(FeatureDescription {
                    name: input_name,
                    r#type: Some(Self::create_feature_type(&operand.descriptor)?),
                    ..Default::default()
                });
            }
        }

        let mut output_descriptions = Vec::new();
        for &output_id in &graph_info.output_operands {
            if let Some(operand) = graph_info.operand(output_id) {
                let output_name = operand_name(graph_info, output_id);
                // For int32 proxy outputs (argmin/argmax, or any wide int64/uint32/uint64
                // output) use Int32 at the model interface to match the function emit type.
                let feature_type = if int32_proxy_output_names.contains(&output_name)
                    || Self::is_wide_int(&operand.descriptor.data_type)
                {
                    use crate::protos::coreml::specification::{
                        ArrayFeatureType, FeatureType, feature_type,
                    };
                    let mut af = ArrayFeatureType {
                        data_type: crate::protos::coreml::specification::array_feature_type::ArrayDataType::Int32 as i32,
                        ..Default::default()
                    };
                    let shape = operand.descriptor.static_or_max_shape();
                    for &d in &shape {
                        af.shape.push(d as i64);
                    }
                    FeatureType {
                        r#type: Some(feature_type::Type::MultiArrayType(af)),
                        is_optional: false,
                    }
                } else {
                    Self::create_feature_type(&operand.descriptor)?
                };
                output_descriptions.push(FeatureDescription {
                    name: output_name,
                    r#type: Some(feature_type),
                    ..Default::default()
                });
            }
        }

        model.description = Some(ModelDescription {
            input: input_descriptions,
            output: output_descriptions,
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
    #[cfg(feature = "dynamic-inputs")]
    use crate::graph::DynamicDimension;
    use crate::graph::{ConstantData, GraphInfo, Operand, OperandDescriptor, OperandKind};
    use crate::operator_options::{OperationExtras, OperatorOptions};
    use crate::operators::Operation;

    /// Build an `Operation` from WebNN-style `op` name, operand indices, and parsed options (tests).
    fn op_from_operator_options(
        op_type: &str,
        input_operands: Vec<u32>,
        output_operand: Option<u32>,
        output_operands: Vec<u32>,
        attributes: OperatorOptions,
    ) -> Operation {
        let output_ids: Vec<u32> = if !output_operands.is_empty() {
            output_operands
        } else if let Some(o) = output_operand {
            vec![o]
        } else {
            Vec::new()
        };
        Operation::from_operator_options(
            op_type,
            &input_operands,
            &attributes,
            &output_ids,
            OperationExtras::default(),
        )
        .expect("valid test op")
    }
    #[cfg(feature = "dynamic-inputs")]
    use crate::protos::coreml::mil_spec::dimension;
    use crate::protos::coreml::specification::Model;
    #[cfg(feature = "dynamic-inputs")]
    use crate::protos::coreml::specification::model::Type;
    use prost::Message;
    use std::collections::HashMap;

    fn s(shape: &[u32]) -> Vec<crate::graph::Dimension> {
        crate::graph::to_dimension_vector(shape)
    }

    fn main_operation_types(graph: &GraphInfo) -> Vec<String> {
        let converted = CoremlMlProgramConverter
            .convert(graph)
            .expect("CoreML conversion should succeed");
        decode_main_block(&converted.data)
            .operations
            .iter()
            .map(|op| op.r#type.clone())
            .collect()
    }

    fn immediate_int_values(argument: &Argument) -> Vec<i32> {
        let binding = argument.arguments.first().expect("argument binding");
        let value = match binding.binding.as_ref().expect("binding value") {
            crate::protos::coreml::mil_spec::argument::binding::Binding::Value(value) => value,
            _ => panic!("expected immediate argument"),
        };
        let immediate = match value.value.as_ref().expect("argument value") {
            crate::protos::coreml::mil_spec::value::Value::ImmediateValue(immediate) => immediate,
            _ => panic!("expected immediate value"),
        };
        let tensor = match immediate.value.as_ref().expect("immediate tensor") {
            crate::protos::coreml::mil_spec::value::immediate_value::Value::Tensor(tensor) => {
                tensor
            }
            _ => panic!("expected tensor value"),
        };
        match tensor.value.as_ref().expect("tensor payload") {
            crate::protos::coreml::mil_spec::tensor_value::Value::Ints(values) => {
                values.values.clone()
            }
            _ => panic!("expected integer tensor"),
        }
    }

    #[test]
    fn test_resample2d_moves_arbitrary_axes_to_spatial_dimensions() {
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[3, 2, 1, 1]),
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[6, 4, 1, 1]),
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "resample2d",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::from_json_with_op_type(
                    "resample2d",
                    &serde_json::json!({ "axes": [0, 1], "sizes": [6, 4] }),
                )
                .expect("resample2d options"),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("CoreML resample2d conversion");
        let main_block = decode_main_block(&converted.data);
        let transposes: Vec<_> = main_block
            .operations
            .iter()
            .filter(|op| op.r#type == "transpose")
            .collect();
        assert_eq!(transposes.len(), 2, "expected pre/post transposes");
        assert_eq!(
            immediate_int_values(transposes[0].inputs.get("perm").expect("pre perm")),
            [2, 3, 0, 1]
        );
        assert_eq!(
            immediate_int_values(transposes[1].inputs.get("perm").expect("post perm")),
            [2, 3, 0, 1]
        );
        assert!(
            main_block
                .operations
                .iter()
                .any(|op| op.r#type == mil_ops::UPSAMPLE_NEAREST_NEIGHBOR)
        );
    }

    fn reduce_log_sum_exp_graph(
        input_shape: &[u32],
        output_shape: &[u32],
        options: serde_json::Value,
    ) -> GraphInfo {
        GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(input_shape),
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(output_shape),
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "reduceLogSumExp",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::from_json_with_op_type("reduceLogSumExp", &options)
                    .expect("reduceLogSumExp options"),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn test_reduce_log_sum_exp_uses_stable_decomposition() {
        let graph = reduce_log_sum_exp_graph(
            &[2, 3],
            &[2],
            serde_json::json!({ "axes": [1], "keepDimensions": false }),
        );

        assert_eq!(
            main_operation_types(&graph),
            vec![
                "reduce_max",
                "sub",
                "exp",
                "reduce_sum",
                "log",
                "add",
                "squeeze",
            ]
        );
    }

    #[test]
    fn test_coreml_conv2d_reads_webnn_padding_option() {
        let graph = GraphInfo {
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[1, 1, 3, 3]),
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("filter".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[1, 1, 1, 1]),
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[1, 1, 6, 10]),
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "conv2d",
                vec![0, 1],
                Some(2),
                vec![],
                OperatorOptions::from_json_with_op_type(
                    "conv2d",
                    &serde_json::json!({ "padding": [1, 2, 3, 4] }),
                )
                .expect("conv2d options"),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("CoreML conv2d conversion");
        let main_block = decode_main_block(&converted.data);
        let conv = main_block
            .operations
            .iter()
            .find(|op| op.r#type == mil_ops::CONV)
            .expect("conv operation");
        assert_eq!(
            immediate_int_values(conv.inputs.get("pad").expect("MIL pad input")),
            [1, 2, 3, 4]
        );
        assert!(!conv.inputs.contains_key("pads"));
    }

    #[test]
    fn test_reduce_log_sum_exp_empty_axes_is_identity() {
        let graph = reduce_log_sum_exp_graph(
            &[2, 3],
            &[2, 3],
            serde_json::json!({ "axes": [], "keepDimensions": false }),
        );

        assert_eq!(main_operation_types(&graph), vec!["identity"]);
    }

    #[test]
    fn test_reduce_log_sum_exp_scalar_uses_stable_decomposition() {
        let graph = reduce_log_sum_exp_graph(&[], &[], serde_json::json!({}));

        assert_eq!(
            main_operation_types(&graph),
            vec![
                "reshape",
                "reduce_max",
                "sub",
                "exp",
                "reduce_sum",
                "log",
                "add",
            ]
        );
    }

    /// Helper to create a simple graph with a Float16 constant
    fn create_graph_with_float16_constant(
        shape: Vec<crate::graph::Dimension>,
        data: Vec<u8>,
    ) -> GraphInfo {
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![1], // Output is operand 1
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
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
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        graph
    }

    #[test]
    fn test_parse_mlnumber_f64_non_finite_strings() {
        let pos_inf = serde_json::json!("Infinity");
        let neg_inf = serde_json::json!("-Infinity");
        let nan = serde_json::json!("NaN");
        let finite = serde_json::json!("3.5");

        let parsed_pos =
            CoremlMlProgramConverter::parse_mlnumber_f64(Some(&pos_inf)).expect("parse +inf");
        assert!(parsed_pos.is_infinite());
        assert!(parsed_pos.is_sign_positive());

        let parsed_neg =
            CoremlMlProgramConverter::parse_mlnumber_f64(Some(&neg_inf)).expect("parse -inf");
        assert!(parsed_neg.is_infinite());
        assert!(parsed_neg.is_sign_negative());

        let parsed_nan =
            CoremlMlProgramConverter::parse_mlnumber_f64(Some(&nan)).expect("parse nan");
        assert!(parsed_nan.is_nan());

        let parsed_finite =
            CoremlMlProgramConverter::parse_mlnumber_f64(Some(&finite)).expect("parse finite");
        assert_eq!(parsed_finite, 3.5);
    }

    #[test]
    fn test_parse_clamp_bound_nan_uses_default() {
        let nan = serde_json::json!("NaN");
        let value = CoremlMlProgramConverter::parse_clamp_bound(Some(&nan), 42.0);
        assert_eq!(value, 42.0);
    }

    #[test]
    fn test_float16_scalar_constant_uses_immediate_value() {
        // Create a scalar Float16 constant (shape = [])
        let f16_val = half::f16::from_f32(1.5);
        let data = f16_val.to_le_bytes().to_vec();

        let graph = create_graph_with_float16_constant(s(&[]), data.clone());

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

        let graph = create_graph_with_float16_constant(s(&[3]), data.clone());

        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        assert!(
            result.weights_data.is_some(),
            "Non-scalar Float16 should use weight file"
        );

        let weights = result.weights_data.unwrap();

        // v2 file layout:
        // [0-63]    64-byte global header: count(u32)=1, version(u32)=2, 56 zero bytes
        // [64-127]  64-byte WeightMetadata: sentinel, FLOAT16 type, size_in_bytes=6, payload_offset=128, zeros
        // [128-133] 6 bytes payload
        // [134-191] padding → total 192 bytes

        // Global header
        assert_eq!(&weights[0..4], &1u32.to_le_bytes(), "Entry count = 1");
        assert_eq!(&weights[4..8], &2u32.to_le_bytes(), "File version = 2");

        // WeightMetadata at offset 64
        assert_eq!(&weights[64..68], &0xDEADBEEFu32.to_le_bytes(), "Sentinel");
        assert_eq!(
            &weights[68..72],
            &1u32.to_le_bytes(), // FLOAT16 = 1
            "BlobDataType::FLOAT16"
        );
        assert_eq!(&weights[72..80], &6u64.to_le_bytes(), "size_in_bytes = 6");
        assert_eq!(
            &weights[80..88],
            &128u64.to_le_bytes(),
            "payload at offset 128"
        );

        // Payload
        assert_eq!(&weights[128..134], &data[..], "payload data");
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

        let graph = create_graph_with_float16_constant(s(&[2, 2]), data.clone());

        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        assert!(
            result.weights_data.is_some(),
            "2D Float16 constant should use weight file"
        );

        let weights = result.weights_data.unwrap();

        // size_in_bytes = 8 (4 elements × 2 bytes each), located at [72..80] of metadata block
        assert_eq!(&weights[72..80], &8u64.to_le_bytes(), "size_in_bytes = 8");
        // Payload at offset 128
        assert_eq!(&weights[128..136], &data[..], "payload data");
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
            quantized: false,
        };

        // Operand 0: First Float16 constant [2]
        let data1 = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0
        graph.operands.push(Operand {
            name: Some("constant1".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: s(&[2]),
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
                shape: s(&[2]),
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
                shape: s(&[2]),
                pending_permutation: vec![],
            },
        });

        // Add operation: output = constant1 + constant2
        graph.operations.push(op_from_operator_options(
            "add",
            vec![0, 1],
            Some(2),
            vec![],
            OperatorOptions::default(),
        ));

        // Convert
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Verify weights_data is present
        assert!(
            result.weights_data.is_some(),
            "Multiple Float16 constants should use weight file"
        );

        let weights = result.weights_data.unwrap();

        // v2 layout for two 2-element Float16 constants (4 bytes each):
        // [0-63]    global header (count=2, version=2, zeros)
        // [64-127]  metadata1 (sentinel, FLOAT16, size=4, payload_offset=128, zeros)
        // [128-131] payload1 (4 bytes), padded to [192]
        // [192-255] metadata2 (sentinel, FLOAT16, size=4, payload_offset=256, zeros)
        // [256-259] payload2 (4 bytes), padded to [320]
        // Total: 320 bytes
        assert_eq!(weights.len(), 320, "Two Float16 constants layout");

        // Global header
        assert_eq!(&weights[0..4], &2u32.to_le_bytes(), "Entry count = 2");
        assert_eq!(&weights[4..8], &2u32.to_le_bytes(), "File version = 2");

        // First entry sentinel at offset 64
        assert_eq!(
            &weights[64..68],
            &0xDEADBEEFu32.to_le_bytes(),
            "First sentinel"
        );

        // Second entry sentinel at offset 192
        assert_eq!(
            &weights[192..196],
            &0xDEADBEEFu32.to_le_bytes(),
            "Second sentinel"
        );
    }

    #[test]
    fn test_float32_constant_uses_weight_file() {
        // Non-scalar Float32 constants go to the weight file like Float16:
        // immediate values are re-serialized through CoreML's textual MIL
        // writer at compile time, which is pathological for large weights.
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        // Float32 constant
        let data = vec![0x00, 0x00, 0x80, 0x3F]; // 1.0 as f32
        graph.operands.push(Operand {
            name: Some("constant".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[1]),
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
                shape: s(&[1]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // Convert
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph).unwrap();

        // Non-scalar Float32 lands in the weight file with FLOAT32 metadata.
        let weights = result
            .weights_data
            .expect("non-scalar Float32 constants should use the weight file");
        assert_eq!(
            &weights[68..72],
            &2u32.to_le_bytes(),
            "BlobDataType::FLOAT32"
        );
        assert_eq!(&weights[72..80], &4u64.to_le_bytes(), "size_in_bytes = 4");
        assert_eq!(&weights[128..132], &1.0f32.to_le_bytes(), "payload data");
    }

    #[test]
    fn test_int4_data_type_supported() {
        let mut graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        // Int4 input
        graph.operands.push(Operand {
            name: Some("input".to_string()),
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
        });

        // Output
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // int4 is now supported via the int32 proxy; conversion should succeed.
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph);
        assert!(result.is_ok(), "int4 should convert: {:?}", result.err());
    }

    #[test]
    fn test_uint4_data_type_supported() {
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        // Uint4 constant
        let data = vec![0x12, 0x34, 0x56, 0x78];
        graph.operands.push(Operand {
            name: Some("constant".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Uint4,
                shape: s(&[8]),
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
                data_type: DataType::Uint4,
                shape: s(&[8]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // uint4 is now supported via the int32 proxy; conversion should succeed.
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph);
        assert!(result.is_ok(), "uint4 should convert: {:?}", result.err());
    }

    #[test]
    fn test_int4_output_supported() {
        let mut graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        // Float32 input
        graph.operands.push(Operand {
            name: Some("input".to_string()),
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
        });

        // Int4 output (this should be rejected when building value info)
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: s(&[10, 10]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // int4 output is now supported via the int32 proxy; conversion should succeed.
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph);
        assert!(
            result.is_ok(),
            "int4 output should convert: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_uint4_input_supported() {
        let mut graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        // Uint4 input
        graph.operands.push(Operand {
            name: Some("input".to_string()),
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Uint4,
                shape: s(&[1, 3, 224, 224]),
                pending_permutation: vec![],
            },
        });

        // Float32 output
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(&[1, 3, 224, 224]),
                pending_permutation: vec![],
            },
        });

        // Add relu operation
        graph.operations.push(op_from_operator_options(
            "relu",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        // uint4 input is now supported via the int32 proxy; conversion should succeed.
        let converter = CoremlMlProgramConverter;
        let result = converter.convert(&graph);
        assert!(
            result.is_ok(),
            "uint4 input should convert: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_linear_float32_converts_to_mul_add_ops() {
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[2, 3]),
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[2, 3]),
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "linear",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::from_json_with_op_type(
                    "linear",
                    &serde_json::json!({ "alpha": 2.0, "beta": -1.0 }),
                )
                .expect("linear options"),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("coreml linear float32 conversion should succeed");
        let model = Model::decode(converted.data.as_slice()).expect("decode coreml model");
        let program = match model.r#type.expect("model type") {
            crate::protos::coreml::specification::model::Type::MlProgram(program) => program,
            _ => panic!("expected MLProgram model"),
        };
        let main_fn = program.functions.get("main").expect("main function");
        let main_block = main_fn
            .block_specializations
            .get("CoreML7")
            .expect("CoreML7 block");

        assert!(main_block.operations.iter().any(|op| op.r#type == "mul"));
        assert!(main_block.operations.iter().any(|op| op.r#type == "add"));
    }

    /// Decode a converted model and return its main CoreML7 block.
    fn decode_main_block(data: &[u8]) -> crate::protos::coreml::mil_spec::Block {
        let model = Model::decode(data).expect("decode coreml model");
        let program = match model.r#type.expect("model type") {
            crate::protos::coreml::specification::model::Type::MlProgram(program) => program,
            _ => panic!("expected MLProgram model"),
        };
        program
            .functions
            .get("main")
            .expect("main function")
            .block_specializations
            .get("CoreML7")
            .expect("CoreML7 block")
            .clone()
    }

    /// Unpack a scalar immediate int argument (e.g. the `axis` input of a
    /// native `dequantize` op) from a decoded MIL operation.
    fn immediate_int_argument(arg: &Argument) -> i32 {
        use crate::protos::coreml::mil_spec::{tensor_value, value};
        let Some(Binding::Value(v)) = arg.arguments.first().and_then(|b| b.binding.as_ref()) else {
            panic!("expected immediate value binding");
        };
        let Some(value::Value::ImmediateValue(imm)) = &v.value else {
            panic!("expected immediate value");
        };
        let Some(value::immediate_value::Value::Tensor(t)) = &imm.value else {
            panic!("expected tensor value");
        };
        let Some(tensor_value::Value::Ints(ints)) = &t.value else {
            panic!("expected int tensor");
        };
        ints.values[0]
    }

    /// Graph: dequantizeLinear(int8 constant of `input_shape`, f32 constant
    /// scale of `scale_shape`, no zero point) -> f32 output. Without a zero
    /// point a per-channel scale can't use constexpr_affine_dequantize, so the
    /// op exercises the native/decomposed dequantize paths.
    fn create_dequantize_graph(input_shape: &[u32], scale_shape: &[u32]) -> GraphInfo {
        let mut graph = GraphInfo {
            input_operands: vec![],
            output_operands: vec![2],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };
        graph.operands.push(Operand {
            name: Some("weights".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Int8,
                shape: s(input_shape),
                pending_permutation: vec![],
            },
        });
        let input_len = input_shape.iter().product::<u32>() as usize;
        graph.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: vec![1u8; input_len],
                label: None,
            },
        );
        graph.operands.push(Operand {
            name: Some("scale".to_string()),
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(scale_shape),
                pending_permutation: vec![],
            },
        });
        let scale_len = scale_shape.iter().product::<u32>() as usize;
        graph.constant_operand_ids_to_handles.insert(
            1,
            ConstantData {
                data: 0.5f32.to_le_bytes().repeat(scale_len),
                label: None,
            },
        );
        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: s(input_shape),
                pending_permutation: vec![],
            },
        });
        graph.operations.push(op_from_operator_options(
            "dequantizeLinear",
            vec![0, 1],
            Some(2),
            vec![],
            OperatorOptions::default(),
        ));
        graph
    }

    #[test]
    fn test_qdq_per_channel_axis_rank_aligned() {
        type C = CoremlMlProgramConverter;
        // Square weight: both input dims match the scale length; rank
        // alignment picks the scale's own non-unit position, not the first
        // coincident input dim.
        assert_eq!(C::qdq_per_channel_axis(&[4, 4], &[1, 4]), Some(1));
        assert_eq!(C::qdq_per_channel_axis(&[4, 4], &[4, 1]), Some(0));
        // Rank-mismatched scales fall back to the first matching input dim.
        assert_eq!(C::qdq_per_channel_axis(&[3, 4], &[4]), Some(1));
        // Blockwise: the scale length divides input dim 1 but coincidentally
        // equals input dim 0 — no valid per-channel axis.
        assert_eq!(C::qdq_per_channel_axis(&[64, 128], &[1, 64]), None);
        // Per-tensor scales have no per-channel axis.
        assert_eq!(C::qdq_per_channel_axis(&[4, 4], &[1, 1]), None);
        assert_eq!(C::qdq_per_channel_axis(&[4, 4], &[]), None);
    }

    #[test]
    fn test_qdq_native_rejects_coincident_blockwise_scale() {
        // Input [64, 128] with rank-aligned scale [1, 64]: blockwise along
        // axis 1 (block 2). The squeezed length 64 coincides with input dim 0,
        // which previously passed the contains() check as "per-channel".
        assert!(!CoremlMlProgramConverter::qdq_native_supported(
            &DataType::Int8,
            &[64, 128],
            &[1, 64],
        ));
        // Exact per-channel and per-tensor scales stay native.
        assert!(CoremlMlProgramConverter::qdq_native_supported(
            &DataType::Int8,
            &[64, 128],
            &[1, 128],
        ));
        assert!(CoremlMlProgramConverter::qdq_native_supported(
            &DataType::Int8,
            &[64, 128],
            &[1, 1],
        ));
    }

    #[test]
    fn test_dequantize_square_matrix_per_channel_axis() {
        // Square int8 weight [4, 4] with rank-aligned scale [1, 4]: the scale
        // names axis 1; first-match derivation would silently pick axis 0.
        let graph = create_dequantize_graph(&[4, 4], &[1, 4]);
        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("square-matrix per-channel dequantize should convert");
        let block = decode_main_block(&converted.data);
        let deq = block
            .operations
            .iter()
            .find(|op| op.r#type == "dequantize")
            .expect("per-channel int8 dequantize should use the native op");
        let axis = deq
            .inputs
            .get("axis")
            .expect("per-channel dequantize emits axis");
        assert_eq!(immediate_int_argument(axis), 1);
    }

    #[test]
    fn test_dequantize_coincident_blockwise_scale_decomposes() {
        // Input [4, 8] with scale [1, 4]: blockwise along axis 1 (block 2)
        // whose length coincides with input dim 0. Native dequantize can't
        // express it; it must lower to the elementwise decomposition.
        let graph = create_dequantize_graph(&[4, 8], &[1, 4]);
        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("blockwise dequantize should convert via decomposition");
        let block = decode_main_block(&converted.data);
        assert!(
            !block
                .operations
                .iter()
                .any(|op| op.r#type == "dequantize" || op.r#type == "constexpr_affine_dequantize"),
            "coincident blockwise scale must not use native/constexpr dequantize"
        );
        assert!(
            block.operations.iter().any(|op| op.r#type == "mul"),
            "elementwise decomposition should emit mul"
        );
    }

    #[cfg(not(feature = "dynamic-inputs"))]
    #[test]
    fn test_dynamic_dimensions_require_feature_opt_in() {
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            crate::graph::Dimension::Dynamic(crate::graph::DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            crate::graph::Dimension::Static(4),
                        ],
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            crate::graph::Dimension::Dynamic(crate::graph::DynamicDimension {
                                name: "batch".to_string(),
                                max_size: 8,
                            }),
                            crate::graph::Dimension::Static(4),
                        ],
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "identity",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::default(),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let err = CoremlMlProgramConverter.convert(&graph).unwrap_err();
        assert!(matches!(err, GraphError::DynamicInputsFeatureDisabled));
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn test_dynamic_input_dim_maps_to_unknown_mil_dimension() {
        let mut graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        graph.operands.push(Operand {
            name: Some("input".to_string()),
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![
                    crate::graph::Dimension::Dynamic(DynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    crate::graph::Dimension::Static(4),
                ],
                pending_permutation: vec![],
            },
        });

        graph.operands.push(Operand {
            name: Some("output".to_string()),
            kind: OperandKind::Output,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![
                    crate::graph::Dimension::Dynamic(DynamicDimension {
                        name: "batch".to_string(),
                        max_size: 8,
                    }),
                    crate::graph::Dimension::Static(4),
                ],
                pending_permutation: vec![],
            },
        });

        graph.operations.push(op_from_operator_options(
            "identity",
            vec![0],
            Some(1),
            vec![],
            OperatorOptions::default(),
        ));

        let converter = CoremlMlProgramConverter;
        let converted = converter.convert(&graph).unwrap();
        let model = Model::decode(converted.data.as_slice()).unwrap();
        let program = match model.r#type.unwrap() {
            Type::MlProgram(p) => p,
            _ => panic!("expected mlProgram"),
        };
        let main = program.functions.get("main").expect("main function");
        let input = main.inputs.first().expect("input");
        let tensor = match input
            .r#type
            .as_ref()
            .and_then(|t| t.r#type.as_ref())
            .expect("input type")
        {
            crate::protos::coreml::mil_spec::value_type::Type::TensorType(t) => t,
            _ => panic!("expected tensor input"),
        };

        match tensor.dimensions[0].dimension.as_ref().expect("dim 0") {
            dimension::Dimension::Unknown(_) => {}
            _ => panic!("expected unknown dimension for dynamic batch"),
        }

        match tensor.dimensions[1].dimension.as_ref().expect("dim 1") {
            dimension::Dimension::Constant(c) => assert_eq!(c.size, 4),
            _ => panic!("expected constant dimension for static axis"),
        }
    }

    #[test]
    fn test_linear_float16_converts_successfully() {
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float16,
                        shape: s(&[4]),
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float16,
                        shape: s(&[4]),
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "linear",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::default(),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("coreml linear float16 conversion should succeed");
        let model = Model::decode(converted.data.as_slice()).expect("decode coreml model");
        assert!(model.r#type.is_some(), "model type should be set");
    }

    #[test]
    fn test_cumulative_sum_converts_to_cumsum_op() {
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            crate::graph::Dimension::Static(2),
                            crate::graph::Dimension::Static(3),
                        ],
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![
                            crate::graph::Dimension::Static(2),
                            crate::graph::Dimension::Static(3),
                        ],
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "cumulativeSum",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::from_json_with_op_type(
                    "cumulativeSum",
                    &serde_json::json!({ "axis": 1, "exclusive": true, "reversed": true }),
                )
                .expect("cumulativeSum options"),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("coreml cumulativeSum conversion should succeed");
        let model = Model::decode(converted.data.as_slice()).expect("decode coreml model");
        let program = match model.r#type.expect("model type") {
            crate::protos::coreml::specification::model::Type::MlProgram(program) => program,
            _ => panic!("expected MLProgram model"),
        };
        let main_fn = program.functions.get("main").expect("main function");
        let main_block = main_fn
            .block_specializations
            .get("CoreML7")
            .expect("CoreML7 block");

        assert!(main_block.operations.iter().any(|op| op.r#type == "cumsum"));
    }

    #[test]
    fn test_gelu_emits_explicit_mode_input() {
        // The watchOS MIL loader rejects gelu without an explicit `mode` input
        // ("Required param 'mode' is missing"); iOS/macOS loaders accept the
        // default. WebNN gelu has no mode parameter — spec default is exact
        // (erf), so emitting mode=EXACT is correct on every platform.
        let graph = GraphInfo {
            input_operands: vec![0],
            output_operands: vec![1],
            operands: vec![
                Operand {
                    name: Some("input".to_string()),
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[1, 4]),
                        pending_permutation: vec![],
                    },
                },
                Operand {
                    name: Some("output".to_string()),
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: s(&[1, 4]),
                        pending_permutation: vec![],
                    },
                },
            ],
            operations: vec![op_from_operator_options(
                "gelu",
                vec![0],
                Some(1),
                vec![],
                OperatorOptions::from_json_with_op_type("gelu", &serde_json::json!({}))
                    .expect("gelu options"),
            )],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        };

        let converted = CoremlMlProgramConverter
            .convert(&graph)
            .expect("coreml gelu conversion should succeed");
        let model = Model::decode(converted.data.as_slice()).expect("decode coreml model");
        let program = match model.r#type.expect("model type") {
            crate::protos::coreml::specification::model::Type::MlProgram(program) => program,
            _ => panic!("expected MLProgram model"),
        };
        let main_fn = program.functions.get("main").expect("main function");
        let main_block = main_fn
            .block_specializations
            .get("CoreML7")
            .expect("CoreML7 block");

        let gelu = main_block
            .operations
            .iter()
            .find(|op| op.r#type == "gelu")
            .expect("gelu op");

        let mode_arg = gelu
            .inputs
            .get("mode")
            .expect("gelu must emit a `mode` input for the watchOS MIL loader");
        let binding = mode_arg
            .arguments
            .first()
            .expect("mode arg should have a binding");
        let value = match binding.binding.as_ref().expect("mode arg binding present") {
            crate::protos::coreml::mil_spec::argument::binding::Binding::Value(v) => v.clone(),
            _ => panic!("mode input should be an immediate Value, not a variable reference"),
        };
        let immediate = match value.value.as_ref().expect("mode value present") {
            crate::protos::coreml::mil_spec::value::Value::ImmediateValue(iv) => iv,
            _ => panic!("mode input should be an ImmediateValue"),
        };
        let tensor = match immediate.value.as_ref().expect("immediate value present") {
            crate::protos::coreml::mil_spec::value::immediate_value::Value::Tensor(t) => t,
            _ => panic!("mode input should wrap a TensorValue"),
        };
        let strings = match tensor.value.as_ref().expect("tensor value present") {
            crate::protos::coreml::mil_spec::tensor_value::Value::Strings(s) => s,
            _ => panic!("mode input tensor should be Strings-typed"),
        };
        assert_eq!(
            strings.values.as_slice(),
            ["EXACT"],
            "mode must be the scalar string \"EXACT\" to match the WebNN spec default",
        );
    }
}
