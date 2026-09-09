//! WebNN [`MLOpSupportLimits`](https://www.w3.org/TR/webnn/#dictdef-mlopsupportlimits) and related
//! limit dictionaries, for JSON interchange and alignment with `web-sys` / wasm-bindgen bindings.

mod types;

pub use types::{
    MLBatchNormalizationSupportLimits, MLBinarySupportLimits, MLConcatSupportLimits,
    MLConv2dSupportLimits, MLGatherSupportLimits, MLGemmSupportLimits, MLGruCellSupportLimits,
    MLGruSupportLimits, MLLogicalNotSupportLimits, MLLstmCellSupportLimits, MLLstmSupportLimits,
    MLNormalizationSupportLimits, MLPreluSupportLimits, MLQuantizeDequantizeLinearSupportLimits,
    MLScatterSupportLimits, MLSingleInputSupportLimits, MLSplitSupportLimits, MLWhereSupportLimits,
};

use serde::{Deserialize, Serialize};

use crate::graph::DataType;

/// `MLInputOperandLayout` — preferred layout for layout-dependent operators (e.g. conv2d).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MLInputOperandLayout {
    Nchw,
    #[default]
    Nhwc,
}

/// `MLRankRange` — inclusive min/max tensor rank supported.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLRankRange {
    pub min: u32,
    pub max: u32,
}

/// `MLTensorLimits` — allowed operand data types and rank range.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLTensorLimits {
    pub data_types: Vec<DataType>, // use &[DataType] ?
    pub rank_range: MLRankRange,
}

/// `MLOpSupportLimits` — merged dictionary from the specification (base + all partials).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct MLOpSupportLimits {
    pub preferred_input_layout: MLInputOperandLayout,
    pub max_tensor_byte_length: u64,
    pub input: MLTensorLimits,
    pub constant: MLTensorLimits,
    pub output: MLTensorLimits,

    #[serde(default)]
    pub arg_min: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub arg_max: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub batch_normalization: Option<MLBatchNormalizationSupportLimits>,
    #[serde(default)]
    pub cast: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub clamp: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub concat: Option<MLConcatSupportLimits>,
    #[serde(default)]
    pub conv2d: Option<MLConv2dSupportLimits>,
    #[serde(default)]
    pub conv_transpose2d: Option<MLConv2dSupportLimits>,
    #[serde(default)]
    pub cumulative_sum: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub add: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub sub: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub mul: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub div: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub max: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub min: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub pow: Option<MLBinarySupportLimits>,

    #[serde(default)]
    pub equal: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub not_equal: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub greater: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub greater_or_equal: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub lesser: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub lesser_or_equal: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub logical_not: Option<MLLogicalNotSupportLimits>,
    #[serde(default)]
    pub logical_and: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub logical_or: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub logical_xor: Option<MLBinarySupportLimits>,
    #[serde(default)]
    pub is_na_n: Option<MLLogicalNotSupportLimits>,
    #[serde(default)]
    pub is_infinite: Option<MLLogicalNotSupportLimits>,

    #[serde(default)]
    pub abs: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub ceil: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub cos: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub erf: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub exp: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub floor: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub identity: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub log: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub neg: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reciprocal: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub round_even: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub sin: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub sign: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub sqrt: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub tan: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub dequantize_linear: Option<MLQuantizeDequantizeLinearSupportLimits>,
    #[serde(default)]
    pub quantize_linear: Option<MLQuantizeDequantizeLinearSupportLimits>,

    #[serde(default)]
    pub elu: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub expand: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub gather: Option<MLGatherSupportLimits>,
    #[serde(default)]
    pub gather_elements: Option<MLGatherSupportLimits>,
    #[serde(default)]
    pub gather_nd: Option<MLGatherSupportLimits>,

    #[serde(default)]
    pub gelu: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub gemm: Option<MLGemmSupportLimits>,

    #[serde(default)]
    pub gru: Option<MLGruSupportLimits>,
    #[serde(default)]
    pub gru_cell: Option<MLGruCellSupportLimits>,

    #[serde(default)]
    pub hard_sigmoid: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub hard_swish: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub instance_normalization: Option<MLNormalizationSupportLimits>,
    #[serde(default)]
    pub layer_normalization: Option<MLNormalizationSupportLimits>,

    #[serde(default)]
    pub leaky_relu: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub linear: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub lstm: Option<MLLstmSupportLimits>,
    #[serde(default)]
    pub lstm_cell: Option<MLLstmCellSupportLimits>,

    #[serde(default)]
    pub matmul: Option<MLBinarySupportLimits>,

    #[serde(default)]
    pub pad: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub average_pool2d: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub l2_pool2d: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub max_pool2d: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub prelu: Option<MLPreluSupportLimits>,

    #[serde(default)]
    pub reduce_l1: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reduce_l2: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reduce_log_sum: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reduce_log_sum_exp: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reduce_max: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reduce_mean: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reduce_min: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reduce_product: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reduce_sum: Option<MLSingleInputSupportLimits>,
    #[serde(default)]
    pub reduce_sum_square: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub relu: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub resample2d: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub reshape: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub reverse: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub scatter_elements: Option<MLScatterSupportLimits>,
    #[serde(default)]
    pub scatter_nd: Option<MLScatterSupportLimits>,

    #[serde(default)]
    pub sigmoid: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub slice: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub softmax: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub softplus: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub softsign: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub split: Option<MLSplitSupportLimits>,

    #[serde(default)]
    pub tanh: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub tile: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub transpose: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub triangular: Option<MLSingleInputSupportLimits>,

    #[serde(default)]
    pub r#where: Option<MLWhereSupportLimits>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tensor_limits() -> MLTensorLimits {
        MLTensorLimits {
            data_types: vec![DataType::Float32],
            rank_range: MLRankRange { min: 1, max: 8 },
        }
    }

    #[test]
    fn serde_ml_op_support_limits_roundtrip() {
        let limits = MLOpSupportLimits {
            preferred_input_layout: MLInputOperandLayout::Nchw,
            max_tensor_byte_length: 268_435_456,
            input: sample_tensor_limits(),
            constant: sample_tensor_limits(),
            output: sample_tensor_limits(),
            conv2d: Some(MLConv2dSupportLimits {
                input: sample_tensor_limits(),
                filter: sample_tensor_limits(),
                bias: sample_tensor_limits(),
                output: sample_tensor_limits(),
            }),
            ..Default::default()
        };

        let json = serde_json::to_string(&limits).unwrap();
        let back: MLOpSupportLimits = serde_json::from_str(&json).unwrap();
        assert_eq!(back.preferred_input_layout, MLInputOperandLayout::Nchw);
        assert_eq!(back.max_tensor_byte_length, 268_435_456);
        assert!(back.conv2d.is_some());
    }
}
