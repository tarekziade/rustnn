//! Nested limit dictionaries from the WebNN specification (MLOpSupportLimits partials).

use serde::{Deserialize, Serialize};

use super::MLTensorLimits;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLBatchNormalizationSupportLimits {
    pub input: MLTensorLimits,
    pub mean: MLTensorLimits,
    pub variance: MLTensorLimits,
    pub scale: MLTensorLimits,
    pub bias: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLBinarySupportLimits {
    pub a: MLTensorLimits,
    pub b: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLConcatSupportLimits {
    pub inputs: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLConv2dSupportLimits {
    pub input: MLTensorLimits,
    pub filter: MLTensorLimits,
    pub bias: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLGatherSupportLimits {
    pub input: MLTensorLimits,
    pub indices: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLGemmSupportLimits {
    pub a: MLTensorLimits,
    pub b: MLTensorLimits,
    pub c: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLGruSupportLimits {
    pub input: MLTensorLimits,
    pub weight: MLTensorLimits,
    pub recurrent_weight: MLTensorLimits,
    pub bias: MLTensorLimits,
    pub recurrent_bias: MLTensorLimits,
    pub initial_hidden_state: MLTensorLimits,
    pub output0: MLTensorLimits,
    pub output1: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLGruCellSupportLimits {
    pub input: MLTensorLimits,
    pub weight: MLTensorLimits,
    pub recurrent_weight: MLTensorLimits,
    pub hidden_state: MLTensorLimits,
    pub bias: MLTensorLimits,
    pub recurrent_bias: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLLogicalNotSupportLimits {
    pub a: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLLstmSupportLimits {
    pub input: MLTensorLimits,
    pub weight: MLTensorLimits,
    pub recurrent_weight: MLTensorLimits,
    pub bias: MLTensorLimits,
    pub recurrent_bias: MLTensorLimits,
    pub peephole_weight: MLTensorLimits,
    pub initial_hidden_state: MLTensorLimits,
    pub initial_cell_state: MLTensorLimits,
    pub output0: MLTensorLimits,
    pub output1: MLTensorLimits,
    pub output2: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLLstmCellSupportLimits {
    pub input: MLTensorLimits,
    pub weight: MLTensorLimits,
    pub recurrent_weight: MLTensorLimits,
    pub hidden_state: MLTensorLimits,
    pub cell_state: MLTensorLimits,
    pub bias: MLTensorLimits,
    pub recurrent_bias: MLTensorLimits,
    pub peephole_weight: MLTensorLimits,
    pub output0: MLTensorLimits,
    pub output1: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLNormalizationSupportLimits {
    pub input: MLTensorLimits,
    pub scale: MLTensorLimits,
    pub bias: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLPreluSupportLimits {
    pub input: MLTensorLimits,
    pub slope: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLQuantizeDequantizeLinearSupportLimits {
    pub input: MLTensorLimits,
    pub scale: MLTensorLimits,
    pub zero_point: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLScatterSupportLimits {
    pub input: MLTensorLimits,
    pub indices: MLTensorLimits,
    pub updates: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLSingleInputSupportLimits {
    pub input: MLTensorLimits,
    pub output: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLSplitSupportLimits {
    pub input: MLTensorLimits,
    pub outputs: MLTensorLimits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MLWhereSupportLimits {
    pub condition: MLTensorLimits,
    pub true_value: MLTensorLimits,
    pub false_value: MLTensorLimits,
    pub output: MLTensorLimits,
}
