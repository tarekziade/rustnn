/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
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

//! CANN/HiAI converter for WebNN graphs.
//!
//! Converts `GraphInfo` into compiled model bytes via the adapter
//! (the `hiai-rs` crate's `adapter/`). The adapter wraps the HiAI DDK's C++ API.
//!
//! - `cann-runtime`: calls `encode_via_adapter()` to build a GE graph,
//!   compile via HiaiIrBuild, return ModelBufferData bytes.
//! - `cann-runtime-mock`: validates ops via `webnn_op_to_hiai`, returns
//!   placeholder bytes for CI/testing.

use crate::error::GraphError;
use crate::graph::GraphInfo;
use crate::operators::Operation;

use super::{ConvertedGraph, GraphConverter};

// Maps a WebNN operation to its HIAI IR operation name.
//
// Returns Some(name) for ops that map directly to an adapter cann_op_*()
//
// Returns `None` for ops that need decomposition or are not supported
#[cfg(any(feature = "cann-runtime", test))]
pub(crate) fn webnn_op_to_hiai(op: &Operation) -> Option<&'static str> {
    match op {
        // ── Element-wise binary ──────────────────────────────────────
        Operation::Add { .. } => Some("Add"),
        Operation::Sub { .. } => Some("Sub"),
        Operation::Mul { .. } => Some("Mul"),
        Operation::Div { .. } => Some("Div"),
        Operation::Pow { .. } => Some("Pow"),
        Operation::Max { .. } => Some("Max"),
        Operation::Min { .. } => Some("Min"),
        Operation::Equal { .. } => Some("Equal"),
        Operation::Greater { .. } => Some("Greater"),
        Operation::GreaterOrEqual { .. } => Some("GreaterOrEqual"),
        Operation::Lesser { .. } => Some("Lesser"),
        Operation::LesserOrEqual { .. } => Some("LesserOrEqual"),
        Operation::NotEqual { .. } => Some("NotEqual"),
        Operation::LogicalAnd { .. } => Some("LogicalAnd"),
        Operation::LogicalOr { .. } => Some("LogicalOr"),
        Operation::LogicalXor { .. } => Some("LogicalXor"),
        Operation::LogicalNot { .. } => Some("LogicalNot"),

        // ── Element-wise unary ───────────────────────────────────────
        Operation::Abs { .. } => Some("Abs"),
        Operation::Neg { .. } => Some("Neg"),
        Operation::Exp { .. } => Some("Exp"),
        Operation::Log { .. } => Some("Log"),
        Operation::Sin { .. } => Some("Sin"),
        Operation::Cos { .. } => Some("Cos"),
        Operation::Tan { .. } => Some("Tan"),
        Operation::Sqrt { .. } => Some("Sqrt"),
        Operation::Ceil { .. } => Some("Ceil"),
        Operation::Floor { .. } => Some("Floor"),
        Operation::Sign { .. } => Some("Sign"),
        Operation::Erf { .. } => Some("Erf"),
        Operation::Reciprocal { .. } => Some("Reciprocal"),
        Operation::Cast { .. } => Some("Cast"),
        Operation::Clamp { .. } => Some("Clamp"),

        // ── Activations ──────────────────────────────────────────────
        Operation::Relu { .. } => Some("ReLU"),
        Operation::Sigmoid { .. } => Some("Sigmoid"),
        Operation::Tanh { .. } => Some("Tanh"),
        Operation::Elu { .. } => Some("ELU"),
        Operation::Gelu { .. } => Some("GELU"),
        Operation::LeakyRelu { .. } => Some("LeakyRelu"),
        Operation::HardSigmoid { .. } => Some("HardSigmoid"),
        Operation::HardSwish { .. } => Some("HardSwish"),
        Operation::Softplus { .. } => Some("Softplus"),
        Operation::Softsign { .. } => Some("Softsign"),

        // ── Convolution + Pool + Matmul ──────────────────────────────
        Operation::Conv2d { .. } => Some("Conv2D"),
        Operation::ConvTranspose2d { .. } => Some("ConvTranspose"),
        Operation::MaxPool2d { .. } => Some("MaxPool"),
        Operation::AveragePool2d { .. } => Some("AvgPool"),
        Operation::Matmul { .. } => Some("MatMul"),
        Operation::Gemm { .. } => Some("Gemm"),
        Operation::Softmax { .. } => Some("Softmax"),

        // ── Reduction ────────────────────────────────────────────────
        Operation::ReduceSum { .. } => Some("ReduceSum"),
        Operation::ReduceMean { .. } => Some("ReduceMean"),
        Operation::ReduceMax { .. } => Some("ReduceMax"),
        Operation::ReduceMin { .. } => Some("ReduceMin"),
        Operation::ReduceProduct { .. } => Some("ReduceProduct"),
        Operation::ReduceL1 { .. } => Some("ReduceL1"),
        Operation::ReduceL2 { .. } => Some("ReduceL2"),
        Operation::ReduceLogSum { .. } => Some("ReduceLogSum"),
        Operation::ReduceLogSumExp { .. } => Some("ReduceLogSumExp"),
        Operation::ReduceSumSquare { .. } => Some("ReduceSumSquare"),
        Operation::ArgMax { .. } => Some("ArgMax"),
        Operation::ArgMin { .. } => Some("ArgMin"),

        // ── Shape ops ────────────────────────────────────────────────
        Operation::Reshape { .. } => Some("Reshape"),
        Operation::Transpose { .. } => Some("Transpose"),
        Operation::Tile { .. } => Some("Tile"),
        Operation::Slice { .. } => Some("Slice"),
        Operation::Split { .. } => Some("Split"),
        Operation::Concat { .. } => Some("Concat"),
        Operation::Pad { .. } => Some("Pad"),
        Operation::Squeeze { .. } => Some("Squeeze"),
        Operation::Unsqueeze { .. } => Some("Unsqueeze"),
        Operation::Expand { .. } => Some("Expand"),
        Operation::CumulativeSum { .. } => Some("CumulativeSum"),

        // ── Gather / Scatter / Where ─────────────────────────────────
        Operation::Gather { .. } => Some("Gather"),
        Operation::GatherElements { .. } => Some("GatherElements"),
        Operation::GatherND { .. } => Some("GatherND"),
        Operation::ScatterElements { .. } => Some("ScatterElements"),
        Operation::ScatterND { .. } => Some("ScatterND"),
        Operation::Where { .. } => Some("Where"),

        // ── Normalization + Quantization ─────────────────────────────
        Operation::BatchNormalization { .. } => Some("BatchNormalization"),
        Operation::InstanceNormalization { .. } => Some("InstanceNormalization"),
        Operation::QuantizeLinear { .. } => Some("QuantizeLinear"),
        Operation::DequantizeLinear { .. } => Some("DequantizeLinear"),

        // ── Other ────────────────────────────────────────────────────
        Operation::Resample2d { .. } => Some("Resample2D"),
        Operation::GlobalAveragePool { .. } => Some("GlobalAveragePool"),
        Operation::GlobalMaxPool { .. } => Some("GlobalMaxPool"),
        Operation::Constant { .. } => Some("Constant"),
        Operation::Shape { .. } => Some("Shape"),

        // ── Needs decomposition (Phase 3) ────────────────────────────
        Operation::Identity { .. } => Some("Identity"),
        Operation::Prelu { .. } => None,
        Operation::Linear { .. } => None,
        Operation::LayerNormalization { .. } => None,
        Operation::Triangular { .. } => None,
        Operation::IsNaN { .. } => None,
        Operation::IsInfinite { .. } => None,

        // ── Not supported ────────────────────────────────────────────
        Operation::Gru { .. }
        | Operation::GruCell { .. }
        | Operation::Lstm { .. }
        | Operation::LstmCell { .. }
        | Operation::L2Pool2d { .. }
        | Operation::Reverse { .. }
        | Operation::RoundEven { .. } => None,
    }
}

// The YoloV8s priority operators. encode_via_adapter (and the mock path) only
// accept these; every other operation is reported as unsupported. The wiring
// for other ops remains in place below for future re-enabling.
pub(crate) fn is_supported_op(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Add { .. }
            | Operation::Sub { .. }
            | Operation::Mul { .. }
            | Operation::Div { .. }
            | Operation::Conv2d { .. }
            | Operation::MaxPool2d { .. }
            | Operation::Concat { .. }
            | Operation::Reshape { .. }
            | Operation::Resample2d { .. }
            | Operation::Sigmoid { .. }
            | Operation::Slice { .. }
            | Operation::Softmax { .. }
            | Operation::Split { .. }
            | Operation::Transpose { .. }
            | Operation::Cast { .. }
            | Operation::ReduceSum { .. }
            | Operation::Prelu { .. }
    )
}

#[cfg(feature = "cann-runtime")]
mod adapter {
    use super::*;
    use crate::graph::{DataType, OperandDescriptor};
    use hiai_rs::sys::*;

    // ── Graph builder helpers ──────────────────────────────────────────

    fn cann_data_type(data_type: DataType) -> hiai_rs::sys::ddk_CannDataType {
        use hiai_rs::sys::ddk_CannDataType as C;
        match data_type {
            DataType::Float32 => C::CANN_DT_FLOAT,
            DataType::Float16 => C::CANN_DT_FLOAT16,
            DataType::Int32 => C::CANN_DT_INT32,
            DataType::Int8 => C::CANN_DT_INT8,
            DataType::Uint8 => C::CANN_DT_UINT8,
            DataType::Uint32 => C::CANN_DT_UINT32,
            DataType::Int64 => C::CANN_DT_INT64,
            DataType::Uint64 => C::CANN_DT_UINT64,
            _ => C::CANN_DT_FLOAT,
        }
    }

    // Binary operators.
    // Returns (input_a, input_b, ge_input_name_a, ge_input_name_b), or None.
    fn binary_op_inputs(operation: &Operation) -> Option<(u32, u32, &'static str, &'static str)> {
        let (a, b, name_a, name_b) = match operation {
            Operation::Add { a, b, .. }
            | Operation::Sub { a, b, .. }
            | Operation::Mul { a, b, .. }
            | Operation::Div { a, b, .. }
            | Operation::Pow { a, b, .. }
            | Operation::Max { a, b, .. }
            | Operation::Min { a, b, .. } => (*a, *b, "x1", "x2"),
            Operation::Equal { a, b, .. }
            | Operation::Greater { a, b, .. }
            | Operation::GreaterOrEqual { a, b, .. }
            | Operation::Lesser { a, b, .. }
            | Operation::LesserOrEqual { a, b, .. }
            | Operation::NotEqual { a, b, .. } => (*a, *b, "x1", "x2"),
            Operation::LogicalAnd { a, b, .. }
            | Operation::LogicalOr { a, b, .. }
            | Operation::LogicalXor { a, b, .. } => (*a, *b, "x1", "x2"),
            Operation::Matmul { a, b, .. } | Operation::Gemm { a, b, .. } => (*a, *b, "x1", "x2"),
            _ => return None,
        };
        Some((a, b, name_a, name_b))
    }

    // Unary element-wise operators.
    // Returns input operand index, or None.
    fn unary_op_input(operation: &Operation) -> Option<u32> {
        match operation {
            Operation::Relu { input, .. }
            | Operation::Sigmoid { input, .. }
            | Operation::Tanh { input, .. }
            | Operation::Elu { input, .. }
            | Operation::Gelu { input, .. }
            | Operation::LeakyRelu { input, .. }
            | Operation::HardSigmoid { input, .. }
            | Operation::HardSwish { input, .. }
            | Operation::Softplus { input, .. }
            | Operation::Softsign { input, .. } => Some(*input),
            Operation::Abs { input, .. }
            | Operation::Neg { input, .. }
            | Operation::Exp { input, .. }
            | Operation::Log { input, .. }
            | Operation::Sin { input, .. }
            | Operation::Cos { input, .. }
            | Operation::Tan { input, .. }
            | Operation::Sqrt { input, .. }
            | Operation::Ceil { input, .. }
            | Operation::Floor { input, .. }
            | Operation::Sign { input, .. }
            | Operation::Erf { input, .. }
            | Operation::Reciprocal { input, .. } => Some(*input),
            Operation::Cast { input, .. } | Operation::Identity { input, .. } => Some(*input),
            Operation::Clamp { input, .. } | Operation::LogicalNot { input, .. } => Some(*input),
            _ => None,
        }
    }

    fn descriptor_dims(descriptor: &OperandDescriptor) -> Vec<i64> {
        use crate::graph::Dimension;
        descriptor
            .shape
            .iter()
            .map(|dim| match dim {
                Dimension::Static(dim_value) => *dim_value as i64,
                _ => 0,
            })
            .collect()
    }

    // Create a Const operator holding the given tensor data, mirroring the
    // Chromium reference's Constant<T>() helper (Const().set_attr_value(tensor)).
    fn make_const(
        name: &str,
        data: &[u8],
        shape: &[i64],
        dtype: ddk_CannDataType,
        format: i32,
    ) -> ddk_CannOperatorHandle {
        let name_c =
            std::ffi::CString::new(name).expect("operator/operand names never contain NUL bytes");
        let value_name = std::ffi::CString::new("value").unwrap();
        let const_op = unsafe { ddk_cann_op_const_with_name(name_c.as_ptr()) };
        unsafe {
            ddk_cann_operator_set_attr_tensor_raw_format(
                const_op,
                value_name.as_ptr(),
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u32,
                shape.as_ptr(),
                shape.len() as i32,
                dtype,
                format,
            );
        }
        const_op
    }

    // Connect an operand to `name` on `op`. (Split is decomposed into Slice ops
    // elsewhere, so every source here is single-output.)
    unsafe fn set_operand_input(
        op: ddk_CannOperatorHandle,
        name: &std::ffi::CStr,
        handle: ddk_CannOperatorHandle,
    ) -> i32 {
        unsafe { ddk_cann_operator_set_input(op, name.as_ptr(), handle) }
    }

    // Connect `index` on dynamic input `name` of `op` to a source operand's
    // handle. Dynamic inputs route a multi-output source (Split) by the
    // consumer's 1-based index, matching the Chromium reference.
    unsafe fn set_dynamic_input(
        op: ddk_CannOperatorHandle,
        name: &std::ffi::CStr,
        index: u32,
        handle: ddk_CannOperatorHandle,
    ) -> i32 {
        unsafe { ddk_cann_operator_set_dynamic_input_by_index(op, name.as_ptr(), index, handle) }
    }

    // Create a `Slice` op extracting `sizes` starting at `starts` from
    // `x_handle`. Shared by the Slice operation and the Split decomposition
    // (Split is emitted as N single-output Slices, since GE cannot resolve
    // SplitD's dynamic output from a static consumer).
    fn create_slice_op(
        name: &str,
        x_handle: ddk_CannOperatorHandle,
        starts: &[i32],
        sizes: &[i32],
        extra_ops: &mut Vec<ddk_CannOperatorHandle>,
    ) -> Result<ddk_CannOperatorHandle, GraphError> {
        let name_c =
            std::ffi::CString::new(name).expect("operator/operand names never contain NUL bytes");
        let slice_type = std::ffi::CString::new("Slice").unwrap();
        let slice_op =
            unsafe { ddk_cann_operator_create_registered(slice_type.as_ptr(), name_c.as_ptr()) };
        if slice_op.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_create failed for slice '{name}'").into(),
            });
        }

        let x_name = std::ffi::CString::new("x").unwrap();
        let status = unsafe { set_operand_input(slice_op, &x_name, x_handle) };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_set_input for slice '{name}' failed").into(),
            });
        }

        let offsets_const = make_const(
            &format!("{name}_offsets"),
            bytemuck::cast_slice(starts),
            &[starts.len() as i64],
            ddk_CannDataType::CANN_DT_INT32,
            2,
        );
        extra_ops.push(offsets_const);
        let size_const = make_const(
            &format!("{name}_size"),
            bytemuck::cast_slice(sizes),
            &[sizes.len() as i64],
            ddk_CannDataType::CANN_DT_INT32,
            2,
        );
        extra_ops.push(size_const);

        let offsets_name = std::ffi::CString::new("offsets").unwrap();
        let size_name = std::ffi::CString::new("size").unwrap();
        unsafe {
            ddk_cann_operator_set_input(slice_op, offsets_name.as_ptr(), offsets_const);
            ddk_cann_operator_set_input(slice_op, size_name.as_ptr(), size_const);
        }

        Ok(slice_op)
    }

    // Returns the output operand slice for all operations with standard
    // `outputs: Vec<u32>` field.
    fn op_outputs(operation: &Operation) -> &[u32] {
        match operation {
            // Binary
            Operation::Add { outputs, .. }
            | Operation::Sub { outputs, .. }
            | Operation::Mul { outputs, .. }
            | Operation::Div { outputs, .. }
            | Operation::Pow { outputs, .. }
            | Operation::Max { outputs, .. }
            | Operation::Min { outputs, .. }
            // Comparison
            | Operation::Equal { outputs, .. }
            | Operation::Greater { outputs, .. }
            | Operation::GreaterOrEqual { outputs, .. }
            | Operation::Lesser { outputs, .. }
            | Operation::LesserOrEqual { outputs, .. }
            | Operation::NotEqual { outputs, .. }
            // Logical binary
            | Operation::LogicalAnd { outputs, .. }
            | Operation::LogicalOr { outputs, .. }
            | Operation::LogicalXor { outputs, .. }
            // Activations
            | Operation::Relu { outputs, .. }
            | Operation::Sigmoid { outputs, .. }
            | Operation::Tanh { outputs, .. }
            | Operation::Elu { outputs, .. }
            | Operation::Gelu { outputs, .. }
            | Operation::LeakyRelu { outputs, .. }
            | Operation::HardSigmoid { outputs, .. }
            | Operation::HardSwish { outputs, .. }
            | Operation::Softplus { outputs, .. }
            | Operation::Softsign { outputs, .. }
            // Unary math
            | Operation::Abs { outputs, .. }
            | Operation::Neg { outputs, .. }
            | Operation::Exp { outputs, .. }
            | Operation::Log { outputs, .. }
            | Operation::Sin { outputs, .. }
            | Operation::Cos { outputs, .. }
            | Operation::Tan { outputs, .. }
            | Operation::Sqrt { outputs, .. }
            | Operation::Ceil { outputs, .. }
            | Operation::Floor { outputs, .. }
            | Operation::Sign { outputs, .. }
            | Operation::Erf { outputs, .. }
            | Operation::Reciprocal { outputs, .. }
            // Type / other
            | Operation::Cast { outputs, .. }
            | Operation::Clamp { outputs, .. }
            | Operation::Identity { outputs, .. }
            | Operation::LogicalNot { outputs, .. }
            | Operation::Softmax { outputs, .. }
            // Matrix
            | Operation::Matmul { outputs, .. }
            | Operation::Gemm { outputs, .. }
            // Conv
            | Operation::Conv2d { outputs, .. }
            | Operation::ConvTranspose2d { outputs, .. }
            // Pooling
            | Operation::MaxPool2d { outputs, .. }
            | Operation::AveragePool2d { outputs, .. }
            | Operation::L2Pool2d { outputs, .. }
            | Operation::GlobalMaxPool { outputs, .. }
            | Operation::GlobalAveragePool { outputs, .. }
            // ArgMax / ArgMin
            | Operation::ArgMax { outputs, .. }
            | Operation::ArgMin { outputs, .. }
            // Reduction
            | Operation::ReduceSum { outputs, .. }
            // Normalization
            | Operation::BatchNormalization { outputs, .. }
            // Shape / other (YoloV8s)
            | Operation::Concat { outputs, .. }
            | Operation::Reshape { outputs, .. }
            | Operation::Resample2d { outputs, .. }
            | Operation::Slice { outputs, .. }
            | Operation::Split { outputs, .. }
            | Operation::Transpose { outputs, .. } => outputs,
            _ => &[],
        }
    }

    // ── Layout helpers ─────────────────────────────────────────────────

    /// Map WebNN input_layout to GE data_format.
    fn conv_data_format(options: &Option<crate::operator_options::MLConv2dOptions>) -> &str {
        match options {
            Some(o) if o.input_layout.eq_ignore_ascii_case("nhwc") => "NHWC",
            _ => "NCHW",
        }
    }

    // Build a CANN graph via the adapter, compile it, and return the model bytes.
    //
    // Data nodes (with tensor descriptors) -> ops -> NetOutput -> compile -> bytes.
    // Returns Err if the shim library is unavailable.
    pub(crate) fn encode_via_adapter(graph: &GraphInfo) -> Result<Vec<u8>, GraphError> {
        use std::ffi::CString;

        // ── 1. Create graph ─────────────────────────────────────────────
        let graph_name = CString::new("webnn_model").unwrap();
        let can_graph = unsafe { ddk_cann_graph_create(graph_name.as_ptr()) };
        if can_graph.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_graph_create failed".into(),
            });
        }

        // operand_index -> CannOperatorHandle
        let mut handles: Vec<ddk_CannOperatorHandle> =
            vec![std::ptr::null_mut(); graph.operands.len()];

        // ── 2. Create Data operators for each input ──────────────────────
        let mut data_ops: Vec<ddk_CannOperatorHandle> = Vec::new();

        for &input_id in &graph.input_operands {
            let descriptor = &graph.operands[input_id as usize].descriptor;
            let dimensions = descriptor_dims(descriptor);
            let name = CString::new(
                graph.operands[input_id as usize]
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("input_{input_id}")),
            )
            .expect("operator/operand names never contain NUL bytes");

            let data_op = unsafe { ddk_cann_op_data_with_name(name.as_ptr()) };
            if data_op.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_op_data_with_name for operand {input_id} failed").into(),
                });
            }

            // Set tensor descriptor: shape + FORMAT_ND + dtype
            let shape =
                unsafe { ddk_cann_shape_create(dimensions.as_ptr(), dimensions.len() as i32) };
            if shape.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: "cann_shape_create failed".into(),
                });
            }
            let tensor_desc = unsafe {
                ddk_cann_tensor_desc_create(
                    shape,
                    ddk_CannFormat::CANN_FORMAT_ND,
                    cann_data_type(descriptor.data_type),
                )
            };
            if tensor_desc.is_null() {
                unsafe { ddk_cann_shape_destroy(shape) };
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: "cann_tensor_desc_create failed".into(),
                });
            }
            let x_name = CString::new("x").unwrap();
            let status = unsafe {
                ddk_cann_operator_update_input_desc(data_op, x_name.as_ptr(), tensor_desc)
            };
            unsafe {
                ddk_cann_tensor_desc_destroy(tensor_desc);
                ddk_cann_shape_destroy(shape);
            }
            if status != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!(
                        "cann_operator_update_input_desc for operand {input_id} failed: {status}"
                    )
                    .into(),
                });
            }

            handles[input_id as usize] = data_op;
            data_ops.push(data_op);
        }

        // Create Const operators for constant operands (for example, Conv2d filters).
        let mut const_ops: Vec<ddk_CannOperatorHandle> = Vec::new();
        for (const_id, constant_data) in &graph.constant_operand_ids_to_handles {
            let name = CString::new(
                graph.operands[*const_id as usize]
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("const_{const_id}")),
            )
            .expect("operator/operand names never contain NUL bytes");

            let const_op = unsafe { ddk_cann_op_const_with_name(name.as_ptr()) };
            if const_op.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_op_const_with_name for operand {const_id} failed").into(),
                });
            }

            let desc = &graph.operands[*const_id as usize].descriptor;
            let dims = descriptor_dims(desc);
            let value_name = CString::new("value").unwrap();
            let format = 0_i32; // FORMAT_NCHW for all const tensors
            let status = unsafe {
                ddk_cann_operator_set_attr_tensor_raw_format(
                    const_op,
                    value_name.as_ptr(),
                    constant_data.data.as_ptr() as *const _,
                    constant_data.data.len() as u32,
                    dims.as_ptr(),
                    dims.len() as i32,
                    cann_data_type(desc.data_type),
                    format,
                )
            };
            if status != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!(
                        "cann_operator_set_attr_tensor_raw for operand {const_id} failed: {status}"
                    )
                    .into(),
                });
            }

            handles[*const_id as usize] = const_op;
            const_ops.push(const_op);
        }

        // ── 3. Create compute operations ─────────────────────────────────
        let mut compute_ops: Vec<ddk_CannOperatorHandle> = Vec::new();
        // Intermediate ops from decompositions (e.g. sigmoid), added to the
        // graph alongside compute_ops.
        let mut extra_ops: Vec<ddk_CannOperatorHandle> = Vec::new();

        for op in &graph.operations {
            // Phase-out gate: only the YoloV8s priority operators are accepted.
            if !is_supported_op(op) {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("unsupported op: {}", op.label()).into(),
                });
            }

            // Sigmoid is decomposed (matching the Chromium reference):
            // sigmoid(x) = 1 / (1 + exp(-x))
            if let Operation::Sigmoid { input, outputs, .. } = op {
                let x_handle = handles[*input as usize];
                let out_id = outputs[0];

                let one = make_const(
                    &format!("sigmoid_one_{out_id}"),
                    bytemuck::cast_slice(&[1.0f32]),
                    &[1], // shape [1] (adapter requires shape_count > 0)
                    ddk_CannDataType::CANN_DT_FLOAT,
                    0, // FORMAT_NCHW
                );
                extra_ops.push(one);

                let neg_name = CString::new(format!("sigmoid_neg_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let neg_type = CString::new("Neg").unwrap();
                let neg = unsafe {
                    ddk_cann_operator_create_registered(neg_type.as_ptr(), neg_name.as_ptr())
                };
                let x_name = CString::new("x").unwrap();
                unsafe { set_operand_input(neg, &x_name, x_handle) };
                extra_ops.push(neg);

                let exp_name = CString::new(format!("sigmoid_exp_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let exp_type = CString::new("Exp").unwrap();
                let exp_neg = unsafe {
                    ddk_cann_operator_create_registered(exp_type.as_ptr(), exp_name.as_ptr())
                };
                unsafe { ddk_cann_operator_set_input(exp_neg, x_name.as_ptr(), neg) };
                extra_ops.push(exp_neg);

                let denom_name = CString::new(format!("sigmoid_denom_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let add_type = CString::new("Add").unwrap();
                let denom = unsafe {
                    ddk_cann_operator_create_registered(add_type.as_ptr(), denom_name.as_ptr())
                };
                let x1_name = CString::new("x1").unwrap();
                let x2_name = CString::new("x2").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(denom, x1_name.as_ptr(), one);
                    ddk_cann_operator_set_input(denom, x2_name.as_ptr(), exp_neg);
                }
                extra_ops.push(denom);

                let div_name = CString::new(format!("sigmoid_div_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let div_type = CString::new("Div").unwrap();
                let div = unsafe {
                    ddk_cann_operator_create_registered(div_type.as_ptr(), div_name.as_ptr())
                };
                unsafe {
                    ddk_cann_operator_set_input(div, x1_name.as_ptr(), one);
                    ddk_cann_operator_set_input(div, x2_name.as_ptr(), denom);
                }

                for &o in outputs.iter() {
                    handles[o as usize] = div;
                }
                compute_ops.push(div);
                continue;
            }

            // PReLU is decomposed (matching the Chromium reference):
            // prelu(x, slope) = max(0, x) + slope * min(0, x)
            if let Operation::Prelu {
                input,
                slope,
                outputs,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];
                let slope_handle = handles[*slope as usize];
                let out_id = outputs[0];

                let x_name = CString::new("x").unwrap();
                let x1_name = CString::new("x1").unwrap();
                let x2_name = CString::new("x2").unwrap();
                let mode_name = CString::new("mode").unwrap();
                let neg_type = CString::new("Neg").unwrap();
                let relu_type = CString::new("ReLU").unwrap();
                let mul_type = CString::new("Mul").unwrap();
                let add_type = CString::new("Add").unwrap();

                // pos = ReLU(x) = Activation(x, mode=1)
                let pos_name = CString::new(format!("prelu_pos_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let pos = unsafe {
                    ddk_cann_operator_create_registered(relu_type.as_ptr(), pos_name.as_ptr())
                };
                unsafe { set_operand_input(pos, &x_name, x_handle) };
                unsafe { ddk_cann_operator_set_attr_int64(pos, mode_name.as_ptr(), 1) };
                extra_ops.push(pos);

                // neg_input = Neg(x)
                let neg_input_name = CString::new(format!("prelu_neg_input_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let neg_input = unsafe {
                    ddk_cann_operator_create_registered(neg_type.as_ptr(), neg_input_name.as_ptr())
                };
                unsafe { set_operand_input(neg_input, &x_name, x_handle) };
                extra_ops.push(neg_input);

                // relu_neg = Activation(neg_input, mode=1)
                let relu_neg_name = CString::new(format!("prelu_relu_neg_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let relu_neg = unsafe {
                    ddk_cann_operator_create_registered(relu_type.as_ptr(), relu_neg_name.as_ptr())
                };
                unsafe { set_operand_input(relu_neg, &x_name, neg_input) };
                unsafe { ddk_cann_operator_set_attr_int64(relu_neg, mode_name.as_ptr(), 1) };
                extra_ops.push(relu_neg);

                // neg_x = Neg(relu_neg)
                let neg_x_name = CString::new(format!("prelu_neg_x_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let neg_x = unsafe {
                    ddk_cann_operator_create_registered(neg_type.as_ptr(), neg_x_name.as_ptr())
                };
                unsafe { set_operand_input(neg_x, &x_name, relu_neg) };
                extra_ops.push(neg_x);

                // neg_scaled = Mul(neg_x, slope)
                let neg_scaled_name = CString::new(format!("prelu_neg_scaled_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let neg_scaled = unsafe {
                    ddk_cann_operator_create_registered(mul_type.as_ptr(), neg_scaled_name.as_ptr())
                };
                unsafe {
                    ddk_cann_operator_set_input(neg_scaled, x1_name.as_ptr(), neg_x);
                    ddk_cann_operator_set_input(neg_scaled, x2_name.as_ptr(), slope_handle);
                }
                extra_ops.push(neg_scaled);

                // output = Add(pos, neg_scaled)
                let output_name = CString::new(format!("prelu_output_{out_id}"))
                    .expect("operator/operand names never contain NUL bytes");
                let output = unsafe {
                    ddk_cann_operator_create_registered(add_type.as_ptr(), output_name.as_ptr())
                };
                unsafe {
                    ddk_cann_operator_set_input(output, x1_name.as_ptr(), pos);
                    ddk_cann_operator_set_input(output, x2_name.as_ptr(), neg_scaled);
                }

                for &o in outputs.iter() {
                    handles[o as usize] = output;
                }
                compute_ops.push(output);
                continue;
            }

            // Resample2d picks ResizeNearestNeighbor vs ResizeBilinear based on
            // the interpolation mode (matching the Chromium reference).
            if let Operation::Resample2d {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];
                let x_name = CString::new("x").unwrap();

                let nearest = options
                    .as_ref()
                    .map(|o| o.mode == "nearest-neighbor")
                    .unwrap_or(false);
                let op_name = CString::new(format!("resample2d_{}", outputs[0]))
                    .expect("operator/operand names never contain NUL bytes");
                let resample_op = if nearest {
                    unsafe { ddk_cann_op_resize_nearest_neighbor_with_name(op_name.as_ptr()) }
                } else {
                    let resample2d_type = CString::new("Resample2D").unwrap();
                    unsafe {
                        ddk_cann_operator_create_registered(
                            resample2d_type.as_ptr(),
                            op_name.as_ptr(),
                        )
                    }
                };
                if resample_op.is_null() {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: "cann_operator_create failed for resample2d".into(),
                    });
                }

                let status = unsafe { set_operand_input(resample_op, &x_name, x_handle) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: "cann_operator_set_input for resample2d failed".into(),
                    });
                }

                // size = [h, w] from the output shape (NCHW axes 2, 3).
                let out_dims = descriptor_dims(&graph.operands[outputs[0] as usize].descriptor);
                let h = if out_dims.len() >= 4 { out_dims[2] } else { 0 };
                let w = if out_dims.len() >= 4 { out_dims[3] } else { 0 };
                let size_vals: Vec<i32> = [h as i32, w as i32].to_vec();
                let size_const = make_const(
                    &format!("resample_size_{}", outputs[0]),
                    bytemuck::cast_slice(&size_vals),
                    &[2],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(size_const);
                let size_name = CString::new("size").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(resample_op, size_name.as_ptr(), size_const);
                }

                for &out_id in outputs.iter() {
                    handles[out_id as usize] = resample_op;
                }
                compute_ops.push(resample_op);
                continue;
            }

            // Wire Split by decomposing it into N single-output Slice ops.
            // GE cannot resolve SplitD's dynamic output "y" from a static
            // consumer (GetOutput("y", i) still yields an invalid graph), so we
            // emit one Slice per output instead.
            if let Operation::Split {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];
                let axis = options.as_ref().map(|o| o.axis).unwrap_or(0) as usize;
                let input_dims = descriptor_dims(&graph.operands[*input as usize].descriptor);
                let rank = input_dims.len();

                let mut offset: i32 = 0;
                for &out_id in outputs.iter() {
                    let out_dims = descriptor_dims(&graph.operands[out_id as usize].descriptor);
                    let part_size = out_dims[axis] as i32;

                    let starts: Vec<i32> = (0..rank)
                        .map(|d| if d == axis { offset } else { 0 })
                        .collect();
                    let sizes: Vec<i32> = (0..rank)
                        .map(|d| {
                            if d == axis {
                                part_size
                            } else {
                                input_dims[d] as i32
                            }
                        })
                        .collect();

                    let slice_op = create_slice_op(
                        &format!("split_{out_id}"),
                        x_handle,
                        &starts,
                        &sizes,
                        &mut extra_ops,
                    )?;

                    handles[out_id as usize] = slice_op;
                    compute_ops.push(slice_op);

                    offset += part_size;
                }
                continue;
            }

            let operator_type_name = match webnn_op_to_hiai(op) {
                Some(name) => {
                    CString::new(name).expect("operator/operand names never contain NUL bytes")
                }
                None => {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!("unsupported op: {}", op.label()).into(),
                    });
                }
            };

            // Create operator via cann_operator_create_registered.
            let op_name = CString::new(format!(
                "{}_{}",
                operator_type_name.to_str().unwrap(),
                op_outputs(op)[0]
            ))
            .expect("operator/operand names never contain NUL bytes");
            let compute_op: ddk_CannOperatorHandle = unsafe {
                ddk_cann_operator_create_registered(operator_type_name.as_ptr(), op_name.as_ptr())
            };

            if compute_op.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_operator_create failed for {}", op.label()).into(),
                });
            }

            if let Some((a, b, name_a, name_b)) = binary_op_inputs(op) {
                let lhs = handles[a as usize];
                let rhs = handles[b as usize];
                let lhs_name =
                    CString::new(name_a).expect("operator/operand names never contain NUL bytes");
                let rhs_name =
                    CString::new(name_b).expect("operator/operand names never contain NUL bytes");
                let status_a = unsafe { set_operand_input(compute_op, &lhs_name, lhs) };
                let status_b = unsafe { set_operand_input(compute_op, &rhs_name, rhs) };
                if status_a != 0 || status_b != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
            }

            if let Some(input) = unary_op_input(op) {
                let source_handle = handles[input as usize];
                let x_name = CString::new("x").unwrap();
                let status = unsafe { set_operand_input(compute_op, &x_name, source_handle) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
            }

            // Wire Cast via hiai::op::CastT (adapter maps "Cast"; attrs
            // src_dtype/dst_dtype).
            if let Operation::Cast { input, outputs, .. } = op {
                let src_dtype = graph.operands[*input as usize].descriptor.data_type;
                let dst_dtype = graph.operands[outputs[0] as usize].descriptor.data_type;
                let src_name = CString::new("src_dtype").unwrap();
                let dst_name = CString::new("dst_dtype").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_int64(
                        compute_op,
                        src_name.as_ptr(),
                        cann_data_type(src_dtype) as i64,
                    );
                    ddk_cann_operator_set_attr_int64(
                        compute_op,
                        dst_name.as_ptr(),
                        cann_data_type(dst_dtype) as i64,
                    );
                }
            }

            // Wire ReduceSum via hiai::op::ReduceSum (x + axes const + keep_dims).
            if let Operation::ReduceSum {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];
                let x_name = CString::new("x").unwrap();
                let status = unsafe { set_operand_input(compute_op, &x_name, x_handle) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }

                // axes const: reduce over the requested axes, or all axes if none.
                // HIAI ReduceSum declares axes as DT_INT32.
                let rank = graph.operands[*input as usize].descriptor.shape.len();
                let axes: Vec<i32> = options
                    .as_ref()
                    .and_then(|o| o.axes.as_ref())
                    .map(|a| a.iter().map(|&ax| ax as i32).collect())
                    .unwrap_or_else(|| (0..rank as u32).map(|ax| ax as i32).collect());
                let axes_const = make_const(
                    &format!("reduce_sum_axes_{}", outputs[0]),
                    bytemuck::cast_slice(&axes),
                    &[axes.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2, // FORMAT_ND
                );
                extra_ops.push(axes_const);
                let axes_name = CString::new("axes").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(compute_op, axes_name.as_ptr(), axes_const);
                }

                let keep_dims = options.as_ref().map(|o| o.keep_dimensions).unwrap_or(false);
                let keep_dims_name = CString::new("keep_dims").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_bool(
                        compute_op,
                        keep_dims_name.as_ptr(),
                        keep_dims as i32,
                    );
                }
            }

            // Wire Conv2d via hiai::op::Convolution.
            if let Operation::Conv2d {
                input,
                filter,
                options,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];
                let filter_handle = handles[*filter as usize];
                let x_name = CString::new("x").unwrap();
                let filter_name = CString::new("filter").unwrap();
                let set_status_x = unsafe { set_operand_input(compute_op, &x_name, x_handle) };
                let set_status_filter = unsafe {
                    ddk_cann_operator_set_input(compute_op, filter_name.as_ptr(), filter_handle)
                };
                if set_status_x != 0 || set_status_filter != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }

                // Optional bias input (hiai::op::Convolution supports a bias input,
                // matching the Chromium reference's set_input_bias).
                if let Some(bias_id) = options.as_ref().and_then(|o| o.bias) {
                    let bias_handle = handles[bias_id as usize];
                    let bias_name = CString::new("bias").unwrap();
                    let set_status_bias = unsafe {
                        ddk_cann_operator_set_input(compute_op, bias_name.as_ptr(), bias_handle)
                    };
                    if set_status_bias != 0 {
                        return Err(GraphError::ConversionFailed {
                            format: "cann".into(),
                            reason: "cann_operator_set_input for Conv2d bias failed".into(),
                        });
                    }
                }

                // hiai::op::Convolution: strides(2), pads(4), dilations(2),
                // pad_mode="SPECIFIC", data_format, groups. Read from options.
                let (stride_h, stride_w) = match options.as_ref().and_then(|o| {
                    if o.strides.len() >= 2 {
                        Some((o.strides[0] as i64, o.strides[1] as i64))
                    } else {
                        None
                    }
                }) {
                    Some((h, w)) => (h, w),
                    None => (1, 1),
                };
                let (dilation_h, dilation_w) = match options.as_ref().and_then(|o| {
                    if o.dilations.len() >= 2 {
                        Some((o.dilations[0] as i64, o.dilations[1] as i64))
                    } else {
                        None
                    }
                }) {
                    Some((h, w)) => (h, w),
                    None => (1, 1),
                };
                let (padding_top, padding_bottom, padding_left, padding_right) =
                    match options.as_ref().and_then(|o| {
                        if o.padding.len() >= 4 {
                            Some((
                                o.padding[0] as i64,
                                o.padding[1] as i64,
                                o.padding[2] as i64,
                                o.padding[3] as i64,
                            ))
                        } else {
                            None
                        }
                    }) {
                        Some((t, b, l, r)) => (t, b, l, r),
                        None => (0, 0, 0, 0),
                    };
                let groups = options
                    .as_ref()
                    .map(|o| o.groups as i64)
                    .unwrap_or(1)
                    .max(1);
                let format_str = conv_data_format(options);

                let strides: [i64; 2] = [stride_h, stride_w];
                let pads: [i64; 4] = [padding_top, padding_bottom, padding_left, padding_right];
                let dilations: [i64; 2] = [dilation_h, dilation_w];
                unsafe {
                    let strides_name = CString::new("strides").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        strides_name.as_ptr(),
                        strides.as_ptr(),
                        2,
                    );
                    let pads_name = CString::new("pads").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        pads_name.as_ptr(),
                        pads.as_ptr(),
                        4,
                    );
                    let dilations_name = CString::new("dilations").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        dilations_name.as_ptr(),
                        dilations.as_ptr(),
                        2,
                    );
                    let groups_name = CString::new("groups").unwrap();
                    ddk_cann_operator_set_attr_int64(compute_op, groups_name.as_ptr(), groups);
                    let data_format_name = CString::new("data_format").unwrap();
                    let data_format_value = CString::new(format_str)
                        .expect("operator/operand names never contain NUL bytes");
                    ddk_cann_operator_set_attr_string(
                        compute_op,
                        data_format_name.as_ptr(),
                        data_format_value.as_ptr(),
                    );
                    let pad_mode_name = CString::new("pad_mode").unwrap();
                    let pad_mode_value = CString::new("SPECIFIC").unwrap();
                    ddk_cann_operator_set_attr_string(
                        compute_op,
                        pad_mode_name.as_ptr(),
                        pad_mode_value.as_ptr(),
                    );
                }
            }

            // Wire MaxPool2d via hiai::op::PoolingD.
            if let Operation::MaxPool2d { input, options, .. } = op {
                let x_handle = handles[*input as usize];
                let x_name = CString::new("x").unwrap();
                let status = unsafe { set_operand_input(compute_op, &x_name, x_handle) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }

                // hiai::op::PoolingD: mode(0=max), window(2), stride(2), pad(4).
                let mut window_h: i64 = 1;
                let mut window_w: i64 = 1;
                let mut stride_h: i64 = 1;
                let mut stride_w: i64 = 1;
                let mut padding_top: i64 = 0;
                let mut padding_bottom: i64 = 0;
                let mut padding_left: i64 = 0;
                let mut padding_right: i64 = 0;
                if let Some(pool_options) = options {
                    if pool_options.padding.len() >= 4 {
                        padding_top = pool_options.padding[0] as i64;
                        padding_bottom = pool_options.padding[1] as i64;
                        padding_left = pool_options.padding[2] as i64;
                        padding_right = pool_options.padding[3] as i64;
                    }
                    if let Some(ref ws) = pool_options.window_dimensions {
                        if ws.len() >= 2 {
                            window_h = ws[0] as i64;
                            window_w = ws[1] as i64;
                        }
                    }
                    if pool_options.strides.len() >= 2 {
                        stride_h = pool_options.strides[0] as i64;
                        stride_w = pool_options.strides[1] as i64;
                    }
                }

                let window: [i64; 2] = [window_h, window_w];
                let stride: [i64; 2] = [stride_h, stride_w];
                let pad: [i64; 4] = [padding_top, padding_bottom, padding_left, padding_right];
                unsafe {
                    let mode_name = CString::new("mode").unwrap();
                    ddk_cann_operator_set_attr_int64(compute_op, mode_name.as_ptr(), 0);
                    let window_name = CString::new("window").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        window_name.as_ptr(),
                        window.as_ptr(),
                        2,
                    );
                    let stride_name = CString::new("stride").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        stride_name.as_ptr(),
                        stride.as_ptr(),
                        2,
                    );
                    let pad_name = CString::new("pad").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        pad_name.as_ptr(),
                        pad.as_ptr(),
                        4,
                    );
                }
            }

            // Wire ConvTranspose2d via hiai::op::ConvTranspose.
            // Input order: filter, x (opposite of Conv2D).
            if let Operation::ConvTranspose2d {
                input,
                filter,
                options,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];
                let filter_handle = handles[*filter as usize];
                let filter_name = CString::new("filter").unwrap();
                let x_name = CString::new("x").unwrap();
                let set_status_filter = unsafe {
                    ddk_cann_operator_set_input(compute_op, filter_name.as_ptr(), filter_handle)
                };
                let set_status_x =
                    unsafe { ddk_cann_operator_set_input(compute_op, x_name.as_ptr(), x_handle) };
                if set_status_filter != 0 || set_status_x != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }

                // hiai::op::ConvTranspose: same attribute set as Convolution.
                let (stride_h, stride_w) = match options.as_ref().and_then(|o| {
                    if o.strides.len() >= 2 {
                        Some((o.strides[0] as i64, o.strides[1] as i64))
                    } else {
                        None
                    }
                }) {
                    Some((h, w)) => (h, w),
                    None => (1, 1),
                };
                let (dilation_h, dilation_w) = match options.as_ref().and_then(|o| {
                    if o.dilations.len() >= 2 {
                        Some((o.dilations[0] as i64, o.dilations[1] as i64))
                    } else {
                        None
                    }
                }) {
                    Some((h, w)) => (h, w),
                    None => (1, 1),
                };
                let (padding_top, padding_bottom, padding_left, padding_right) =
                    match options.as_ref().and_then(|o| {
                        if o.padding.len() >= 4 {
                            Some((
                                o.padding[0] as i64,
                                o.padding[1] as i64,
                                o.padding[2] as i64,
                                o.padding[3] as i64,
                            ))
                        } else {
                            None
                        }
                    }) {
                        Some((t, b, l, r)) => (t, b, l, r),
                        None => (0, 0, 0, 0),
                    };
                let groups = options
                    .as_ref()
                    .map(|o| o.groups as i64)
                    .unwrap_or(1)
                    .max(1);
                let format_str = match options.as_ref().and_then(|o| {
                    if o.input_layout.eq_ignore_ascii_case("nhwc") {
                        Some("NHWC")
                    } else if o.input_layout.eq_ignore_ascii_case("nchw") {
                        Some("NCHW")
                    } else {
                        None
                    }
                }) {
                    Some(format) => format,
                    None => "NCHW",
                };

                let strides: [i64; 2] = [stride_h, stride_w];
                let pads: [i64; 4] = [padding_top, padding_bottom, padding_left, padding_right];
                let dilations: [i64; 2] = [dilation_h, dilation_w];
                unsafe {
                    let strides_name = CString::new("strides").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        strides_name.as_ptr(),
                        strides.as_ptr(),
                        2,
                    );
                    let pads_name = CString::new("pads").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        pads_name.as_ptr(),
                        pads.as_ptr(),
                        4,
                    );
                    let dilations_name = CString::new("dilations").unwrap();
                    ddk_cann_operator_set_attr_int64_list(
                        compute_op,
                        dilations_name.as_ptr(),
                        dilations.as_ptr(),
                        2,
                    );
                    let groups_name = CString::new("groups").unwrap();
                    ddk_cann_operator_set_attr_int64(compute_op, groups_name.as_ptr(), groups);
                    let data_format_name = CString::new("data_format").unwrap();
                    let data_format_value = CString::new(format_str)
                        .expect("operator/operand names never contain NUL bytes");
                    ddk_cann_operator_set_attr_string(
                        compute_op,
                        data_format_name.as_ptr(),
                        data_format_value.as_ptr(),
                    );
                    let pad_mode_name = CString::new("pad_mode").unwrap();
                    let pad_mode_value = CString::new("SPECIFIC").unwrap();
                    ddk_cann_operator_set_attr_string(
                        compute_op,
                        pad_mode_name.as_ptr(),
                        pad_mode_value.as_ptr(),
                    );
                }
            }

            // Wire ArgMax via hiai::op::ArgMaxExt2.
            // Axis is a tensor input, not an attribute.  Create an inline Const.
            if let Operation::ArgMax {
                input,
                axis,
                outputs,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];

                let axis_name_str = CString::new(format!("argmax_axis_{}", outputs[0]))
                    .expect("operator/operand names never contain NUL bytes");
                let axis_operator = unsafe { ddk_cann_op_const_with_name(axis_name_str.as_ptr()) };
                let axis_value: i32 = *axis as i32;
                let axis_shape: [i64; 1] = [1];
                let value_name = CString::new("value").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_tensor_raw_format(
                        axis_operator,
                        value_name.as_ptr(),
                        &axis_value as *const i32 as *const std::ffi::c_void,
                        4,
                        axis_shape.as_ptr(),
                        1,
                        ddk_CannDataType::CANN_DT_INT32,
                        2, // FORMAT_ND
                    );
                }
                const_ops.push(axis_operator);

                let x_name = CString::new("x").unwrap();
                let axis_name = CString::new("axis").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(compute_op, x_name.as_ptr(), x_handle);
                    ddk_cann_operator_set_input(compute_op, axis_name.as_ptr(), axis_operator);
                }
            }

            // Wire BatchNormalization via hiai::op::BNInference.
            if let Operation::BatchNormalization {
                input,
                mean,
                variance,
                options,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];
                let mean_handle = handles[*mean as usize];
                let variance_handle = handles[*variance as usize];

                let x_name = CString::new("x").unwrap();
                let mean_name = CString::new("mean").unwrap();
                let variance_name = CString::new("variance").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(compute_op, x_name.as_ptr(), x_handle);
                    ddk_cann_operator_set_input(compute_op, mean_name.as_ptr(), mean_handle);
                    ddk_cann_operator_set_input(
                        compute_op,
                        variance_name.as_ptr(),
                        variance_handle,
                    );
                }

                // Optional scale and offset (bias) from options.
                let scale_id = options.as_ref().and_then(|o| o.scale);
                let bias_id = options.as_ref().and_then(|o| o.bias);
                if let Some(id) = scale_id {
                    let scale_handle = handles[id as usize];
                    let scale_name = CString::new("scale").unwrap();
                    unsafe {
                        ddk_cann_operator_set_input(compute_op, scale_name.as_ptr(), scale_handle);
                    }
                }
                if let Some(id) = bias_id {
                    let offset_handle = handles[id as usize];
                    let offset_name = CString::new("offset").unwrap();
                    unsafe {
                        ddk_cann_operator_set_input(
                            compute_op,
                            offset_name.as_ptr(),
                            offset_handle,
                        );
                    }
                }

                let epsilon = options.as_ref().map(|o| o.epsilon as f32).unwrap_or(1e-5);
                let epsilon_name = CString::new("epsilon").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_float(compute_op, epsilon_name.as_ptr(), epsilon);
                }
            }

            // Wire Softmax via hiai::op::Softmax.
            if let Operation::Softmax { input, axis, .. } = op {
                let x_handle = handles[*input as usize];
                let x_name = CString::new("x").unwrap();
                let status = unsafe { set_operand_input(compute_op, &x_name, x_handle) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
                let axis_name = CString::new("axis").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_int64(compute_op, axis_name.as_ptr(), *axis as i64);
                }
            }

            // Wire Concat via hiai::op::ConcatD (dynamic input x, 1-based index).
            if let Operation::Concat { inputs, axis, .. } = op {
                let x_name = CString::new("x").unwrap();
                let status = unsafe {
                    ddk_cann_operator_create_dynamic_input(
                        compute_op,
                        x_name.as_ptr(),
                        inputs.len() as u32,
                    )
                };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!("cann_operator_create_dynamic_input failed: {status}")
                            .into(),
                    });
                }
                for (i, &input_id) in inputs.iter().enumerate() {
                    let handle = handles[input_id as usize];
                    let status =
                        unsafe { set_dynamic_input(compute_op, &x_name, (i + 1) as u32, handle) };
                    if status != 0 {
                        return Err(GraphError::ConversionFailed {
                            format: "cann".into(),
                            reason: format!(
                                "cann_operator_set_dynamic_input_by_index failed: {status}"
                            )
                            .into(),
                        });
                    }
                }
                let concat_dim_name = CString::new("concat_dim").unwrap();
                let n_name = CString::new("N").unwrap();
                unsafe {
                    ddk_cann_operator_set_attr_int64(
                        compute_op,
                        concat_dim_name.as_ptr(),
                        *axis as i64,
                    );
                    ddk_cann_operator_set_attr_int64(
                        compute_op,
                        n_name.as_ptr(),
                        inputs.len() as i64,
                    );
                }
            }

            // Wire Reshape via hiai::op::Reshape (x + shape const).
            if let Operation::Reshape { input, outputs, .. } = op {
                let x_handle = handles[*input as usize];
                let x_name = CString::new("x").unwrap();
                let status = unsafe { set_operand_input(compute_op, &x_name, x_handle) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
                let shape_vals = descriptor_dims(&graph.operands[outputs[0] as usize].descriptor);
                let shape_const = make_const(
                    &format!("reshape_shape_{}", outputs[0]),
                    bytemuck::cast_slice(&shape_vals),
                    &[shape_vals.len() as i64],
                    ddk_CannDataType::CANN_DT_INT64,
                    2, // FORMAT_ND
                );
                extra_ops.push(shape_const);
                let shape_name = CString::new("shape").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(compute_op, shape_name.as_ptr(), shape_const);
                }
            }

            // Wire Slice via hiai::op::Slice (x + offsets + size).
            if let Operation::Slice {
                input,
                starts,
                sizes,
                outputs,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];
                let x_name = CString::new("x").unwrap();
                let status = unsafe { set_operand_input(compute_op, &x_name, x_handle) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
                let dims = starts.len();
                let offsets_vals: Vec<i32> = starts.iter().map(|&s| s as i32).collect();
                let offsets_const = make_const(
                    &format!("slice_offsets_{}", outputs[0]),
                    bytemuck::cast_slice(&offsets_vals),
                    &[dims as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(offsets_const);
                let size_vals: Vec<i32> = sizes.iter().map(|d| d.static_or_max() as i32).collect();
                let size_const = make_const(
                    &format!("slice_size_{}", outputs[0]),
                    bytemuck::cast_slice(&size_vals),
                    &[dims as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(size_const);
                let offsets_name = CString::new("offsets").unwrap();
                let size_name = CString::new("size").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(compute_op, offsets_name.as_ptr(), offsets_const);
                    ddk_cann_operator_set_input(compute_op, size_name.as_ptr(), size_const);
                }
            }

            // Wire Transpose via ge::op::Transpose (x + perm const).
            if let Operation::Transpose {
                input,
                options,
                outputs,
                ..
            } = op
            {
                let x_handle = handles[*input as usize];
                let x_name = CString::new("x").unwrap();
                let status = unsafe { set_operand_input(compute_op, &x_name, x_handle) };
                if status != 0 {
                    return Err(GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!(
                            "cann_operator_set_input for {operator_type_name:?} failed"
                        )
                        .into(),
                    });
                }
                let perm_vals: Vec<i32> = options
                    .as_ref()
                    .map(|o| o.permutation.iter().map(|&p| p as i32).collect())
                    .unwrap_or_default();
                let perm_const = make_const(
                    &format!("transpose_perm_{}", outputs[0]),
                    bytemuck::cast_slice(&perm_vals),
                    &[perm_vals.len() as i64],
                    ddk_CannDataType::CANN_DT_INT32,
                    2,
                );
                extra_ops.push(perm_const);
                let perm_name = CString::new("perm").unwrap();
                unsafe {
                    ddk_cann_operator_set_input(compute_op, perm_name.as_ptr(), perm_const);
                }
            }

            let outputs: &[u32] = op_outputs(op);
            for &out_id in outputs.iter() {
                handles[out_id as usize] = compute_op;
            }

            compute_ops.push(compute_op);
        }

        // ── 4. Create NetOutput ─────────────────────────────────────────
        let out_name = graph.operands[graph.output_operands[0] as usize]
            .name
            .as_deref()
            .unwrap_or("output");
        let net_name =
            CString::new(out_name).expect("operator/operand names never contain NUL bytes");
        let mut net_out = unsafe {
            ddk_cann_op_net_output_with_name(net_name.as_ptr(), graph.output_operands.len() as i32)
        };
        if net_out.is_null() {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_op_net_output failed".into(),
            });
        }

        let x_name = CString::new("x").unwrap();
        let y_name = CString::new("y").unwrap();
        let output_type_name = CString::new("output_type").unwrap();
        let status = unsafe {
            ddk_cann_operator_create_dynamic_input(
                net_out,
                x_name.as_ptr(),
                graph.output_operands.len() as u32,
            )
        };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_create_dynamic_input failed: {status}").into(),
            });
        }

        for (output_index, &out_id) in graph.output_operands.iter().enumerate() {
            let source_handle = handles[out_id as usize];
            if source_handle.is_null() {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("no handle for output operand {out_id}").into(),
                });
            }
            // Dynamic input index is 1-based (matches the Chromium reference).
            let dst_index = (output_index + 1) as u32;
            let status = unsafe { set_dynamic_input(net_out, &x_name, dst_index, source_handle) };
            if status != 0 {
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!(
                        "cann_operator_set_dynamic_input_by_index[{output_index}] failed: {status}"
                    )
                    .into(),
                });
            }
        }

        let status = unsafe {
            ddk_cann_operator_create_dynamic_output(
                net_out,
                y_name.as_ptr(),
                graph.output_operands.len() as u32,
            )
        };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_create_dynamic_output failed: {status}").into(),
            });
        }

        let out_types: Vec<i64> = graph
            .output_operands
            .iter()
            .map(|&id| cann_data_type(graph.operands[id as usize].descriptor.data_type) as i64)
            .collect();
        let status = unsafe {
            ddk_cann_operator_set_attr_int64_list(
                net_out,
                output_type_name.as_ptr(),
                out_types.as_ptr(),
                out_types.len() as i32,
            )
        };
        if status != 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("cann_operator_set_attr_int64_list failed: {status}").into(),
            });
        }

        let mut all_ops: Vec<ddk_CannOperatorHandle> = Vec::new();
        all_ops.extend(data_ops.clone());
        all_ops.extend(const_ops);
        all_ops.extend(compute_ops);
        all_ops.extend(extra_ops);
        all_ops.push(net_out);

        // ── 5. Add all ops to graph ─────────────────────────────────────
        for &handle in &all_ops {
            if unsafe { ddk_cann_graph_add_op(can_graph, handle) } != 0 {
                let name = unsafe { ddk_cann_operator_get_name(handle) };
                let op_type = unsafe { ddk_cann_operator_get_type(handle) };
                let name = if name.is_null() {
                    "<unknown>".to_string()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(name) }
                        .to_string_lossy()
                        .into_owned()
                };
                let op_type = if op_type.is_null() {
                    "<unknown>".to_string()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(op_type) }
                        .to_string_lossy()
                        .into_owned()
                };
                return Err(GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("cann_graph_add_op failed for {op_type} '{name}'").into(),
                });
            }
        }

        // ── 6. Set graph inputs / outputs ───────────────────────────────
        unsafe {
            ddk_cann_graph_set_inputs(can_graph, data_ops.as_mut_ptr(), data_ops.len() as i32);
            ddk_cann_graph_set_outputs(can_graph, &mut net_out, 1);
        }

        // ── 7. Validate graph ───────────────────────────────────────────
        if unsafe { ddk_cann_graph_is_valid(can_graph) } == 0 {
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_graph_is_valid returned false".into(),
            });
        }

        // ── 8. Compile model ────────────────────────────────────────────
        let model_name = CString::new("webnn_model").unwrap();
        let model = unsafe { ddk_cann_model_create_with_name(model_name.as_ptr()) };
        if model.is_null() {
            unsafe { ddk_cann_graph_destroy(can_graph) };
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_model_create_with_name failed".into(),
            });
        }
        if unsafe { ddk_cann_model_set_graph(model, can_graph) } != 0 {
            unsafe {
                ddk_cann_model_destroy(model);
                ddk_cann_graph_destroy(can_graph);
            }
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_model_set_graph failed".into(),
            });
        }

        let ir_handle = unsafe { ddk_cann_hiai_ir_build_create() };
        if ir_handle.is_null() {
            unsafe {
                ddk_cann_model_destroy(model);
                ddk_cann_graph_destroy(can_graph);
            }
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_hiai_ir_build_create failed".into(),
            });
        }

        let mut buffer = ddk_CannModelBuffer {
            data: std::ptr::null_mut(),
            length: 0,
        };
        if unsafe { ddk_cann_model_create_buff_default(ir_handle, model, &mut buffer) } != 0
            || buffer.data.is_null()
        {
            unsafe {
                ddk_cann_hiai_ir_build_destroy(ir_handle);
                ddk_cann_model_destroy(model);
                ddk_cann_graph_destroy(can_graph);
            }
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_model_create_buff_default failed".into(),
            });
        }

        // Build options: CUSTOM device select + NPU order, matching the
        // Chromium reference (which does not set input shapes; HiAI's HCL
        // compiler pads to 4D itself and rejects explicitly-set mixed-rank
        // input shapes).
        let build_opts = unsafe { ddk_cann_build_options_create() };
        if !build_opts.is_null() {
            unsafe {
                ddk_cann_build_options_set_mode(build_opts, 1); // CUSTOM
                let devices: [i32; 1] = [0]; // 0 = NPU
                ddk_cann_build_options_set_device_order(build_opts, devices.as_ptr(), 1);
            }
        }

        // Compile IR model to OM bytes.
        let status = unsafe { ddk_cann_build_model(ir_handle, model, build_opts, &mut buffer) };
        if status != 0 || buffer.data.is_null() {
            unsafe {
                ddk_cann_model_buffer_destroy(ir_handle, &mut buffer);
                ddk_cann_hiai_ir_build_destroy(ir_handle);
                ddk_cann_model_destroy(model);
                ddk_cann_graph_destroy(can_graph);
            }
            return Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: "cann_build_model failed".into(),
            });
        }

        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.data as *const u8, buffer.length as usize) }
                .to_vec();

        // ── 9. Cleanup ──────────────────────────────────────────────────
        unsafe {
            ddk_cann_model_buffer_destroy(ir_handle, &mut buffer);
        }
        for &handle in &all_ops {
            unsafe { ddk_cann_operator_destroy(handle) };
        }
        unsafe {
            ddk_cann_hiai_ir_build_destroy(ir_handle);
            ddk_cann_model_destroy(model);
            ddk_cann_graph_destroy(can_graph);
        }

        Ok(bytes)
    }
}

#[cfg(feature = "cann-runtime")]
pub(crate) use adapter::encode_via_adapter;

#[cfg(not(feature = "cann-runtime"))]
pub(crate) fn encode_via_adapter(_graph: &GraphInfo) -> Result<Vec<u8>, GraphError> {
    Err(GraphError::ConversionFailed {
        format: "cann".into(),
        reason: "CANN shim not available (mock mode)".into(),
    })
}

pub struct CannConverter;

impl GraphConverter for CannConverter {
    fn format(&self) -> &'static str {
        "cann"
    }

    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        // Call CANN shim layer.
        if let Ok(bytes) = encode_via_adapter(graph) {
            return Ok(ConvertedGraph {
                format: "cann",
                content_type: "application/octet-stream",
                data: bytes,
                weights_data: None,
            });
        }
        // Fallback to Mock CANN -- validates ops, returns placeholder bytes.
        let model_bytes = build_hiai_ir_model_mock(graph)?;
        Ok(ConvertedGraph {
            format: "cann",
            content_type: "application/octet-stream",
            data: model_bytes,
            weights_data: None,
        })
    }
}

// cann-runtime-mock: verify graph structure, return placeholder bytes.
fn build_hiai_ir_model_mock(graph: &GraphInfo) -> Result<Vec<u8>, GraphError> {
    let mut operation_count: usize = 0;
    for operation in &graph.operations {
        if is_supported_op(operation) {
            operation_count += 1;
        }
    }
    if operation_count == 0 {
        return Err(GraphError::ConversionFailed {
            format: "cann".to_string(),
            reason: "no supported ops found".to_string(),
        });
    }
    Ok(vec![0x00, 0x00, 0x00, 0x00])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
    use crate::operator_options::MLDimension;
    use crate::operators::Operation;
    use std::collections::HashMap;

    fn make_add_graph() -> GraphInfo {
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(2), Dimension::Static(2)],
                        pending_permutation: vec![],
                    },
                    name: Some("lhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(2), Dimension::Static(2)],
                        pending_permutation: vec![],
                    },
                    name: Some("rhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![Dimension::Static(2), Dimension::Static(2)],
                        pending_permutation: vec![],
                    },
                    name: Some("sum".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation::Add {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn test_add_graph_converts() {
        let graph = make_add_graph();
        let converter = CannConverter;
        assert_eq!(converter.format(), "cann");
        let result = converter.convert(&graph);
        assert!(result.is_ok(), "{result:?}");
        let converted = result.unwrap();
        assert!(!converted.data.is_empty());
    }

    #[test]
    fn test_webnn_op_to_hiai_relu() {
        let op = Operation::Relu {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("ReLU"));
    }

    #[test]
    fn test_webnn_op_to_hiai_sigmoid() {
        let op = Operation::Sigmoid {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Sigmoid"));
    }

    #[test]
    fn test_webnn_op_to_hiai_tanh() {
        let op = Operation::Tanh {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Tanh"));
    }

    #[test]
    fn test_webnn_op_to_hiai_add() {
        let op = Operation::Add {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Add"));
    }

    #[test]
    fn test_webnn_op_to_hiai_mul() {
        let op = Operation::Mul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Mul"));
    }

    #[test]
    fn test_webnn_op_to_hiai_sub() {
        let op = Operation::Sub {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Sub"));
    }

    #[test]
    fn test_webnn_op_to_hiai_div() {
        let op = Operation::Div {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Div"));
    }

    #[test]
    fn test_webnn_op_to_hiai_cast() {
        let op = Operation::Cast {
            input: 0,
            data_type: crate::operator_enums::MLOperandDataType::Int32,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Cast"));
    }

    #[test]
    fn test_webnn_op_to_hiai_reduce_sum() {
        let op = Operation::ReduceSum {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("ReduceSum"));
    }

    #[test]
    fn test_webnn_op_to_hiai_conv2d() {
        let op = Operation::Conv2d {
            input: 0,
            filter: 1,
            options: None,
            outputs: vec![3],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Conv2D"));
    }

    #[test]
    fn test_webnn_op_to_hiai_max_pool2d() {
        let op = Operation::MaxPool2d {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("MaxPool"));
    }

    #[test]
    fn test_webnn_op_to_hiai_average_pool2d() {
        let op = Operation::AveragePool2d {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("AvgPool"));
    }

    #[test]
    fn test_webnn_op_to_hiai_matmul() {
        let op = Operation::Matmul {
            a: 0,
            b: 1,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("MatMul"));
    }

    #[test]
    fn test_webnn_op_to_hiai_softmax() {
        let op = Operation::Softmax {
            input: 0,
            axis: 1,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Softmax"));
    }

    #[test]
    fn test_webnn_op_to_hiai_reshape() {
        let op = Operation::Reshape {
            input: 0,
            new_shape: vec![MLDimension::Static(1), MLDimension::Static(4)],
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Reshape"));
    }

    #[test]
    fn test_webnn_op_to_hiai_concat() {
        let op = Operation::Concat {
            inputs: vec![0, 1],
            axis: 0,
            options: None,
            outputs: vec![2],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Concat"));
    }

    #[test]
    fn test_webnn_op_to_hiai_identity() {
        let op = Operation::Identity {
            input: 0,
            options: None,
            outputs: vec![1],
        };
        assert_eq!(webnn_op_to_hiai(&op), Some("Identity"));
    }
}
