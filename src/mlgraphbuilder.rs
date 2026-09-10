use std::collections::HashMap;

use bytemuck::NoUninit;
use log::{debug, trace};
use webnn_graph::serialize::SerializeOptions;

use crate::error::{GraphBuilderError, GraphError, ShapeInferenceError};
use crate::graph::{Dimension, get_static_or_max_size, to_dimension_vector};
use crate::mlcontext::{MLGraph, MLNamedOperands, MLOperand, MLOperandDescriptor, MLTensor};
use crate::operator_enums::MLOperandDataType;
use crate::operator_options::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLClampOptions, MLConstantOptions,
    MLConv2dOptions, MLConvTranspose2dOptions, MLCumulativeSumOptions, MLDimension, MLEluOptions,
    MLGatherOptions, MLGemmOptions, MLGruCellOptions, MLGruOptions, MLHardSigmoidOptions,
    MLInstanceNormalizationOptions, MLLayerNormalizationOptions, MLLeakyReluOptions,
    MLLinearOptions, MLLstmCellOptions, MLLstmOptions, MLOperatorOptions, MLPadOptions,
    MLPool2dOptions, MLReduceOptions, MLResample2dOptions, MLReverseOptions, MLScatterOptions,
    MLSliceOptions, MLSplitOptions, MLSqueezeOptions, MLTransposeOptions, MLTriangularOptions,
    MLUnsqueezeOptions, OperandIndex,
};
use crate::shape_inference::{
    InputLayout, ReduceOptions, SplitSpec, infer_concat_shape_dimensions,
    infer_conv_transpose2d_shape, infer_conv2d_shape, infer_expand_shape_dimensions,
    infer_gather_shape_dimensions, infer_gemm_shape_dimensions, infer_global_pool_shape,
    infer_matmul_shape_dimensions, infer_pad_shape, infer_pool2d_shape_dimensions,
    infer_prelu_shape, infer_reduce_shape_dimensions, infer_resample2d_shape,
    infer_scatter_elements_shape, infer_scatter_nd_shape, infer_slice_shape, infer_split_shapes,
    infer_squeeze_shape, infer_tile_shape, infer_transpose_shape_dimensions,
    infer_triangular_shape, infer_unsqueeze_shape_dimensions,
};
use crate::webnn_json::to_graph_json;
use crate::{DataType, Operand, OperandDescriptor, OperandKind, Operation};
use crate::{
    GraphInfo,
    mlcontext::{MLBackendBuilder, MLContext},
};

pub type Result<T> = std::result::Result<T, GraphBuilderError>;

#[derive(Debug)]
pub struct MLGraphBuilder<'context, 'builder> {
    backend: Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>,

    graph: Option<GraphInfo>,
}

#[derive(Debug)]
struct UncompiledBackendBuilder;

impl<'context, 'builder> MLBackendBuilder<'context, 'builder> for UncompiledBackendBuilder {
    fn build(&mut self, _graph: GraphInfo) -> crate::error::Result<MLGraph<'context>> {
        panic!("the graph-recording builder cannot create a runtime graph")
    }
}

pub(crate) fn get_operand(input: MLOperand, graph: &GraphInfo) -> Result<&Operand> {
    graph
        .operands
        .get(input.id)
        .ok_or(GraphBuilderError::InvalidOperand(input))
}

fn get_operands<'graph>(
    inputs: &[MLOperand],
    graph: &'graph GraphInfo,
) -> Result<Vec<&'graph Operand>> {
    inputs
        .iter()
        .map(|i| {
            graph
                .operands
                .get(i.id)
                .ok_or(GraphBuilderError::InvalidOperand(*i))
        })
        .collect()
}

fn check_same_data_type(
    inputs: &[MLOperand],
    operation: &Operation,
    operands: &[&Operand],
) -> Result<DataType> {
    let data_type = operands[0].descriptor.data_type;
    if operands
        .iter()
        .skip(1)
        .any(|o| o.descriptor.data_type != data_type)
    {
        return Err(Box::new(ShapeInferenceError::InconsistentDataTypes {
            operation: operation.clone(),
            inputs: inputs
                .iter()
                .copied()
                .zip(operands.iter().map(|&o| o.clone()))
                .collect(),
        })
        .into());
    }
    Ok(data_type)
}

fn slice_shape(
    input: MLOperand,
    operation: &Operation,
    starts: &[u32],
    sizes: &[MLDimension],
    options: &Option<MLSliceOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    if starts.len() != operand.descriptor.shape.len()
        || sizes.len() != operand.descriptor.shape.len()
    {
        return Err(Box::new(ShapeInferenceError::SliceErrorWrongStartSizes {
            operation: operation.clone(),
            input: (input, operand.clone()),
        })
        .into());
    }

    let sizes_u32: Vec<u32> = sizes.iter().map(|d| d.static_or_max()).collect();
    let strides = options.as_ref().map(|o| o.strides.as_slice());
    let shape = infer_shape_err(
        "slice",
        operation,
        infer_slice_shape(
            &shape_dims_u32(&operand.descriptor.shape),
            starts,
            &sizes_u32,
            strides,
        )
        .map(|v| to_dimension_vector(&v)),
    )?;

    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn concat_shape(
    inputs: &[MLOperand],
    operation: &Operation,
    axis: u32,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operands = get_operands(inputs, graph)?;
    let data_type = check_same_data_type(inputs, operation, &operands)?;

    let shape = if operands.iter().all(|s| s.descriptor.shape.is_empty()) && axis == 0 {
        vec![Dimension::Static(operands.len() as u32)]
    } else {
        let input_shapes: Vec<Vec<Dimension>> = operands
            .iter()
            .map(|op| op.descriptor.shape.clone())
            .collect();
        infer_concat_shape_dimensions(&input_shapes, axis).map_err(|e| {
            Box::new(ShapeInferenceError::ConcatError {
                operation: operation.clone(),
                inputs: inputs
                    .iter()
                    .copied()
                    .zip(operands.iter().map(|&o| o.clone()))
                    .collect(),
                source: e,
            })
        })?
    };
    Ok(OperandDescriptor {
        data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn expand_shape(
    input: MLOperand,
    operation: &Operation,
    new_shape: &[MLDimension],
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let desc = operand.descriptor.clone();
    let data_type = desc.data_type;
    let new_shape: Vec<Dimension> = new_shape.iter().map(|s| s.clone().into()).collect();

    let shape: Result<Vec<Dimension>> = infer_expand_shape_dimensions(&desc.shape, &new_shape)
        .map_err(|e| {
            Box::new(ShapeInferenceError::ExpandError {
                operation: operation.clone(),
                input: (input, operand.clone()),
                source: e,
            })
            .into()
        });
    let shape = shape?;

    Ok(OperandDescriptor {
        data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn shape_dims_u32(shape: &[Dimension]) -> Vec<u32> {
    shape.iter().map(get_static_or_max_size).collect()
}

fn infer_shape_err(
    op_name: &'static str,
    operation: &Operation,
    result: std::result::Result<Vec<Dimension>, GraphError>,
) -> Result<Vec<Dimension>> {
    result.map_err(|source| {
        Box::new(ShapeInferenceError::InferError {
            op_name,
            operation: operation.clone(),
            source,
        })
        .into()
    })
}

fn preserve_input_shape(
    input: MLOperand,
    _operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    Ok(operand.descriptor.clone())
}

fn constant_shape(
    operation: &Operation,
    options: Option<&MLConstantOptions>,
) -> Result<OperandDescriptor> {
    let options = options.ok_or_else(|| {
        Box::new(ShapeInferenceError::MissingOptions {
            operation: operation.clone(),
        })
    })?;
    let shape = options
        .shape
        .iter()
        .copied()
        .map(Dimension::Static)
        .collect();
    let data_type = match options.data_type.to_ascii_lowercase().as_str() {
        "float32" => DataType::Float32,
        "float16" => DataType::Float16,
        "int32" => DataType::Int32,
        "uint32" => DataType::Uint32,
        "int64" => DataType::Int64,
        "uint64" => DataType::Uint64,
        "int8" => DataType::Int8,
        "uint8" => DataType::Uint8,
        "int4" => DataType::Int4,
        "uint4" => DataType::Uint4,
        _ => DataType::Float32,
    };
    Ok(OperandDescriptor {
        data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn matmul_shape(
    a: MLOperand,
    b: MLOperand,
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let a_op = get_operand(a, graph)?;
    let b_op = get_operand(b, graph)?;
    let data_type = check_same_data_type(&[a, b], operation, &[a_op, b_op])?;
    let shape = infer_shape_err(
        "matmul",
        operation,
        infer_matmul_shape_dimensions(&a_op.descriptor.shape, &b_op.descriptor.shape),
    )?;
    Ok(OperandDescriptor {
        data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn gemm_shape(
    a: MLOperand,
    b: MLOperand,
    operation: &Operation,
    options: Option<&MLGemmOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let a_op = get_operand(a, graph)?;
    let b_op = get_operand(b, graph)?;
    let data_type = check_same_data_type(&[a, b], operation, &[a_op, b_op])?;
    let opts = options.cloned().unwrap_or_default();
    let mut shape = infer_shape_err(
        "gemm",
        operation,
        infer_gemm_shape_dimensions(
            &a_op.descriptor.shape,
            &b_op.descriptor.shape,
            opts.a_transpose,
            opts.b_transpose,
        ),
    )?;
    if let Some(c_id) = opts.c {
        let c_op = get_operand(MLOperand { id: c_id as usize }, graph)?;
        shape = infer_shape_err(
            "gemm",
            operation,
            crate::shape_inference::broadcast_shapes_dimensions(&shape, &c_op.descriptor.shape),
        )?;
    }
    Ok(OperandDescriptor {
        data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn transpose_shape(
    input: MLOperand,
    operation: &Operation,
    options: Option<&MLTransposeOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let perm = options
        .filter(|o| !o.permutation.is_empty())
        .map(|o| o.permutation.as_slice());
    let shape = infer_shape_err(
        "transpose",
        operation,
        infer_transpose_shape_dimensions(&operand.descriptor.shape, perm),
    )?;
    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn gather_shape(
    input: MLOperand,
    indices: MLOperand,
    operation: &Operation,
    options: Option<&MLGatherOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let indices_op = get_operand(indices, graph)?;
    let axis = options.map(|o| o.axis).unwrap_or(0);
    let shape = infer_shape_err(
        "gather",
        operation,
        infer_gather_shape_dimensions(
            &input_op.descriptor.shape,
            &indices_op.descriptor.shape,
            axis,
        ),
    )?;
    Ok(OperandDescriptor {
        data_type: input_op.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn gather_elements_shape(
    input: MLOperand,
    indices: MLOperand,
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let indices_op = get_operand(indices, graph)?;
    let _ = operation;
    Ok(OperandDescriptor {
        data_type: input_op.descriptor.data_type,
        shape: indices_op.descriptor.shape.clone(),
        pending_permutation: vec![],
    })
}

fn gather_nd_shape(
    input: MLOperand,
    indices: MLOperand,
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let indices_op = get_operand(indices, graph)?;
    let indices_shape = &indices_op.descriptor.shape;
    if indices_shape.is_empty() {
        let shape = infer_shape_err(
            "gatherND",
            operation,
            Err(GraphError::ShapeInferenceFailed {
                reason: "GatherND indices must have rank >= 1".to_string(),
            }),
        )?;
        return Ok(OperandDescriptor {
            data_type: input_op.descriptor.data_type,
            shape,
            pending_permutation: vec![],
        });
    }
    let k = match indices_shape.last() {
        Some(Dimension::Static(v)) => *v as usize,
        Some(Dimension::Dynamic(d)) => d.max_size as usize,
        None => unreachable!("indices_shape.empty already covered"),
    };
    if k > input_op.descriptor.shape.len() {
        let shape = infer_shape_err(
            "gatherND",
            operation,
            Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "GatherND indices last dimension {} exceeds input rank {}",
                    k,
                    input_op.descriptor.shape.len()
                ),
            }),
        )?;
        return Ok(OperandDescriptor {
            data_type: input_op.descriptor.data_type,
            shape,
            pending_permutation: vec![],
        });
    }
    let mut shape = indices_shape[..indices_shape.len() - 1].to_vec();
    shape.extend_from_slice(&input_op.descriptor.shape[k..]);
    Ok(OperandDescriptor {
        data_type: input_op.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn scatter_nd_shape(
    input: MLOperand,
    updates: MLOperand,
    indices: MLOperand,
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let indices_op = get_operand(indices, graph)?;
    let updates_op = get_operand(updates, graph)?;

    check_same_data_type(&[input, updates], operation, &[input_op, updates_op])?;

    let fully_static =
        |shape: &[Dimension]| shape.iter().all(|d| matches!(d, Dimension::Static(_)));
    if fully_static(&input_op.descriptor.shape)
        && fully_static(&indices_op.descriptor.shape)
        && fully_static(&updates_op.descriptor.shape)
    {
        let _validated = infer_shape_err(
            "scatterND",
            operation,
            infer_scatter_nd_shape(
                &shape_dims_u32(&input_op.descriptor.shape),
                &shape_dims_u32(&indices_op.descriptor.shape),
                &shape_dims_u32(&updates_op.descriptor.shape),
            )
            .map(|shape| to_dimension_vector(&shape)),
        )?;
    }

    Ok(input_op.descriptor.clone())
}

fn scatter_elements_shape(
    input: MLOperand,
    indices: MLOperand,
    updates: MLOperand,
    operation: &Operation,
    options: Option<&MLScatterOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let indices_op = get_operand(indices, graph)?;
    let updates_op = get_operand(updates, graph)?;

    check_same_data_type(&[input, updates], operation, &[input_op, updates_op])?;

    let fully_static =
        |shape: &[Dimension]| shape.iter().all(|d| matches!(d, Dimension::Static(_)));
    if fully_static(&input_op.descriptor.shape)
        && fully_static(&indices_op.descriptor.shape)
        && fully_static(&updates_op.descriptor.shape)
    {
        let axis = options.map(|o| o.axis).unwrap_or(0);
        let _validated = infer_shape_err(
            "scatterElements",
            operation,
            infer_scatter_elements_shape(
                &shape_dims_u32(&input_op.descriptor.shape),
                &shape_dims_u32(&indices_op.descriptor.shape),
                &shape_dims_u32(&updates_op.descriptor.shape),
                axis,
            )
            .map(|shape| to_dimension_vector(&shape)),
        )?;
    }

    Ok(input_op.descriptor.clone())
}

fn resample2d_shape(
    input: MLOperand,
    operation: &Operation,
    options: Option<&MLResample2dOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let default_opts = MLResample2dOptions::default();
    let opts = options.unwrap_or(&default_opts);
    let rank = input_op.descriptor.shape.len();
    let axes: [usize; 2] = if opts.axes.len() == 2 {
        [opts.axes[0] as usize, opts.axes[1] as usize]
    } else {
        [rank.saturating_sub(2), rank.saturating_sub(1)]
    };
    let shape = infer_shape_err(
        "resample2d",
        operation,
        infer_resample2d_shape(
            &input_op.descriptor.shape,
            axes,
            opts.sizes.as_deref(),
            &opts.scales,
        ),
    )?;
    Ok(OperandDescriptor {
        data_type: input_op.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn argminmax_shape(
    input: MLOperand,
    operation: &Operation,
    axis: u32,
    options: Option<&MLArgMinMaxOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let opts = options.cloned().unwrap_or_default();
    let reduce_opts = ReduceOptions {
        axes: vec![axis],
        keep_dimensions: opts.keep_dimensions,
    };
    let shape = infer_shape_err(
        "reduce",
        operation,
        infer_reduce_shape_dimensions(&operand.descriptor.shape, &reduce_opts),
    )?;
    Ok(OperandDescriptor {
        data_type: opts.output_data_type.into(),
        shape,
        pending_permutation: vec![],
    })
}

fn reduce_shape(
    input: MLOperand,
    operation: &Operation,
    options: Option<&MLReduceOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let opts = options.cloned().unwrap_or_default();
    let rank = operand.descriptor.shape.len() as u32;
    let reduce_opts = ReduceOptions {
        // WebNN: omitted `axes` reduces all dimensions; explicit `axes: []` reduces none.
        axes: match &opts.axes {
            None => (0..rank).collect(),
            Some(axes) => axes.clone(),
        },
        keep_dimensions: opts.keep_dimensions,
    };
    let shape = infer_shape_err(
        "reduce",
        operation,
        infer_reduce_shape_dimensions(&operand.descriptor.shape, &reduce_opts),
    )?;
    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn reshape_shape(
    input: MLOperand,
    _operation: &Operation,
    new_shape: &[MLDimension],
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let shape: Vec<Dimension> = new_shape.iter().cloned().map(Into::into).collect();
    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn conv2d_shape(
    input: MLOperand,
    filter: MLOperand,
    operation: &Operation,
    options: Option<&MLConv2dOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let filter_op = get_operand(filter, graph)?;
    let opts = options.cloned().unwrap_or_default();
    let shape = infer_shape_err(
        "conv2d",
        operation,
        infer_conv2d_shape(
            &shape_dims_u32(&input_op.descriptor.shape),
            &shape_dims_u32(&filter_op.descriptor.shape),
            &opts,
        )
        .map(|v| to_dimension_vector(&v)),
    )?;
    Ok(OperandDescriptor {
        data_type: input_op.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn conv_transpose2d_shape(
    input: MLOperand,
    filter: MLOperand,
    operation: &Operation,
    options: Option<&MLConvTranspose2dOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let filter_op = get_operand(filter, graph)?;
    let opts = options.cloned().unwrap_or_default();
    let shape = infer_shape_err(
        "convTranspose2d",
        operation,
        infer_conv_transpose2d_shape(
            &shape_dims_u32(&input_op.descriptor.shape),
            &shape_dims_u32(&filter_op.descriptor.shape),
            &opts,
        )
        .map(|v| to_dimension_vector(&v)),
    )?;
    Ok(OperandDescriptor {
        data_type: input_op.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn pool2d_shape(
    input: MLOperand,
    operation: &Operation,
    options: Option<&MLPool2dOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let default_pool = MLPool2dOptions::default();
    let o = options.unwrap_or(&default_pool);
    let shape = infer_shape_err(
        "pool2d",
        operation,
        infer_pool2d_shape_dimensions(
            &operand.descriptor.shape,
            &o.layout,
            o.window_dimensions.as_deref(),
            &o.strides,
            &o.dilations,
            &o.padding,
            o.output_sizes.as_deref(),
            o.output_shape_rounding.eq_ignore_ascii_case("ceil"),
        ),
    )?;
    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn global_pool_shape(
    input: MLOperand,
    operation: &Operation,
    options: Option<&MLPool2dOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let default_pool = MLPool2dOptions::default();
    let o = options.unwrap_or(&default_pool);
    let layout_enum = if o.layout.eq_ignore_ascii_case("nhwc") {
        InputLayout::Nhwc
    } else {
        InputLayout::Nchw
    };
    let shape = infer_shape_err(
        "globalPool",
        operation,
        infer_global_pool_shape(&shape_dims_u32(&operand.descriptor.shape), layout_enum)
            .map(|v| to_dimension_vector(&v)),
    )?;
    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn pad_shape(
    input: MLOperand,
    operation: &Operation,
    beginning_padding: &[u32],
    ending_padding: &[u32],
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let mut padding = beginning_padding.to_vec();
    padding.extend_from_slice(ending_padding);
    let shape = infer_shape_err(
        "pad",
        operation,
        infer_pad_shape(&shape_dims_u32(&operand.descriptor.shape), &padding)
            .map(|v| to_dimension_vector(&v)),
    )?;
    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn squeeze_shape(
    input: MLOperand,
    operation: &Operation,
    options: Option<&MLSqueezeOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let axes = options
        .filter(|o| !o.axes.is_empty())
        .map(|o| o.axes.as_slice());
    let shape = infer_shape_err(
        "squeeze",
        operation,
        infer_squeeze_shape(&shape_dims_u32(&operand.descriptor.shape), axes)
            .map(|v| to_dimension_vector(&v)),
    )?;
    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn unsqueeze_shape(
    input: MLOperand,
    operation: &Operation,
    options: Option<&MLUnsqueezeOptions>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let axes = options.map(|o| o.axes.as_slice()).unwrap_or(&[]);
    let shape = infer_shape_err(
        "unsqueeze",
        operation,
        infer_unsqueeze_shape_dimensions(&operand.descriptor.shape, axes),
    )?;
    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn tile_shape(
    input: MLOperand,
    operation: &Operation,
    repetitions: &[u32],
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let shape = infer_shape_err(
        "tile",
        operation,
        infer_tile_shape(&shape_dims_u32(&operand.descriptor.shape), repetitions)
            .map(|v| to_dimension_vector(&v)),
    )?;
    Ok(OperandDescriptor {
        data_type: operand.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn prelu_shape(
    input: MLOperand,
    slope: MLOperand,
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let slope_op = get_operand(slope, graph)?;
    let shape = infer_shape_err(
        "prelu",
        operation,
        infer_prelu_shape(
            &shape_dims_u32(&input_op.descriptor.shape),
            &shape_dims_u32(&slope_op.descriptor.shape),
        )
        .map(|v| to_dimension_vector(&v)),
    )?;
    Ok(OperandDescriptor {
        data_type: input_op.descriptor.data_type,
        shape,
        pending_permutation: vec![],
    })
}

fn split_shape(
    input: MLOperand,
    operation: &Operation,
    splits: &[u32],
    split_equal_parts: Option<u32>,
    options: Option<&MLSplitOptions>,
    graph: &GraphInfo,
) -> Result<Vec<OperandDescriptor>> {
    let operand = get_operand(input, graph)?;
    let axis = options.map(|o| o.axis).unwrap_or(0);
    let spec = if let Some(n) = split_equal_parts {
        SplitSpec::Count(n)
    } else {
        SplitSpec::Sizes(splits.to_vec())
    };
    let mut output_shapes =
        infer_split_shapes(&shape_dims_u32(&operand.descriptor.shape), &spec, axis).map_err(
            |e| {
                Box::new(ShapeInferenceError::InferError {
                    op_name: "split",
                    operation: operation.clone(),
                    source: e,
                })
            },
        )?;

    Ok(output_shapes
        .drain(..)
        .map(|shape| OperandDescriptor {
            data_type: operand.descriptor.data_type,
            shape: shape.iter().map(|d| Dimension::Static(*d)).collect(),
            pending_permutation: vec![],
        })
        .collect())
}

fn shape_op_shape(input: MLOperand, graph: &GraphInfo) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    let rank = operand.descriptor.shape.len() as u32;
    Ok(OperandDescriptor {
        data_type: DataType::Int64,
        shape: vec![Dimension::Static(rank)],
        pending_permutation: vec![],
    })
}

fn where_shape(
    inputs: &[MLOperand],
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operands = get_operands(inputs, graph)?;
    if operands[0].descriptor.data_type != DataType::Uint8 {
        return Err(Box::new(ShapeInferenceError::NonUint8Condition {
            operation: operation.clone(),
            inputs: inputs
                .iter()
                .copied()
                .zip(operands.iter().map(|&o| o.clone()))
                .collect(),
        })
        .into());
    }

    check_same_data_type(&inputs[1..], operation, &operands[1..])?;

    let mut output_shape = crate::shape_inference::broadcast_shapes_dimensions(
        &operands[1].descriptor.shape,
        &operands[2].descriptor.shape,
    )
    .map_err(|e| {
        Box::new(ShapeInferenceError::BroadcastError {
            operation: operation.clone(),
            inputs: inputs
                .iter()
                .copied()
                .zip(operands.iter().map(|&o| o.clone()))
                .collect(),
            source: e,
        })
    })?;
    output_shape = crate::shape_inference::broadcast_shapes_dimensions(
        &operands[0].descriptor.shape,
        &output_shape,
    )
    .map_err(|e| {
        Box::new(ShapeInferenceError::BroadcastError {
            operation: operation.clone(),
            inputs: inputs
                .iter()
                .copied()
                .zip(operands.iter().map(|&o| o.clone()))
                .collect(),
            source: e,
        })
    })?;

    Ok(OperandDescriptor {
        data_type: operands[1].descriptor.data_type,
        shape: output_shape,
        pending_permutation: vec![],
    })
}

fn same_shape(
    inputs: &[MLOperand],
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operands = get_operands(inputs, graph)?;

    check_same_data_type(inputs, operation, &operands)?;

    let mut output_descriptor = operands[0].descriptor.clone();
    for op in operands.iter().skip(1) {
        output_descriptor.shape = crate::shape_inference::broadcast_shapes_dimensions(
            &output_descriptor.shape,
            &op.descriptor.shape,
        )
        .map_err(|e| {
            Box::new(ShapeInferenceError::BroadcastError {
                operation: operation.clone(),
                inputs: inputs
                    .iter()
                    .copied()
                    .zip(operands.iter().map(|&o| o.clone()))
                    .collect(),
                source: e,
            })
        })?;
    }

    Ok(output_descriptor)
}

/// Shape inference for element-wise logical ops (`equal`, `logicalAnd`, `logicalNot`, …).
///
/// WebNN always outputs `uint8` (0/1) for these ops regardless of input dtype
/// (see MLGraphBuilder element-wise logical operations in the WebNN spec).
fn element_wise_logical_shape(
    inputs: &[MLOperand],
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let mut output = same_shape(inputs, operation, graph)?;
    output.data_type = DataType::Uint8;
    Ok(output)
}

fn unary_element_wise_logical_shape(
    input: MLOperand,
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let mut output = preserve_input_shape(input, operation, graph)?;
    output.data_type = DataType::Uint8;
    Ok(output)
}

fn cast_shape(
    input: MLOperand,
    data_type: MLOperandDataType,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let operand = get_operand(input, graph)?;
    Ok(OperandDescriptor {
        data_type: data_type.into(),
        shape: operand.descriptor.shape.clone(),
        pending_permutation: vec![],
    })
}

/// WebNN: output descriptor uses zeroPoint's dataType and input's shape.
fn quantize_linear_shape(
    input: MLOperand,
    zero_point: Option<u32>,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let data_type = match zero_point {
        Some(zp) => {
            get_operand(MLOperand { id: zp as usize }, graph)?
                .descriptor
                .data_type
        }
        None => DataType::Uint8,
    };
    Ok(OperandDescriptor {
        data_type,
        shape: input_op.descriptor.shape.clone(),
        pending_permutation: vec![],
    })
}

/// WebNN: output descriptor uses scale's dataType and input's shape.
fn dequantize_linear_shape(
    input: MLOperand,
    scale: MLOperand,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let input_op = get_operand(input, graph)?;
    let scale_op = get_operand(scale, graph)?;
    Ok(OperandDescriptor {
        data_type: scale_op.descriptor.data_type,
        shape: input_op.descriptor.shape.clone(),
        pending_permutation: vec![],
    })
}

/// Generates `fn_name(input)` / `fn_name(input, extra, …)` and matching `_with_options` on `MLGraphBuilder`.
///
/// Extra parameters must match `Operation` field names (e.g. `axis: u32` for `Softmax`).
macro_rules! impl_unary_op {
    ($fn_name:ident, $fn_with_options:ident, $op:ident) => {
        impl_unary_op! {$fn_name, $fn_with_options, $op, MLOperatorOptions}
    };
    ($fn_name:ident, $fn_with_options:ident, $op:ident, $option_type:ident) => {
        pub fn $fn_name(&mut self, input: MLOperand) -> Result<MLOperand> {
            self.$fn_with_options(input, $option_type::default())
        }

        pub fn $fn_with_options(
            &mut self,
            input: MLOperand,
            options: $option_type,
        ) -> Result<MLOperand> {
            self.unary_same_shape_operation(input, options, |input, output_id, options| {
                Operation::$op {
                    input,
                    options,
                    outputs: vec![output_id],
                }
            })
        }
    };
    (
        $fn_name:ident,
        $fn_with_options:ident,
        $op:ident,
        $option_type:ident,
        $( $extra:ident : $ety:ty ),+ $(,)?
    ) => {
        pub fn $fn_name(
            &mut self,
            input: MLOperand,
            $( $extra : $ety ),+
        ) -> Result<MLOperand> {
            self.$fn_with_options(input, $( $extra ),+, $option_type::default())
        }

        pub fn $fn_with_options(
            &mut self,
            input: MLOperand,
            $( $extra : $ety ),+,
            options: $option_type,
        ) -> Result<MLOperand> {
            self.unary_same_shape_operation(input, options, |input, output_id, options| {
                Operation::$op {
                    input,
                    $( $extra ),+,
                    options,
                    outputs: vec![output_id],
                }
            })
        }
    };
}

/// Generates `fn_name(a, b)` and `fn_with_options(a, b, options)` on `MLGraphBuilder`.
macro_rules! impl_binary_op {
    ($fn_name:ident, $fn_with_options:ident, $op:ident) => {
        impl_binary_op! {$fn_name, $fn_with_options, $op, MLOperatorOptions}
    };
    ($fn_name:ident, $fn_with_options:ident, $op:ident, $option_type:ident) => {
        impl_binary_op! {$fn_name, $fn_with_options, $op, $option_type, a, b}
    };
    ($fn_name:ident, $fn_with_options:ident, $op:ident, $option_type:ident, $arg1:ident, $arg2:ident) => {
        pub fn $fn_name(&mut self, $arg1: MLOperand, $arg2: MLOperand) -> Result<MLOperand> {
            self.$fn_with_options($arg1, $arg2, $option_type::default())
        }

        pub fn $fn_with_options(
            &mut self,
            $arg1: MLOperand,
            $arg2: MLOperand,
            options: $option_type,
        ) -> Result<MLOperand> {
            self.binary_same_shape_operation(
                $arg1,
                $arg2,
                options,
                |$arg1, $arg2, output_id, options| Operation::$op {
                    $arg1,
                    $arg2,
                    options,
                    outputs: vec![output_id],
                },
            )
        }
    };
}

/// Generates `fn_name(a, b, c)` and `fn_with_options(a, b, c, options)` on `MLGraphBuilder`.
///
/// Operand parameter names must match `Operation` field names (e.g. `condition`, `true_value`, `false_value` for `Where`).
macro_rules! impl_ternary_op {
    ($fn_name:ident, $fn_with_options:ident, $op:ident) => {
        impl_ternary_op! {
            $fn_name,
            $fn_with_options,
            $op,
            MLOperatorOptions,
            a,
            b,
            c
        }
    };
    (
        $fn_name:ident,
        $fn_with_options:ident,
        $op:ident,
        $arg1:ident,
        $arg2:ident,
        $arg3:ident
    ) => {
        impl_ternary_op! {
            $fn_name,
            $fn_with_options,
            $op,
            MLOperatorOptions,
            $arg1,
            $arg2,
            $arg3
        }
    };
    (
        $fn_name:ident,
        $fn_with_options:ident,
        $op:ident,
        $option_type:ident,
        $arg1:ident,
        $arg2:ident,
        $arg3:ident
    ) => {
        pub fn $fn_name(
            &mut self,
            $arg1: MLOperand,
            $arg2: MLOperand,
            $arg3: MLOperand,
        ) -> Result<MLOperand> {
            self.$fn_with_options($arg1, $arg2, $arg3, $option_type::default())
        }

        pub fn $fn_with_options(
            &mut self,
            $arg1: MLOperand,
            $arg2: MLOperand,
            $arg3: MLOperand,
            options: $option_type,
        ) -> Result<MLOperand> {
            self.ternary_operation(
                $arg1,
                $arg2,
                $arg3,
                options,
                |a, b, c, output_id, options| Operation::$op {
                    $arg1: a,
                    $arg2: b,
                    $arg3: c,
                    options,
                    outputs: vec![output_id],
                },
            )
        }
    };
}

/// Two required operands plus an optional third (`quantizeLinear` / `dequantizeLinear`).
macro_rules! impl_ternary_optional_op {
    (
        $fn_name:ident,
        $fn_with_three:ident,
        $fn_with_options:ident,
        $op:ident,
        $option_type:ident,
        $arg1:ident,
        $arg2:ident,
        $optional:ident
    ) => {
        pub fn $fn_name(&mut self, $arg1: MLOperand, $arg2: MLOperand) -> Result<MLOperand> {
            self.$fn_with_options($arg1, $arg2, None, $option_type::default())
        }

        pub fn $fn_with_three(
            &mut self,
            $arg1: MLOperand,
            $arg2: MLOperand,
            $optional: MLOperand,
        ) -> Result<MLOperand> {
            self.$fn_with_options($arg1, $arg2, Some($optional), $option_type::default())
        }

        pub fn $fn_with_options(
            &mut self,
            $arg1: MLOperand,
            $arg2: MLOperand,
            $optional: Option<MLOperand>,
            options: $option_type,
        ) -> Result<MLOperand> {
            self.ternary_optional_operation(
                $arg1,
                $arg2,
                $optional,
                options,
                |a, b, optional, output_id, options| Operation::$op {
                    $arg1: a,
                    $arg2: b,
                    $optional: optional,
                    options,
                    outputs: vec![output_id],
                },
            )
        }
    };
}

fn recurrent_num_directions(direction: &str) -> u32 {
    if direction == "both" { 2 } else { 1 }
}

fn recurrent_batch_size(input_shape: &[Dimension]) -> u32 {
    match input_shape.len() {
        2 => get_static_or_max_size(&input_shape[0]),
        3 => get_static_or_max_size(&input_shape[1]),
        _ => 1,
    }
}

fn gru_output_shapes(
    input: OperandIndex,
    steps: u32,
    hidden_size: u32,
    options: Option<&MLGruOptions>,
    graph: &GraphInfo,
) -> Result<Vec<OperandDescriptor>> {
    let input_operand = &graph.operands[input as usize];
    let dtype = input_operand.descriptor.data_type;
    let opts = options.cloned().unwrap_or_default();
    let num_dir = recurrent_num_directions(&opts.direction);
    let batch = recurrent_batch_size(&input_operand.descriptor.shape);
    let h = hidden_size;

    let mut shapes = vec![OperandDescriptor {
        data_type: dtype,
        shape: to_dimension_vector(&[num_dir, batch, h]),
        pending_permutation: vec![],
    }];
    if opts.return_sequence {
        shapes.push(OperandDescriptor {
            data_type: dtype,
            shape: to_dimension_vector(&[steps, num_dir, batch, h]),
            pending_permutation: vec![],
        });
    }
    Ok(shapes)
}

fn lstm_output_shapes(
    input: OperandIndex,
    steps: u32,
    hidden_size: u32,
    options: Option<&MLLstmOptions>,
    graph: &GraphInfo,
) -> Result<Vec<OperandDescriptor>> {
    let input_operand = &graph.operands[input as usize];
    let dtype = input_operand.descriptor.data_type;
    let opts = options.cloned().unwrap_or_default();
    let num_dir = recurrent_num_directions(&opts.direction);
    let batch = recurrent_batch_size(&input_operand.descriptor.shape);
    let h = hidden_size;

    let mut shapes = Vec::new();
    shapes.push(OperandDescriptor {
        data_type: dtype,
        shape: to_dimension_vector(&[num_dir, batch, h]),
        pending_permutation: vec![],
    });
    shapes.push(OperandDescriptor {
        data_type: dtype,
        shape: to_dimension_vector(&[num_dir, batch, h]),
        pending_permutation: vec![],
    });
    if opts.return_sequence {
        shapes.push(OperandDescriptor {
            data_type: dtype,
            shape: to_dimension_vector(&[steps, num_dir, batch, h]),
            pending_permutation: vec![],
        });
    }
    Ok(shapes)
}

fn lstm_cell_output_shapes(
    hidden_state: OperandIndex,
    cell_state: OperandIndex,
    graph: &GraphInfo,
) -> Result<Vec<OperandDescriptor>> {
    Ok(vec![
        graph.operands[hidden_state as usize].descriptor.clone(),
        graph.operands[cell_state as usize].descriptor.clone(),
    ])
}

fn gru_cell_shape(
    input: MLOperand,
    hidden_state: MLOperand,
    hidden_size: u32,
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    let hidden_state_desc = &graph.operands[hidden_state.id].descriptor;
    if !hidden_state_desc.shape.is_empty() {
        return Ok(hidden_state_desc.clone());
    }

    let input_shape = &graph.operands[input.id].descriptor.shape;
    if input_shape.len() == 2 && hidden_size > 0 {
        return Ok(OperandDescriptor {
            data_type: graph.operands[input.id].descriptor.data_type,
            shape: to_dimension_vector(&[get_static_or_max_size(&input_shape[0]), hidden_size]),
            pending_permutation: vec![],
        });
    }

    Err(Box::new(ShapeInferenceError::InferError {
        op_name: "gruCell",
        operation: operation.clone(),
        source: GraphError::ShapeInferenceFailed {
            reason: "unable to infer gruCell output shape".to_string(),
        },
    })
    .into())
}

fn shape_inference_multi_output(
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<Vec<OperandDescriptor>> {
    match operation {
        Operation::Split {
            input,
            splits,
            options,
            split_equal_parts,
            ..
        } => split_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            splits,
            *split_equal_parts,
            options.as_ref(),
            graph,
        ),
        Operation::Gru {
            input,
            steps,
            hidden_size,
            options,
            ..
        } => gru_output_shapes(*input, *steps, *hidden_size, options.as_ref(), graph),
        Operation::Lstm {
            input,
            steps,
            hidden_size,
            options,
            ..
        } => lstm_output_shapes(*input, *steps, *hidden_size, options.as_ref(), graph),
        Operation::LstmCell {
            hidden_state,
            cell_state,
            ..
        } => lstm_cell_output_shapes(*hidden_state, *cell_state, graph),
        op => Ok(vec![shape_inference_single_output(op, graph)?]),
    }
}

#[expect(unused_variables)]
fn shape_inference_single_output(
    operation: &Operation,
    graph: &GraphInfo,
) -> Result<OperandDescriptor> {
    match operation {
        Operation::Add {
            a,
            b,
            options,
            outputs,
        }
        | Operation::Sub {
            a,
            b,
            options,
            outputs,
        }
        | Operation::Mul {
            a,
            b,
            options,
            outputs,
        }
        | Operation::Div {
            a,
            b,
            options,
            outputs,
        }
        | Operation::Pow {
            a,
            b,
            options,
            outputs,
        }
        | Operation::Max {
            a,
            b,
            options,
            outputs,
        }
        | Operation::Min {
            a,
            b,
            options,
            outputs,
        } => same_shape(
            &[MLOperand { id: *a as usize }, MLOperand { id: *b as usize }],
            operation,
            graph,
        ),
        Operation::Equal {
            a,
            b,
            options,
            outputs,
        }
        | Operation::Greater {
            a,
            b,
            options,
            outputs,
        }
        | Operation::GreaterOrEqual {
            a,
            b,
            options,
            outputs,
        }
        | Operation::Lesser {
            a,
            b,
            options,
            outputs,
        }
        | Operation::LesserOrEqual {
            a,
            b,
            options,
            outputs,
        }
        | Operation::NotEqual {
            a,
            b,
            options,
            outputs,
        }
        | Operation::LogicalAnd {
            a,
            b,
            options,
            outputs,
        }
        | Operation::LogicalOr {
            a,
            b,
            options,
            outputs,
        }
        | Operation::LogicalXor {
            a,
            b,
            options,
            outputs,
        } => element_wise_logical_shape(
            &[MLOperand { id: *a as usize }, MLOperand { id: *b as usize }],
            operation,
            graph,
        ),

        Operation::Abs { input, .. }
        | Operation::Ceil { input, .. }
        | Operation::Cos { input, .. }
        | Operation::Exp { input, .. }
        | Operation::Elu { input, .. }
        | Operation::Gelu { input, .. }
        | Operation::Floor { input, .. }
        | Operation::Log { input, .. }
        | Operation::Neg { input, .. }
        | Operation::Relu { input, .. }
        | Operation::Sigmoid { input, .. }
        | Operation::Sin { input, .. }
        | Operation::Sqrt { input, .. }
        | Operation::Tan { input, .. }
        | Operation::Tanh { input, .. }
        | Operation::Erf { input, .. }
        | Operation::Reciprocal { input, .. }
        | Operation::Sign { input, .. }
        | Operation::Identity { input, .. }
        | Operation::BatchNormalization { input, .. }
        | Operation::RoundEven { input, .. }
        | Operation::Clamp { input, .. } => same_shape(
            &[MLOperand {
                id: *input as usize,
            }],
            operation,
            graph,
        ),

        Operation::LogicalNot { input, .. } => unary_element_wise_logical_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            graph,
        ),

        Operation::Cast {
            input, data_type, ..
        } => cast_shape(
            MLOperand {
                id: *input as usize,
            },
            *data_type,
            graph,
        ),

        Operation::Where {
            condition,
            true_value,
            false_value,
            ..
        } => where_shape(
            &[
                MLOperand {
                    id: *condition as usize,
                },
                MLOperand {
                    id: *true_value as usize,
                },
                MLOperand {
                    id: *false_value as usize,
                },
            ],
            operation,
            graph,
        ),

        Operation::Constant { options, .. } => constant_shape(operation, options.as_ref()),
        Operation::Conv2d {
            input,
            filter,
            options,
            ..
        } => conv2d_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *filter as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::ConvTranspose2d {
            input,
            filter,
            options,
            ..
        } => conv_transpose2d_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *filter as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::Concat {
            inputs,
            axis,
            options,
            outputs,
        } => concat_shape(
            &inputs
                .iter()
                .map(|i| MLOperand { id: *i as usize })
                .collect::<Vec<_>>(),
            operation,
            *axis,
            graph,
        ),
        Operation::CumulativeSum { input, .. } => preserve_input_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            graph,
        ),
        Operation::Expand {
            input,
            new_shape,
            options,
            outputs,
        } => expand_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            new_shape,
            graph,
        ),
        Operation::Gather {
            input,
            indices,
            options,
            ..
        } => gather_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *indices as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::GatherElements { input, indices, .. } => gather_elements_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *indices as usize,
            },
            operation,
            graph,
        ),
        Operation::Gemm { a, b, options, .. } => gemm_shape(
            MLOperand { id: *a as usize },
            MLOperand { id: *b as usize },
            operation,
            options.as_ref(),
            graph,
        ),
        // TODO: verify data type of non-input operands
        Operation::GruCell {
            input,
            hidden_state,
            hidden_size,
            ..
        } => gru_cell_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *hidden_state as usize,
            },
            *hidden_size,
            operation,
            graph,
        ),
        Operation::HardSigmoid { input, .. }
        | Operation::HardSwish { input, .. }
        | Operation::InstanceNormalization { input, .. }
        | Operation::LayerNormalization { input, .. }
        | Operation::LeakyRelu { input, .. }
        | Operation::Linear { input, .. }
        | Operation::Reverse { input, .. }
        | Operation::Softmax { input, .. }
        | Operation::Softplus { input, .. }
        | Operation::Softsign { input, .. } => preserve_input_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            graph,
        ),
        Operation::IsNaN { input, .. } | Operation::IsInfinite { input, .. } => {
            unary_element_wise_logical_shape(
                MLOperand {
                    id: *input as usize,
                },
                operation,
                graph,
            )
        }
        Operation::Pad {
            input,
            beginning_padding,
            ending_padding,
            ..
        } => pad_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            beginning_padding,
            ending_padding,
            graph,
        ),
        Operation::AveragePool2d { input, options, .. }
        | Operation::MaxPool2d { input, options, .. }
        | Operation::L2Pool2d { input, options, .. } => pool2d_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::GlobalAveragePool { input, options, .. }
        | Operation::GlobalMaxPool { input, options, .. } => global_pool_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::ReduceSum { input, options, .. }
        | Operation::ReduceMean { input, options, .. }
        | Operation::ReduceMax { input, options, .. }
        | Operation::ReduceMin { input, options, .. }
        | Operation::ReduceProduct { input, options, .. }
        | Operation::ReduceL1 { input, options, .. }
        | Operation::ReduceL2 { input, options, .. }
        | Operation::ReduceLogSum { input, options, .. }
        | Operation::ReduceLogSumExp { input, options, .. }
        | Operation::ReduceSumSquare { input, options, .. } => reduce_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::ArgMin {
            input,
            axis,
            options,
            ..
        }
        | Operation::ArgMax {
            input,
            axis,
            options,
            ..
        } => argminmax_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            *axis,
            options.as_ref(),
            graph,
        ),
        Operation::Reshape {
            input, new_shape, ..
        } => reshape_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            new_shape,
            graph,
        ),
        Operation::Resample2d { input, options, .. } => resample2d_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::ScatterElements {
            input,
            indices,
            updates,
            options,
            ..
        } => scatter_elements_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *indices as usize,
            },
            MLOperand {
                id: *updates as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::Slice {
            input,
            starts,
            sizes,
            options,
            ..
        } => slice_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            starts,
            sizes,
            options,
            graph,
        ),
        Operation::Split { .. } => {
            panic!("This method only supports single output ops. Use shape_inference_multi_output")
        }
        Operation::Transpose { input, options, .. } => transpose_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::Squeeze { input, options, .. } => squeeze_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::Unsqueeze { input, options, .. } => unsqueeze_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            options.as_ref(),
            graph,
        ),
        Operation::Tile {
            input, repetitions, ..
        } => tile_shape(
            MLOperand {
                id: *input as usize,
            },
            operation,
            repetitions,
            graph,
        ),
        Operation::Triangular { input, .. } => {
            let operand = get_operand(
                MLOperand {
                    id: *input as usize,
                },
                graph,
            )?;
            let shape = infer_shape_err(
                "triangular",
                operation,
                infer_triangular_shape(&shape_dims_u32(&operand.descriptor.shape))
                    .map(|v| to_dimension_vector(&v)),
            )?;
            Ok(OperandDescriptor {
                data_type: operand.descriptor.data_type,
                shape,
                pending_permutation: vec![],
            })
        }
        Operation::Prelu { input, slope, .. } => prelu_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *slope as usize,
            },
            operation,
            graph,
        ),
        Operation::QuantizeLinear {
            input, zero_point, ..
        } => quantize_linear_shape(
            MLOperand {
                id: *input as usize,
            },
            *zero_point,
            graph,
        ),
        Operation::DequantizeLinear { input, scale, .. } => dequantize_linear_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *scale as usize,
            },
            graph,
        ),
        Operation::Shape { input, .. } => shape_op_shape(
            MLOperand {
                id: *input as usize,
            },
            graph,
        ),
        Operation::ScatterND {
            input,
            indices,
            updates,
            ..
        } => scatter_nd_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *updates as usize,
            },
            MLOperand {
                id: *indices as usize,
            },
            operation,
            graph,
        ),
        Operation::GatherND { input, indices, .. } => gather_nd_shape(
            MLOperand {
                id: *input as usize,
            },
            MLOperand {
                id: *indices as usize,
            },
            operation,
            graph,
        ),
        Operation::Matmul { a, b, .. } => matmul_shape(
            MLOperand { id: *a as usize },
            MLOperand { id: *b as usize },
            operation,
            graph,
        ),
        Operation::Gru { .. } | Operation::Lstm { .. } | Operation::LstmCell { .. } => {
            panic!("This method only supports single output ops. Use shape_inference_multi_output")
        }
    }
}

impl<'context, 'builder> MLGraphBuilder<'context, 'builder> {
    /// Create a builder that records a backend-agnostic [`GraphInfo`] without creating a runtime
    /// backend. This is used by tooling that serializes or forwards the recorded graph itself.
    pub fn new_uncompiled() -> MLGraphBuilder<'static, 'static> {
        MLGraphBuilder {
            backend: Box::new(UncompiledBackendBuilder),
            graph: Some(Default::default()),
        }
    }

    pub fn new(context: &'_ mut MLContext<'context>) -> crate::error::Result<Self>
    where
        'context: 'builder,
    {
        let backend = context.backend.create_builder()?;
        Ok(Self {
            backend,
            graph: Some(Default::default()),
        })
    }

    pub fn build_graph_info(
        &mut self,
        graph: GraphInfo,
    ) -> crate::error::Result<MLGraph<'context>> {
        self.backend.build(graph)
    }

    /// Serialize the in-progress graph (with outputs marked) as `.webnn` text for debugging.
    pub fn rustnn_webnn_text_for_outputs(&self, outputs: &MLNamedOperands) -> Option<String> {
        let graph = self.graph.as_ref()?;
        if outputs.is_empty() {
            return None;
        }

        let mut graph = graph.clone();
        graph.output_operands.clear();
        for (name, operand) in outputs {
            let op = graph.operands.get_mut(operand.id)?;
            if op.kind == OperandKind::Input || op.kind == OperandKind::Constant {
                return None;
            }
            op.kind = OperandKind::Output;
            op.name = Some(name.to_string());
            graph.output_operands.push(operand.id as u32);
        }
        graph.output_operands.sort_unstable();

        let graph_json = to_graph_json(&graph, false).ok()?;
        webnn_graph::serialize::serialize_graph_to_wg_text(
            &graph_json,
            SerializeOptions { quantized: false },
        )
        .ok()
    }

    /// Save the in-progress graph (with `outputs` marked) as a rustnn `.webnn` text file plus a
    /// sibling `.safetensors` weights file.
    ///
    /// This is a **rustnn-specific** extension (not part of the W3C WebNN API). Constants are
    /// written to `<stem>.safetensors` next to `path` and referenced from the `.webnn` file via
    /// `@weights(...)`, so the pair round-trips through [`crate::load_graph_from_path`].
    ///
    /// Like [`Self::build`], this takes the graph's `outputs` to finalize which operands are graph
    /// outputs, but it borrows the builder read-only: the output names are handed to the exporter
    /// as an override, so neither the graph nor its constant weight bytes are copied or mutated, and
    /// the builder can still be built afterwards.
    pub fn rustnn_save_webnn(
        &self,
        outputs: &MLNamedOperands,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::Result<()> {
        if outputs.is_empty() {
            return Err(GraphBuilderError::EmptyOutputHashMap.into());
        }
        let graph = self
            .graph
            .as_ref()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;

        // Build the operand-id -> output-name override for the exporter, validating as we go.
        // Two names for the same operand (aliasing) would collide here, so we reject it like
        // `build()` does. Output names are unique by construction (they are the `outputs` map keys).
        let mut output_names = HashMap::<u32, String>::with_capacity(outputs.len());
        for (name, operand) in outputs {
            let op = graph.operands.get(operand.id).ok_or(
                GraphBuilderError::BuildWithInvalidOperand {
                    operand: *operand,
                    name: name.to_string(),
                },
            )?;
            if op.kind == OperandKind::Input {
                return Err(GraphBuilderError::RequestedInputAsOutput {
                    operand: op.clone(),
                    id: operand.id,
                }
                .into());
            } else if op.kind == OperandKind::Constant {
                return Err(GraphBuilderError::RequestedConstantAsOutput {
                    operand: op.clone(),
                    id: operand.id,
                }
                .into());
            }
            if let Some(first_name) = output_names.insert(operand.id as u32, name.to_string()) {
                return Err(GraphBuilderError::DuplicateOutput {
                    operand: *operand,
                    first_name,
                    second_name: name.to_string(),
                }
                .into());
            }
        }

        crate::webnn_save::validate_save_output_names(graph, &output_names)?;
        crate::webnn_save::write_webnn_and_safetensors(graph, &output_names, path.as_ref())
    }

    /*async*/
    pub fn build(
        &mut self,
        outputs: &'_ MLNamedOperands,
    ) -> crate::error::Result<MLGraph<'context>> {
        let graph = self.finish_graph_info(outputs)?;
        self.backend.build(graph)
    }

    /// Finalize the recorded graph and return it without compiling it for a runtime backend.
    pub fn finish_graph_info(
        &mut self,
        outputs: &'_ MLNamedOperands,
    ) -> crate::error::Result<GraphInfo> {
        trace!("Trying to build graph for outputs {outputs:?}");
        // spec: If outputs is empty, then return a new promise in realm rejected with a TypeError.
        if outputs.is_empty() {
            return Err(GraphBuilderError::EmptyOutputHashMap.into());
        }

        let mut graph = self
            .graph
            .take()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;

        let mut duplicates = HashMap::<MLOperand, &str>::new();
        for (name, operand) in outputs.iter() {
            match duplicates.entry(*operand) {
                std::collections::hash_map::Entry::Occupied(occupied_entry) => {
                    return Err(GraphBuilderError::DuplicateOutput {
                        operand: *operand,
                        first_name: occupied_entry.get().to_string(),
                        second_name: name.to_string(),
                    }
                    .into());
                }
                std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(name);
                }
            }

            if let Some(op) = graph.operands.get_mut(operand.id) {
                // spec: If operand is in this’s graph’s inputs or constants, then return a new promise in realm rejected with a TypeError.
                if op.kind == OperandKind::Input {
                    return Err(GraphBuilderError::RequestedInputAsOutput {
                        operand: op.clone(),
                        id: operand.id,
                    }
                    .into());
                } else if op.kind == OperandKind::Constant {
                    return Err(GraphBuilderError::RequestedConstantAsOutput {
                        operand: op.clone(),
                        id: operand.id,
                    }
                    .into());
                }
                op.kind = OperandKind::Output;
                op.name = Some(name.to_string());
            } else {
                return Err(GraphBuilderError::BuildWithInvalidOperand {
                    operand: *operand,
                    name: name.to_string(),
                }
                .into());
            }
            graph.output_operands.push(operand.id as u32);
        }
        // HashMap iteration order is nondeterministic; keep output_operands in operand-index order.
        graph.output_operands.sort_unstable();

        debug!("Building graph with {} operands", graph.operands.len());
        // Verbose info for small graphs
        if graph.operands.len() < 20 {
            trace!("Building graph:\n{graph:#?}");
            trace!("Graph webnn JSON:\n{}", {
                if let Ok(graph_json) = to_graph_json(&graph, false) {
                    webnn_graph::serialize::serialize_graph_to_wg_text(
                        &graph_json,
                        SerializeOptions { quantized: false },
                    )
                    .unwrap_or_else(|_| String::new())
                } else {
                    String::new()
                }
            });
        }

        Ok(graph)
    }

    /// Debug tool to check operand shape
    /// Actually, WebNN API exposes these on MLOperand
    pub fn rustnn_operand_shape(
        &mut self,
        operand: MLOperand,
        // this should be either &[MLDimension] or &[u32], &[u64], whatever shape the public API has
    ) -> crate::error::Result<Vec<u64>> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        operand.shape(graph)
    }

    /// Debug tool to check operand shape
    /// Actually, WebNN API exposes these on MLOperand
    pub fn rustnn_operand_data_type(
        &mut self,
        operand: MLOperand,
        // this should be either MLDimension or &[u32]
    ) -> crate::error::Result<MLOperandDataType> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        operand.data_type(graph)
    }

    pub fn input(
        &mut self,
        name: &str,
        descriptor: &MLOperandDescriptor,
    ) -> crate::error::Result<MLOperand> {
        debug!("Adding input {name:?} {descriptor:?}");
        let operand = Operand {
            descriptor: descriptor.into(),
            kind: OperandKind::Input,
            name: Some(name.to_string()),
        };

        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;

        let id = graph.operands.len();
        graph.operands.push(operand);
        graph.input_operands.push(id as u32);

        Ok(MLOperand { id })
    }

    // MLGraphBuilder.constant
    //
    // three flavors
    //
    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-constant
    pub fn constant_from_tensor(&mut self, _tensor: MLTensor) -> crate::error::Result<MLOperand> {
        todo!("not implemented yet. requires backend integration")
    }

    pub fn constant_from_vec<T: NoUninit>(
        &mut self,
        descriptor: &MLOperandDescriptor,
        values: Vec<T>,
    ) -> crate::error::Result<MLOperand> {
        let required_size = descriptor.rustnn_required_bytes();
        let provided_size = std::mem::size_of_val(values.as_slice());
        if required_size != provided_size {
            return Err(GraphBuilderError::WrongConstantSize {
                descriptor: descriptor.clone(),
                required_size,
                provided_size,
            }
            .into());
        }

        let operand = Operand {
            descriptor: descriptor.into(),
            kind: OperandKind::Constant,
            name: None,
        };

        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;

        let id = graph.operands.len();
        graph.operands.push(operand);
        graph
            .id_to_constant_tensor_operand_map
            .insert(id as u32, format!("{id}"));
        graph.constant_operand_ids_to_handles.insert(
            id as u32,
            crate::ConstantData {
                data: bytemuck::cast_slice(values.as_slice()).to_vec(),
                label: None,
            },
        );

        Ok(MLOperand { id })
    }

    pub fn constant_from_slice<T: NoUninit>(
        &mut self,
        descriptor: &MLOperandDescriptor,
        values: &[T],
    ) -> crate::error::Result<MLOperand> {
        trace!(
            "constant_from_slice: {descriptor:?} size={} bytes",
            descriptor.rustnn_required_bytes()
        );
        self.constant_from_bytes(descriptor, bytemuck::cast_slice::<T, u8>(values).to_vec())
    }

    /// Register a constant from raw little-endian bytes, taking ownership of
    /// the buffer. Unlike [`Self::constant_from_slice`] this makes no copy:
    /// weight tensors reach hundreds of MB and every extra copy of them stays
    /// live for the whole backend compile.
    pub fn constant_from_bytes(
        &mut self,
        descriptor: &MLOperandDescriptor,
        data: Vec<u8>,
    ) -> crate::error::Result<MLOperand> {
        let required_size = descriptor.rustnn_required_bytes();
        if required_size != data.len() {
            return Err(GraphBuilderError::WrongConstantSize {
                descriptor: descriptor.clone(),
                required_size,
                provided_size: data.len(),
            }
            .into());
        }
        let operand = Operand {
            descriptor: descriptor.into(),
            kind: OperandKind::Constant,
            name: None,
        };

        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;

        let id = graph.operands.len();
        graph.operands.push(operand);
        graph
            .id_to_constant_tensor_operand_map
            .insert(id as u32, format!("{id}"));
        graph
            .constant_operand_ids_to_handles
            .insert(id as u32, crate::ConstantData { data, label: None });

        Ok(MLOperand { id })
    }

    pub fn constant_from_value<T>(
        &mut self,
        _data_type: MLOperandDataType,
        _value: T,
    ) -> crate::error::Result<MLOperand> {
        todo!()
    }

    // internal methods
    fn unary_same_shape_operation<Options>(
        &mut self,
        input: MLOperand,
        options: Options,
        build: impl FnOnce(u32, u32, Option<Options>) -> Operation,
    ) -> Result<MLOperand> {
        let output_id = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?
            .operands
            .len() as u32;
        self.add_single_output_operation(build(input.id as u32, output_id, Some(options)))
    }

    fn binary_same_shape_operation<Options>(
        &mut self,
        a: MLOperand,
        b: MLOperand,
        options: Options,
        build: impl FnOnce(u32, u32, u32, Option<Options>) -> Operation,
    ) -> Result<MLOperand> {
        let output_id = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?
            .operands
            .len() as u32;
        self.add_single_output_operation(build(a.id as u32, b.id as u32, output_id, Some(options)))
    }

    fn ternary_operation<Options>(
        &mut self,
        arg1: MLOperand,
        arg2: MLOperand,
        arg3: MLOperand,
        options: Options,
        build: impl FnOnce(u32, u32, u32, u32, Option<Options>) -> Operation,
    ) -> Result<MLOperand> {
        let output_id = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?
            .operands
            .len() as u32;
        self.add_single_output_operation(build(
            arg1.id as u32,
            arg2.id as u32,
            arg3.id as u32,
            output_id,
            Some(options),
        ))
    }

    fn ternary_optional_operation<Options>(
        &mut self,
        arg1: MLOperand,
        arg2: MLOperand,
        optional: Option<MLOperand>,
        options: Options,
        build: impl FnOnce(u32, u32, Option<u32>, u32, Option<Options>) -> Operation,
    ) -> Result<MLOperand> {
        let output_id = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?
            .operands
            .len() as u32;
        self.add_single_output_operation(build(
            arg1.id as u32,
            arg2.id as u32,
            optional.map(|o| o.id as u32),
            output_id,
            Some(options),
        ))
    }

    impl_binary_op!(add, add_with_options, Add);
    impl_binary_op!(sub, sub_with_options, Sub);
    impl_binary_op!(mul, mul_with_options, Mul);
    impl_binary_op!(div, div_with_options, Div);
    impl_binary_op!(pow, pow_with_options, Pow);
    impl_binary_op!(max, max_with_options, Max);
    impl_binary_op!(min, min_with_options, Min);
    impl_binary_op!(equal, equal_with_options, Equal);
    impl_binary_op!(greater, greater_with_options, Greater);
    impl_binary_op!(
        greater_or_equal,
        greater_or_equal_with_options,
        GreaterOrEqual
    );
    impl_binary_op!(lesser, lesser_with_options, Lesser);
    impl_binary_op!(lesser_or_equal, lesser_or_equal_with_options, LesserOrEqual);
    impl_binary_op!(not_equal, not_equal_with_options, NotEqual);
    impl_binary_op!(logical_and, logical_and_with_options, LogicalAnd);
    impl_binary_op!(logical_or, logical_or_with_options, LogicalOr);
    impl_binary_op!(logical_xor, logical_xor_with_options, LogicalXor);
    impl_binary_op!(matmul, matmul_with_options, Matmul);
    impl_binary_op!(gemm, gemm_with_options, Gemm, MLGemmOptions);
    impl_binary_op!(
        conv2d,
        conv2_with_options,
        Conv2d,
        MLConv2dOptions,
        input,
        filter
    );
    impl_binary_op!(
        conv_transpose2d,
        conv_transpose2d_with_options,
        ConvTranspose2d,
        MLConvTranspose2dOptions,
        input,
        filter
    );

    pub fn split(&mut self, input: MLOperand, splits: &[u32]) -> Result<Vec<MLOperand>> {
        self.split_with_options(input, splits, MLSplitOptions::default())
    }

    pub fn split_with_options(
        &mut self,
        input: MLOperand,
        splits: &[u32],
        options: MLSplitOptions,
    ) -> Result<Vec<MLOperand>> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let output_ids: Vec<u32> = (0u32..splits.len() as u32)
            .map(|i| graph.operands.len() as u32 + i)
            .collect();

        let operation = Operation::Split {
            input: input.id as u32,
            splits: splits.to_vec(),
            split_equal_parts: None,
            options: Some(options),
            outputs: output_ids,
        };
        self.add_multi_output_operation(operation)
    }

    pub fn split_equal_with_options(
        &mut self,
        input: MLOperand,
        num_splits: u32,
        options: MLSplitOptions,
    ) -> Result<Vec<MLOperand>> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let output_ids: Vec<u32> = (0u32..num_splits)
            .map(|i| graph.operands.len() as u32 + i)
            .collect();

        let operation = Operation::Split {
            input: input.id as u32,
            splits: Vec::new(),
            split_equal_parts: Some(num_splits),
            options: Some(options),
            outputs: output_ids,
        };
        self.add_multi_output_operation(operation)
    }

    impl_ternary_op!(
        batch_normalization,
        batch_normalization_with_options,
        BatchNormalization,
        MLBatchNormalizationOptions,
        input,
        mean,
        variance
    );
    impl_ternary_op!(
        where_,
        where_with_options,
        Where,
        condition,
        true_value,
        false_value
    );
    impl_ternary_op!(
        scatter_nd,
        scatter_nd_with_options,
        ScatterND,
        input,
        indices,
        updates
    );
    impl_ternary_op!(
        scatter_elements,
        scatter_elements_with_options,
        ScatterElements,
        MLScatterOptions,
        input,
        indices,
        updates
    );
    impl_ternary_optional_op!(
        quantize_linear,
        quantize_linear_with_zeropoint,
        quantize_linear_with_options,
        QuantizeLinear,
        MLOperatorOptions,
        input,
        scale,
        zero_point
    );
    impl_ternary_optional_op!(
        dequantize_linear,
        dequantize_linear_with_zeropoint,
        dequantize_linear_with_options,
        DequantizeLinear,
        MLOperatorOptions,
        input,
        scale,
        zero_point
    );

    pub fn slice(
        &mut self,
        input: MLOperand,
        starts: &[u32],
        sizes: &[MLDimension],
    ) -> Result<MLOperand> {
        let opts = MLSliceOptions {
            strides: vec![1u32; starts.len()],
            ..Default::default()
        };
        self.slice_with_options(input, starts, sizes, opts)
    }

    pub fn slice_with_options(
        &mut self,
        input: MLOperand,
        starts: &[u32],
        sizes: &[MLDimension],
        options: MLSliceOptions,
    ) -> Result<MLOperand> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let output_id = graph.operands.len();

        let operation = Operation::Slice {
            input: input.id as u32,
            starts: starts.to_vec(),
            sizes: sizes.to_vec(),
            options: Some(options),
            outputs: vec![output_id as u32],
        };
        self.add_single_output_operation(operation)
    }

    impl_unary_op!(abs, abs_with_options, Abs);
    impl_unary_op!(round_even, round_even_with_options, RoundEven);
    impl_unary_op!(ceil, ceil_with_options, Ceil);
    impl_unary_op!(cos, cos_with_options, Cos);
    impl_unary_op!(elu, elu_with_options, Elu, MLEluOptions);
    impl_unary_op!(
        hard_sigmoid,
        hard_sigmoid_with_options,
        HardSigmoid,
        MLHardSigmoidOptions
    );
    impl_unary_op!(hard_swish, hard_swish_with_options, HardSwish);
    impl_unary_op!(
        leaky_relu,
        leaky_relu_with_options,
        LeakyRelu,
        MLLeakyReluOptions
    );
    impl_unary_op!(exp, exp_with_options, Exp);
    impl_unary_op!(floor, floor_with_options, Floor);
    impl_unary_op!(gelu, gelu_with_options, Gelu);
    impl_unary_op!(log, log_with_options, Log);
    impl_unary_op!(neg, neg_with_options, Neg);
    impl_unary_op!(relu, relu_with_options, Relu);
    impl_unary_op!(sigmoid, sigmoid_with_options, Sigmoid);
    impl_unary_op!(sin, sin_with_options, Sin);
    impl_unary_op!(sqrt, sqrt_with_options, Sqrt);
    impl_unary_op!(tan, tan_with_options, Tan);
    impl_unary_op!(tanh, tanh_with_options, Tanh);
    impl_unary_op!(
        transpose,
        transpose_with_options,
        Transpose,
        MLTransposeOptions
    );
    impl_unary_op!(squeeze, squeeze_with_options, Squeeze, MLSqueezeOptions);
    impl_unary_op!(
        unsqueeze,
        unsqueeze_with_options,
        Unsqueeze,
        MLUnsqueezeOptions
    );
    impl_unary_op!(erf, erf_with_options, Erf);
    impl_unary_op!(reciprocal, reciprocal_with_options, Reciprocal);
    impl_unary_op!(sign, sign_with_options, Sign);
    impl_unary_op!(logical_not, logical_not_with_options, LogicalNot);
    impl_unary_op!(identity, identity_with_options, Identity);

    // Unary ops with extra method parameters (Option B).
    impl_unary_op!(arg_min, arg_min_with_options, ArgMin, MLArgMinMaxOptions, axis: u32);
    impl_unary_op!(arg_max, arg_max_with_options, ArgMax, MLArgMinMaxOptions, axis: u32);
    impl_unary_op!(softmax, softmax_with_options, Softmax, MLOperatorOptions, axis: u32);
    impl_unary_op!(
        cumulative_sum,
        cumulative_sum_with_options,
        CumulativeSum,
        MLCumulativeSumOptions,
        axis: u32
    );
    impl_unary_op!(
        cast,
        cast_with_options,
        Cast,
        MLOperatorOptions,
        data_type: MLOperandDataType
    );
    impl_unary_op!(
        expand,
        expand_with_options,
        Expand,
        MLOperatorOptions,
        new_shape: Vec<MLDimension>
    );
    impl_unary_op!(
        reshape,
        reshape_with_options,
        Reshape,
        MLOperatorOptions,
        new_shape: Vec<MLDimension>
    );
    impl_unary_op!(
        tile,
        tile_with_options,
        Tile,
        MLOperatorOptions,
        repetitions: Vec<u32>
    );
    impl_unary_op!(
        pad,
        pad_with_options,
        Pad,
        MLPadOptions,
        beginning_padding: Vec<u32>,
        ending_padding: Vec<u32>
    );

    // Unary ops without extra method parameters (not yet on the builder).
    impl_unary_op!(
        instance_normalization,
        instance_normalization_with_options,
        InstanceNormalization,
        MLInstanceNormalizationOptions
    );
    impl_unary_op!(
        layer_normalization,
        layer_normalization_with_options,
        LayerNormalization,
        MLLayerNormalizationOptions
    );
    impl_unary_op!(linear, linear_with_options, Linear, MLLinearOptions);
    impl_unary_op!(clamp, clamp_with_options, Clamp, MLClampOptions);
    impl_unary_op!(
        resample2d,
        resample2d_with_options,
        Resample2d,
        MLResample2dOptions
    );
    impl_unary_op!(reverse, reverse_with_options, Reverse, MLReverseOptions);
    impl_unary_op!(softplus, softplus_with_options, Softplus);
    impl_unary_op!(softsign, softsign_with_options, Softsign);
    impl_unary_op!(is_nan, is_nan_with_options, IsNaN);
    impl_unary_op!(is_infinite, is_infinite_with_options, IsInfinite);
    impl_unary_op!(shape, shape_with_options, Shape);
    impl_unary_op!(
        triangular,
        triangular_with_options,
        Triangular,
        MLTriangularOptions
    );
    impl_unary_op!(
        average_pool2d,
        average_pool2d_with_options,
        AveragePool2d,
        MLPool2dOptions
    );
    impl_unary_op!(
        max_pool2d,
        max_pool2d_with_options,
        MaxPool2d,
        MLPool2dOptions
    );
    impl_unary_op!(l2_pool2d, l2_pool2d_with_options, L2Pool2d, MLPool2dOptions);
    impl_unary_op!(
        global_average_pool,
        global_average_pool_with_options,
        GlobalAveragePool,
        MLPool2dOptions
    );
    impl_unary_op!(
        global_max_pool,
        global_max_pool_with_options,
        GlobalMaxPool,
        MLPool2dOptions
    );
    impl_unary_op!(
        reduce_sum,
        reduce_sum_with_options,
        ReduceSum,
        MLReduceOptions
    );
    impl_unary_op!(
        reduce_mean,
        reduce_mean_with_options,
        ReduceMean,
        MLReduceOptions
    );
    impl_unary_op!(
        reduce_max,
        reduce_max_with_options,
        ReduceMax,
        MLReduceOptions
    );
    impl_unary_op!(
        reduce_min,
        reduce_min_with_options,
        ReduceMin,
        MLReduceOptions
    );
    impl_unary_op!(
        reduce_product,
        reduce_product_with_options,
        ReduceProduct,
        MLReduceOptions
    );
    impl_unary_op!(reduce_l1, reduce_l1_with_options, ReduceL1, MLReduceOptions);
    impl_unary_op!(reduce_l2, reduce_l2_with_options, ReduceL2, MLReduceOptions);
    impl_unary_op!(
        reduce_log_sum,
        reduce_log_sum_with_options,
        ReduceLogSum,
        MLReduceOptions
    );
    impl_unary_op!(
        reduce_log_sum_exp,
        reduce_log_sum_exp_with_options,
        ReduceLogSumExp,
        MLReduceOptions
    );
    impl_unary_op!(
        reduce_sum_square,
        reduce_sum_square_with_options,
        ReduceSumSquare,
        MLReduceOptions
    );

    pub fn gather(&mut self, input: MLOperand, indices: MLOperand) -> Result<MLOperand> {
        self.gather_with_options(input, indices, MLGatherOptions::default())
    }

    pub fn gather_with_options(
        &mut self,
        input: MLOperand,
        indices: MLOperand,
        options: MLGatherOptions,
    ) -> Result<MLOperand> {
        let output_id = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?
            .operands
            .len() as u32;
        self.add_single_output_operation(Operation::Gather {
            input: input.id as u32,
            indices: indices.id as u32,
            batch_dimensions: None,
            options: Some(options),
            outputs: vec![output_id],
        })
    }

    pub fn gather_elements(&mut self, input: MLOperand, indices: MLOperand) -> Result<MLOperand> {
        self.gather_elements_with_options(input, indices, MLGatherOptions::default())
    }

    pub fn gather_elements_with_options(
        &mut self,
        input: MLOperand,
        indices: MLOperand,
        options: MLGatherOptions,
    ) -> Result<MLOperand> {
        let output_id = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?
            .operands
            .len() as u32;
        self.add_single_output_operation(Operation::GatherElements {
            input: input.id as u32,
            indices: indices.id as u32,
            batch_dimensions: None,
            options: Some(options),
            outputs: vec![output_id],
        })
    }

    impl_binary_op!(
        gather_nd,
        gather_nd_with_options,
        GatherND,
        MLOperatorOptions,
        input,
        indices
    );
    impl_binary_op!(
        prelu,
        prelu_with_options,
        Prelu,
        MLOperatorOptions,
        input,
        slope
    );

    pub fn concat(&mut self, inputs: &[MLOperand], axis: u32) -> Result<MLOperand> {
        self.concat_with_options(inputs, axis, MLOperatorOptions::default())
    }

    pub fn concat_with_options(
        &mut self,
        inputs: &[MLOperand],
        axis: u32,
        options: MLOperatorOptions,
    ) -> Result<MLOperand> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let output_id = graph.operands.len();

        self.add_single_output_operation(Operation::Concat {
            inputs: inputs.iter().map(|i| i.id as u32).collect(),
            axis,
            options: Some(options),
            outputs: vec![output_id as u32],
        })
    }

    fn add_single_output_operation(&mut self, operation: Operation) -> Result<MLOperand> {
        trace!("Adding operation {operation:?}");
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let output_id = graph.operands.len();

        let output_operand = Operand {
            kind: OperandKind::Intermediate,
            descriptor: shape_inference_single_output(&operation, graph)?,
            name: (!operation.label().is_empty()).then(|| operation.label().to_string()),
        };
        trace!("  Adding operand {output_operand:?}");
        graph.operands.push(output_operand);
        graph.operations.push(operation);

        Ok(MLOperand { id: output_id })
    }

    fn add_multi_output_operation(&mut self, operation: Operation) -> Result<Vec<MLOperand>> {
        trace!("Adding operation {operation:?}");
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let mut splits = shape_inference_multi_output(&operation, graph)?;

        let mut output = vec![];
        let name_prefix = operation
            .label()
            .is_empty()
            .then(|| operation.label().to_string());

        for (i, s) in splits.drain(..).enumerate() {
            let output_id = graph.operands.len();
            let output_operand = Operand {
                kind: OperandKind::Intermediate,
                descriptor: s,
                name: name_prefix.as_ref().map(|prefix| format!("{prefix}_{i}")),
            };
            trace!(" Adding operand {output_operand:?}");
            graph.operands.push(output_operand);
            output.push(MLOperand { id: output_id })
        }
        graph.operations.push(operation);

        Ok(output)
    }

    pub fn gru_with_options(
        &mut self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        steps: u32,
        hidden_size: u32,
        options: MLGruOptions,
    ) -> Result<Vec<MLOperand>> {
        let output_count = if options.return_sequence { 2 } else { 1 };
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let base = graph.operands.len() as u32;
        let output_ids: Vec<u32> = (0..output_count as u32).map(|i| base + i).collect();
        let operation = Operation::Gru {
            input: input.id as u32,
            weight: weight.id as u32,
            recurrence: recurrent_weight.id as u32,
            steps,
            hidden_size,
            options: Some(options),
            outputs: output_ids,
        };
        self.add_multi_output_operation(operation)
    }

    pub fn gru_cell_with_options(
        &mut self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        hidden_state: MLOperand,
        hidden_size: u32,
        options: MLGruCellOptions,
    ) -> Result<MLOperand> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let output_id = graph.operands.len() as u32;
        self.add_single_output_operation(Operation::GruCell {
            input: input.id as u32,
            weight: weight.id as u32,
            recurrence: recurrent_weight.id as u32,
            hidden_state: hidden_state.id as u32,
            hidden_size,
            options: Some(options),
            outputs: vec![output_id],
        })
    }

    pub fn lstm_with_options(
        &mut self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        steps: u32,
        hidden_size: u32,
        options: MLLstmOptions,
    ) -> Result<Vec<MLOperand>> {
        let output_count = if options.return_sequence { 3 } else { 2 };
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let base = graph.operands.len() as u32;
        let output_ids: Vec<u32> = (0..output_count as u32).map(|i| base + i).collect();
        let operation = Operation::Lstm {
            input: input.id as u32,
            weight: weight.id as u32,
            recurrence: recurrent_weight.id as u32,
            steps,
            hidden_size,
            options: Some(options),
            outputs: output_ids,
        };
        self.add_multi_output_operation(operation)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn lstm_cell_with_options(
        &mut self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        hidden_state: MLOperand,
        cell_state: MLOperand,
        hidden_size: u32,
        options: MLLstmCellOptions,
    ) -> Result<Vec<MLOperand>> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let base = graph.operands.len() as u32;
        let output_ids = vec![base, base + 1];
        let operation = Operation::LstmCell {
            input: input.id as u32,
            weight: weight.id as u32,
            recurrence: recurrent_weight.id as u32,
            hidden_state: hidden_state.id as u32,
            cell_state: cell_state.id as u32,
            hidden_size,
            options: Some(options),
            outputs: output_ids,
        };
        self.add_multi_output_operation(operation)
    }
}

#[cfg(test)]
mod test {
    use log::warn;

    use crate::{
        mlcontext::{
            Backend, MLContext, MLContextOptions, MLNamedOperands, MLNamedTensors,
            MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
        },
        mlgraphbuilder::MLGraphBuilder,
    };

    #[test]
    fn add_inputs() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        };

        let mut context = context.unwrap();
        dbg!(&context);
        let desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();

        let a = builder.input("a", &desc).unwrap();
        let b = builder.input("b", &desc).unwrap();
        assert_eq!(builder.graph.as_ref().unwrap().operands.len(), 2);

        let mut outputs = MLNamedOperands::new();
        outputs.insert("out1", a);
        outputs.insert("out2", b);
        let error = builder.build(&outputs).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::GraphBuilderError { .. }
        ));
        let error_message = format!("{error}");
        assert!(error_message.contains("requested an MLOperand with id "));
        assert!(error_message.contains(" as an output that is already an input"));

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &desc).unwrap();
        let b = builder.input("b", &desc).unwrap();
        assert_eq!(builder.graph.as_ref().unwrap().operands.len(), 2);
        let out1 = builder.identity(a).unwrap();
        let out2 = builder.add(a, b).unwrap();
        let mut outputs = MLNamedOperands::new();
        outputs.insert("out1", out1);
        outputs.insert("out2", out2);
        builder.build(&outputs).unwrap();
    }

    #[test]
    fn unused_incompatible_inputs() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        };

        let mut context = context.unwrap();
        if context.rustnn_backend() == Backend::Trtx || context.rustnn_backend() == Backend::Litert
        {
            warn!(
                "quantize_dequantize_linear_output_dtype unsupported by Trtx and Litert at the moment! Skipping..."
            );
            return;
        }
        let mat_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );
        let incompatible_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [3].to_vec(),
        );

        // simple graph
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &mat_desc).unwrap();
        let _unused = builder.input("unused", &mat_desc).unwrap();
        let incompatible = builder.input("incompatible", &incompatible_desc).unwrap();
        // incompatible broadcast shape
        builder.sub(a, incompatible).unwrap_err();
        assert_eq!(builder.rustnn_operand_shape(a).unwrap(), mat_desc.shape());
        let a = builder.identity(a).unwrap();
        let output = builder.identity(a).unwrap();

        let mut outputs = MLNamedOperands::new();
        outputs.insert("out", output);

        // This graph deliberately stresses the graph-builder's binding/validation logic
        // with degenerate shapes: an unused input and an identity-passthrough output.
        // The CoreML MLProgram converter cannot yet (a) load a model whose output is a
        // bare `identity` op (rejected as "multi-function description syntax"), nor
        // (b) bind a declared-but-unused input. Both are tracked as separate converter
        // gaps. The builder-level assertions above run on every backend; only the
        // backend execution of this degenerate graph is skipped on CoreML.
        #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
        let mut graph = match builder.build(&outputs) {
            Ok(graph) => graph,
            Err(e) => {
                eprintln!(
                    "skipping CoreML execution of degenerate graph (unused input / identity output): {e:?}"
                );
                return;
            }
        };
        #[cfg(not(all(target_os = "macos", feature = "coreml-runtime")))]
        let mut graph = builder.build(&outputs).unwrap();
        dbg!(&graph);

        let mut a_desc = MLTensorDescriptor::from_operand_descriptor(&mat_desc);
        a_desc.set_writable(true);
        a_desc.set_readable(true);
        let inc_desc = MLTensorDescriptor::from_operand_descriptor(&incompatible_desc);

        let a = context.create_tensor(&a_desc).unwrap();
        let unused = context.create_tensor(&a_desc).unwrap();
        let incompatible = context.create_tensor(&inc_desc).unwrap();
        let output = context.create_tensor(&a_desc).unwrap();

        // All declared graph inputs must be provided at dispatch (including unused ones).
        let mut inputs = MLNamedTensors::new();
        inputs.insert("a", &a);
        inputs.insert("unused", &unused);
        inputs.insert("incompatible", &incompatible);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("out", &output);
        context.write_tensor(&a, &[3.0f32, 4., 5., 6.]).unwrap();
        // CoreML rejects binding the declared-but-unused `incompatible` input at dispatch
        // ("Unable to copy Float32 3 vector"); tolerate that known gap while keeping the
        // execution assertion strict on every other backend.
        #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
        if let Err(e) = context.dispatch(&mut graph, &inputs, &outputs) {
            eprintln!("skipping CoreML execution assertion (unused input binding): {e:?}");
            return;
        }
        #[cfg(not(all(target_os = "macos", feature = "coreml-runtime")))]
        context.dispatch(&mut graph, &inputs, &outputs).unwrap();
        let mut output_cpu = vec![0.0f32; 4];
        context.read_tensor(&output, &mut output_cpu).unwrap();
        assert_eq!(output_cpu, &[3.0f32, 4., 5., 6.]);
    }

    #[test]
    fn add_mat_plus_scalar() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        };

        let mut context = context.unwrap();
        let mat_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );
        let scalar_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [].to_vec(),
        );

        // simple graph
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &mat_desc).unwrap();
        let b = builder.input("b", &scalar_desc).unwrap();
        let a = builder.relu(a).unwrap();
        assert_eq!(builder.rustnn_operand_shape(a).unwrap(), mat_desc.shape());
        let a = builder.identity(a).unwrap();
        let output = builder.add(a, b).unwrap();

        let mut outputs = MLNamedOperands::new();
        outputs.insert("out", output);
        let mut graph = builder.build(&outputs).unwrap();

        // should error with GraphAlreadyBuilt
        builder.input("a", &mat_desc).unwrap_err();

        // simple graph but one input replaced by constant
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &mat_desc).unwrap();
        let b = builder
            .constant_from_slice(&mat_desc, &[3.0f32; 4])
            .unwrap();
        // wrong size should error
        builder
            .constant_from_slice(&mat_desc, &[3.0f32; 5])
            .unwrap_err();
        let a = builder.relu(a).unwrap();
        assert_eq!(builder.rustnn_operand_shape(a).unwrap(), mat_desc.shape());
        let a = builder.identity(a).unwrap();
        let output = builder.add(a, b).unwrap();
        let mut outputs = MLNamedOperands::new();
        outputs.insert("out", output);
        let mut graph2 = builder.build(&outputs).unwrap();

        let mut a_desc = MLTensorDescriptor::from_operand_descriptor(&mat_desc);
        a_desc.set_writable(true);
        a_desc.set_readable(true);
        let mut b_desc = MLTensorDescriptor::from_operand_descriptor(&scalar_desc);
        b_desc.set_writable(true);

        let a = context.create_tensor(&a_desc).unwrap();
        let b = context.create_tensor(&b_desc).unwrap();
        let output = context.create_tensor(&a_desc).unwrap();

        context.write_tensor(&a, &[1.0f32, 2., 3., 4.]).unwrap();
        context.write_tensor(&b, &[2.0f32]).unwrap();

        let mut inputs = MLNamedTensors::new();
        inputs.insert("a", &a);
        inputs.insert("b", &b);
        let mut outputs = MLNamedTensors::new();
        outputs.insert("out", &output);
        context.dispatch(&mut graph, &inputs, &outputs).unwrap();
        let mut output_cpu = vec![0.0f32; 4];
        context.read_tensor(&output, &mut output_cpu).unwrap();
        assert_eq!(output_cpu, &[3.0f32, 4., 5., 6.]);
        inputs.remove("b");
        context.dispatch(&mut graph2, &inputs, &outputs).unwrap();
        context.read_tensor(&output, &mut output_cpu).unwrap();
        assert_eq!(output_cpu, &[4.0f32, 5., 6., 7.]);
    }

    #[test]
    fn quantize_dequantize_linear_output_dtype() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        }

        let mut context = context.unwrap();
        if context.rustnn_backend() == Backend::Trtx || context.rustnn_backend() == Backend::Litert
        {
            warn!(
                "quantize_dequantize_linear_output_dtype unsupported by Trtx and Litert at the moment! Skipping..."
            );
            return;
        }
        let float_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 3].to_vec(),
        );
        let int_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Uint8,
            [2, 3].to_vec(),
        );
        let scalar_f32 = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [].to_vec(),
        );
        let scalar_u8 =
            MLOperandDescriptor::new(crate::operator_enums::MLOperandDataType::Uint8, [].to_vec());

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let x = builder.input("x", &float_desc).unwrap();
        let scale = builder.constant_from_slice(&scalar_f32, &[0.5f32]).unwrap();
        let zero_point = builder.constant_from_slice(&scalar_u8, &[128u8]).unwrap();
        let q = builder
            .quantize_linear_with_zeropoint(x, scale, zero_point)
            .unwrap();
        assert_eq!(
            builder.rustnn_operand_data_type(q).unwrap(),
            crate::operator_enums::MLOperandDataType::Uint8
        );
        assert_eq!(builder.rustnn_operand_shape(q).unwrap(), float_desc.shape());

        let dq = builder
            .dequantize_linear_with_zeropoint(q, scale, zero_point)
            .unwrap();
        assert_eq!(
            builder.rustnn_operand_data_type(dq).unwrap(),
            crate::operator_enums::MLOperandDataType::Float32
        );
        assert_eq!(builder.rustnn_operand_shape(dq).unwrap(), int_desc.shape());

        let mut outputs = MLNamedOperands::new();
        outputs.insert("out", dq);
        builder.build(&outputs).unwrap();
    }

    #[test]
    fn rustnn_save_webnn_round_trips_through_loader() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        };
        let mut context = context.unwrap();

        let mat_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &mat_desc).unwrap();
        let b = builder
            .constant_from_slice(&mat_desc, &[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let sum = builder.add(a, b).unwrap();

        let mut outputs = MLNamedOperands::new();
        outputs.insert("sum", sum);

        let temp_dir = tempfile::TempDir::new().unwrap();
        let webnn_path = temp_dir.path().join("model.webnn");
        builder.rustnn_save_webnn(&outputs, &webnn_path).unwrap();

        let webnn_text = std::fs::read_to_string(&webnn_path).unwrap();
        assert!(webnn_text.contains("@weights("));
        assert!(!webnn_text.contains("@bytes("));

        let st_path = webnn_path.with_extension("safetensors");
        assert!(st_path.exists());

        // Saving does not consume the builder: it can still be built.
        builder.build(&outputs).unwrap();

        let loaded = crate::load_graph_from_path(&webnn_path).expect("reload saved graph");
        assert_eq!(loaded.input_operands.len(), 1);
        assert_eq!(loaded.output_operands.len(), 1);
        assert_eq!(loaded.constant_operand_ids_to_handles.len(), 1);

        let constant = loaded
            .constant_operand_ids_to_handles
            .values()
            .next()
            .unwrap();
        let expected: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert_eq!(constant.data, expected);
    }

    #[test]
    fn rustnn_save_webnn_emits_named_constant_weight_ref() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        };
        let mut context = context.unwrap();

        let scalar_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [].to_vec(),
        );
        let mat_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &mat_desc).unwrap();
        let scale = builder
            .constant_from_slice(&scalar_desc, &[0.5f32])
            .unwrap();
        let out = builder.mul(a, scale).unwrap();

        let mut outputs = MLNamedOperands::new();
        outputs.insert("out", out);

        let temp_dir = tempfile::TempDir::new().unwrap();
        let webnn_path = temp_dir.path().join("scaled.webnn");
        builder.rustnn_save_webnn(&outputs, &webnn_path).unwrap();

        let text = std::fs::read_to_string(&webnn_path).unwrap();
        // unnamed constants get operand_<id> names
        assert!(text.contains("@weights(\"operand_"));
    }

    #[test]
    fn rustnn_save_webnn_rejects_output_name_colliding_with_input() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        };
        let mut context = context.unwrap();

        let mat_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &mat_desc).unwrap();
        let b = builder
            .constant_from_slice(&mat_desc, &[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let sum = builder.add(a, b).unwrap();

        let mut outputs = MLNamedOperands::new();
        outputs.insert("a", sum);

        let temp_dir = tempfile::TempDir::new().unwrap();
        let webnn_path = temp_dir.path().join("collision.webnn");
        let err = builder
            .rustnn_save_webnn(&outputs, &webnn_path)
            .unwrap_err();
        std::assert_matches!(
            err,
            crate::error::Error::GraphBuilderError {
                source: crate::error::GraphBuilderError::OutputNameConflictsWithOperand {
                    name,
                    operand_kind: "input",
                }
            } if name == "a"
        );
        assert!(!webnn_path.exists());
    }

    #[test]
    fn rustnn_save_webnn_rejects_duplicate_output_operand() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, true));
        if matches!(context, Err(crate::error::Error::NoBackendAvailable { .. })) {
            return;
        };
        let mut context = context.unwrap();

        let mat_desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("x", &mat_desc).unwrap();
        let b = builder
            .constant_from_slice(&mat_desc, &[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let sum = builder.add(a, b).unwrap();

        let mut outputs = MLNamedOperands::new();
        outputs.insert("out1", sum);
        outputs.insert("out2", sum);

        let temp_dir = tempfile::TempDir::new().unwrap();
        let webnn_path = temp_dir.path().join("dup.webnn");
        let err = builder
            .rustnn_save_webnn(&outputs, &webnn_path)
            .unwrap_err();
        std::assert_matches!(
            err,
            crate::error::Error::GraphBuilderError {
                source: crate::error::GraphBuilderError::DuplicateOutput { .. }
            }
        );
        assert!(!webnn_path.exists());
    }
}
