/// Shape inference and validation for WebNN operations
use crate::error::GraphError;

/// Compute the broadcasted shape for two operands following NumPy broadcasting rules
///
/// Broadcasting rules:
/// 1. If arrays have different ranks, prepend 1s to the smaller rank
/// 2. Two dimensions are compatible if they are equal or one of them is 1
/// 3. Output shape is the maximum of each dimension
pub fn broadcast_shapes(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    let max_rank = shape_a.len().max(shape_b.len());
    let mut result = Vec::with_capacity(max_rank);

    // Iterate from right to left (least significant dimension first)
    for i in 0..max_rank {
        let dim_a = if i < shape_a.len() {
            shape_a[shape_a.len() - 1 - i]
        } else {
            1
        };

        let dim_b = if i < shape_b.len() {
            shape_b[shape_b.len() - 1 - i]
        } else {
            1
        };

        if dim_a == dim_b || dim_a == 1 || dim_b == 1 {
            result.push(dim_a.max(dim_b));
        } else {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "Incompatible shapes for broadcasting: {:?} and {:?} (dimension {} incompatible: {} vs {})",
                    shape_a, shape_b, i, dim_a, dim_b
                ),
            });
        }
    }

    // Reverse to get back to original order
    result.reverse();
    Ok(result)
}

/// Infer output shape for matrix multiplication (matmul)
///
/// For 2D matrices: [M, K] @ [K, N] -> [M, N]
/// For batched matmul: broadcasting is applied to batch dimensions
pub fn infer_matmul_shape(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    if shape_a.len() < 2 || shape_b.len() < 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Matmul requires at least 2D tensors, got shapes {:?} and {:?}",
                shape_a, shape_b
            ),
        });
    }

    let a_rows = shape_a[shape_a.len() - 2];
    let a_cols = shape_a[shape_a.len() - 1];
    let b_rows = shape_b[shape_b.len() - 2];
    let b_cols = shape_b[shape_b.len() - 1];

    if a_cols != b_rows {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Incompatible shapes for matmul: {:?} and {:?} (inner dimensions {} != {})",
                shape_a, shape_b, a_cols, b_rows
            ),
        });
    }

    // For simple 2D case
    if shape_a.len() == 2 && shape_b.len() == 2 {
        return Ok(vec![a_rows, b_cols]);
    }

    // For batched matmul, broadcast batch dimensions
    let batch_a = &shape_a[..shape_a.len() - 2];
    let batch_b = &shape_b[..shape_b.len() - 2];
    let mut batch_dims = broadcast_shapes(batch_a, batch_b)?;
    batch_dims.push(a_rows);
    batch_dims.push(b_cols);

    Ok(batch_dims)
}

/// Validate that a reshape operation is valid
pub fn validate_reshape(input_shape: &[u32], output_shape: &[u32]) -> Result<(), GraphError> {
    let input_size: u32 = input_shape.iter().product();
    let output_size: u32 = output_shape.iter().product();

    if input_size != output_size {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Reshape requires same number of elements: input {:?} ({} elements) != output {:?} ({} elements)",
                input_shape, input_size, output_shape, output_size
            ),
        });
    }

    Ok(())
}

/// Layout for conv2d input tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv2dInputLayout {
    /// Channels-first: [batch, channels, height, width]
    Nchw,
    /// Channels-last: [batch, height, width, channels]
    Nhwc,
}

/// Layout for conv2d filter tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv2dFilterLayout {
    /// [out_channels, in_channels, height, width]
    Oihw,
    /// [height, width, in_channels, out_channels]
    Hwio,
    /// [out_channels, height, width, in_channels]
    Ohwi,
    /// [in_channels, height, width, out_channels]
    Ihwo,
}

/// Parameters for conv2d shape inference
pub struct Conv2dOptions {
    pub strides: Vec<u32>,
    pub dilations: Vec<u32>,
    pub pads: Vec<u32>,
    pub groups: u32,
    pub input_layout: Conv2dInputLayout,
    pub filter_layout: Conv2dFilterLayout,
}

/// Infer output shape for 2D convolution
///
/// Following the W3C WebNN specification for conv2d:
/// https://www.w3.org/TR/webnn/#api-mlgraphbuilder-conv2d
pub fn infer_conv2d_shape(
    input_shape: &[u32],
    filter_shape: &[u32],
    options: &Conv2dOptions,
) -> Result<Vec<u32>, GraphError> {
    // Input must be 4D: [batch, channels, height, width] or [batch, height, width, channels]
    if input_shape.len() != 4 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!("Conv2d input must be 4D, got shape {:?}", input_shape),
        });
    }

    // Filter must be 4D
    if filter_shape.len() != 4 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!("Conv2d filter must be 4D, got shape {:?}", filter_shape),
        });
    }

    // Extract dimensions based on layout
    let (batch, in_channels, input_h, input_w) = match options.input_layout {
        Conv2dInputLayout::Nchw => (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ),
        Conv2dInputLayout::Nhwc => (
            input_shape[0],
            input_shape[3],
            input_shape[1],
            input_shape[2],
        ),
    };

    let (out_channels, filter_in_channels, kernel_h, kernel_w) = match options.filter_layout {
        Conv2dFilterLayout::Oihw => (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        ),
        Conv2dFilterLayout::Hwio => (
            filter_shape[3],
            filter_shape[2],
            filter_shape[0],
            filter_shape[1],
        ),
        Conv2dFilterLayout::Ohwi => (
            filter_shape[0],
            filter_shape[3],
            filter_shape[1],
            filter_shape[2],
        ),
        Conv2dFilterLayout::Ihwo => (
            filter_shape[3],
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
        ),
    };

    // Validate groups
    if options.groups == 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "Conv2d groups must be > 0".to_string(),
        });
    }

    if in_channels % options.groups != 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Conv2d input channels {} must be divisible by groups {}",
                in_channels, options.groups
            ),
        });
    }

    if filter_in_channels * options.groups != in_channels {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Conv2d filter input channels {} * groups {} must equal input channels {}",
                filter_in_channels, options.groups, in_channels
            ),
        });
    }

    // Validate strides
    if options.strides.len() != 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Conv2d strides must have 2 elements, got {:?}",
                options.strides
            ),
        });
    }
    let stride_h = options.strides[0];
    let stride_w = options.strides[1];

    if stride_h == 0 || stride_w == 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "Conv2d strides must be > 0".to_string(),
        });
    }

    // Validate dilations
    if options.dilations.len() != 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Conv2d dilations must have 2 elements, got {:?}",
                options.dilations
            ),
        });
    }
    let dilation_h = options.dilations[0];
    let dilation_w = options.dilations[1];

    if dilation_h == 0 || dilation_w == 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "Conv2d dilations must be > 0".to_string(),
        });
    }

    // Validate pads
    if options.pads.len() != 4 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!("Conv2d pads must have 4 elements, got {:?}", options.pads),
        });
    }
    let pad_begin_h = options.pads[0];
    let pad_begin_w = options.pads[1];
    let pad_end_h = options.pads[2];
    let pad_end_w = options.pads[3];

    // Compute effective kernel size with dilation
    let effective_kernel_h = dilation_h * (kernel_h - 1) + 1;
    let effective_kernel_w = dilation_w * (kernel_w - 1) + 1;

    // Compute output spatial dimensions
    // Formula: floor((input_size + pad_begin + pad_end - effective_kernel_size) / stride) + 1
    let padded_h = input_h + pad_begin_h + pad_end_h;
    let padded_w = input_w + pad_begin_w + pad_end_w;

    if padded_h < effective_kernel_h || padded_w < effective_kernel_w {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Conv2d: padded input size [{}, {}] is smaller than effective kernel size [{}, {}]",
                padded_h, padded_w, effective_kernel_h, effective_kernel_w
            ),
        });
    }

    let output_h = (padded_h - effective_kernel_h) / stride_h + 1;
    let output_w = (padded_w - effective_kernel_w) / stride_w + 1;

    // Build output shape based on input layout
    let output_shape = match options.input_layout {
        Conv2dInputLayout::Nchw => vec![batch, out_channels, output_h, output_w],
        Conv2dInputLayout::Nhwc => vec![batch, output_h, output_w, out_channels],
    };

    Ok(output_shape)
}

/// Parameters for convTranspose2d shape inference
pub struct ConvTranspose2dOptions {
    pub strides: Vec<u32>,
    pub dilations: Vec<u32>,
    pub pads: Vec<u32>,
    pub output_padding: Vec<u32>,
    pub output_sizes: Option<Vec<u32>>,
    pub groups: u32,
    pub input_layout: Conv2dInputLayout,
    pub filter_layout: Conv2dFilterLayout,
}

/// Infer output shape for 2D transposed convolution (deconvolution)
///
/// Following the W3C WebNN specification for convTranspose2d:
/// https://www.w3.org/TR/webnn/#api-mlgraphbuilder-convtranspose2d
pub fn infer_conv_transpose2d_shape(
    input_shape: &[u32],
    filter_shape: &[u32],
    options: &ConvTranspose2dOptions,
) -> Result<Vec<u32>, GraphError> {
    // Input must be 4D: [batch, channels, height, width] or [batch, height, width, channels]
    if input_shape.len() != 4 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "ConvTranspose2d input must be 4D, got shape {:?}",
                input_shape
            ),
        });
    }

    // Filter must be 4D
    if filter_shape.len() != 4 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "ConvTranspose2d filter must be 4D, got shape {:?}",
                filter_shape
            ),
        });
    }

    // Extract dimensions based on layout
    let (batch, in_channels, input_h, input_w) = match options.input_layout {
        Conv2dInputLayout::Nchw => (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ),
        Conv2dInputLayout::Nhwc => (
            input_shape[0],
            input_shape[3],
            input_shape[1],
            input_shape[2],
        ),
    };

    // For transposed convolution, filter layout interpretation is different
    // Filter is [in_channels, out_channels/groups, height, width] for OIHW-like layout
    let (filter_in_channels, out_channels_per_group, kernel_h, kernel_w) =
        match options.filter_layout {
            Conv2dFilterLayout::Oihw => {
                // For transpose: [in_channels, out_channels/groups, h, w]
                (
                    filter_shape[0],
                    filter_shape[1],
                    filter_shape[2],
                    filter_shape[3],
                )
            }
            Conv2dFilterLayout::Hwio => {
                // [h, w, in_channels, out_channels/groups]
                (
                    filter_shape[2],
                    filter_shape[3],
                    filter_shape[0],
                    filter_shape[1],
                )
            }
            Conv2dFilterLayout::Ohwi => {
                // [in_channels, h, w, out_channels/groups]
                (
                    filter_shape[0],
                    filter_shape[3],
                    filter_shape[1],
                    filter_shape[2],
                )
            }
            Conv2dFilterLayout::Ihwo => {
                // [in_channels, h, w, out_channels/groups]
                (
                    filter_shape[0],
                    filter_shape[3],
                    filter_shape[1],
                    filter_shape[2],
                )
            }
        };

    // Validate groups
    if options.groups == 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "ConvTranspose2d groups must be > 0".to_string(),
        });
    }

    if in_channels % options.groups != 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "ConvTranspose2d input channels {} must be divisible by groups {}",
                in_channels, options.groups
            ),
        });
    }

    if filter_in_channels != in_channels {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "ConvTranspose2d filter input channels {} must equal input channels {}",
                filter_in_channels, in_channels
            ),
        });
    }

    let out_channels = out_channels_per_group * options.groups;

    // Validate strides
    if options.strides.len() != 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "ConvTranspose2d strides must have 2 elements, got {:?}",
                options.strides
            ),
        });
    }
    let stride_h = options.strides[0];
    let stride_w = options.strides[1];

    if stride_h == 0 || stride_w == 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "ConvTranspose2d strides must be > 0".to_string(),
        });
    }

    // Validate dilations
    if options.dilations.len() != 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "ConvTranspose2d dilations must have 2 elements, got {:?}",
                options.dilations
            ),
        });
    }
    let dilation_h = options.dilations[0];
    let dilation_w = options.dilations[1];

    if dilation_h == 0 || dilation_w == 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "ConvTranspose2d dilations must be > 0".to_string(),
        });
    }

    // Validate pads
    if options.pads.len() != 4 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "ConvTranspose2d pads must have 4 elements, got {:?}",
                options.pads
            ),
        });
    }
    let pad_begin_h = options.pads[0];
    let pad_begin_w = options.pads[1];
    let pad_end_h = options.pads[2];
    let pad_end_w = options.pads[3];

    // Validate output_padding
    if options.output_padding.len() != 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "ConvTranspose2d output_padding must have 2 elements, got {:?}",
                options.output_padding
            ),
        });
    }
    let output_pad_h = options.output_padding[0];
    let output_pad_w = options.output_padding[1];

    // Compute effective kernel size with dilation
    let effective_kernel_h = dilation_h * (kernel_h - 1) + 1;
    let effective_kernel_w = dilation_w * (kernel_w - 1) + 1;

    // Compute output spatial dimensions
    // Formula for transposed convolution:
    // output_size = (input_size - 1) * stride + effective_kernel_size - pad_begin - pad_end + output_padding
    let output_h = if let Some(ref sizes) = options.output_sizes {
        if sizes.len() != 2 {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "ConvTranspose2d output_sizes must have 2 elements, got {:?}",
                    sizes
                ),
            });
        }
        sizes[0]
    } else {
        (input_h - 1) * stride_h + effective_kernel_h - pad_begin_h - pad_end_h + output_pad_h
    };

    let output_w = if let Some(ref sizes) = options.output_sizes {
        sizes[1]
    } else {
        (input_w - 1) * stride_w + effective_kernel_w - pad_begin_w - pad_end_w + output_pad_w
    };

    // Build output shape based on input layout
    let output_shape = match options.input_layout {
        Conv2dInputLayout::Nchw => vec![batch, out_channels, output_h, output_w],
        Conv2dInputLayout::Nhwc => vec![batch, output_h, output_w, out_channels],
    };

    Ok(output_shape)
}

/// Parameters for pool2d shape inference
pub struct Pool2dOptions {
    pub window_dimensions: Vec<u32>,
    pub strides: Vec<u32>,
    pub dilations: Vec<u32>,
    pub pads: Vec<u32>,
    pub layout: Conv2dInputLayout,
}

/// Infer output shape for 2D pooling operations (average, max)
///
/// Following the W3C WebNN specification for pool2d:
/// https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d
pub fn infer_pool2d_shape(
    input_shape: &[u32],
    options: &Pool2dOptions,
) -> Result<Vec<u32>, GraphError> {
    // Input must be 4D: [batch, channels, height, width] or [batch, height, width, channels]
    if input_shape.len() != 4 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!("Pool2d input must be 4D, got shape {:?}", input_shape),
        });
    }

    // Extract dimensions based on layout
    let (batch, channels, input_h, input_w) = match options.layout {
        Conv2dInputLayout::Nchw => (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ),
        Conv2dInputLayout::Nhwc => (
            input_shape[0],
            input_shape[3],
            input_shape[1],
            input_shape[2],
        ),
    };

    // Validate window dimensions
    if options.window_dimensions.len() != 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Pool2d window_dimensions must have 2 elements, got {:?}",
                options.window_dimensions
            ),
        });
    }
    let window_h = options.window_dimensions[0];
    let window_w = options.window_dimensions[1];

    if window_h == 0 || window_w == 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "Pool2d window dimensions must be > 0".to_string(),
        });
    }

    // Validate strides
    if options.strides.len() != 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Pool2d strides must have 2 elements, got {:?}",
                options.strides
            ),
        });
    }
    let stride_h = options.strides[0];
    let stride_w = options.strides[1];

    if stride_h == 0 || stride_w == 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "Pool2d strides must be > 0".to_string(),
        });
    }

    // Validate dilations
    if options.dilations.len() != 2 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Pool2d dilations must have 2 elements, got {:?}",
                options.dilations
            ),
        });
    }
    let dilation_h = options.dilations[0];
    let dilation_w = options.dilations[1];

    if dilation_h == 0 || dilation_w == 0 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "Pool2d dilations must be > 0".to_string(),
        });
    }

    // Validate pads
    if options.pads.len() != 4 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!("Pool2d pads must have 4 elements, got {:?}", options.pads),
        });
    }
    let pad_begin_h = options.pads[0];
    let pad_begin_w = options.pads[1];
    let pad_end_h = options.pads[2];
    let pad_end_w = options.pads[3];

    // Compute effective window size with dilation
    let effective_window_h = dilation_h * (window_h - 1) + 1;
    let effective_window_w = dilation_w * (window_w - 1) + 1;

    // Compute output spatial dimensions
    // Formula: floor((input_size + pad_begin + pad_end - effective_window_size) / stride) + 1
    let padded_h = input_h + pad_begin_h + pad_end_h;
    let padded_w = input_w + pad_begin_w + pad_end_w;

    if padded_h < effective_window_h || padded_w < effective_window_w {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Pool2d: padded input size [{}, {}] is smaller than effective window size [{}, {}]",
                padded_h, padded_w, effective_window_h, effective_window_w
            ),
        });
    }

    let output_h = (padded_h - effective_window_h) / stride_h + 1;
    let output_w = (padded_w - effective_window_w) / stride_w + 1;

    // Build output shape based on layout (channels remain unchanged)
    let output_shape = match options.layout {
        Conv2dInputLayout::Nchw => vec![batch, channels, output_h, output_w],
        Conv2dInputLayout::Nhwc => vec![batch, output_h, output_w, channels],
    };

    Ok(output_shape)
}

/// Options for global pooling operations
#[derive(Debug, Clone)]
pub struct GlobalPoolOptions {
    pub layout: Conv2dInputLayout,
}

/// Infer the output shape for global pooling operations
/// Global pooling reduces spatial dimensions to 1x1
pub fn infer_global_pool_shape(
    input_shape: &[u32],
    options: &GlobalPoolOptions,
) -> Result<Vec<u32>, GraphError> {
    // Validate input is 4D
    if input_shape.len() != 4 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Global pooling input must be 4D, got {}D tensor {:?}",
                input_shape.len(),
                input_shape
            ),
        });
    }

    // Global pooling reduces spatial dimensions to 1x1
    // Output shape depends on layout
    let output_shape = match options.layout {
        Conv2dInputLayout::Nchw => {
            // [N, C, H, W] -> [N, C, 1, 1]
            vec![input_shape[0], input_shape[1], 1, 1]
        }
        Conv2dInputLayout::Nhwc => {
            // [N, H, W, C] -> [N, 1, 1, C]
            vec![input_shape[0], 1, 1, input_shape[3]]
        }
    };

    Ok(output_shape)
}

/// Infer the output shape for batchNormalization
/// Batch normalization output has the same shape as input
pub fn infer_batch_normalization_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    // Batch normalization preserves the input shape
    Ok(input_shape.to_vec())
}

/// Infer the output shape for instanceNormalization
/// Instance normalization output has the same shape as input
pub fn infer_instance_normalization_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    // Instance normalization preserves the input shape
    Ok(input_shape.to_vec())
}

/// Infer the output shape for layerNormalization
/// Layer normalization output has the same shape as input
pub fn infer_layer_normalization_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    // Layer normalization preserves the input shape
    Ok(input_shape.to_vec())
}

/// Options for reduction operations
#[derive(Debug, Clone)]
pub struct ReduceOptions {
    pub axes: Vec<u32>,
    pub keep_dimensions: bool,
}

/// Infer the output shape for reduction operations
///
/// Reduction operations reduce input tensor dimensions by applying a reduction function
/// across specified axes.
///
/// Following the W3C WebNN specification for reduction operations:
/// https://www.w3.org/TR/webnn/#api-mlgraphbuilder-reduce
pub fn infer_reduce_shape(
    input_shape: &[u32],
    options: &ReduceOptions,
) -> Result<Vec<u32>, GraphError> {
    // If axes is empty, reduce all dimensions
    let axes_to_reduce: Vec<u32> = if options.axes.is_empty() {
        (0..input_shape.len() as u32).collect()
    } else {
        options.axes.clone()
    };

    // Validate that all axes are within bounds
    for &axis in &axes_to_reduce {
        if axis >= input_shape.len() as u32 {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "Reduce axis {} out of bounds for shape {:?} (rank {})",
                    axis,
                    input_shape,
                    input_shape.len()
                ),
            });
        }
    }

    // Check for duplicate axes
    let mut sorted_axes = axes_to_reduce.clone();
    sorted_axes.sort_unstable();
    for i in 1..sorted_axes.len() {
        if sorted_axes[i] == sorted_axes[i - 1] {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!("Duplicate axis {} in reduction axes", sorted_axes[i]),
            });
        }
    }

    // Build output shape
    let mut output_shape = Vec::new();
    for (idx, &dim) in input_shape.iter().enumerate() {
        let is_reduced = axes_to_reduce.contains(&(idx as u32));
        if is_reduced {
            if options.keep_dimensions {
                output_shape.push(1);
            }
            // else: omit this dimension
        } else {
            output_shape.push(dim);
        }
    }

    Ok(output_shape)
}

/// Infer the output shape for element-wise unary operations
/// All element-wise unary operations preserve the input shape

pub fn infer_abs_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_ceil_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_floor_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_round_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_neg_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_sign_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_exp_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_log_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_sqrt_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_reciprocal_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_sin_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_cos_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_tan_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_asin_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_acos_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_atan_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_sinh_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_cosh_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_asinh_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_acosh_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_atanh_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_erf_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

pub fn infer_identity_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

/// Infer the output shape for binary comparison operations
/// Comparison operations use NumPy-style broadcasting and return uint8 output
pub fn infer_equal_shape(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    broadcast_shapes(shape_a, shape_b)
}

pub fn infer_greater_shape(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    broadcast_shapes(shape_a, shape_b)
}

pub fn infer_greater_or_equal_shape(
    shape_a: &[u32],
    shape_b: &[u32],
) -> Result<Vec<u32>, GraphError> {
    broadcast_shapes(shape_a, shape_b)
}

pub fn infer_lesser_shape(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    broadcast_shapes(shape_a, shape_b)
}

pub fn infer_lesser_or_equal_shape(
    shape_a: &[u32],
    shape_b: &[u32],
) -> Result<Vec<u32>, GraphError> {
    broadcast_shapes(shape_a, shape_b)
}

/// Infer the output shape for unary logical operations
/// logicalNot preserves input shape and returns uint8 output
pub fn infer_logical_not_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

/// Infer the output shape for binary logical operations
/// Logical operations use NumPy-style broadcasting and return uint8 output
pub fn infer_logical_and_shape(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    broadcast_shapes(shape_a, shape_b)
}

pub fn infer_logical_or_shape(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    broadcast_shapes(shape_a, shape_b)
}

pub fn infer_logical_xor_shape(shape_a: &[u32], shape_b: &[u32]) -> Result<Vec<u32>, GraphError> {
    broadcast_shapes(shape_a, shape_b)
}

/// Infer the output shape for dequantizeLinear
/// Converts quantized integer values to floating-point, preserving shape
/// Formula: output = (input - zeroPoint) * scale
pub fn infer_dequantize_linear_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

/// Infer the output shape for quantizeLinear
/// Converts floating-point values to quantized integers, preserving shape
/// Formula: output = input / scale + zeroPoint
pub fn infer_quantize_linear_shape(input_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    Ok(input_shape.to_vec())
}

/// Infer output shape for transpose operation
///
/// Transpose reorders tensor dimensions according to a permutation.
/// If no permutation is provided, dimensions are reversed.
pub fn infer_transpose_shape(
    input_shape: &[u32],
    permutation: Option<&[u32]>,
) -> Result<Vec<u32>, GraphError> {
    let rank = input_shape.len();

    // If no permutation, reverse dimensions (default WebNN behavior)
    if permutation.is_none() {
        let mut output_shape = input_shape.to_vec();
        output_shape.reverse();
        return Ok(output_shape);
    }

    let perm = permutation.unwrap();

    // Validate permutation length matches input rank
    if perm.len() != rank {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Transpose permutation length {} must match input rank {}, input shape: {:?}",
                perm.len(),
                rank,
                input_shape
            ),
        });
    }

    // Validate permutation contains unique values in range [0, rank)
    let mut seen = vec![false; rank];
    for &axis in perm {
        if axis >= rank as u32 {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "Transpose permutation axis {} out of bounds for rank {}, input shape: {:?}",
                    axis, rank, input_shape
                ),
            });
        }
        if seen[axis as usize] {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!("Transpose permutation contains duplicate axis {}", axis),
            });
        }
        seen[axis as usize] = true;
    }

    // Build output shape by permuting dimensions
    let output_shape: Vec<u32> = perm.iter().map(|&i| input_shape[i as usize]).collect();

    Ok(output_shape)
}

/// Infer output shape for concat operation
///
/// Concatenates multiple tensors along a specified axis.
pub fn infer_concat_shape(input_shapes: &[Vec<u32>], axis: u32) -> Result<Vec<u32>, GraphError> {
    if input_shapes.is_empty() {
        return Err(GraphError::ShapeInferenceFailed {
            reason: "Concat requires at least one input".to_string(),
        });
    }

    let first_shape = &input_shapes[0];
    let rank = first_shape.len();

    // Validate axis is within bounds
    if axis >= rank as u32 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Concat axis {} out of bounds for rank {}, shape: {:?}",
                axis, rank, first_shape
            ),
        });
    }

    // Validate all inputs have same rank
    for (idx, shape) in input_shapes.iter().enumerate() {
        if shape.len() != rank {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "Concat input {} has rank {} but expected rank {}, shapes: {:?}",
                    idx,
                    shape.len(),
                    rank,
                    input_shapes
                ),
            });
        }
    }

    // Validate all dimensions except concat axis match
    for dim_idx in 0..rank {
        if dim_idx == axis as usize {
            continue;
        }
        let expected_dim = first_shape[dim_idx];
        for (input_idx, shape) in input_shapes.iter().enumerate() {
            if shape[dim_idx] != expected_dim {
                return Err(GraphError::ShapeInferenceFailed {
                    reason: format!(
                        "Concat input {} dimension {} is {} but expected {} (all non-concat dimensions must match)",
                        input_idx, dim_idx, shape[dim_idx], expected_dim
                    ),
                });
            }
        }
    }

    // Compute output shape: sum concat axis, others match first input
    let mut output_shape = first_shape.clone();
    let concat_dim_size: u32 = input_shapes.iter().map(|shape| shape[axis as usize]).sum();
    output_shape[axis as usize] = concat_dim_size;

    Ok(output_shape)
}

/// Infer output shape for slice operation
///
/// Extracts a contiguous sub-tensor from the input.
pub fn infer_slice_shape(
    input_shape: &[u32],
    starts: &[u32],
    sizes: &[u32],
) -> Result<Vec<u32>, GraphError> {
    let rank = input_shape.len();

    // Validate starts and sizes have same length as input rank
    if starts.len() != rank {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Slice starts length {} must match input rank {}, input shape: {:?}",
                starts.len(),
                rank,
                input_shape
            ),
        });
    }

    if sizes.len() != rank {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Slice sizes length {} must match input rank {}, input shape: {:?}",
                sizes.len(),
                rank,
                input_shape
            ),
        });
    }

    // Validate starts and sizes are within bounds
    for (dim_idx, (&start, &size)) in starts.iter().zip(sizes.iter()).enumerate() {
        let input_dim = input_shape[dim_idx];

        if start >= input_dim {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "Slice start {} for dimension {} exceeds input dimension size {}",
                    start, dim_idx, input_dim
                ),
            });
        }

        if start + size > input_dim {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "Slice end {} (start {} + size {}) for dimension {} exceeds input dimension size {}",
                    start + size,
                    start,
                    size,
                    dim_idx,
                    input_dim
                ),
            });
        }
    }

    // Output shape is simply the sizes
    Ok(sizes.to_vec())
}

/// Infer output shape for expand operation
///
/// Broadcasts a tensor to a larger shape. Dimensions of size 1 can be expanded to larger sizes.
pub fn infer_expand_shape(input_shape: &[u32], new_shape: &[u32]) -> Result<Vec<u32>, GraphError> {
    let input_rank = input_shape.len();
    let output_rank = new_shape.len();

    // Output rank must be >= input rank
    if output_rank < input_rank {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Expand new_shape rank {} must be >= input rank {}, input shape: {:?}, new_shape: {:?}",
                output_rank, input_rank, input_shape, new_shape
            ),
        });
    }

    // Align shapes from the right (trailing dimensions)
    let offset = output_rank - input_rank;

    for i in 0..input_rank {
        let input_dim = input_shape[i];
        let output_dim = new_shape[offset + i];

        // Input dimension must be 1 or match output dimension
        if input_dim != 1 && input_dim != output_dim {
            return Err(GraphError::ShapeInferenceFailed {
                reason: format!(
                    "Expand dimension {} mismatch: input {} can only expand if it's 1, but new_shape specifies {}, input shape: {:?}, new_shape: {:?}",
                    i, input_dim, output_dim, input_shape, new_shape
                ),
            });
        }
    }

    // Output shape is the new_shape
    Ok(new_shape.to_vec())
}

/// Infer output shape for gather operation
///
/// Gathers values from input tensor along an axis according to indices.
pub fn infer_gather_shape(
    input_shape: &[u32],
    indices_shape: &[u32],
    axis: u32,
) -> Result<Vec<u32>, GraphError> {
    let input_rank = input_shape.len();

    // Validate axis is within bounds
    if axis >= input_rank as u32 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Gather axis {} out of bounds for input rank {}, input shape: {:?}",
                axis, input_rank, input_shape
            ),
        });
    }

    // Output shape = input_shape[0:axis] + indices_shape + input_shape[axis+1:]
    let mut output_shape = Vec::new();
    output_shape.extend_from_slice(&input_shape[..axis as usize]);
    output_shape.extend_from_slice(indices_shape);
    output_shape.extend_from_slice(&input_shape[(axis as usize + 1)..]);

    Ok(output_shape)
}

/// Represents the split specification
#[derive(Debug, Clone)]
pub enum SplitSpec {
    /// Split into N equal parts
    Count(u32),
    /// Split into parts of specified sizes
    Sizes(Vec<u32>),
}

/// Infer output shapes for split operation
///
/// Splits a tensor into multiple sub-tensors along an axis.
pub fn infer_split_shapes(
    input_shape: &[u32],
    split_spec: &SplitSpec,
    axis: u32,
) -> Result<Vec<Vec<u32>>, GraphError> {
    let input_rank = input_shape.len();

    // Validate axis is within bounds
    if axis >= input_rank as u32 {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Split axis {} out of bounds for input rank {}, input shape: {:?}",
                axis, input_rank, input_shape
            ),
        });
    }

    let axis_size = input_shape[axis as usize];

    let split_sizes: Vec<u32> = match split_spec {
        SplitSpec::Count(count) => {
            if *count == 0 {
                return Err(GraphError::ShapeInferenceFailed {
                    reason: "Split count must be > 0".to_string(),
                });
            }

            if axis_size % count != 0 {
                return Err(GraphError::ShapeInferenceFailed {
                    reason: format!(
                        "Split count {} does not evenly divide axis size {}, input shape: {:?}",
                        count, axis_size, input_shape
                    ),
                });
            }

            let size_per_split = axis_size / count;
            vec![size_per_split; *count as usize]
        }
        SplitSpec::Sizes(sizes) => {
            let total: u32 = sizes.iter().sum();
            if total != axis_size {
                return Err(GraphError::ShapeInferenceFailed {
                    reason: format!(
                        "Split sizes {:?} sum to {} but axis size is {}, input shape: {:?}",
                        sizes, total, axis_size, input_shape
                    ),
                });
            }
            sizes.clone()
        }
    };

    // Create output shapes
    let mut output_shapes = Vec::new();
    for &split_size in &split_sizes {
        let mut shape = input_shape.to_vec();
        shape[axis as usize] = split_size;
        output_shapes.push(shape);
    }

    Ok(output_shapes)
}

/// Infer output shape for where operation
///
/// Selects elements from trueValue or falseValue based on condition.
/// All inputs are broadcast to a common shape.
pub fn infer_where_shape(
    condition_shape: &[u32],
    true_value_shape: &[u32],
    false_value_shape: &[u32],
) -> Result<Vec<u32>, GraphError> {
    // All three inputs are broadcast to a common shape
    let temp_shape = broadcast_shapes(condition_shape, true_value_shape)?;
    let output_shape = broadcast_shapes(&temp_shape, false_value_shape)?;

    Ok(output_shape)
}

/// Pad mode for pad operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Constant,
    Edge,
    Reflection,
    Symmetric,
}

/// Infer output shape for pad operation
///
/// Adds padding around the input tensor.
/// Padding is specified as [begin_0, begin_1, ..., begin_n, end_0, end_1, ..., end_n]
pub fn infer_pad_shape(input_shape: &[u32], padding: &[u32]) -> Result<Vec<u32>, GraphError> {
    let rank = input_shape.len();

    // Padding must have length 2 * rank (begin and end for each dimension)
    if padding.len() != 2 * rank {
        return Err(GraphError::ShapeInferenceFailed {
            reason: format!(
                "Pad padding length {} must be 2 * input rank {}, input shape: {:?}",
                padding.len(),
                rank,
                input_shape
            ),
        });
    }

    // Compute output shape
    let mut output_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let begin_pad = padding[i];
        let end_pad = padding[rank + i];
        output_shape.push(input_shape[i] + begin_pad + end_pad);
    }

    Ok(output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_same_shape() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_with_ones() {
        assert_eq!(broadcast_shapes(&[2, 3], &[1, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[1, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_different_ranks() {
        assert_eq!(
            broadcast_shapes(&[2, 3, 4], &[3, 4]).unwrap(),
            vec![2, 3, 4]
        );
        assert_eq!(
            broadcast_shapes(&[3, 4], &[2, 3, 4]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_scalar() {
        assert_eq!(broadcast_shapes(&[2, 3], &[1]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        assert!(broadcast_shapes(&[2, 3], &[2, 4]).is_err());
        assert!(broadcast_shapes(&[2, 3, 4], &[2, 5, 4]).is_err());
    }

    #[test]
    fn test_matmul_2d() {
        assert_eq!(infer_matmul_shape(&[2, 3], &[3, 4]).unwrap(), vec![2, 4]);
    }

    #[test]
    fn test_matmul_batched() {
        assert_eq!(
            infer_matmul_shape(&[5, 2, 3], &[5, 3, 4]).unwrap(),
            vec![5, 2, 4]
        );
    }

    #[test]
    fn test_matmul_incompatible() {
        assert!(infer_matmul_shape(&[2, 3], &[4, 5]).is_err());
        assert!(infer_matmul_shape(&[2], &[3, 4]).is_err());
    }

    #[test]
    fn test_validate_reshape_valid() {
        assert!(validate_reshape(&[2, 3], &[6]).is_ok());
        assert!(validate_reshape(&[2, 3, 4], &[6, 4]).is_ok());
        assert!(validate_reshape(&[6], &[2, 3]).is_ok());
    }

    #[test]
    fn test_validate_reshape_invalid() {
        assert!(validate_reshape(&[2, 3], &[5]).is_err());
        assert!(validate_reshape(&[2, 3, 4], &[5, 5]).is_err());
    }

    #[test]
    fn test_conv2d_nchw_basic() {
        // Input: [1, 3, 32, 32], Filter: [64, 3, 3, 3]
        // Stride: [1, 1], Dilation: [1, 1], Pads: [1, 1, 1, 1]
        // Expected output: [1, 64, 32, 32]
        let options = Conv2dOptions {
            strides: vec![1, 1],
            dilations: vec![1, 1],
            pads: vec![1, 1, 1, 1],
            groups: 1,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        let output = infer_conv2d_shape(&[1, 3, 32, 32], &[64, 3, 3, 3], &options).unwrap();
        assert_eq!(output, vec![1, 64, 32, 32]);
    }

    #[test]
    fn test_conv2d_nhwc_basic() {
        // Input: [1, 32, 32, 3], Filter: [64, 3, 3, 3]
        // Stride: [1, 1], Dilation: [1, 1], Pads: [1, 1, 1, 1]
        // Expected output: [1, 32, 32, 64]
        let options = Conv2dOptions {
            strides: vec![1, 1],
            dilations: vec![1, 1],
            pads: vec![1, 1, 1, 1],
            groups: 1,
            input_layout: Conv2dInputLayout::Nhwc,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        let output = infer_conv2d_shape(&[1, 32, 32, 3], &[64, 3, 3, 3], &options).unwrap();
        assert_eq!(output, vec![1, 32, 32, 64]);
    }

    #[test]
    fn test_conv2d_with_stride() {
        // Input: [1, 3, 28, 28], Filter: [32, 3, 5, 5]
        // Stride: [2, 2], Dilation: [1, 1], Pads: [0, 0, 0, 0]
        // Output: [1, 32, 12, 12]
        let options = Conv2dOptions {
            strides: vec![2, 2],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            groups: 1,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        let output = infer_conv2d_shape(&[1, 3, 28, 28], &[32, 3, 5, 5], &options).unwrap();
        assert_eq!(output, vec![1, 32, 12, 12]);
    }

    #[test]
    fn test_conv2d_with_dilation() {
        // Input: [1, 3, 32, 32], Filter: [64, 3, 3, 3]
        // Stride: [1, 1], Dilation: [2, 2], Pads: [2, 2, 2, 2]
        // Effective kernel: 5x5, Output: [1, 64, 32, 32]
        let options = Conv2dOptions {
            strides: vec![1, 1],
            dilations: vec![2, 2],
            pads: vec![2, 2, 2, 2],
            groups: 1,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        let output = infer_conv2d_shape(&[1, 3, 32, 32], &[64, 3, 3, 3], &options).unwrap();
        assert_eq!(output, vec![1, 64, 32, 32]);
    }

    #[test]
    fn test_conv2d_depthwise() {
        // Depthwise convolution: groups = in_channels
        // Input: [1, 32, 28, 28], Filter: [32, 1, 3, 3]
        // Stride: [1, 1], Dilation: [1, 1], Pads: [1, 1, 1, 1], Groups: 32
        let options = Conv2dOptions {
            strides: vec![1, 1],
            dilations: vec![1, 1],
            pads: vec![1, 1, 1, 1],
            groups: 32,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        let output = infer_conv2d_shape(&[1, 32, 28, 28], &[32, 1, 3, 3], &options).unwrap();
        assert_eq!(output, vec![1, 32, 28, 28]);
    }

    #[test]
    fn test_conv2d_invalid_input_dim() {
        let options = Conv2dOptions {
            strides: vec![1, 1],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            groups: 1,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        // Input must be 4D
        assert!(infer_conv2d_shape(&[3, 32, 32], &[64, 3, 3, 3], &options).is_err());
    }

    #[test]
    fn test_conv2d_invalid_groups() {
        let options = Conv2dOptions {
            strides: vec![1, 1],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            groups: 2,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        // Groups must divide input channels evenly
        assert!(infer_conv2d_shape(&[1, 3, 32, 32], &[64, 1, 3, 3], &options).is_err());
    }

    // ConvTranspose2d tests
    #[test]
    fn test_conv_transpose2d_basic() {
        let options = ConvTranspose2dOptions {
            strides: vec![1, 1],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            output_padding: vec![0, 0],
            output_sizes: None,
            groups: 1,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        // Input: [1, 64, 14, 14], Filter: [64, 32, 3, 3]
        // Output: (14-1)*1 + 3 - 0 - 0 + 0 = 16
        let output =
            infer_conv_transpose2d_shape(&[1, 64, 14, 14], &[64, 32, 3, 3], &options).unwrap();
        assert_eq!(output, vec![1, 32, 16, 16]);
    }

    #[test]
    fn test_conv_transpose2d_with_stride() {
        let options = ConvTranspose2dOptions {
            strides: vec![2, 2],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            output_padding: vec![0, 0],
            output_sizes: None,
            groups: 1,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        // Input: [1, 64, 14, 14], Filter: [64, 32, 3, 3]
        // Output: (14-1)*2 + 3 - 0 - 0 + 0 = 29
        let output =
            infer_conv_transpose2d_shape(&[1, 64, 14, 14], &[64, 32, 3, 3], &options).unwrap();
        assert_eq!(output, vec![1, 32, 29, 29]);
    }

    #[test]
    fn test_conv_transpose2d_with_output_padding() {
        let options = ConvTranspose2dOptions {
            strides: vec![2, 2],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            output_padding: vec![1, 1],
            output_sizes: None,
            groups: 1,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        // Input: [1, 64, 14, 14], Filter: [64, 32, 3, 3]
        // Output: (14-1)*2 + 3 - 0 - 0 + 1 = 30
        let output =
            infer_conv_transpose2d_shape(&[1, 64, 14, 14], &[64, 32, 3, 3], &options).unwrap();
        assert_eq!(output, vec![1, 32, 30, 30]);
    }

    #[test]
    fn test_conv_transpose2d_with_output_sizes() {
        let options = ConvTranspose2dOptions {
            strides: vec![2, 2],
            dilations: vec![1, 1],
            pads: vec![1, 1, 1, 1],
            output_padding: vec![0, 0],
            output_sizes: Some(vec![28, 28]),
            groups: 1,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        // When output_sizes is specified, use it directly
        let output =
            infer_conv_transpose2d_shape(&[1, 64, 14, 14], &[64, 32, 3, 3], &options).unwrap();
        assert_eq!(output, vec![1, 32, 28, 28]);
    }

    #[test]
    fn test_conv_transpose2d_nhwc_layout() {
        let options = ConvTranspose2dOptions {
            strides: vec![2, 2],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            output_padding: vec![0, 0],
            output_sizes: None,
            groups: 1,
            input_layout: Conv2dInputLayout::Nhwc,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        // Input: [1, 14, 14, 64] (NHWC), Filter: [64, 32, 3, 3]
        // Output: [1, 29, 29, 32] (NHWC)
        let output =
            infer_conv_transpose2d_shape(&[1, 14, 14, 64], &[64, 32, 3, 3], &options).unwrap();
        assert_eq!(output, vec![1, 29, 29, 32]);
    }

    #[test]
    fn test_conv_transpose2d_invalid_input_dim() {
        let options = ConvTranspose2dOptions {
            strides: vec![1, 1],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            output_padding: vec![0, 0],
            output_sizes: None,
            groups: 1,
            input_layout: Conv2dInputLayout::Nchw,
            filter_layout: Conv2dFilterLayout::Oihw,
        };
        // Input must be 4D
        assert!(infer_conv_transpose2d_shape(&[64, 14, 14], &[64, 32, 3, 3], &options).is_err());
    }

    // Pool2d tests
    #[test]
    fn test_pool2d_basic() {
        let options = Pool2dOptions {
            window_dimensions: vec![2, 2],
            strides: vec![2, 2],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            layout: Conv2dInputLayout::Nchw,
        };
        // Input: [1, 64, 32, 32], Window: [2, 2], Stride: [2, 2]
        // Output: (32 - 2) / 2 + 1 = 16
        let output = infer_pool2d_shape(&[1, 64, 32, 32], &options).unwrap();
        assert_eq!(output, vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_pool2d_with_padding() {
        let options = Pool2dOptions {
            window_dimensions: vec![3, 3],
            strides: vec![1, 1],
            dilations: vec![1, 1],
            pads: vec![1, 1, 1, 1],
            layout: Conv2dInputLayout::Nchw,
        };
        // Input: [1, 64, 32, 32], Window: [3, 3], Padding: 1
        // Output: (32 + 1 + 1 - 3) / 1 + 1 = 32
        let output = infer_pool2d_shape(&[1, 64, 32, 32], &options).unwrap();
        assert_eq!(output, vec![1, 64, 32, 32]);
    }

    #[test]
    fn test_pool2d_nhwc_layout() {
        let options = Pool2dOptions {
            window_dimensions: vec![2, 2],
            strides: vec![2, 2],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            layout: Conv2dInputLayout::Nhwc,
        };
        // Input: [1, 32, 32, 64] (NHWC)
        // Output: [1, 16, 16, 64] (NHWC)
        let output = infer_pool2d_shape(&[1, 32, 32, 64], &options).unwrap();
        assert_eq!(output, vec![1, 16, 16, 64]);
    }

    #[test]
    fn test_pool2d_with_stride() {
        let options = Pool2dOptions {
            window_dimensions: vec![3, 3],
            strides: vec![2, 2],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            layout: Conv2dInputLayout::Nchw,
        };
        // Input: [1, 64, 28, 28], Window: [3, 3], Stride: [2, 2]
        // Output: (28 - 3) / 2 + 1 = 13
        let output = infer_pool2d_shape(&[1, 64, 28, 28], &options).unwrap();
        assert_eq!(output, vec![1, 64, 13, 13]);
    }

    #[test]
    fn test_pool2d_invalid_input_dim() {
        let options = Pool2dOptions {
            window_dimensions: vec![2, 2],
            strides: vec![2, 2],
            dilations: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            layout: Conv2dInputLayout::Nchw,
        };
        // Input must be 4D
        assert!(infer_pool2d_shape(&[64, 32, 32], &options).is_err());
    }

    // Global pooling tests
    #[test]
    fn test_global_pool_nchw() {
        let options = GlobalPoolOptions {
            layout: Conv2dInputLayout::Nchw,
        };
        // Input: [1, 64, 28, 28] -> Output: [1, 64, 1, 1]
        let output = infer_global_pool_shape(&[1, 64, 28, 28], &options).unwrap();
        assert_eq!(output, vec![1, 64, 1, 1]);
    }

    #[test]
    fn test_global_pool_nhwc() {
        let options = GlobalPoolOptions {
            layout: Conv2dInputLayout::Nhwc,
        };
        // Input: [1, 28, 28, 64] -> Output: [1, 1, 1, 64]
        let output = infer_global_pool_shape(&[1, 28, 28, 64], &options).unwrap();
        assert_eq!(output, vec![1, 1, 1, 64]);
    }

    #[test]
    fn test_global_pool_various_sizes() {
        let options = GlobalPoolOptions {
            layout: Conv2dInputLayout::Nchw,
        };
        // Different spatial sizes should all reduce to 1x1
        let output = infer_global_pool_shape(&[2, 128, 7, 7], &options).unwrap();
        assert_eq!(output, vec![2, 128, 1, 1]);

        let output = infer_global_pool_shape(&[1, 512, 14, 14], &options).unwrap();
        assert_eq!(output, vec![1, 512, 1, 1]);
    }

    #[test]
    fn test_global_pool_invalid_input_dim() {
        let options = GlobalPoolOptions {
            layout: Conv2dInputLayout::Nchw,
        };
        // Input must be 4D
        assert!(infer_global_pool_shape(&[64, 32, 32], &options).is_err());
        assert!(infer_global_pool_shape(&[1, 64, 32, 32, 32], &options).is_err());
    }

    // Normalization tests
    #[test]
    fn test_batch_normalization_shape() {
        // Batch normalization preserves input shape
        let output = infer_batch_normalization_shape(&[1, 64, 28, 28]).unwrap();
        assert_eq!(output, vec![1, 64, 28, 28]);

        let output = infer_batch_normalization_shape(&[8, 128, 14, 14]).unwrap();
        assert_eq!(output, vec![8, 128, 14, 14]);
    }

    #[test]
    fn test_instance_normalization_shape() {
        // Instance normalization preserves input shape
        let output = infer_instance_normalization_shape(&[1, 64, 28, 28]).unwrap();
        assert_eq!(output, vec![1, 64, 28, 28]);

        let output = infer_instance_normalization_shape(&[4, 32, 56, 56]).unwrap();
        assert_eq!(output, vec![4, 32, 56, 56]);
    }

    #[test]
    fn test_layer_normalization_shape() {
        // Layer normalization preserves input shape
        let output = infer_layer_normalization_shape(&[1, 64, 28, 28]).unwrap();
        assert_eq!(output, vec![1, 64, 28, 28]);

        // Works with any dimensional input
        let output = infer_layer_normalization_shape(&[8, 512]).unwrap();
        assert_eq!(output, vec![8, 512]);

        let output = infer_layer_normalization_shape(&[2, 10, 768]).unwrap();
        assert_eq!(output, vec![2, 10, 768]);
    }

    // Reduction operation tests
    #[test]
    fn test_reduce_single_axis() {
        let options = ReduceOptions {
            axes: vec![1],
            keep_dimensions: false,
        };
        // [2, 3, 4] reduce axis 1 -> [2, 4]
        let output = infer_reduce_shape(&[2, 3, 4], &options).unwrap();
        assert_eq!(output, vec![2, 4]);
    }

    #[test]
    fn test_reduce_single_axis_keep_dims() {
        let options = ReduceOptions {
            axes: vec![1],
            keep_dimensions: true,
        };
        // [2, 3, 4] reduce axis 1 keep_dims -> [2, 1, 4]
        let output = infer_reduce_shape(&[2, 3, 4], &options).unwrap();
        assert_eq!(output, vec![2, 1, 4]);
    }

    #[test]
    fn test_reduce_multiple_axes() {
        let options = ReduceOptions {
            axes: vec![1, 2],
            keep_dimensions: false,
        };
        // [2, 3, 4, 5] reduce axes [1,2] -> [2, 5]
        let output = infer_reduce_shape(&[2, 3, 4, 5], &options).unwrap();
        assert_eq!(output, vec![2, 5]);
    }

    #[test]
    fn test_reduce_multiple_axes_keep_dims() {
        let options = ReduceOptions {
            axes: vec![1, 2],
            keep_dimensions: true,
        };
        // [2, 3, 4, 5] reduce axes [1,2] keep_dims -> [2, 1, 1, 5]
        let output = infer_reduce_shape(&[2, 3, 4, 5], &options).unwrap();
        assert_eq!(output, vec![2, 1, 1, 5]);
    }

    #[test]
    fn test_reduce_all_axes() {
        let options = ReduceOptions {
            axes: vec![],
            keep_dimensions: false,
        };
        // [2, 3, 4] reduce all axes -> [] (scalar)
        let output = infer_reduce_shape(&[2, 3, 4], &options).unwrap();
        assert_eq!(output, Vec::<u32>::new());
    }

    #[test]
    fn test_reduce_all_axes_keep_dims() {
        let options = ReduceOptions {
            axes: vec![],
            keep_dimensions: true,
        };
        // [2, 3, 4] reduce all axes keep_dims -> [1, 1, 1]
        let output = infer_reduce_shape(&[2, 3, 4], &options).unwrap();
        assert_eq!(output, vec![1, 1, 1]);
    }

    #[test]
    fn test_reduce_last_axis() {
        let options = ReduceOptions {
            axes: vec![2],
            keep_dimensions: false,
        };
        // [2, 3, 4] reduce axis 2 -> [2, 3]
        let output = infer_reduce_shape(&[2, 3, 4], &options).unwrap();
        assert_eq!(output, vec![2, 3]);
    }

    #[test]
    fn test_reduce_first_axis() {
        let options = ReduceOptions {
            axes: vec![0],
            keep_dimensions: false,
        };
        // [2, 3, 4] reduce axis 0 -> [3, 4]
        let output = infer_reduce_shape(&[2, 3, 4], &options).unwrap();
        assert_eq!(output, vec![3, 4]);
    }

    #[test]
    fn test_reduce_invalid_axis() {
        let options = ReduceOptions {
            axes: vec![3],
            keep_dimensions: false,
        };
        // [2, 3, 4] has only axes 0,1,2; axis 3 is out of bounds
        assert!(infer_reduce_shape(&[2, 3, 4], &options).is_err());
    }

    #[test]
    fn test_reduce_duplicate_axes() {
        let options = ReduceOptions {
            axes: vec![1, 1],
            keep_dimensions: false,
        };
        // Duplicate axes should error
        assert!(infer_reduce_shape(&[2, 3, 4], &options).is_err());
    }

    #[test]
    fn test_reduce_non_contiguous_axes() {
        let options = ReduceOptions {
            axes: vec![0, 2],
            keep_dimensions: false,
        };
        // [2, 3, 4, 5] reduce axes [0, 2] -> [3, 5]
        let output = infer_reduce_shape(&[2, 3, 4, 5], &options).unwrap();
        assert_eq!(output, vec![3, 5]);
    }

    #[test]
    fn test_reduce_non_contiguous_axes_keep_dims() {
        let options = ReduceOptions {
            axes: vec![0, 2],
            keep_dimensions: true,
        };
        // [2, 3, 4, 5] reduce axes [0, 2] keep_dims -> [1, 3, 1, 5]
        let output = infer_reduce_shape(&[2, 3, 4, 5], &options).unwrap();
        assert_eq!(output, vec![1, 3, 1, 5]);
    }

    // Quantization operation tests
    #[test]
    fn test_dequantize_linear_shape() {
        // dequantizeLinear preserves input shape
        let output = infer_dequantize_linear_shape(&[1, 3, 224, 224]).unwrap();
        assert_eq!(output, vec![1, 3, 224, 224]);

        let output = infer_dequantize_linear_shape(&[8, 128]).unwrap();
        assert_eq!(output, vec![8, 128]);

        // Works with any dimensional input
        let output = infer_dequantize_linear_shape(&[10]).unwrap();
        assert_eq!(output, vec![10]);
    }

    #[test]
    fn test_quantize_linear_shape() {
        // quantizeLinear preserves input shape
        let output = infer_quantize_linear_shape(&[1, 3, 224, 224]).unwrap();
        assert_eq!(output, vec![1, 3, 224, 224]);

        let output = infer_quantize_linear_shape(&[8, 128]).unwrap();
        assert_eq!(output, vec![8, 128]);

        // Works with any dimensional input
        let output = infer_quantize_linear_shape(&[10]).unwrap();
        assert_eq!(output, vec![10]);
    }

    // Transpose tests
    #[test]
    fn test_transpose_default_permutation() {
        // Default: reverse dimensions
        assert_eq!(infer_transpose_shape(&[4, 6], None).unwrap(), vec![6, 4]);
        assert_eq!(
            infer_transpose_shape(&[2, 3, 4], None).unwrap(),
            vec![4, 3, 2]
        );
    }

    #[test]
    fn test_transpose_custom_permutation() {
        // 2D with explicit permutation
        assert_eq!(
            infer_transpose_shape(&[4, 6], Some(&[1, 0])).unwrap(),
            vec![6, 4]
        );

        // 3D with custom permutation
        assert_eq!(
            infer_transpose_shape(&[2, 3, 4], Some(&[2, 0, 1])).unwrap(),
            vec![4, 2, 3]
        );
    }

    #[test]
    fn test_transpose_invalid_permutation() {
        // Wrong length
        assert!(infer_transpose_shape(&[2, 3, 4], Some(&[0, 1])).is_err());

        // Out of bounds axis
        assert!(infer_transpose_shape(&[2, 3], Some(&[0, 3])).is_err());

        // Duplicate axis
        assert!(infer_transpose_shape(&[2, 3], Some(&[0, 0])).is_err());
    }

    // Concat tests
    #[test]
    fn test_concat_basic() {
        let shapes = vec![vec![2, 3], vec![2, 3]];
        assert_eq!(infer_concat_shape(&shapes, 0).unwrap(), vec![4, 3]);

        let shapes = vec![vec![2, 3], vec![2, 3]];
        assert_eq!(infer_concat_shape(&shapes, 1).unwrap(), vec![2, 6]);
    }

    #[test]
    fn test_concat_multiple_inputs() {
        let shapes = vec![vec![1, 3], vec![2, 3], vec![3, 3]];
        assert_eq!(infer_concat_shape(&shapes, 0).unwrap(), vec![6, 3]);
    }

    #[test]
    fn test_concat_3d() {
        let shapes = vec![vec![2, 3, 4], vec![2, 3, 4]];
        assert_eq!(infer_concat_shape(&shapes, 2).unwrap(), vec![2, 3, 8]);
    }

    #[test]
    fn test_concat_invalid() {
        // Empty inputs
        assert!(infer_concat_shape(&[], 0).is_err());

        // Mismatched ranks
        let shapes = vec![vec![2, 3], vec![2, 3, 4]];
        assert!(infer_concat_shape(&shapes, 0).is_err());

        // Mismatched non-concat dimensions
        let shapes = vec![vec![2, 3], vec![2, 4]];
        assert!(infer_concat_shape(&shapes, 0).is_err());

        // Axis out of bounds
        let shapes = vec![vec![2, 3]];
        assert!(infer_concat_shape(&shapes, 2).is_err());
    }

    // Slice tests
    #[test]
    fn test_slice_basic() {
        // 1D slice
        assert_eq!(infer_slice_shape(&[24], &[12], &[12]).unwrap(), vec![12]);

        // 2D slice
        assert_eq!(
            infer_slice_shape(&[4, 6], &[2, 2], &[2, 4]).unwrap(),
            vec![2, 4]
        );

        // 3D slice
        assert_eq!(
            infer_slice_shape(&[4, 3, 2], &[1, 1, 1], &[3, 2, 1]).unwrap(),
            vec![3, 2, 1]
        );
    }

    #[test]
    fn test_slice_invalid() {
        // starts length mismatch
        assert!(infer_slice_shape(&[4, 6], &[2], &[2, 4]).is_err());

        // sizes length mismatch
        assert!(infer_slice_shape(&[4, 6], &[2, 2], &[2]).is_err());

        // start out of bounds
        assert!(infer_slice_shape(&[4, 6], &[5, 2], &[1, 4]).is_err());

        // end out of bounds
        assert!(infer_slice_shape(&[4, 6], &[2, 2], &[3, 4]).is_err());
    }

    // Expand tests
    #[test]
    fn test_expand_basic() {
        // Expand 1D to larger 1D
        assert_eq!(infer_expand_shape(&[1], &[24]).unwrap(), vec![24]);

        // Expand to higher dimensions
        assert_eq!(infer_expand_shape(&[1], &[4, 6]).unwrap(), vec![4, 6]);

        // Expand some dimensions
        assert_eq!(infer_expand_shape(&[1, 6], &[4, 6]).unwrap(), vec![4, 6]);
    }

    #[test]
    fn test_expand_scalar() {
        // 0D (scalar) to various shapes
        assert_eq!(infer_expand_shape(&[], &[24]).unwrap(), vec![24]);
        assert_eq!(infer_expand_shape(&[], &[4, 6]).unwrap(), vec![4, 6]);
    }

    #[test]
    fn test_expand_invalid() {
        // Output rank < input rank
        assert!(infer_expand_shape(&[2, 3], &[6]).is_err());

        // Non-1 dimension can't be expanded
        assert!(infer_expand_shape(&[2, 3], &[4, 3]).is_err());
    }

    // Gather tests
    #[test]
    fn test_gather_basic() {
        // 1D input, 1D indices
        assert_eq!(infer_gather_shape(&[24], &[8], 0).unwrap(), vec![8]);

        // 2D input, 1D indices, axis=0
        assert_eq!(infer_gather_shape(&[12, 2], &[8], 0).unwrap(), vec![8, 2]);

        // 3D input, 2D indices, axis=1
        assert_eq!(
            infer_gather_shape(&[3, 4, 2], &[2, 2], 1).unwrap(),
            vec![3, 2, 2, 2]
        );
    }

    #[test]
    fn test_gather_scalar_indices() {
        // Scalar indices (0D)
        assert_eq!(
            infer_gather_shape(&[24], &[], 0).unwrap(),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn test_gather_invalid() {
        // Axis out of bounds
        assert!(infer_gather_shape(&[24], &[8], 1).is_err());
    }

    // Split tests
    #[test]
    fn test_split_by_count() {
        // Split 1D into 3 equal parts
        let shapes = infer_split_shapes(&[24], &SplitSpec::Count(3), 0).unwrap();
        assert_eq!(shapes, vec![vec![8], vec![8], vec![8]]);

        // Split 2D along axis 0
        let shapes = infer_split_shapes(&[8, 3], &SplitSpec::Count(2), 0).unwrap();
        assert_eq!(shapes, vec![vec![4, 3], vec![4, 3]]);
    }

    #[test]
    fn test_split_by_sizes() {
        // Split with custom sizes
        let shapes = infer_split_shapes(&[24], &SplitSpec::Sizes(vec![8, 8, 8]), 0).unwrap();
        assert_eq!(shapes, vec![vec![8], vec![8], vec![8]]);

        // Unequal split sizes
        let shapes = infer_split_shapes(&[12], &SplitSpec::Sizes(vec![3, 3, 3, 3]), 0).unwrap();
        assert_eq!(shapes, vec![vec![3], vec![3], vec![3], vec![3]]);
    }

    #[test]
    fn test_split_invalid() {
        // Count doesn't divide evenly
        assert!(infer_split_shapes(&[24], &SplitSpec::Count(5), 0).is_err());

        // Sizes don't sum to axis size
        assert!(infer_split_shapes(&[24], &SplitSpec::Sizes(vec![10, 10]), 0).is_err());

        // Axis out of bounds
        assert!(infer_split_shapes(&[24], &SplitSpec::Count(3), 1).is_err());

        // Zero count
        assert!(infer_split_shapes(&[24], &SplitSpec::Count(0), 0).is_err());
    }

    // Where tests
    #[test]
    fn test_where_basic() {
        // Same shapes
        assert_eq!(
            infer_where_shape(&[2, 3], &[2, 3], &[2, 3]).unwrap(),
            vec![2, 3]
        );

        // Broadcasting
        assert_eq!(
            infer_where_shape(&[2, 3], &[1, 3], &[2, 1]).unwrap(),
            vec![2, 3]
        );
    }

    #[test]
    fn test_where_broadcast_complex() {
        // All inputs broadcast to common shape
        assert_eq!(
            infer_where_shape(&[1, 3], &[2, 1], &[2, 3]).unwrap(),
            vec![2, 3]
        );
    }

    #[test]
    fn test_where_invalid() {
        // Incompatible shapes
        assert!(infer_where_shape(&[2, 3], &[2, 4], &[2, 3]).is_err());
    }

    // Pad tests
    #[test]
    fn test_pad_basic() {
        // 1D padding [begin, end]
        assert_eq!(infer_pad_shape(&[9], &[1, 1]).unwrap(), vec![11]);

        // 2D padding [begin_0, begin_1, end_0, end_1]
        assert_eq!(infer_pad_shape(&[3, 3], &[1, 1, 1, 1]).unwrap(), vec![5, 5]);

        // 4D padding
        assert_eq!(
            infer_pad_shape(&[1, 3, 3, 1], &[0, 2, 2, 0, 0, 2, 2, 0]).unwrap(),
            vec![1, 7, 7, 1]
        );
    }

    #[test]
    fn test_pad_no_padding() {
        // Zero padding
        assert_eq!(infer_pad_shape(&[3, 3], &[0, 0, 0, 0]).unwrap(), vec![3, 3]);
    }

    #[test]
    fn test_pad_invalid() {
        // Wrong padding length
        assert!(infer_pad_shape(&[3, 3], &[1, 1, 1]).is_err());
    }
}
