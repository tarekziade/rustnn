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
}
