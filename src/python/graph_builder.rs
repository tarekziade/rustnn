//! Graph builder for constructing WebNN computational graphs
//!
//! PyO3 macros generate unsafe code that triggers unsafe_op_in_unsafe_fn warnings.
//! This is expected behavior from the macro-generated code.
#![allow(unsafe_op_in_unsafe_fn)]

use super::graph::PyMLGraph;
use super::operand::{PyMLOperand, parse_data_type};
use crate::graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation,
};
use crate::shape_inference::{broadcast_shapes, infer_matmul_shape, validate_reshape};
use crate::validator::GraphValidator;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Builder for constructing WebNN computational graphs
#[pyclass(name = "MLGraphBuilder")]
pub struct PyMLGraphBuilder {
    operands: Vec<Operand>,
    operations: Vec<Operation>,
    input_operands: Vec<u32>,
    next_operand_id: u32,
    operand_map: HashMap<u32, PyMLOperand>,
    constant_data_map: HashMap<u32, ConstantData>,
}

#[pymethods]
impl PyMLGraphBuilder {
    #[new]
    fn new() -> Self {
        Self {
            operands: Vec::new(),
            operations: Vec::new(),
            input_operands: Vec::new(),
            next_operand_id: 0,
            operand_map: HashMap::new(),
            constant_data_map: HashMap::new(),
        }
    }

    /// Create an input operand
    ///
    /// Args:
    ///     name: Name of the input
    ///     shape: List of dimensions
    ///     data_type: Data type string (e.g., "float32")
    ///
    /// Returns:
    ///     MLOperand: The created input operand
    fn input(&mut self, name: String, shape: Vec<u32>, data_type: &str) -> PyResult<PyMLOperand> {
        let dtype = parse_data_type(data_type)?;
        let descriptor = OperandDescriptor {
            data_type: dtype,
            shape: shape.clone(),
            pending_permutation: Vec::new(),
        };

        let operand = Operand {
            descriptor: descriptor.clone(),
            kind: OperandKind::Input,
            name: Some(name.clone()),
        };

        let id = self.next_operand_id;
        self.operands.push(operand);
        self.input_operands.push(id);

        let py_operand = PyMLOperand::new(id, descriptor, OperandKind::Input, Some(name));
        self.operand_map.insert(id, py_operand.clone());
        self.next_operand_id += 1;

        Ok(py_operand)
    }

    /// Create a constant operand from numpy array
    ///
    /// Args:
    ///     value: NumPy array or Python list
    ///     shape: Optional shape override
    ///     data_type: Data type string (e.g., "float32")
    ///
    /// Returns:
    ///     MLOperand: The created constant operand
    #[pyo3(signature = (value, shape=None, data_type=None))]
    fn constant(
        &mut self,
        py: Python,
        value: &Bound<'_, PyAny>,
        shape: Option<Vec<u32>>,
        data_type: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        // Try to import numpy and convert to array
        let numpy = py.import_bound("numpy")?;
        let array = numpy.call_method1("asarray", (value,))?;

        // Get shape from array if not provided
        let actual_shape = if let Some(s) = shape {
            s
        } else {
            array.getattr("shape")?.extract::<Vec<u32>>()?
        };

        // Get dtype from array if not provided
        let actual_dtype = if let Some(dt) = data_type {
            parse_data_type(dt)?
        } else {
            let dtype_name: String = array.getattr("dtype")?.getattr("name")?.extract()?;
            parse_data_type(&dtype_name)?
        };

        let descriptor = OperandDescriptor {
            data_type: actual_dtype,
            shape: actual_shape.clone(),
            pending_permutation: Vec::new(),
        };

        // Convert array to bytes
        let bytes: Vec<u8> = array.call_method0("tobytes")?.extract()?;
        let constant_data = ConstantData {
            data: bytes,
            label: None,
        };

        let operand = Operand {
            descriptor: descriptor.clone(),
            kind: OperandKind::Constant,
            name: None,
        };

        let id = self.next_operand_id;
        self.operands.push(operand);
        self.constant_data_map.insert(id, constant_data);

        let py_operand = PyMLOperand::new(id, descriptor, OperandKind::Constant, None);
        self.operand_map.insert(id, py_operand.clone());
        self.next_operand_id += 1;

        Ok(py_operand)
    }

    // Binary operations

    /// Element-wise addition
    fn add(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("add", a, b)
    }

    /// Element-wise subtraction
    fn sub(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("sub", a, b)
    }

    /// Element-wise multiplication
    fn mul(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("mul", a, b)
    }

    /// Element-wise division
    fn div(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.binary_op("div", a, b)
    }

    /// Matrix multiplication
    fn matmul(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        // Use proper matmul shape inference
        let output_shape = infer_matmul_shape(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: a.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "matmul".to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// General Matrix Multiplication (GEMM)
    ///
    /// Computes: alpha * A' * B' + beta * C
    /// where A' and B' are optionally transposed versions of A and B.
    ///
    /// Args:
    ///     a: First input matrix (2D tensor)
    ///     b: Second input matrix (2D tensor)
    ///     c: Optional bias matrix (2D tensor, default: None)
    ///     alpha: Scalar multiplier for A*B (default: 1.0)
    ///     beta: Scalar multiplier for C (default: 1.0)
    ///     a_transpose: Whether to transpose A (default: False)
    ///     b_transpose: Whether to transpose B (default: False)
    ///
    ///Returns:
    ///     MLOperand: Output matrix [M, N]
    ///
    /// Example:
    ///     # Standard multiplication: Y = A * B
    ///     y = builder.gemm(a, b)
    ///
    ///     # With bias and transposed B: Y = A * B^T + C
    ///     y = builder.gemm(a, b, c=bias, b_transpose=True)
    #[pyo3(signature = (a, b, c=None, alpha=1.0, beta=1.0, a_transpose=false, b_transpose=false))]
    fn gemm(
        &mut self,
        a: &PyMLOperand,
        b: &PyMLOperand,
        c: Option<&PyMLOperand>,
        alpha: f32,
        beta: f32,
        a_transpose: bool,
        b_transpose: bool,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_gemm_shape;

        let output_shape = infer_gemm_shape(
            &a.descriptor.shape,
            &b.descriptor.shape,
            a_transpose,
            b_transpose,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: a.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let mut attributes = serde_json::json!({
            "alpha": alpha,
            "beta": beta,
            "a_transpose": a_transpose,
            "b_transpose": b_transpose,
        });

        let mut input_operands = vec![a.id, b.id];

        // Add optional bias operand
        if let Some(c_operand) = c {
            input_operands.push(c_operand.id);
            attributes["has_bias"] = serde_json::json!(true);
        } else {
            attributes["has_bias"] = serde_json::json!(false);
        }

        let operation = Operation {
            op_type: "gemm".to_string(),
            input_operands,
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// 2D Convolution operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     filter: Filter operand (4D tensor)
    ///     strides: Stride along each spatial axis (default: [1, 1])
    ///     dilations: Dilation along each spatial axis (default: [1, 1])
    ///     pads: Padding [begin_h, begin_w, end_h, end_w] (default: [0, 0, 0, 0])
    ///     groups: Number of groups (default: 1)
    ///     input_layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///     filter_layout: Filter layout "oihw", "hwio", "ohwi", or "ihwo" (default: "oihw")
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, filter, strides=None, dilations=None, pads=None, groups=None, input_layout=None, filter_layout=None, bias=None))]
    fn conv2d(
        &mut self,
        input: &PyMLOperand,
        filter: &PyMLOperand,
        strides: Option<Vec<u32>>,
        dilations: Option<Vec<u32>>,
        pads: Option<Vec<u32>>,
        groups: Option<u32>,
        input_layout: Option<&str>,
        filter_layout: Option<&str>,
        bias: Option<&PyMLOperand>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::{
            Conv2dFilterLayout, Conv2dInputLayout, Conv2dOptions, infer_conv2d_shape,
        };

        // Default values matching WebNN spec
        let strides = strides.unwrap_or_else(|| vec![1, 1]);
        let dilations = dilations.unwrap_or_else(|| vec![1, 1]);
        let pads = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);
        let groups = groups.unwrap_or(1);

        // Parse layout strings
        let input_layout_enum = match input_layout.unwrap_or("nchw") {
            "nchw" => Conv2dInputLayout::Nchw,
            "nhwc" => Conv2dInputLayout::Nhwc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid input_layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        let filter_layout_enum = match filter_layout.unwrap_or("oihw") {
            "oihw" => Conv2dFilterLayout::Oihw,
            "hwio" => Conv2dFilterLayout::Hwio,
            "ohwi" => Conv2dFilterLayout::Ohwi,
            "ihwo" => Conv2dFilterLayout::Ihwo,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid filter_layout '{}', must be 'oihw', 'hwio', 'ohwi', or 'ihwo'",
                    other
                )));
            }
        };

        // Create options for shape inference
        let options = Conv2dOptions {
            strides: strides.clone(),
            dilations: dilations.clone(),
            pads: pads.clone(),
            groups,
            input_layout: input_layout_enum,
            filter_layout: filter_layout_enum,
        };

        // Infer output shape
        let output_shape =
            infer_conv2d_shape(&input.descriptor.shape, &filter.descriptor.shape, &options)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Build input_operands list: [input, filter, bias?]
        let mut input_operands = vec![input.id, filter.id];
        if let Some(bias_op) = bias {
            input_operands.push(bias_op.id);
        }

        // Store parameters as JSON attributes
        let mut attributes_map = serde_json::json!({
            "strides": strides,
            "dilations": dilations,
            "pads": pads,
            "groups": groups,
            "inputLayout": input_layout.unwrap_or("nchw"),
            "filterLayout": filter_layout.unwrap_or("oihw"),
        });

        // Add bias flag if present
        if bias.is_some() {
            attributes_map["hasBias"] = serde_json::json!(true);
        }

        let operation = Operation {
            op_type: "conv2d".to_string(),
            input_operands,
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: attributes_map,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// 2D Transposed Convolution operation (deconvolution)
    #[pyo3(signature = (input, filter, strides=None, dilations=None, pads=None, output_padding=None, output_sizes=None, groups=None, input_layout=None, filter_layout=None, bias=None))]
    fn conv_transpose2d(
        &mut self,
        input: &PyMLOperand,
        filter: &PyMLOperand,
        strides: Option<Vec<u32>>,
        dilations: Option<Vec<u32>>,
        pads: Option<Vec<u32>>,
        output_padding: Option<Vec<u32>>,
        output_sizes: Option<Vec<u32>>,
        groups: Option<u32>,
        input_layout: Option<&str>,
        filter_layout: Option<&str>,
        bias: Option<&PyMLOperand>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::{
            Conv2dFilterLayout, Conv2dInputLayout, ConvTranspose2dOptions,
            infer_conv_transpose2d_shape,
        };

        // Default values matching WebNN spec
        let strides = strides.unwrap_or_else(|| vec![1, 1]);
        let dilations = dilations.unwrap_or_else(|| vec![1, 1]);
        let pads = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);
        let output_padding = output_padding.unwrap_or_else(|| vec![0, 0]);
        let groups = groups.unwrap_or(1);

        // Parse layout strings
        let input_layout_enum = match input_layout.unwrap_or("nchw") {
            "nchw" => Conv2dInputLayout::Nchw,
            "nhwc" => Conv2dInputLayout::Nhwc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid input_layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        let filter_layout_enum = match filter_layout.unwrap_or("iohw") {
            "iohw" => Conv2dFilterLayout::Oihw, // Input-Output-Height-Width (reinterpreted for transpose)
            "hwoi" => Conv2dFilterLayout::Ihwo, // Height-Width-Output-Input (reinterpreted for transpose)
            "ohwi" => Conv2dFilterLayout::Ohwi, // Output-Height-Width-Input
            "oihw" => Conv2dFilterLayout::Hwio, // Output-Input-Height-Width (reinterpreted for transpose)
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid filter_layout '{}', must be 'iohw', 'hwoi', 'ohwi', or 'oihw'",
                    other
                )));
            }
        };

        // Create options for shape inference
        let options = ConvTranspose2dOptions {
            strides: strides.clone(),
            dilations: dilations.clone(),
            pads: pads.clone(),
            output_padding: output_padding.clone(),
            output_sizes: output_sizes.clone(),
            groups,
            input_layout: input_layout_enum,
            filter_layout: filter_layout_enum,
        };

        // Infer output shape
        let output_shape = infer_conv_transpose2d_shape(
            &input.descriptor.shape,
            &filter.descriptor.shape,
            &options,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Build input_operands list: [input, filter, bias?]
        let mut input_operands = vec![input.id, filter.id];
        if let Some(bias_op) = bias {
            input_operands.push(bias_op.id);
        }

        // Store parameters as JSON attributes
        let mut attributes = serde_json::json!({
            "strides": strides,
            "dilations": dilations,
            "pads": pads,
            "outputPadding": output_padding,
            "groups": groups,
            "inputLayout": input_layout.unwrap_or("nchw"),
            "filterLayout": filter_layout.unwrap_or("iohw"),
        });

        // Add output_sizes if specified
        if let Some(ref sizes) = output_sizes {
            attributes["outputSizes"] = serde_json::json!(sizes);
        }

        // Add bias flag if present
        if bias.is_some() {
            attributes["hasBias"] = serde_json::json!(true);
        }

        let operation = Operation {
            op_type: "convTranspose2d".to_string(),
            input_operands,
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// 2D Average Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     window_dimensions: Size of the pooling window [height, width] (default: [1, 1])
    ///     strides: Stride along each spatial axis (default: [1, 1])
    ///     dilations: Dilation along each spatial axis (default: [1, 1])
    ///     pads: Padding [begin_h, begin_w, end_h, end_w] (default: [0, 0, 0, 0])
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None))]
    fn average_pool2d(
        &mut self,
        input: &PyMLOperand,
        window_dimensions: Option<Vec<u32>>,
        strides: Option<Vec<u32>>,
        dilations: Option<Vec<u32>>,
        pads: Option<Vec<u32>>,
        layout: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::{Conv2dInputLayout, Pool2dOptions, infer_pool2d_shape};

        // Default values matching WebNN spec
        let window_dimensions = window_dimensions.unwrap_or_else(|| vec![1, 1]);
        let strides = strides.unwrap_or_else(|| vec![1, 1]);
        let dilations = dilations.unwrap_or_else(|| vec![1, 1]);
        let pads = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);

        // Parse layout string
        let layout_enum = match layout.unwrap_or("nchw") {
            "nchw" => Conv2dInputLayout::Nchw,
            "nhwc" => Conv2dInputLayout::Nhwc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        // Create options for shape inference
        let options = Pool2dOptions {
            window_dimensions: window_dimensions.clone(),
            strides: strides.clone(),
            dilations: dilations.clone(),
            pads: pads.clone(),
            layout: layout_enum,
        };

        // Infer output shape
        let output_shape = infer_pool2d_shape(&input.descriptor.shape, &options)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Store parameters as JSON attributes
        let attributes = serde_json::json!({
            "windowDimensions": window_dimensions,
            "strides": strides,
            "dilations": dilations,
            "pads": pads,
            "layout": layout.unwrap_or("nchw"),
        });

        let operation = Operation {
            op_type: "averagePool2d".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// 2D Max Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     window_dimensions: Size of the pooling window [height, width] (default: [1, 1])
    ///     strides: Stride along each spatial axis (default: [1, 1])
    ///     dilations: Dilation along each spatial axis (default: [1, 1])
    ///     pads: Padding [begin_h, begin_w, end_h, end_w] (default: [0, 0, 0, 0])
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, window_dimensions=None, strides=None, dilations=None, pads=None, layout=None))]
    fn max_pool2d(
        &mut self,
        input: &PyMLOperand,
        window_dimensions: Option<Vec<u32>>,
        strides: Option<Vec<u32>>,
        dilations: Option<Vec<u32>>,
        pads: Option<Vec<u32>>,
        layout: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::{Conv2dInputLayout, Pool2dOptions, infer_pool2d_shape};

        // Default values matching WebNN spec
        let window_dimensions = window_dimensions.unwrap_or_else(|| vec![1, 1]);
        let strides = strides.unwrap_or_else(|| vec![1, 1]);
        let dilations = dilations.unwrap_or_else(|| vec![1, 1]);
        let pads = pads.unwrap_or_else(|| vec![0, 0, 0, 0]);

        // Parse layout string
        let layout_enum = match layout.unwrap_or("nchw") {
            "nchw" => Conv2dInputLayout::Nchw,
            "nhwc" => Conv2dInputLayout::Nhwc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        // Create options for shape inference
        let options = Pool2dOptions {
            window_dimensions: window_dimensions.clone(),
            strides: strides.clone(),
            dilations: dilations.clone(),
            pads: pads.clone(),
            layout: layout_enum,
        };

        // Infer output shape
        let output_shape = infer_pool2d_shape(&input.descriptor.shape, &options)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Store parameters as JSON attributes
        let attributes = serde_json::json!({
            "windowDimensions": window_dimensions,
            "strides": strides,
            "dilations": dilations,
            "pads": pads,
            "layout": layout.unwrap_or("nchw"),
        });

        let operation = Operation {
            op_type: "maxPool2d".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Global Average Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand with spatial dimensions reduced to 1x1
    #[pyo3(signature = (input, layout=None))]
    fn global_average_pool(
        &mut self,
        input: &PyMLOperand,
        layout: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::{
            Conv2dInputLayout, GlobalPoolOptions, infer_global_pool_shape,
        };

        // Parse layout string
        let layout_enum = match layout.unwrap_or("nchw") {
            "nchw" => Conv2dInputLayout::Nchw,
            "nhwc" => Conv2dInputLayout::Nhwc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        // Create options for shape inference
        let options = GlobalPoolOptions {
            layout: layout_enum,
        };

        // Infer output shape
        let output_shape = infer_global_pool_shape(&input.descriptor.shape, &options)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Store parameters as JSON attributes
        let attributes = serde_json::json!({
            "layout": layout.unwrap_or("nchw"),
        });

        let operation = Operation {
            op_type: "globalAveragePool".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Global Max Pooling operation
    ///
    /// Args:
    ///     input: Input operand (4D tensor)
    ///     layout: Input layout "nchw" or "nhwc" (default: "nchw")
    ///
    /// Returns:
    ///     MLOperand: The output operand with spatial dimensions reduced to 1x1
    #[pyo3(signature = (input, layout=None))]
    fn global_max_pool(
        &mut self,
        input: &PyMLOperand,
        layout: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::{
            Conv2dInputLayout, GlobalPoolOptions, infer_global_pool_shape,
        };

        // Parse layout string
        let layout_enum = match layout.unwrap_or("nchw") {
            "nchw" => Conv2dInputLayout::Nchw,
            "nhwc" => Conv2dInputLayout::Nhwc,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid layout '{}', must be 'nchw' or 'nhwc'",
                    other
                )));
            }
        };

        // Create options for shape inference
        let options = GlobalPoolOptions {
            layout: layout_enum,
        };

        // Infer output shape
        let output_shape = infer_global_pool_shape(&input.descriptor.shape, &options)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Store parameters as JSON attributes
        let attributes = serde_json::json!({
            "layout": layout.unwrap_or("nchw"),
        });

        let operation = Operation {
            op_type: "globalMaxPool".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    // Normalization operations

    /// Batch Normalization operation
    #[pyo3(signature = (input, mean, variance, scale=None, bias=None, epsilon=1e-5, axis=1))]
    fn batch_normalization(
        &mut self,
        input: &PyMLOperand,
        mean: &PyMLOperand,
        variance: &PyMLOperand,
        scale: Option<&PyMLOperand>,
        bias: Option<&PyMLOperand>,
        epsilon: f32,
        axis: i32,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_batch_normalization_shape;

        // Infer output shape (same as input for batch normalization)
        let output_shape = infer_batch_normalization_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Create output descriptor
        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Build input operands list
        let mut input_operands = vec![input.id, mean.id, variance.id];
        if let Some(s) = scale {
            input_operands.push(s.id);
        }
        if let Some(b) = bias {
            input_operands.push(b.id);
        }

        let attributes = serde_json::json!({
            "epsilon": epsilon,
            "axis": axis,
            "has_scale": scale.is_some(),
            "has_bias": bias.is_some(),
        });

        let operation = Operation {
            op_type: "batchNormalization".to_string(),
            input_operands,
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Instance Normalization operation
    #[pyo3(signature = (input, scale=None, bias=None, epsilon=1e-5, layout=None))]
    fn instance_normalization(
        &mut self,
        input: &PyMLOperand,
        scale: Option<&PyMLOperand>,
        bias: Option<&PyMLOperand>,
        epsilon: f32,
        layout: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_instance_normalization_shape;

        // Infer output shape (same as input for instance normalization)
        let output_shape = infer_instance_normalization_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Create output descriptor
        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Build input operands list
        let mut input_operands = vec![input.id];
        if let Some(s) = scale {
            input_operands.push(s.id);
        }
        if let Some(b) = bias {
            input_operands.push(b.id);
        }

        let attributes = serde_json::json!({
            "epsilon": epsilon,
            "layout": layout.unwrap_or("nchw"),
            "has_scale": scale.is_some(),
            "has_bias": bias.is_some(),
        });

        let operation = Operation {
            op_type: "instanceNormalization".to_string(),
            input_operands,
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Layer Normalization operation
    #[pyo3(signature = (input, scale=None, bias=None, epsilon=1e-5, axes=None))]
    fn layer_normalization(
        &mut self,
        input: &PyMLOperand,
        scale: Option<&PyMLOperand>,
        bias: Option<&PyMLOperand>,
        epsilon: f32,
        axes: Option<Vec<i32>>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_layer_normalization_shape;

        // Infer output shape (same as input for layer normalization)
        let output_shape = infer_layer_normalization_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Create output descriptor
        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Build input operands list
        let mut input_operands = vec![input.id];
        if let Some(s) = scale {
            input_operands.push(s.id);
        }
        if let Some(b) = bias {
            input_operands.push(b.id);
        }

        // Default to normalizing over last dimension if axes not specified
        let norm_axes = axes.unwrap_or_else(|| vec![-1]);

        let attributes = serde_json::json!({
            "epsilon": epsilon,
            "axes": norm_axes,
            "has_scale": scale.is_some(),
            "has_bias": bias.is_some(),
        });

        let operation = Operation {
            op_type: "layerNormalization".to_string(),
            input_operands,
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    // Unary operations

    /// ReLU activation
    fn relu(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("relu", x)
    }

    /// Sigmoid activation
    fn sigmoid(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("sigmoid", x)
    }

    /// Tanh activation
    fn tanh(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("tanh", x)
    }

    /// Softmax activation
    fn softmax(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("softmax", x)
    }

    // Element-wise operations - Basic math

    /// Element-wise absolute value
    fn abs(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("abs", x)
    }

    /// Element-wise ceiling (round up)
    fn ceil(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("ceil", x)
    }

    /// Element-wise floor (round down)
    fn floor(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("floor", x)
    }

    /// Element-wise rounding to nearest integer
    fn round(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("round", x)
    }

    /// Element-wise negation
    fn neg(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("neg", x)
    }

    /// Element-wise sign (-1, 0, 1)
    fn sign(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("sign", x)
    }

    // Element-wise operations - Exponential and logarithmic

    /// Element-wise natural exponential (e^x)
    fn exp(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("exp", x)
    }

    /// Element-wise natural logarithm
    fn log(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("log", x)
    }

    /// Element-wise square root
    fn sqrt(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("sqrt", x)
    }

    /// Element-wise reciprocal (1/x)
    fn reciprocal(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("reciprocal", x)
    }

    // Element-wise operations - Trigonometric

    /// Element-wise sine
    fn sin(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("sin", x)
    }

    /// Element-wise cosine
    fn cos(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("cos", x)
    }

    /// Element-wise tangent
    fn tan(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("tan", x)
    }

    /// Element-wise arcsine
    fn asin(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("asin", x)
    }

    /// Element-wise arccosine
    fn acos(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("acos", x)
    }

    /// Element-wise arctangent
    fn atan(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("atan", x)
    }

    // Element-wise operations - Hyperbolic

    /// Element-wise hyperbolic sine
    fn sinh(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("sinh", x)
    }

    /// Element-wise hyperbolic cosine
    fn cosh(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("cosh", x)
    }

    /// Element-wise hyperbolic arcsine
    fn asinh(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("asinh", x)
    }

    /// Element-wise hyperbolic arccosine
    fn acosh(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("acosh", x)
    }

    /// Element-wise hyperbolic arctangent
    fn atanh(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("atanh", x)
    }

    // Element-wise operations - Special functions

    /// Element-wise error function
    fn erf(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("erf", x)
    }

    /// Identity operation (returns input unchanged)
    fn identity(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        self.unary_op("identity", x)
    }

    // Logic operations

    /// Element-wise equality comparison
    fn equal(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_equal_shape;

        let output_shape = infer_equal_shape(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8, // Boolean output as uint8
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "equal".to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise greater than comparison
    fn greater(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_greater_shape;

        let output_shape = infer_greater_shape(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "greater".to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise greater than or equal comparison
    fn greater_or_equal(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_greater_or_equal_shape;

        let output_shape = infer_greater_or_equal_shape(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "greaterOrEqual".to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise less than comparison
    fn lesser(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_lesser_shape;

        let output_shape = infer_lesser_shape(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "lesser".to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise less than or equal comparison
    fn lesser_or_equal(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_lesser_or_equal_shape;

        let output_shape = infer_lesser_or_equal_shape(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "lesserOrEqual".to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise logical NOT
    fn logical_not(&mut self, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_logical_not_shape;

        let output_shape = infer_logical_not_shape(&x.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "logicalNot".to_string(),
            input_operands: vec![x.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise logical AND
    fn logical_and(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_logical_and_shape;

        let output_shape = infer_logical_and_shape(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "logicalAnd".to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise logical OR
    fn logical_or(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_logical_or_shape;

        let output_shape = infer_logical_or_shape(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "logicalOr".to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Element-wise logical XOR
    fn logical_xor(&mut self, a: &PyMLOperand, b: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_logical_xor_shape;

        let output_shape = infer_logical_xor_shape(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: DataType::Uint8,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "logicalXor".to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// DequantizeLinear operation
    /// Converts quantized integer values to floating-point representation
    /// Formula: output = (input - zeroPoint) * scale
    fn dequantize_linear(
        &mut self,
        input: &PyMLOperand,
        scale: &PyMLOperand,
        zero_point: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_dequantize_linear_shape;

        let output_shape = infer_dequantize_linear_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Output is always float32 for dequantization
        let output_descriptor = OperandDescriptor {
            data_type: DataType::Float32,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "dequantizeLinear".to_string(),
            input_operands: vec![input.id, scale.id, zero_point.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// QuantizeLinear operation
    /// Converts floating-point values to quantized integer representation
    /// Formula: output = input / scale + zeroPoint
    fn quantize_linear(
        &mut self,
        input: &PyMLOperand,
        scale: &PyMLOperand,
        zero_point: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_quantize_linear_shape;

        let output_shape = infer_quantize_linear_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Output data type matches zero_point's data type (typically int8 or uint8)
        let output_descriptor = OperandDescriptor {
            data_type: zero_point.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "quantizeLinear".to_string(),
            input_operands: vec![input.id, scale.id, zero_point.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Reshape operation
    fn reshape(&mut self, x: &PyMLOperand, new_shape: Vec<u32>) -> PyResult<PyMLOperand> {
        // Validate that reshape is possible
        validate_reshape(&x.descriptor.shape, &new_shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: x.descriptor.data_type,
            shape: new_shape.clone(),
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "reshape".to_string(),
            input_operands: vec![x.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({
                "newShape": new_shape
            }),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    // Reduction operations

    /// Reduce Sum operation
    ///
    /// Reduces the input tensor by summing elements along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_sum(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceSum", input, axes, keep_dimensions)
    }

    /// Reduce Mean operation
    ///
    /// Reduces the input tensor by computing the arithmetic mean along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_mean(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceMean", input, axes, keep_dimensions)
    }

    /// Reduce Max operation
    ///
    /// Reduces the input tensor by computing the maximum value along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_max(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceMax", input, axes, keep_dimensions)
    }

    /// Reduce Min operation
    ///
    /// Reduces the input tensor by computing the minimum value along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_min(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceMin", input, axes, keep_dimensions)
    }

    /// Reduce Product operation
    ///
    /// Reduces the input tensor by computing the product of elements along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_product(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceProduct", input, axes, keep_dimensions)
    }

    /// Reduce L1 operation
    ///
    /// Reduces the input tensor by computing the L1 norm (sum of absolute values) along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_l1(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceL1", input, axes, keep_dimensions)
    }

    /// Reduce L2 operation
    ///
    /// Reduces the input tensor by computing the L2 norm (Euclidean norm) along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_l2(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceL2", input, axes, keep_dimensions)
    }

    /// Reduce Log Sum operation
    ///
    /// Reduces the input tensor by computing the natural logarithm of the sum along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_log_sum(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceLogSum", input, axes, keep_dimensions)
    }

    /// Reduce Log Sum Exp operation
    ///
    /// Reduces the input tensor by computing the log of the sum of exponentials along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_log_sum_exp(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceLogSumExp", input, axes, keep_dimensions)
    }

    /// Reduce Sum Square operation
    ///
    /// Reduces the input tensor by computing the sum of squares along specified axes.
    ///
    /// Args:
    ///     input: Input operand
    ///     axes: Axes to reduce (default: all axes)
    ///     keep_dimensions: Keep reduced dimensions with size 1 (default: false)
    ///
    /// Returns:
    ///     MLOperand: The output operand
    #[pyo3(signature = (input, axes=None, keep_dimensions=false))]
    fn reduce_sum_square(
        &mut self,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        self.reduce_op("reduceSumSquare", input, axes, keep_dimensions)
    }

    // Tensor manipulation operations

    /// Transpose operation
    ///
    /// Reorders the dimensions of a tensor according to a permutation.
    /// If no permutation is provided, reverses the dimensions.
    ///
    /// Args:
    ///     input: Input operand
    ///     permutation: Optional permutation of dimensions (default: reverse dimensions)
    ///
    /// Returns:
    ///     MLOperand: The transposed output operand
    #[pyo3(signature = (input, permutation=None))]
    fn transpose(
        &mut self,
        input: &PyMLOperand,
        permutation: Option<Vec<u32>>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_transpose_shape;

        // Infer output shape
        let output_shape = infer_transpose_shape(&input.descriptor.shape, permutation.as_deref())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Store parameters as JSON attributes
        let mut attributes = serde_json::json!({});
        if let Some(perm) = permutation {
            attributes["permutation"] = serde_json::json!(perm);
        }

        let operation = Operation {
            op_type: "transpose".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Concat operation
    ///
    /// Concatenates multiple tensors along a specified axis.
    ///
    /// Args:
    ///     inputs: List of input operands to concatenate
    ///     axis: Axis along which to concatenate
    ///
    /// Returns:
    ///     MLOperand: The concatenated output operand
    fn concat(&mut self, inputs: Vec<PyMLOperand>, axis: u32) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_concat_shape;

        if inputs.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Concat requires at least one input",
            ));
        }

        // Collect input shapes
        let input_shapes: Vec<Vec<u32>> = inputs
            .iter()
            .map(|op| op.descriptor.shape.clone())
            .collect();

        // Infer output shape
        let output_shape = infer_concat_shape(&input_shapes, axis)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: inputs[0].descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Collect input IDs
        let input_ids: Vec<u32> = inputs.iter().map(|op| op.id).collect();

        let attributes = serde_json::json!({
            "axis": axis,
        });

        let operation = Operation {
            op_type: "concat".to_string(),
            input_operands: input_ids,
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Slice operation
    ///
    /// Extracts a contiguous sub-tensor from the input.
    ///
    /// Args:
    ///     input: Input operand
    ///     starts: Starting indices for each dimension
    ///     sizes: Size of the slice for each dimension
    ///     strides: Optional stride values for each dimension (defaults to 1)
    ///
    /// Returns:
    ///     MLOperand: The sliced output operand
    #[pyo3(signature = (input, starts, sizes, strides=None))]
    fn slice(
        &mut self,
        input: &PyMLOperand,
        starts: Vec<u32>,
        sizes: Vec<u32>,
        strides: Option<Vec<i32>>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_slice_shape;

        // Infer output shape
        let output_shape = infer_slice_shape(&input.descriptor.shape, &starts, &sizes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let mut attributes = serde_json::json!({
            "starts": starts,
            "sizes": sizes,
        });

        // Add strides if provided
        if let Some(strides_val) = strides {
            attributes["strides"] = serde_json::json!(strides_val);
        }

        let attributes = attributes;

        let operation = Operation {
            op_type: "slice".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Expand operation
    ///
    /// Broadcasts a tensor to a larger shape. Dimensions of size 1 can be expanded to larger sizes.
    ///
    /// Args:
    ///     input: Input operand
    ///     new_shape: Target shape for expansion
    ///
    /// Returns:
    ///     MLOperand: The expanded output operand
    fn expand(&mut self, input: &PyMLOperand, new_shape: Vec<u32>) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_expand_shape;

        // Infer output shape
        let output_shape = infer_expand_shape(&input.descriptor.shape, &new_shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "newShape": new_shape,
        });

        let operation = Operation {
            op_type: "expand".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Gather operation
    ///
    /// Gathers values from input tensor along an axis according to indices.
    ///
    /// Args:
    ///     input: Input operand
    ///     indices: Indices tensor
    ///     axis: Axis along which to gather (default: 0)
    ///
    /// Returns:
    ///     MLOperand: The gathered output operand
    #[pyo3(signature = (input, indices, axis=0))]
    fn gather(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        axis: u32,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_gather_shape;

        // Infer output shape
        let output_shape =
            infer_gather_shape(&input.descriptor.shape, &indices.descriptor.shape, axis)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "axis": axis,
        });

        let operation = Operation {
            op_type: "gather".to_string(),
            input_operands: vec![input.id, indices.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Split operation
    ///
    /// Splits a tensor into multiple sub-tensors along an axis.
    ///
    /// Args:
    ///     input: Input operand
    ///     splits: Either number of equal splits (int) or list of split sizes
    ///     axis: Axis along which to split (default: 0)
    ///
    /// Returns:
    ///     List[MLOperand]: List of output operands
    #[pyo3(signature = (input, splits, axis=0))]
    fn split(
        &mut self,
        _py: Python,
        input: &PyMLOperand,
        splits: &Bound<'_, PyAny>,
        axis: u32,
    ) -> PyResult<Vec<PyMLOperand>> {
        use crate::shape_inference::{SplitSpec, infer_split_shapes};

        // Determine split specification
        let split_spec = if let Ok(count) = splits.extract::<u32>() {
            SplitSpec::Count(count)
        } else if let Ok(sizes) = splits.extract::<Vec<u32>>() {
            SplitSpec::Sizes(sizes)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "splits must be either an integer or a list of integers",
            ));
        };

        // Infer output shapes
        let output_shapes = infer_split_shapes(&input.descriptor.shape, &split_spec, axis)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Create output operands
        let mut py_operands = Vec::new();
        let mut output_ids = Vec::new();

        for output_shape in &output_shapes {
            let output_descriptor = OperandDescriptor {
                data_type: input.descriptor.data_type,
                shape: output_shape.clone(),
                pending_permutation: Vec::new(),
            };

            let output_id = self.next_operand_id;
            self.next_operand_id += 1;

            let output_operand = Operand {
                descriptor: output_descriptor.clone(),
                kind: OperandKind::Output,
                name: None,
            };
            self.operands.push(output_operand);

            let py_operand =
                PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
            self.operand_map.insert(output_id, py_operand.clone());
            py_operands.push(py_operand);
            output_ids.push(output_id);
        }

        // Create operation with multiple outputs
        let attributes = match split_spec {
            SplitSpec::Count(count) => serde_json::json!({
                "axis": axis,
                "splits": count,
            }),
            SplitSpec::Sizes(sizes) => serde_json::json!({
                "axis": axis,
                "splits": sizes,
            }),
        };

        let operation = Operation {
            op_type: "split".to_string(),
            input_operands: vec![input.id],
            output_operand: None,
            output_operands: output_ids,
            attributes,
            label: None,
        };

        self.operations.push(operation);

        Ok(py_operands)
    }

    /// Where operation
    ///
    /// Selects elements from trueValue or falseValue based on condition.
    /// All inputs are broadcast to a common shape.
    ///
    /// Args:
    ///     condition: Boolean condition tensor
    ///     true_value: Values to select when condition is true
    ///     false_value: Values to select when condition is false
    ///
    /// Returns:
    ///     MLOperand: The output operand
    fn where_(
        &mut self,
        condition: &PyMLOperand,
        true_value: &PyMLOperand,
        false_value: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_where_shape;

        // Infer output shape
        let output_shape = infer_where_shape(
            &condition.descriptor.shape,
            &true_value.descriptor.shape,
            &false_value.descriptor.shape,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: true_value.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "where".to_string(),
            input_operands: vec![condition.id, true_value.id, false_value.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Pad operation
    ///
    /// Adds padding around the input tensor.
    ///
    /// Args:
    ///     input: Input operand
    ///     padding: Padding values [begin_0, begin_1, ..., end_0, end_1, ...]
    ///     mode: Padding mode ("constant", "edge", "reflection", "symmetric") (default: "constant")
    ///     value: Padding value for constant mode (default: 0.0)
    ///
    /// Returns:
    ///     MLOperand: The padded output operand
    #[pyo3(signature = (input, padding, mode=None, value=None))]
    fn pad(
        &mut self,
        input: &PyMLOperand,
        padding: Vec<u32>,
        mode: Option<&str>,
        value: Option<f32>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_pad_shape;

        // Infer output shape
        let output_shape = infer_pad_shape(&input.descriptor.shape, &padding)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Validate mode
        let mode_str = mode.unwrap_or("constant");
        if !["constant", "edge", "reflection", "symmetric"].contains(&mode_str) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid pad mode '{}', must be 'constant', 'edge', 'reflection', or 'symmetric'",
                mode_str
            )));
        }

        let mut attributes = serde_json::json!({
            "padding": padding,
            "mode": mode_str,
        });

        if let Some(v) = value {
            attributes["value"] = serde_json::json!(v);
        }

        let operation = Operation {
            op_type: "pad".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// GELU activation operation
    ///
    /// Args:
    ///     input: The input tensor
    ///
    /// Returns:
    ///     MLOperand: Output operand with GELU activation applied
    fn gelu(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_gelu_shape;

        let output_shape = infer_gelu_shape(&input.descriptor.shape);

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "gelu".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Squeeze operation (remove dimensions of size 1)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axes: Optional sequence of axes to squeeze. If None, all dimensions of size 1 are removed
    ///
    /// Returns:
    ///     MLOperand: Output operand with dimensions squeezed
    #[pyo3(signature = (input, axes=None))]
    fn squeeze(&mut self, input: &PyMLOperand, axes: Option<Vec<u32>>) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_squeeze_shape;

        let output_shape = infer_squeeze_shape(&input.descriptor.shape, axes.as_deref())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let mut attributes = serde_json::json!({});
        if let Some(axes) = axes {
            attributes["axes"] = serde_json::json!(axes);
        }

        let operation = Operation {
            op_type: "squeeze".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Unsqueeze operation (add dimensions of size 1)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axes: Sequence of axes where dimensions of size 1 should be inserted
    ///
    /// Returns:
    ///     MLOperand: Output operand with dimensions unsqueezed
    fn unsqueeze(&mut self, input: &PyMLOperand, axes: Vec<u32>) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_unsqueeze_shape;

        let output_shape = infer_unsqueeze_shape(&input.descriptor.shape, &axes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "axes": axes
        });

        let operation = Operation {
            op_type: "unsqueeze".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// ArgMax operation (find indices of maximum values)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axis: The axis to reduce along
    ///     keep_dimensions: If True, keep the reduced axis with size 1. Default is False
    ///     output_data_type: Output data type for indices ("int32" or "int64"). Default is "int64"
    ///
    /// Returns:
    ///     MLOperand: Output operand containing indices of maximum values
    #[pyo3(signature = (input, axis, keep_dimensions=false, output_data_type=None))]
    fn arg_max(
        &mut self,
        input: &PyMLOperand,
        axis: u32,
        keep_dimensions: bool,
        output_data_type: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_arg_reduce_shape;

        let output_shape = infer_arg_reduce_shape(&input.descriptor.shape, axis, keep_dimensions)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Parse output data type, default to int64
        let output_type = match output_data_type {
            Some("int32") => DataType::Int32,
            Some("int64") | None => DataType::Int64,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid output_data_type '{}', must be 'int32' or 'int64'",
                    other
                )));
            }
        };

        let output_descriptor = OperandDescriptor {
            data_type: output_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let mut attributes = serde_json::json!({
            "axis": axis,
            "keepDimensions": keep_dimensions
        });
        if let Some(dtype) = output_data_type {
            attributes["outputDataType"] = serde_json::json!(dtype);
        }

        let operation = Operation {
            op_type: "argMax".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// ArgMin operation (find indices of minimum values)
    ///
    /// Args:
    ///     input: The input tensor
    ///     axis: The axis to reduce along
    ///     keep_dimensions: If True, keep the reduced axis with size 1. Default is False
    ///     output_data_type: Output data type for indices ("int32" or "int64"). Default is "int64"
    ///
    /// Returns:
    ///     MLOperand: Output operand containing indices of minimum values
    #[pyo3(signature = (input, axis, keep_dimensions=false, output_data_type=None))]
    fn arg_min(
        &mut self,
        input: &PyMLOperand,
        axis: u32,
        keep_dimensions: bool,
        output_data_type: Option<&str>,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_arg_reduce_shape;

        let output_shape = infer_arg_reduce_shape(&input.descriptor.shape, axis, keep_dimensions)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Parse output data type, default to int64
        let output_type = match output_data_type {
            Some("int32") => DataType::Int32,
            Some("int64") | None => DataType::Int64,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid output_data_type '{}', must be 'int32' or 'int64'",
                    other
                )));
            }
        };

        let output_descriptor = OperandDescriptor {
            data_type: output_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let mut attributes = serde_json::json!({
            "axis": axis,
            "keepDimensions": keep_dimensions
        });
        if let Some(dtype) = output_data_type {
            attributes["outputDataType"] = serde_json::json!(dtype);
        }

        let operation = Operation {
            op_type: "argMin".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Cast operation (type conversion)
    ///
    /// Args:
    ///     input: The input tensor
    ///     data_type: Target data type ("float32", "float16", "int32", "uint32", "int8", "uint8", "int64")
    ///
    /// Returns:
    ///     MLOperand: Output operand with converted type
    fn cast(&mut self, input: &PyMLOperand, data_type: &str) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_cast_shape;

        let output_shape = infer_cast_shape(&input.descriptor.shape);

        // Parse target data type
        let target_type = match data_type {
            "float32" => DataType::Float32,
            "float16" => DataType::Float16,
            "int32" => DataType::Int32,
            "uint32" => DataType::Uint32,
            "int8" => DataType::Int8,
            "uint8" => DataType::Uint8,
            "int64" => DataType::Int64,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid data_type '{}', must be one of: float32, float16, int32, uint32, int8, uint8, int64",
                    other
                )));
            }
        };

        let output_descriptor = OperandDescriptor {
            data_type: target_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "to": data_type
        });

        let operation = Operation {
            op_type: "cast".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Build the computational graph
    ///
    /// Args:
    ///     outputs: Dictionary mapping output names to MLOperand objects
    ///
    /// Returns:
    ///     MLGraph: The compiled graph
    fn build(&mut self, outputs: &Bound<'_, PyDict>) -> PyResult<PyMLGraph> {
        let mut output_operands = Vec::new();

        // Mark outputs and collect output IDs
        for (name, operand_obj) in outputs.iter() {
            let name_str: String = name.extract()?;
            let operand: PyMLOperand = operand_obj.extract()?;

            // Update the operand to mark it as an output with the given name
            if let Some(op) = self.operands.get_mut(operand.id as usize) {
                op.kind = OperandKind::Output;
                op.name = Some(name_str);
            }
            output_operands.push(operand.id);
        }

        // Create GraphInfo
        let graph_info = GraphInfo {
            operands: self.operands.clone(),
            input_operands: self.input_operands.clone(),
            output_operands,
            operations: self.operations.clone(),
            constant_operand_ids_to_handles: self.constant_data_map.clone(),
            id_to_constant_tensor_operand_map: HashMap::new(),
        };

        // Validate the graph
        let validator = GraphValidator::new(&graph_info, Default::default());
        validator.validate().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Graph validation failed: {}", e))
        })?;

        Ok(PyMLGraph::new(graph_info))
    }

    /// Scatter elements operation
    ///
    /// Updates values in input tensor at indices specified by indices tensor.
    ///
    /// Args:
    ///     input: The base tensor to scatter values into
    ///     indices: Integer tensor of same rank as input, containing indices
    ///     updates: Tensor of same rank as input, containing values to scatter
    ///     axis: Axis along which to scatter (can be negative)
    ///
    /// Returns:
    ///     MLOperand: Output operand with scattered values
    fn scatter_elements(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        updates: &PyMLOperand,
        axis: i32,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_scatter_elements_shape;

        let output_shape = infer_scatter_elements_shape(
            &input.descriptor.shape,
            &indices.descriptor.shape,
            &updates.descriptor.shape,
            axis,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "axis": axis,
        });

        let operation = Operation {
            op_type: "scatterElements".to_string(),
            input_operands: vec![input.id, indices.id, updates.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// ScatterND operation
    ///
    /// Scatter updates into a tensor using multi-dimensional indices.
    ///
    /// Args:
    ///     input: Base tensor of rank r >= 1
    ///     indices: Integer tensor of rank q >= 1
    ///     updates: Tensor containing values to scatter
    ///
    /// Returns:
    ///     MLOperand: Output operand with scattered values
    fn scatter_nd(
        &mut self,
        input: &PyMLOperand,
        indices: &PyMLOperand,
        updates: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_scatter_nd_shape;

        let output_shape = infer_scatter_nd_shape(
            &input.descriptor.shape,
            &indices.descriptor.shape,
            &updates.descriptor.shape,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({});

        let operation = Operation {
            op_type: "scatterND".to_string(),
            input_operands: vec![input.id, indices.id, updates.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Tile operation
    ///
    /// Repeats tensor along each dimension according to repetitions.
    ///
    /// Args:
    ///     input: Input tensor to tile
    ///     repetitions: Number of repetitions for each dimension
    ///
    /// Returns:
    ///     MLOperand: Output operand with tiled values
    fn tile(&mut self, input: &PyMLOperand, repetitions: Vec<u32>) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_tile_shape;

        let output_shape = infer_tile_shape(&input.descriptor.shape, &repetitions)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "repetitions": repetitions,
        });

        let operation = Operation {
            op_type: "tile".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Triangular operation
    ///
    /// Extract upper or lower triangular part of matrix (last 2 dimensions).
    ///
    /// Args:
    ///     input: Input tensor (rank >= 2)
    ///     upper: Extract upper triangle if true, lower if false
    ///     diagonal: Diagonal offset (0=main, positive=above, negative=below)
    ///
    /// Returns:
    ///     MLOperand: Output operand with non-triangular elements zeroed
    fn triangular(
        &mut self,
        input: &PyMLOperand,
        upper: bool,
        diagonal: i32,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_triangular_shape;

        let output_shape = infer_triangular_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "upper": upper,
            "diagonal": diagonal,
        });

        let operation = Operation {
            op_type: "triangular".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// hardSigmoid activation operation
    ///
    /// Computes element-wise: y = max(0, min(1, alpha * x + beta))
    ///
    /// Args:
    ///     input: Input tensor
    ///     alpha: Multiplicative coefficient (default: 0.2)
    ///     beta: Additive offset (default: 0.5)
    ///
    /// Returns:
    ///     MLOperand: Output operand with values clipped to [0, 1]
    #[pyo3(signature = (input, alpha=0.2, beta=0.5))]
    fn hard_sigmoid(
        &mut self,
        input: &PyMLOperand,
        alpha: f32,
        beta: f32,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_hardsigmoid_shape;

        let output_shape = infer_hardsigmoid_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "alpha": alpha,
            "beta": beta,
        });

        let operation = Operation {
            op_type: "hardSigmoid".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// hardSwish activation operation
    ///
    /// Computes element-wise: y = x * max(0, min(1, alpha * x + beta))
    ///
    /// Args:
    ///     input: Input tensor
    ///     alpha: Multiplicative coefficient for hardSigmoid (default: 1/6)
    ///     beta: Additive offset for hardSigmoid (default: 0.5)
    ///
    /// Returns:
    ///     MLOperand: Output operand
    #[pyo3(signature = (input, alpha=0.16666666666666666, beta=0.5))]
    fn hard_swish(&mut self, input: &PyMLOperand, alpha: f32, beta: f32) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_hardswish_shape;

        let output_shape = infer_hardswish_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "alpha": alpha,
            "beta": beta,
        });

        let operation = Operation {
            op_type: "hardSwish".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// softplus activation operation
    ///
    /// Computes element-wise: y = log(1 + exp(x))
    /// Smooth approximation of ReLU
    ///
    /// Args:
    ///     input: Input tensor
    ///
    /// Returns:
    ///     MLOperand: Output operand with positive values
    fn softplus(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_softplus_shape;

        let output_shape = infer_softplus_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({});

        let operation = Operation {
            op_type: "softplus".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// softsign activation operation
    ///
    /// Computes element-wise: y = x / (1 + |x|)
    /// Bounded activation with output in (-1, 1)
    ///
    /// Args:
    ///     input: Input tensor
    ///
    /// Returns:
    ///     MLOperand: Output operand with values in (-1, 1)
    fn softsign(&mut self, input: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_softsign_shape;

        let output_shape = infer_softsign_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({});

        let operation = Operation {
            op_type: "softsign".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// clamp operation (element-wise clamping)
    ///
    /// Constrains every element in the input tensor between min and max values:
    ///   y = max(min_value, min(x, max_value))
    ///
    /// Args:
    ///     input: Input tensor
    ///     min_value: Minimum value (default: negative infinity)
    ///     max_value: Maximum value (default: positive infinity)
    ///
    /// Returns:
    ///     MLOperand: Output operand with values clamped to [min_value, max_value]
    ///
    /// Example:
    ///     # ReLU6: clamp(x, 0, 6)
    ///     relu6 = builder.clamp(x, min_value=0.0, max_value=6.0)
    #[pyo3(signature = (input, min_value=f32::NEG_INFINITY, max_value=f32::INFINITY))]
    fn clamp(
        &mut self,
        input: &PyMLOperand,
        min_value: f32,
        max_value: f32,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_clamp_shape;

        // Validate min <= max
        if min_value > max_value {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "clamp min_value ({}) must be <= max_value ({})",
                min_value, max_value
            )));
        }

        let output_shape = infer_clamp_shape(&input.descriptor.shape);

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "min_value": min_value,
            "max_value": max_value,
        });

        let operation = Operation {
            op_type: "clamp".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// elu activation operation (Exponential Linear Unit)
    ///
    /// Computes element-wise:
    ///   y = x if x >= 0
    ///   y = alpha * (exp(x) - 1) if x < 0
    ///
    /// Args:
    ///     input: Input tensor
    ///     alpha: Coefficient for negative values (default: 1.0)
    ///
    /// Returns:
    ///     MLOperand: Output operand
    #[pyo3(signature = (input, alpha=1.0))]
    fn elu(&mut self, input: &PyMLOperand, alpha: f32) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_elu_shape;

        let output_shape = infer_elu_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "alpha": alpha,
        });

        let operation = Operation {
            op_type: "elu".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// leakyRelu activation operation (Leaky Rectified Linear Unit)
    ///
    /// Computes element-wise:
    ///   y = x if x >= 0
    ///   y = alpha * x if x < 0
    ///
    /// Args:
    ///     input: Input tensor
    ///     alpha: Leakage coefficient for negative values (default: 0.01)
    ///
    /// Returns:
    ///     MLOperand: Output operand
    #[pyo3(signature = (input, alpha=0.01))]
    fn leaky_relu(&mut self, input: &PyMLOperand, alpha: f32) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_leakyrelu_shape;

        let output_shape = infer_leakyrelu_shape(&input.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({
            "alpha": alpha,
        });

        let operation = Operation {
            op_type: "leakyRelu".to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// prelu activation operation (Parametric Rectified Linear Unit)
    ///
    /// Computes element-wise:
    ///   y = x if x >= 0
    ///   y = slope * x if x < 0
    ///
    /// Args:
    ///     input: Input tensor
    ///     slope: Learnable slope tensor (must be unidirectionally broadcastable to input)
    ///
    /// Returns:
    ///     MLOperand: Output operand
    fn prelu(&mut self, input: &PyMLOperand, slope: &PyMLOperand) -> PyResult<PyMLOperand> {
        use crate::shape_inference::infer_prelu_shape;

        let output_shape = infer_prelu_shape(&input.descriptor.shape, &slope.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let attributes = serde_json::json!({});

        let operation = Operation {
            op_type: "prelu".to_string(),
            input_operands: vec![input.id, slope.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }
}

impl PyMLGraphBuilder {
    /// Create a new graph builder (Rust-accessible constructor)
    pub fn create() -> Self {
        Self {
            operands: Vec::new(),
            operations: Vec::new(),
            input_operands: Vec::new(),
            next_operand_id: 0,
            operand_map: HashMap::new(),
            constant_data_map: HashMap::new(),
        }
    }

    /// Helper for binary operations with broadcasting
    fn binary_op(
        &mut self,
        op_type: &str,
        a: &PyMLOperand,
        b: &PyMLOperand,
    ) -> PyResult<PyMLOperand> {
        // Compute broadcasted output shape
        let output_shape = broadcast_shapes(&a.descriptor.shape, &b.descriptor.shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: a.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: op_type.to_string(),
            input_operands: vec![a.id, b.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Helper for unary operations
    fn unary_op(&mut self, op_type: &str, x: &PyMLOperand) -> PyResult<PyMLOperand> {
        let output_descriptor = x.descriptor.clone();

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: op_type.to_string(),
            input_operands: vec![x.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes: serde_json::json!({}),
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }

    /// Helper for reduction operations
    fn reduce_op(
        &mut self,
        op_type: &str,
        input: &PyMLOperand,
        axes: Option<Vec<u32>>,
        keep_dimensions: bool,
    ) -> PyResult<PyMLOperand> {
        use crate::shape_inference::{ReduceOptions, infer_reduce_shape};

        // Create reduction options
        let options = ReduceOptions {
            axes: axes.clone().unwrap_or_default(),
            keep_dimensions,
        };

        // Infer output shape
        let output_shape = infer_reduce_shape(&input.descriptor.shape, &options)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: input.descriptor.data_type,
            shape: output_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        // Store parameters as JSON attributes
        let attributes = serde_json::json!({
            "axes": axes.unwrap_or_default(),
            "keepDimensions": keep_dimensions,
        });

        let operation = Operation {
            op_type: op_type.to_string(),
            input_operands: vec![input.id],
            output_operand: Some(output_id),
            output_operands: Vec::new(),
            attributes,
            label: None,
        };

        self.operations.push(operation);

        let output_operand = Operand {
            descriptor: output_descriptor.clone(),
            kind: OperandKind::Output,
            name: None,
        };
        self.operands.push(output_operand);

        let py_operand = PyMLOperand::new(output_id, output_descriptor, OperandKind::Output, None);
        self.operand_map.insert(output_id, py_operand.clone());

        Ok(py_operand)
    }
}
