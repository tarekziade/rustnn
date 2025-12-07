use super::graph::PyMLGraph;
use super::operand::{PyMLOperand, parse_data_type};
use crate::graph::{ConstantData, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation};
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
            output_operand: output_id,
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
    #[pyo3(signature = (input, filter, strides=None, dilations=None, pads=None, groups=None, input_layout=None, filter_layout=None))]
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

        // Store parameters as JSON attributes
        let attributes = serde_json::json!({
            "strides": strides,
            "dilations": dilations,
            "pads": pads,
            "groups": groups,
            "inputLayout": input_layout.unwrap_or("nchw"),
            "filterLayout": filter_layout.unwrap_or("oihw"),
        });

        let operation = Operation {
            op_type: "conv2d".to_string(),
            input_operands: vec![input.id, filter.id],
            output_operand: output_id,
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

    /// 2D Transposed Convolution operation (deconvolution)
    #[pyo3(signature = (input, filter, strides=None, dilations=None, pads=None, output_padding=None, output_sizes=None, groups=None, input_layout=None, filter_layout=None))]
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

        // Store parameters as JSON attributes
        let mut attributes = serde_json::json!({
            "strides": strides,
            "dilations": dilations,
            "pads": pads,
            "outputPadding": output_padding,
            "groups": groups,
            "inputLayout": input_layout.unwrap_or("nchw"),
            "filterLayout": filter_layout.unwrap_or("oihw"),
        });

        // Add output_sizes if specified
        if let Some(ref sizes) = output_sizes {
            attributes["outputSizes"] = serde_json::json!(sizes);
        }

        let operation = Operation {
            op_type: "convTranspose2d".to_string(),
            input_operands: vec![input.id, filter.id],
            output_operand: output_id,
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
            output_operand: output_id,
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
            output_operand: output_id,
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

    /// Reshape operation
    fn reshape(&mut self, x: &PyMLOperand, new_shape: Vec<u32>) -> PyResult<PyMLOperand> {
        // Validate that reshape is possible
        validate_reshape(&x.descriptor.shape, &new_shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let output_descriptor = OperandDescriptor {
            data_type: x.descriptor.data_type,
            shape: new_shape,
            pending_permutation: Vec::new(),
        };

        let output_id = self.next_operand_id;
        self.next_operand_id += 1;

        let operation = Operation {
            op_type: "reshape".to_string(),
            input_operands: vec![x.id],
            output_operand: output_id,
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
            output_operand: output_id,
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
            output_operand: output_id,
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
}
