use crate::graph::{DataType, OperandDescriptor};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

/// MLTensorDescriptor - Describes tensor properties and usage flags
///
/// Following the W3C WebNN MLTensor Explainer:
/// https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md
#[derive(Clone, Debug)]
pub struct MLTensorDescriptor {
    pub descriptor: OperandDescriptor,
    /// If true, tensor data can be read back to CPU
    pub readable: bool,
    /// If true, tensor data can be written from CPU
    pub writable: bool,
    /// If true, tensor can be exported for use as GPU texture (future use)
    pub exportable_to_gpu: bool,
}

/// MLTensor - Represents an opaque typed tensor with data storage
///
/// MLTensor is used for explicit tensor management in WebNN, allowing
/// pre-allocation of input/output buffers and explicit data transfer.
///
/// Following the W3C WebNN MLTensor Explainer:
/// https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md
#[pyclass(name = "MLTensor")]
#[derive(Clone)]
pub struct PyMLTensor {
    pub(crate) tensor_descriptor: MLTensorDescriptor,
    pub(crate) data: Arc<Mutex<Vec<f32>>>,
    destroyed: Arc<Mutex<bool>>,
}

impl PyMLTensor {
    /// Create a new tensor with the given tensor descriptor
    pub fn new(tensor_descriptor: MLTensorDescriptor) -> Self {
        let total_elements: usize = tensor_descriptor
            .descriptor
            .shape
            .iter()
            .map(|&d| d as usize)
            .product();
        let data = vec![0.0f32; total_elements];

        Self {
            tensor_descriptor,
            data: Arc::new(Mutex::new(data)),
            destroyed: Arc::new(Mutex::new(false)),
        }
    }

    /// Create a tensor from existing data
    pub fn from_data(tensor_descriptor: MLTensorDescriptor, data: Vec<f32>) -> Self {
        Self {
            tensor_descriptor,
            data: Arc::new(Mutex::new(data)),
            destroyed: Arc::new(Mutex::new(false)),
        }
    }

    /// Check if tensor has been destroyed
    fn check_destroyed(&self) -> PyResult<()> {
        if *self.destroyed.lock().unwrap() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tensor has been destroyed",
            ));
        }
        Ok(())
    }

    /// Get the data as a vector
    pub fn get_data(&self) -> PyResult<Vec<f32>> {
        self.check_destroyed()?;
        if !self.tensor_descriptor.readable {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tensor is not readable (readable=false)",
            ));
        }
        Ok(self.data.lock().unwrap().clone())
    }

    /// Set the data from a vector
    pub fn set_data(&self, data: Vec<f32>) -> PyResult<()> {
        self.check_destroyed()?;
        if !self.tensor_descriptor.writable {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tensor is not writable (writable=false)",
            ));
        }
        let expected_size: usize = self
            .tensor_descriptor
            .descriptor
            .shape
            .iter()
            .map(|&d| d as usize)
            .product();
        if data.len() != expected_size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data size mismatch: expected {} elements, got {}",
                expected_size,
                data.len()
            )));
        }
        *self.data.lock().unwrap() = data;
        Ok(())
    }
}

#[pymethods]
impl PyMLTensor {
    /// Get the data type of the tensor
    #[getter]
    fn data_type(&self) -> String {
        match self.tensor_descriptor.descriptor.data_type {
            DataType::Float32 => "float32".to_string(),
            DataType::Float16 => "float16".to_string(),
            DataType::Int32 => "int32".to_string(),
            DataType::Uint32 => "uint32".to_string(),
            DataType::Int8 => "int8".to_string(),
            DataType::Uint8 => "uint8".to_string(),
        }
    }

    /// Get the shape of the tensor
    #[getter]
    fn shape(&self) -> Vec<u32> {
        self.tensor_descriptor.descriptor.shape.clone()
    }

    /// Get the number of elements in the tensor
    #[getter]
    fn size(&self) -> usize {
        self.tensor_descriptor
            .descriptor
            .shape
            .iter()
            .map(|&d| d as usize)
            .product()
    }

    /// Check if tensor data can be read back to CPU
    #[getter]
    fn readable(&self) -> bool {
        self.tensor_descriptor.readable
    }

    /// Check if tensor data can be written from CPU
    #[getter]
    fn writable(&self) -> bool {
        self.tensor_descriptor.writable
    }

    /// Check if tensor can be exported for use as GPU texture
    #[getter]
    fn exportable_to_gpu(&self) -> bool {
        self.tensor_descriptor.exportable_to_gpu
    }

    /// Destroy the tensor and release its resources
    ///
    /// After calling destroy(), the tensor cannot be used for any operations.
    /// This follows the W3C WebNN MLTensor Explainer for explicit resource management.
    fn destroy(&self) -> PyResult<()> {
        let mut destroyed = self.destroyed.lock().unwrap();
        if *destroyed {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tensor already destroyed",
            ));
        }
        *destroyed = true;
        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "MLTensor(shape={:?}, dtype={}, readable={}, writable={}, exportable_to_gpu={})",
            self.tensor_descriptor.descriptor.shape,
            self.data_type(),
            self.readable(),
            self.writable(),
            self.exportable_to_gpu()
        )
    }
}
