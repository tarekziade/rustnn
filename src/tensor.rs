//! Device-resident tensor abstractions for zero-copy execution
//!
//! This module provides unified representation for host and device tensors,
//! enabling persistent GPU/NPU tensor storage across inference steps to eliminate
//! host round-trips for iterative GenAI workloads (e.g., KV cache).

use crate::error::GraphError;
use crate::graph::DataType;

/// Unified representation for host and device tensors
#[derive(Debug)]
pub enum TensorValue {
    /// Host-resident tensor stored in CPU memory
    Host(HostTensor),
    /// Device-resident tensor stored in GPU/NPU memory
    Device(DeviceTensorHandle),
}

/// Host-resident tensor stored in CPU memory
#[derive(Debug, Clone)]
pub struct HostTensor {
    /// Tensor data stored as f32 (will expand to support other types later)
    pub data: Vec<f32>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
}

impl HostTensor {
    /// Create a new host tensor with the given shape and data type
    pub fn new(shape: Vec<usize>, dtype: DataType) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![0.0f32; total_elements];
        Self { data, shape, dtype }
    }

    /// Create a host tensor from existing data
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>, dtype: DataType) -> Self {
        Self { data, shape, dtype }
    }

    /// Get the total number of elements
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Handle to a device-resident tensor with backend-specific implementation
#[derive(Debug)]
pub struct DeviceTensorHandle {
    /// Backend-specific tensor implementation
    pub inner: Box<dyn DeviceTensorBackend>,
    /// Data type
    pub dtype: DataType,
    /// Tensor shape
    pub shape: Vec<usize>,
}

impl DeviceTensorHandle {
    /// Create a new device tensor handle
    pub fn new(inner: Box<dyn DeviceTensorBackend>) -> Self {
        let dtype = inner.dtype();
        let shape = inner.shape().to_vec();
        Self {
            inner,
            dtype,
            shape,
        }
    }

    /// Get the device kind (CPU, CUDA, CoreML, etc.)
    pub fn device_kind(&self) -> DeviceKind {
        self.inner.device_kind()
    }

    /// Get the backend kind (OnnxCpu, OnnxGpu, TensorRT, etc.)
    pub fn backend_kind(&self) -> BackendKind {
        self.inner.backend_kind()
    }
}

/// Trait for backend-specific device tensor implementations
///
/// Each backend (ONNX Runtime, CoreML, TensorRT) implements this trait
/// to provide device-resident tensor storage and host transfer operations.
pub trait DeviceTensorBackend: Send + Sync + std::fmt::Debug {
    /// Get the data type of the tensor
    fn dtype(&self) -> DataType;

    /// Get the shape of the tensor
    fn shape(&self) -> &[usize];

    /// Get the device kind where this tensor resides
    fn device_kind(&self) -> DeviceKind;

    /// Get the backend that created this tensor
    fn backend_kind(&self) -> BackendKind;

    /// Read tensor data from device to host
    ///
    /// This performs a device-to-host memory transfer.
    /// Returns data as Vec<f32> (will expand to support other types later).
    fn read_to_host(&self) -> Result<Vec<f32>, GraphError>;

    /// Write tensor data from host to device
    ///
    /// This performs a host-to-device memory transfer.
    /// Accepts data as &[f32] (will expand to support other types later).
    fn write_from_host(&mut self, data: &[f32]) -> Result<(), GraphError>;

    /// Get a reference to self as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Device kind where a tensor resides
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// CPU device
    Cpu,
    /// NVIDIA CUDA GPU
    Cuda,
    /// DirectML GPU (Windows)
    DirectML,
    /// Apple CoreML (GPU/Neural Engine)
    CoreML,
}

/// Backend that created a device tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// ONNX Runtime CPU backend
    OnnxCpu,
    /// ONNX Runtime GPU backend
    OnnxGpu,
    /// Apple CoreML backend
    CoreML,
    /// NVIDIA TensorRT backend
    TensorRT,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceKind::Cpu => write!(f, "cpu"),
            DeviceKind::Cuda => write!(f, "cuda"),
            DeviceKind::DirectML => write!(f, "directml"),
            DeviceKind::CoreML => write!(f, "coreml"),
        }
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::OnnxCpu => write!(f, "onnx_cpu"),
            BackendKind::OnnxGpu => write!(f, "onnx_gpu"),
            BackendKind::CoreML => write!(f, "coreml"),
            BackendKind::TensorRT => write!(f, "tensorrt"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_tensor_creation() {
        let tensor = HostTensor::new(vec![2, 3], DataType::Float32);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.dtype, DataType::Float32);
        assert_eq!(tensor.element_count(), 6);
        assert_eq!(tensor.data.len(), 6);
    }

    #[test]
    fn test_host_tensor_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = HostTensor::from_data(data.clone(), vec![2, 2], DataType::Float32);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_device_kind_display() {
        assert_eq!(DeviceKind::Cpu.to_string(), "cpu");
        assert_eq!(DeviceKind::Cuda.to_string(), "cuda");
        assert_eq!(DeviceKind::CoreML.to_string(), "coreml");
    }

    #[test]
    fn test_backend_kind_display() {
        assert_eq!(BackendKind::OnnxCpu.to_string(), "onnx_cpu");
        assert_eq!(BackendKind::OnnxGpu.to_string(), "onnx_gpu");
        assert_eq!(BackendKind::CoreML.to_string(), "coreml");
        assert_eq!(BackendKind::TensorRT.to_string(), "tensorrt");
    }

    #[test]
    fn test_device_kind_display_all_variants() {
        assert_eq!(DeviceKind::Cpu.to_string(), "cpu");
        assert_eq!(DeviceKind::Cuda.to_string(), "cuda");
        assert_eq!(DeviceKind::DirectML.to_string(), "directml");
        assert_eq!(DeviceKind::CoreML.to_string(), "coreml");
    }

    #[test]
    fn test_device_kind_equality() {
        assert_eq!(DeviceKind::Cpu, DeviceKind::Cpu);
        assert_eq!(DeviceKind::Cuda, DeviceKind::Cuda);
        assert_ne!(DeviceKind::Cpu, DeviceKind::Cuda);
        assert_ne!(DeviceKind::CoreML, DeviceKind::Cuda);
    }

    #[test]
    fn test_backend_kind_equality() {
        assert_eq!(BackendKind::OnnxCpu, BackendKind::OnnxCpu);
        assert_eq!(BackendKind::TensorRT, BackendKind::TensorRT);
        assert_ne!(BackendKind::OnnxCpu, BackendKind::OnnxGpu);
        assert_ne!(BackendKind::CoreML, BackendKind::TensorRT);
    }

    #[test]
    fn test_host_tensor_element_count_scalar() {
        let tensor = HostTensor::new(vec![], DataType::Float32);
        assert_eq!(tensor.element_count(), 1); // Scalar has 1 element
    }

    #[test]
    fn test_host_tensor_element_count_1d() {
        let tensor = HostTensor::new(vec![5], DataType::Float32);
        assert_eq!(tensor.element_count(), 5);
    }

    #[test]
    fn test_host_tensor_element_count_3d() {
        let tensor = HostTensor::new(vec![2, 3, 4], DataType::Float32);
        assert_eq!(tensor.element_count(), 24);
    }

    #[test]
    fn test_host_tensor_element_count_4d() {
        let tensor = HostTensor::new(vec![1, 3, 224, 224], DataType::Float32);
        assert_eq!(tensor.element_count(), 150528);
    }

    #[test]
    fn test_host_tensor_clone() {
        let original = HostTensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DataType::Float32);
        let cloned = original.clone();

        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.data, original.data);
        assert_eq!(cloned.dtype, original.dtype);
    }

    #[test]
    fn test_host_tensor_different_data_types() {
        let f32_tensor = HostTensor::new(vec![2, 2], DataType::Float32);
        assert_eq!(f32_tensor.dtype, DataType::Float32);

        let f16_tensor = HostTensor::new(vec![2, 2], DataType::Float16);
        assert_eq!(f16_tensor.dtype, DataType::Float16);

        let int8_tensor = HostTensor::new(vec![2, 2], DataType::Int8);
        assert_eq!(int8_tensor.dtype, DataType::Int8);
    }

    #[test]
    fn test_tensor_value_host_variant() {
        let host_tensor = HostTensor::new(vec![2, 2], DataType::Float32);
        let tensor_value = TensorValue::Host(host_tensor);

        match tensor_value {
            TensorValue::Host(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.dtype, DataType::Float32);
            }
            _ => panic!("Expected Host variant"),
        }
    }

    // Mock DeviceTensorBackend for testing
    #[derive(Debug)]
    struct MockDeviceTensor {
        dtype: DataType,
        shape: Vec<usize>,
        device: DeviceKind,
        backend: BackendKind,
    }

    impl DeviceTensorBackend for MockDeviceTensor {
        fn dtype(&self) -> DataType {
            self.dtype.clone()
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn device_kind(&self) -> DeviceKind {
            self.device
        }

        fn backend_kind(&self) -> BackendKind {
            self.backend
        }

        fn read_to_host(&self) -> Result<Vec<f32>, GraphError> {
            Ok(vec![0.0; self.shape.iter().product()])
        }

        fn write_from_host(&mut self, _data: &[f32]) -> Result<(), GraphError> {
            Ok(())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_device_tensor_handle_creation() {
        let mock = MockDeviceTensor {
            dtype: DataType::Float32,
            shape: vec![2, 3],
            device: DeviceKind::Cuda,
            backend: BackendKind::OnnxGpu,
        };

        let handle = DeviceTensorHandle::new(Box::new(mock));

        assert_eq!(handle.dtype, DataType::Float32);
        assert_eq!(handle.shape, vec![2, 3]);
        assert_eq!(handle.device_kind(), DeviceKind::Cuda);
        assert_eq!(handle.backend_kind(), BackendKind::OnnxGpu);
    }

    #[test]
    fn test_tensor_value_device_variant() {
        let mock = MockDeviceTensor {
            dtype: DataType::Float32,
            shape: vec![1, 1],
            device: DeviceKind::Cpu,
            backend: BackendKind::OnnxCpu,
        };

        let handle = DeviceTensorHandle::new(Box::new(mock));
        let tensor_value = TensorValue::Device(handle);

        match tensor_value {
            TensorValue::Device(h) => {
                assert_eq!(h.dtype, DataType::Float32);
                assert_eq!(h.device_kind(), DeviceKind::Cpu);
            }
            _ => panic!("Expected Device variant"),
        }
    }

    #[test]
    fn test_host_tensor_zero_initialization() {
        let tensor = HostTensor::new(vec![3, 3], DataType::Float32);

        // Verify all elements are initialized to 0.0
        for &val in &tensor.data {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_device_kind_debug() {
        // Test that Debug is implemented
        let device = DeviceKind::Cuda;
        let debug_str = format!("{:?}", device);
        assert!(debug_str.contains("Cuda"));
    }

    #[test]
    fn test_backend_kind_debug() {
        // Test that Debug is implemented
        let backend = BackendKind::TensorRT;
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("TensorRT"));
    }
}
