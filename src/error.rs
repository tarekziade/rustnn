use std::path::PathBuf;

use crate::graph::DataType;
use serde_json::Error as JsonError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("graph file {path} could not be read: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("graph JSON could not be parsed: {source}")]
    Parse {
        #[from]
        source: JsonError,
    },
    #[error("graph must declare operands, operations, and outputs")]
    EmptyGraph,
    #[error("graph declares {count} operands which exceeds the u32 id space")]
    TooManyOperands { count: usize },
    #[error("operand {operand} has a shape that overflows element count")]
    OperandElementCountOverflow { operand: u32 },
    #[error("operand {operand} exceeds tensor byte limit ({byte_length} > {limit})")]
    TensorLimit {
        operand: u32,
        byte_length: usize,
        limit: usize,
    },
    #[error("input operand {operand} is missing a name")]
    MissingInputName { operand: u32 },
    #[error("input operand name `{name}` is duplicated")]
    DuplicateInputName { name: String },
    #[error("output operand {operand} is missing a name")]
    MissingOutputName { operand: u32 },
    #[error("output operand name `{name}` is duplicated")]
    DuplicateOutputName { name: String },
    #[error("operand {operand} uses unsupported IO data type {data_type:?}")]
    UnsupportedIoDataType { operand: u32, data_type: DataType },
    #[error("constant operand {operand} does not have data associated with it")]
    MissingConstantData { operand: u32 },
    #[error("constant operand {operand} byte mismatch (expected {expected}, got {actual})")]
    ConstantLengthMismatch {
        operand: u32,
        expected: usize,
        actual: usize,
    },
    #[error("graph input operand list does not match operand table")]
    InputOperandListMismatch,
    #[error("graph output operand list does not match operand table")]
    OutputOperandListMismatch,
    #[error("operand id {operand} referenced by `{operation}` is invalid")]
    InvalidOperandReference { operation: String, operand: u32 },
    #[error("operation `{operation}` consumes operand {operand} before it is produced")]
    OperandNotReady { operation: String, operand: u32 },
    #[error("operation `{operation}` attempts to reuse operand {operand} as output")]
    OperandProducedTwice { operation: String, operand: u32 },
    #[error("graph output operand {operand} is never produced by any operation")]
    OutputNotProduced { operand: u32 },
    #[error("operand {operand} never feeds any operation")]
    OperandNeverUsed { operand: u32 },
    #[error("graph contains unused constant data entries")]
    UnusedConstantHandles,
    #[error("quantization validation failed for `{operation}`: {reason}")]
    QuantizationValidation { operation: String, reason: String },
    #[error("graph converter `{requested}` is not available. Supported: {available:?}")]
    UnknownConverter {
        requested: String,
        available: Vec<&'static str>,
    },
    #[error("graph conversion failed for {format}: {reason}")]
    ConversionFailed { format: String, reason: String },
    #[error("operand id {operand} is invalid for conversion")]
    InvalidConversionOperand { operand: u32 },
    #[error("graph could not be exported to {path}: {source}")]
    ExportIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("coreml runtime is only available on macOS with the `coreml-runtime` feature enabled")]
    CoremlRuntimeUnavailable,
    #[error("coreml runtime failed: {reason}")]
    CoremlRuntimeFailed { reason: String },
    #[error("coreml runtime only supports the coreml converter (got {format})")]
    UnsupportedRuntimeFormat { format: String },
    #[error("onnx runtime is only available with the `onnx-runtime` feature enabled")]
    OnnxRuntimeUnavailable,
    #[error("onnx runtime failed: {reason}")]
    OnnxRuntimeFailed { reason: String },
    #[error("tensorrt runtime is only available with the `trtx-runtime` feature enabled")]
    TrtxRuntimeUnavailable,
    #[error("tensorrt runtime failed: {reason}")]
    TrtxRuntimeFailed { reason: String },
    #[error("shape inference failed: {reason}")]
    ShapeInferenceFailed { reason: String },
    #[error("device tensor operation failed: {reason}")]
    DeviceTensorFailed { reason: String },
    #[error("device tensor has been destroyed")]
    DeviceTensorDestroyed,
    #[error("device tensor is not readable")]
    DeviceTensorNotReadable,
    #[error("device tensor is not writable")]
    DeviceTensorNotWritable,
}

impl GraphError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        GraphError::Io {
            path: path.into(),
            source,
        }
    }

    pub fn export(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        GraphError::ExportIo {
            path: path.into(),
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_io_error_helper() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let graph_err = GraphError::io("test.json", io_err);

        match graph_err {
            GraphError::Io { path, .. } => {
                assert_eq!(path, PathBuf::from("test.json"));
            }
            _ => panic!("Expected Io error variant"),
        }
    }

    #[test]
    fn test_export_error_helper() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "permission denied");
        let graph_err = GraphError::export("/tmp/output.onnx", io_err);

        match graph_err {
            GraphError::ExportIo { path, .. } => {
                assert_eq!(path, PathBuf::from("/tmp/output.onnx"));
            }
            _ => panic!("Expected ExportIo error variant"),
        }
    }

    #[test]
    fn test_empty_graph_error_message() {
        let err = GraphError::EmptyGraph;
        let msg = format!("{}", err);
        assert_eq!(msg, "graph must declare operands, operations, and outputs");
    }

    #[test]
    fn test_too_many_operands_error() {
        let err = GraphError::TooManyOperands { count: 5_000_000 };
        let msg = format!("{}", err);
        assert!(msg.contains("5000000"));
        assert!(msg.contains("exceeds the u32 id space"));
    }

    #[test]
    fn test_tensor_limit_error() {
        let err = GraphError::TensorLimit {
            operand: 42,
            byte_length: 2_000_000,
            limit: 1_000_000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
        assert!(msg.contains("2000000"));
        assert!(msg.contains("1000000"));
    }

    #[test]
    fn test_duplicate_input_name_error() {
        let err = GraphError::DuplicateInputName {
            name: "input_tensor".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("input_tensor"));
        assert!(msg.contains("duplicated"));
    }

    #[test]
    fn test_constant_length_mismatch_error() {
        let err = GraphError::ConstantLengthMismatch {
            operand: 10,
            expected: 1024,
            actual: 512,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("10"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_invalid_operand_reference_error() {
        let err = GraphError::InvalidOperandReference {
            operation: "relu_1".to_string(),
            operand: 99,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("relu_1"));
        assert!(msg.contains("99"));
        assert!(msg.contains("invalid"));
    }

    #[test]
    fn test_operand_not_ready_error() {
        let err = GraphError::OperandNotReady {
            operation: "conv2d".to_string(),
            operand: 5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("conv2d"));
        assert!(msg.contains("before it is produced"));
    }

    #[test]
    fn test_conversion_failed_error() {
        let err = GraphError::ConversionFailed {
            format: "ONNX".to_string(),
            reason: "unsupported operation".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("ONNX"));
        assert!(msg.contains("unsupported operation"));
    }

    #[test]
    fn test_runtime_unavailable_errors() {
        let coreml_err = GraphError::CoremlRuntimeUnavailable;
        assert!(format!("{}", coreml_err).contains("coreml runtime"));
        assert!(format!("{}", coreml_err).contains("macOS"));

        let onnx_err = GraphError::OnnxRuntimeUnavailable;
        assert!(format!("{}", onnx_err).contains("onnx runtime"));

        let trtx_err = GraphError::TrtxRuntimeUnavailable;
        assert!(format!("{}", trtx_err).contains("tensorrt runtime"));
    }

    #[test]
    fn test_device_tensor_errors() {
        let destroyed_err = GraphError::DeviceTensorDestroyed;
        assert!(format!("{}", destroyed_err).contains("destroyed"));

        let not_readable_err = GraphError::DeviceTensorNotReadable;
        assert!(format!("{}", not_readable_err).contains("not readable"));

        let not_writable_err = GraphError::DeviceTensorNotWritable;
        assert!(format!("{}", not_writable_err).contains("not writable"));
    }

    #[test]
    fn test_shape_inference_failed_error() {
        let err = GraphError::ShapeInferenceFailed {
            reason: "incompatible dimensions".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("shape inference failed"));
        assert!(msg.contains("incompatible dimensions"));
    }

    #[test]
    fn test_quantization_validation_error() {
        let err = GraphError::QuantizationValidation {
            operation: "quantizeLinear".to_string(),
            reason: "scale must be float32".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("quantizeLinear"));
        assert!(msg.contains("scale must be float32"));
    }
}
