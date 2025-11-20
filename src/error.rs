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
