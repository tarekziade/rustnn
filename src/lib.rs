pub mod converters;
pub mod error;
pub mod executors;
pub mod graph;
pub mod graphviz;
pub mod loader;
pub mod protos;
pub mod shape_inference;
pub mod validator;
pub mod webnn_json;

#[cfg(feature = "python")]
pub mod python;

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use executors::coreml;

pub use converters::{ConvertedGraph, ConverterRegistry, GraphConverter};
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub use coreml::{CoremlOutput, CoremlRunAttempt, run_coreml_zeroed, run_coreml_zeroed_cached};
pub use error::GraphError;
#[cfg(feature = "onnx-runtime")]
pub use executors::onnx::{OnnxOutput, run_onnx_zeroed};
pub use graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation,
};
pub use graphviz::graph_to_dot;
pub use loader::load_graph_from_path;
pub use validator::{ContextProperties, GraphValidator, ValidationArtifacts};
