use std::collections::HashMap;

use crate::error::GraphError;
use crate::graph::GraphInfo;

mod coreml_mlprogram;
pub mod onnx;
mod weight_file_builder;

pub use coreml_mlprogram::CoremlMlProgramConverter;
pub use onnx::OnnxConverter;
pub(crate) use weight_file_builder::WeightFileBuilder;

/// Get operand name for an operand ID, or generate a default name
///
/// This is a shared helper used by all converters to ensure consistent
/// operand naming across different backend formats.
pub(crate) fn operand_name(graph: &GraphInfo, id: u32) -> String {
    graph
        .operand(id)
        .and_then(|op| op.name.clone())
        .unwrap_or_else(|| format!("operand_{}", id))
}

#[derive(Debug, Clone)]
pub struct ConvertedGraph {
    pub format: &'static str,
    pub content_type: &'static str,
    pub data: Vec<u8>,
    /// Optional weight file data for formats that require external weights (e.g., CoreML Float16)
    pub weights_data: Option<Vec<u8>>,
}

pub trait GraphConverter {
    fn format(&self) -> &'static str;
    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError>;
}

pub struct ConverterRegistry {
    converters: HashMap<&'static str, Box<dyn GraphConverter + Send + Sync>>,
}

impl ConverterRegistry {
    pub fn with_defaults() -> Self {
        let mut registry = Self {
            converters: HashMap::new(),
        };
        registry.register(Box::new(OnnxConverter));
        registry.register(Box::new(CoremlMlProgramConverter));
        registry
    }

    pub fn register(&mut self, converter: Box<dyn GraphConverter + Send + Sync>) {
        self.converters.insert(converter.format(), converter);
    }

    pub fn available_formats(&self) -> Vec<&'static str> {
        let mut keys: Vec<_> = self.converters.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    pub fn convert(&self, format: &str, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let key = format.to_ascii_lowercase();
        let Some(converter) = self.converters.get(key.as_str()) else {
            return Err(GraphError::UnknownConverter {
                requested: format.to_string(),
                available: self.available_formats(),
            });
        };
        converter.convert(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::{ConverterRegistry, GraphConverter};
    use crate::error::GraphError;
    use crate::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind};

    struct DummyConverter;

    impl GraphConverter for DummyConverter {
        fn format(&self) -> &'static str {
            "dummy"
        }

        fn convert(&self, _graph: &GraphInfo) -> Result<super::ConvertedGraph, GraphError> {
            Ok(super::ConvertedGraph {
                format: "dummy",
                content_type: "application/octet-stream",
                data: vec![1, 2, 3],
                weights_data: None,
            })
        }
    }

    #[test]
    fn converts_via_registry() {
        let mut registry = ConverterRegistry {
            converters: Default::default(),
        };
        registry.register(Box::new(DummyConverter));

        let graph = GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![],
                    pending_permutation: vec![],
                },
                name: Some("x".to_string()),
            }],
            input_operands: vec![0],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let converted = registry.convert("dummy", &graph).unwrap();
        assert_eq!(converted.format, "dummy");
        assert_eq!(converted.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_with_defaults_registers_converters() {
        let registry = ConverterRegistry::with_defaults();
        let formats = registry.available_formats();

        // Should have at least ONNX and CoreML converters
        assert!(formats.contains(&"onnx"));
        assert!(formats.contains(&"coreml"));
        assert!(formats.len() >= 2);
    }

    #[test]
    fn test_available_formats_sorted() {
        let registry = ConverterRegistry::with_defaults();
        let formats = registry.available_formats();

        // Verify formats are sorted
        let mut sorted_formats = formats.clone();
        sorted_formats.sort_unstable();
        assert_eq!(formats, sorted_formats);
    }

    #[test]
    fn test_convert_unknown_format_returns_error() {
        let registry = ConverterRegistry::with_defaults();
        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let result = registry.convert("unknown_format", &graph);
        assert!(result.is_err());

        match result.unwrap_err() {
            GraphError::UnknownConverter {
                requested,
                available,
            } => {
                assert_eq!(requested, "unknown_format");
                assert!(!available.is_empty());
            }
            _ => panic!("Expected UnknownConverter error"),
        }
    }

    #[test]
    fn test_convert_case_insensitive() {
        let mut registry = ConverterRegistry {
            converters: Default::default(),
        };
        registry.register(Box::new(DummyConverter));

        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        // Should work with different cases
        assert!(registry.convert("dummy", &graph).is_ok());
        assert!(registry.convert("DUMMY", &graph).is_ok());
        assert!(registry.convert("Dummy", &graph).is_ok());
    }

    #[test]
    fn test_operand_name_with_named_operand() {
        let graph = GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![1, 2],
                    pending_permutation: vec![],
                },
                name: Some("input_tensor".to_string()),
            }],
            input_operands: vec![0],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let name = super::operand_name(&graph, 0);
        assert_eq!(name, "input_tensor");
    }

    #[test]
    fn test_operand_name_with_unnamed_operand() {
        let graph = GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![1, 2],
                    pending_permutation: vec![],
                },
                name: None,
            }],
            input_operands: vec![0],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let name = super::operand_name(&graph, 0);
        assert_eq!(name, "operand_0");
    }

    #[test]
    fn test_operand_name_invalid_id() {
        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let name = super::operand_name(&graph, 999);
        assert_eq!(name, "operand_999");
    }

    struct DummyConverterWithWeights;

    impl GraphConverter for DummyConverterWithWeights {
        fn format(&self) -> &'static str {
            "dummy_with_weights"
        }

        fn convert(&self, _graph: &GraphInfo) -> Result<super::ConvertedGraph, GraphError> {
            Ok(super::ConvertedGraph {
                format: "dummy_with_weights",
                content_type: "application/octet-stream",
                data: vec![1, 2, 3],
                weights_data: Some(vec![4, 5, 6, 7, 8]),
            })
        }
    }

    #[test]
    fn test_converted_graph_with_weights() {
        let mut registry = ConverterRegistry {
            converters: Default::default(),
        };
        registry.register(Box::new(DummyConverterWithWeights));

        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let converted = registry.convert("dummy_with_weights", &graph).unwrap();
        assert_eq!(converted.format, "dummy_with_weights");
        assert_eq!(converted.data, vec![1, 2, 3]);
        assert_eq!(converted.weights_data, Some(vec![4, 5, 6, 7, 8]));
    }

    #[test]
    fn test_converted_graph_clone() {
        let original = super::ConvertedGraph {
            format: "test",
            content_type: "application/octet-stream",
            data: vec![1, 2, 3],
            weights_data: Some(vec![4, 5]),
        };

        let cloned = original.clone();
        assert_eq!(cloned.format, original.format);
        assert_eq!(cloned.content_type, original.content_type);
        assert_eq!(cloned.data, original.data);
        assert_eq!(cloned.weights_data, original.weights_data);
    }
}
