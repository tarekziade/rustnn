use std::collections::HashMap;

use crate::error::GraphError;
use crate::graph::GraphInfo;

mod coreml_mlprogram;
mod onnx;

pub use coreml_mlprogram::CoremlMlProgramConverter;
pub use onnx::OnnxConverter;

#[derive(Debug, Clone)]
pub struct ConvertedGraph {
    pub format: &'static str,
    pub content_type: &'static str,
    pub data: Vec<u8>,
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
        registry.register(Box::new(OnnxConverter::default()));
        registry.register(Box::new(CoremlMlProgramConverter::default()));
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
        };

        let converted = registry.convert("dummy", &graph).unwrap();
        assert_eq!(converted.format, "dummy");
        assert_eq!(converted.data, vec![1, 2, 3]);
    }
}
