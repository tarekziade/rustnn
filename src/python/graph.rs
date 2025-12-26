//! Compiled computational graph representation
//!
//! PyO3 macros generate unsafe code that triggers unsafe_op_in_unsafe_fn warnings.
//! This is expected behavior from the macro-generated code.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::graph::GraphInfo;
use crate::webnn_json;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use std::fs;
use std::path::Path;

/// Represents a compiled computational graph
#[pyclass(name = "MLGraph")]
pub struct PyMLGraph {
    pub(crate) graph_info: GraphInfo,
}

#[pymethods]
impl PyMLGraph {
    fn __repr__(&self) -> String {
        format!(
            "MLGraph(operands={}, operations={})",
            self.graph_info.operands.len(),
            self.graph_info.operations.len()
        )
    }

    /// Get the number of operands in the graph
    #[getter]
    fn operand_count(&self) -> usize {
        self.graph_info.operands.len()
    }

    /// Get the number of operations in the graph
    #[getter]
    fn operation_count(&self) -> usize {
        self.graph_info.operations.len()
    }

    /// Get input names
    fn get_input_names(&self) -> Vec<String> {
        self.graph_info
            .operands
            .iter()
            .filter_map(|op| {
                if matches!(op.kind, crate::graph::OperandKind::Input) {
                    op.name.clone()
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get output names
    fn get_output_names(&self) -> Vec<String> {
        // Use output_operands list instead of filtering by kind
        self.graph_info
            .output_operands
            .iter()
            .filter_map(|&idx| {
                self.graph_info
                    .operands
                    .get(idx as usize)
                    .and_then(|op| op.name.clone())
            })
            .collect()
    }

    /// Debug method to inspect operand at index (for debugging)
    fn debug_operand(&self, idx: usize) -> String {
        if let Some(op) = self.graph_info.operands.get(idx) {
            format!(
                "Operand[{}]: name={:?}, kind={:?}, type={:?}, shape={:?}",
                idx, op.name, op.kind, op.descriptor.data_type, op.descriptor.shape
            )
        } else {
            format!("Operand[{}]: not found", idx)
        }
    }

    /// Count operands with empty shapes
    fn count_empty_shapes(&self) -> usize {
        self.graph_info
            .operands
            .iter()
            .filter(|op| op.descriptor.shape.is_empty())
            .count()
    }

    /// Save the graph to a .webnn JSON file
    ///
    /// Args:
    ///     path: File path to save the graph (e.g., "model.webnn")
    ///
    /// Example:
    ///     graph.save("my_model.webnn")
    fn save(&self, path: &str) -> PyResult<()> {
        // Convert GraphInfo to GraphJson
        let graph_json = webnn_json::to_graph_json(&self.graph_info)
            .map_err(|e| PyIOError::new_err(format!("Failed to convert graph: {}", e)))?;

        // Serialize to JSON
        let json_string = serde_json::to_string_pretty(&graph_json)
            .map_err(|e| PyIOError::new_err(format!("Failed to serialize to JSON: {}", e)))?;

        // Write to file
        fs::write(path, json_string)
            .map_err(|e| PyIOError::new_err(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load a graph from a .webnn file (JSON or text format)
    ///
    /// Args:
    ///     path: File path to load the graph from (e.g., "model.webnn")
    ///     manifest_path: Optional path to manifest.json file for external weights
    ///     weights_path: Optional path to weights file for external weights
    ///
    /// Returns:
    ///     MLGraph: The loaded graph
    ///
    /// The loader automatically detects the format:
    /// - JSON format: Legacy format with embedded base64 weights
    /// - Text format: WebNN DSL format (automatically detected)
    ///
    /// Example:
    ///     graph = MLGraph.load("my_model.webnn")
    ///     graph = MLGraph.load("model.webnn", manifest_path="manifest.json", weights_path="model.weights")
    #[staticmethod]
    #[pyo3(signature = (path, manifest_path=None, weights_path=None))]
    fn load(path: &str, manifest_path: Option<&str>, weights_path: Option<&str>) -> PyResult<Self> {
        // Check if file exists
        let path_obj = Path::new(path);
        if !path_obj.exists() {
            return Err(PyIOError::new_err(format!("File not found: {}", path)));
        }

        // Read file
        let content = fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read file: {}", e)))?;

        // Try to detect format and parse accordingly
        let mut graph_json: webnn_graph::ast::GraphJson = if content.trim().starts_with('{') {
            // JSON format
            serde_json::from_str(&content)
                .map_err(|e| PyIOError::new_err(format!("Failed to parse JSON: {}", e)))?
        } else {
            // WebNN text DSL format - sanitize identifiers first
            let sanitized = crate::loader::sanitize_webnn_identifiers(&content);
            webnn_graph::parser::parse_wg_text(&sanitized).map_err(|e| {
                PyIOError::new_err(format!("Failed to parse WebNN text format: {}", e))
            })?
        };

        // Resolve external weight references if present
        Self::resolve_external_weights(&mut graph_json, manifest_path, weights_path)?;

        // Convert GraphJson to GraphInfo
        let graph_info = webnn_json::from_graph_json(&graph_json)
            .map_err(|e| PyIOError::new_err(format!("Failed to convert graph: {}", e)))?;

        Ok(PyMLGraph { graph_info })
    }
}

impl PyMLGraph {
    pub fn new(graph_info: GraphInfo) -> Self {
        Self { graph_info }
    }

    /// Resolve external weight references in a GraphJson
    ///
    /// This function loads manifest and weights files and resolves all weight references to inline bytes.
    ///
    /// If manifest_path and weights_path are not provided, returns immediately (no external weights).
    fn resolve_external_weights(
        graph_json: &mut webnn_graph::ast::GraphJson,
        manifest_path: Option<&str>,
        weights_path: Option<&str>,
    ) -> PyResult<()> {
        use webnn_graph::ast::ConstInit;
        use webnn_graph::weights::WeightsManifest;

        // If no manifest path provided, assume no external weights
        let manifest_path = match manifest_path {
            Some(p) => p,
            None => return Ok(()),
        };

        // If no weights path provided, assume no external weights
        let weights_path = match weights_path {
            Some(p) => p,
            None => return Ok(()),
        };

        // Load manifest
        let manifest_content = fs::read_to_string(manifest_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read manifest: {}", e)))?;
        let manifest: WeightsManifest = serde_json::from_str(&manifest_content)
            .map_err(|e| PyIOError::new_err(format!("Failed to parse manifest: {}", e)))?;

        // Load weights file
        let weights_data = fs::read(weights_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read weights: {}", e)))?;

        // Create a sanitized lookup map: dots and colons in manifest keys -> underscores
        // This allows the sanitized graph references to match manifest entries
        use std::collections::HashMap;
        let sanitized_manifest: HashMap<String, _> = manifest
            .tensors
            .iter()
            .map(|(key, value)| (key.replace("::", "__").replace('.', "_"), value))
            .collect();

        // Resolve weight references in constants
        for (_name, const_decl) in graph_json.consts.iter_mut() {
            if let ConstInit::Weights { r#ref } = &const_decl.init {
                // Look up weight in sanitized manifest (all keys have underscores)
                let tensor_entry = sanitized_manifest.get(r#ref);

                if let Some(tensor_entry) = tensor_entry {
                    let offset = tensor_entry.byte_offset as usize;
                    let length = tensor_entry.byte_length as usize;

                    // Extract bytes from weights file
                    if offset + length > weights_data.len() {
                        return Err(PyIOError::new_err(format!(
                            "Weight '{}' offset/length exceeds weights file size",
                            r#ref
                        )));
                    }
                    let bytes = weights_data[offset..offset + length].to_vec();

                    // Replace weight reference with inline bytes
                    const_decl.init = ConstInit::InlineBytes { bytes };
                } else {
                    return Err(PyIOError::new_err(format!(
                        "Weight '{}' not found in manifest",
                        r#ref
                    )));
                }
            }
        }

        Ok(())
    }
}
