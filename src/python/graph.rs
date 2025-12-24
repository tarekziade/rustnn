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
        self.graph_info
            .operands
            .iter()
            .filter_map(|op| {
                if matches!(op.kind, crate::graph::OperandKind::Output) {
                    op.name.clone()
                } else {
                    None
                }
            })
            .collect()
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

    /// Load a graph from a .webnn JSON file
    ///
    /// Args:
    ///     path: File path to load the graph from (e.g., "model.webnn")
    ///
    /// Returns:
    ///     MLGraph: The loaded graph
    ///
    /// Example:
    ///     graph = MLGraph.load("my_model.webnn")
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        // Check if file exists
        if !Path::new(path).exists() {
            return Err(PyIOError::new_err(format!("File not found: {}", path)));
        }

        // Read file
        let json_string = fs::read_to_string(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read file: {}", e)))?;

        // Deserialize JSON
        let graph_json: webnn_graph::ast::GraphJson = serde_json::from_str(&json_string)
            .map_err(|e| PyIOError::new_err(format!("Failed to parse JSON: {}", e)))?;

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
}
