use std::fs;
use std::path::Path;

use crate::error::GraphError;
use crate::graph::GraphInfo;
use crate::webnn_json;

/// Load a graph from a webnn-graph file (.webnn text or .json)
///
/// Supports two formats:
/// - `.webnn` - Text DSL format (parsed and converted to JSON)
/// - `.json` - Direct JSON format (webnn-graph-json)
pub fn load_graph_from_path(path: impl AsRef<Path>) -> Result<GraphInfo, GraphError> {
    let path_ref = path.as_ref();
    let contents = fs::read_to_string(path_ref).map_err(|err| GraphError::io(path_ref, err))?;

    // Determine format based on file extension
    let graph_json = if let Some(ext) = path_ref.extension() {
        match ext.to_str() {
            Some("webnn") => {
                // Parse .webnn text format
                webnn_graph::parser::parse_wg_text(&contents).map_err(|e| {
                    GraphError::ConversionFailed {
                        format: "webnn-text".to_string(),
                        reason: format!("Failed to parse .webnn file: {}", e),
                    }
                })?
            }
            Some("json") => {
                // Parse JSON format
                serde_json::from_str(&contents)?
            }
            _ => {
                return Err(GraphError::ConversionFailed {
                    format: "unknown".to_string(),
                    reason: format!("Unsupported file extension: {:?}. Use .webnn or .json", ext),
                });
            }
        }
    } else {
        return Err(GraphError::ConversionFailed {
            format: "unknown".to_string(),
            reason: "No file extension found. Use .webnn or .json".to_string(),
        });
    };

    // Convert to internal GraphInfo format
    webnn_json::from_graph_json(&graph_json)
}
