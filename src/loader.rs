use regex::Regex;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use crate::error::GraphError;
use crate::graph::GraphInfo;
use crate::webnn_json;

/// Sanitize WebNN text format identifiers by replacing dots and colons with underscores.
///
/// This function preprocesses WebNN text format to ensure all identifiers follow
/// valid Rust naming conventions (alphanumeric + underscores only). It handles:
/// - Variable declarations: `embeddings.LayerNorm.bias:` -> `embeddings_LayerNorm_bias:`
/// - Weight references: `@weights("embeddings.LayerNorm.bias")` -> `@weights("embeddings_LayerNorm_bias")`
/// - Operand references: `%embeddings.LayerNorm.bias` -> `%embeddings_LayerNorm_bias`
/// - Namespace separators: `onnx::MatMul_0` -> `onnx__MatMul_0`
///
/// This allows models exported from tools like onnx2webnn to be loaded without
/// manual identifier sanitization.
pub fn sanitize_webnn_identifiers(text: &str) -> String {
    static PATTERNS: OnceLock<(Regex, Regex, Regex, Regex)> = OnceLock::new();

    let (decl_re, weights_re, operand_re, bare_id_re) = PATTERNS.get_or_init(|| {
        (
            // Match identifier declarations: `name.with.dots:` -> `name_with_dots:`
            Regex::new(r"([a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z0-9_.]+):").unwrap(),
            // Match weight references: `@weights("name.with.dots")` -> `@weights("name_with_dots")`
            Regex::new(r#"@weights\("([^"]+)"\)"#).unwrap(),
            // Match operand references: `%name.with.dots` -> `%name_with_dots`
            Regex::new(r"%([a-zA-Z_][a-zA-Z0-9_.]*)").unwrap(),
            // Match bare identifiers in operations (but not in strings or declarations)
            // Matches identifiers that contain dots in contexts like function arguments
            Regex::new(r"([,(=])\s*([a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z0-9_.]+)(\s*[,)])").unwrap(),
        )
    });

    let mut result = text.to_string();

    // First, replace :: with __ (namespace separators like onnx::MatMul)
    result = result.replace("::", "__");

    // Replace dots in identifier declarations
    result = decl_re
        .replace_all(&result, |caps: &regex::Captures| {
            format!("{}:", caps[1].replace('.', "_"))
        })
        .to_string();

    // Replace dots in weight references
    result = weights_re
        .replace_all(&result, |caps: &regex::Captures| {
            format!(r#"@weights("{}")"#, caps[1].replace('.', "_"))
        })
        .to_string();

    // Replace dots in operand references
    result = operand_re
        .replace_all(&result, |caps: &regex::Captures| {
            format!("%{}", caps[1].replace('.', "_"))
        })
        .to_string();

    // Replace dots in bare identifiers (operation arguments)
    result = bare_id_re
        .replace_all(&result, |caps: &regex::Captures| {
            format!("{}{}{}", &caps[1], caps[2].replace('.', "_"), &caps[3])
        })
        .to_string();

    result
}

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
                // Sanitize identifiers (replace dots with underscores)
                let sanitized = sanitize_webnn_identifiers(&contents);
                // Parse .webnn text format
                webnn_graph::parser::parse_wg_text(&sanitized).map_err(|e| {
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
