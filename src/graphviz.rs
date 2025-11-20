use std::fmt::Write;

use crate::graph::{GraphInfo, OperandKind};

pub fn graph_to_dot(graph: &GraphInfo) -> String {
    let mut dot = String::from("digraph webnn {\n");
    dot.push_str("  rankdir=LR;\n");
    dot.push_str("  node [fontname=\"Helvetica\"];\n");
    dot.push_str("  edge [fontname=\"Helvetica\"];\n\n");

    for (idx, operand) in graph.operands.iter().enumerate() {
        let node_id = format!("operand_{}", idx);
        let shape = match operand.kind {
            OperandKind::Input => "oval",
            OperandKind::Constant => "diamond",
            OperandKind::Output => "doublecircle",
        };
        let fill = match operand.kind {
            OperandKind::Input => "#d0e6ff",
            OperandKind::Constant => "#f0f0f0",
            OperandKind::Output => "#d6f5d6",
        };
        let mut label_lines = vec![format!(
            "{} operand {}",
            match operand.kind {
                OperandKind::Input => "Input",
                OperandKind::Constant => "Constant",
                OperandKind::Output => "Output",
            },
            idx
        )];
        if let Some(name) = &operand.name {
            if !name.is_empty() {
                label_lines.push(format!("name: {}", name));
            }
        }
        label_lines.push(format!("{:?}", operand.descriptor.data_type));
        label_lines.push(format_shape(&operand.descriptor.shape));
        if !operand.descriptor.pending_permutation.is_empty() {
            label_lines.push(format!("perm {:?}", operand.descriptor.pending_permutation));
        }
        let label = escape_label(&label_lines.join("\n"));
        let _ = writeln!(
            dot,
            "  {} [shape={},style=filled,fillcolor=\"{}\",label=\"{}\"];",
            node_id, shape, fill, label
        );
    }

    dot.push('\n');

    for (idx, operation) in graph.operations.iter().enumerate() {
        let node_id = format!("op_{}", idx);
        let mut label_lines = vec![format!("{} (#{})", operation.display_name(), idx)];
        if let Some(label) = &operation.label {
            if !label.is_empty() && label != &operation.op_type {
                label_lines.push(label.clone());
            }
        }
        let label = escape_label(&label_lines.join("\n"));
        let _ = writeln!(
            dot,
            "  {} [shape=box,style=rounded,label=\"{}\"];",
            node_id, label
        );

        for (input_idx, operand_id) in operation.input_operands.iter().enumerate() {
            let _ = writeln!(
                dot,
                "  operand_{} -> {} [label=\"in{}\"];",
                operand_id, node_id, input_idx
            );
        }
        let _ = writeln!(
            dot,
            "  {} -> operand_{} [label=\"out\"];",
            node_id, operation.output_operand
        );
    }

    dot.push_str("}\n");
    dot
}

fn escape_label(label: &str) -> String {
    label
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn format_shape(shape: &[u32]) -> String {
    if shape.is_empty() {
        "scalar".to_string()
    } else {
        shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("x")
    }
}

#[cfg(test)]
mod tests {
    use super::graph_to_dot;
    use crate::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation};

    #[test]
    fn exports_graphviz_with_operands_and_operations() {
        let graph = GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 3],
                        pending_permutation: vec![],
                    },
                    name: Some("lhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 3],
                        pending_permutation: vec![],
                    },
                    name: Some("rhs".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: vec![1, 3],
                        pending_permutation: vec![],
                    },
                    name: Some("sum".to_string()),
                },
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation {
                op_type: "add".to_string(),
                input_operands: vec![0, 1],
                output_operand: 2,
                attributes: serde_json::Value::Null,
                label: None,
            }],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
        };

        let dot = graph_to_dot(&graph);

        assert!(dot.contains("operand_0 [shape=oval"));
        assert!(dot.contains("operand_2 [shape=doublecircle"));
        assert!(dot.contains("op_0 [shape=box"));
        assert!(dot.contains("operand_0 -> op_0 [label=\"in0\"]"));
        assert!(dot.contains("op_0 -> operand_2 [label=\"out\"]"));
        assert!(dot.contains("Float32"));
        assert!(dot.contains("1x3"));
    }
}
