use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use rust_webnn_graph::{
    ContextProperties, GraphError, GraphValidator, graph_to_dot, load_graph_from_path,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Validate WebNN graph descriptions", long_about = None)]
struct Cli {
    /// Path to a JSON file describing a mojom-like GraphInfo structure.
    graph: PathBuf,
    /// Optional override for the tensor byte length limit.
    #[arg(long)]
    tensor_limit: Option<usize>,
    /// Optional path to write a Graphviz DOT export of the graph.
    #[arg(long)]
    export_dot: Option<PathBuf>,
    /// Convert the graph to a different format (e.g. `onnx`).
    #[arg(long)]
    convert: Option<String>,
    /// Path to write the converted graph (stdout if omitted).
    #[arg(long)]
    convert_output: Option<PathBuf>,
}

fn run() -> Result<(), GraphError> {
    let cli = Cli::parse();
    let graph = load_graph_from_path(&cli.graph)?;
    let mut context = ContextProperties::default();
    if let Some(limit) = cli.tensor_limit {
        context.tensor_byte_length_limit = limit;
    }
    let validator = GraphValidator::new(&graph, context);
    let artifacts = validator.validate()?;

    println!(
        "Validated graph from `{}` with {} operands and {} operations.",
        cli.graph.display(),
        graph.operands.len(),
        graph.operations.len()
    );
    println!("Inputs:");
    for (name, descriptor) in artifacts.input_names_to_descriptors.iter() {
        println!(
            "  - {}: {:?} {:?}",
            name, descriptor.data_type, descriptor.shape
        );
    }
    println!("Outputs:");
    for (name, descriptor) in artifacts.output_names_to_descriptors.iter() {
        println!(
            "  - {}: {:?} {:?}",
            name, descriptor.data_type, descriptor.shape
        );
    }
    println!("Dependency fan-out:");
    for (operand, deps) in artifacts.operand_to_dependent_operations.iter() {
        println!("  - operand {} -> {}", operand, deps.join(", "));
    }

    if let Some(dot_path) = cli.export_dot {
        let dot = graph_to_dot(&graph);
        std::fs::write(&dot_path, dot).map_err(|err| GraphError::export(dot_path.clone(), err))?;
        println!("Exported Graphviz DOT to `{}`.", dot_path.display());
    }

    if let Some(format) = cli.convert {
        let converted =
            rust_webnn_graph::ConverterRegistry::with_defaults().convert(&format, &graph)?;
        if let Some(path) = cli.convert_output {
            std::fs::write(&path, converted.data)
                .map_err(|err| GraphError::export(path.clone(), err))?;
            println!(
                "Converted graph to `{}` format at `{}` (type {}).",
                converted.format,
                path.display(),
                converted.content_type
            );
        } else {
            std::io::stdout()
                .write_all(&converted.data)
                .map_err(|err| GraphError::ConversionFailed {
                    format: converted.format.to_string(),
                    reason: err.to_string(),
                })?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {}", err);
        std::process::exit(1);
    }
}
