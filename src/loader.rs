use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::error::GraphError;
use crate::graph::GraphInfo;

pub fn load_graph_from_path(path: impl AsRef<Path>) -> Result<GraphInfo, GraphError> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref).map_err(|err| GraphError::io(path_ref, err))?;
    let reader = BufReader::new(file);
    Ok(serde_json::from_reader(reader)?)
}
