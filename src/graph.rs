use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_with::{base64::Base64, serde_as};

use crate::error::GraphError;
use std::hash::{Hash, Hasher};

use crate::operator_options::{MLDimension, MLDynamicDimension};
use crate::operators::Operation;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "camelCase")]
pub struct DynamicDimension {
    pub name: String,
    pub max_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(untagged)]
pub enum Dimension {
    Static(u32),
    Dynamic(DynamicDimension),
}

impl Dimension {
    pub fn get_static_or_max_size(&self) -> u32 {
        match self {
            Self::Static(value) => *value,
            Self::Dynamic(dimension) => dimension.max_size,
        }
    }
}

pub fn to_dimension_vector(shape: &[u32]) -> Vec<Dimension> {
    shape.iter().copied().map(Dimension::Static).collect()
}

pub fn get_static_or_max_size(dim: &Dimension) -> u32 {
    dim.get_static_or_max_size()
}

impl From<MLDimension> for Dimension {
    fn from(m: MLDimension) -> Self {
        match m {
            MLDimension::Static(n) => Dimension::Static(n),
            MLDimension::Dynamic(d) => Dimension::Dynamic(DynamicDimension {
                name: d.name,
                max_size: d.max_size,
            }),
        }
    }
}

impl From<MLDynamicDimension> for DynamicDimension {
    fn from(d: MLDynamicDimension) -> Self {
        DynamicDimension {
            name: d.name,
            max_size: d.max_size,
        }
    }
}

impl From<Dimension> for MLDimension {
    fn from(d: Dimension) -> Self {
        match d {
            Dimension::Static(n) => MLDimension::Static(n),
            Dimension::Dynamic(d) => MLDimension::Dynamic(MLDynamicDimension {
                name: d.name,
                max_size: d.max_size,
            }),
        }
    }
}

impl From<DynamicDimension> for MLDynamicDimension {
    fn from(d: DynamicDimension) -> Self {
        MLDynamicDimension {
            name: d.name,
            max_size: d.max_size,
        }
    }
}

pub fn dynamic_inputs_enabled() -> bool {
    cfg!(feature = "dynamic-inputs")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Int4,
    Uint4,
    Float16,
    Float32,
    Int32,
    Uint32,
    Int8,
    Uint8,
    Int64,
    Uint64,
}

impl DataType {
    pub const fn bits_per_element(self) -> usize {
        match self {
            DataType::Int4 | DataType::Uint4 => 4,
            DataType::Int8 | DataType::Uint8 => 8,
            DataType::Float16 => 16,
            DataType::Float32 | DataType::Int32 | DataType::Uint32 => 32,
            DataType::Int64 | DataType::Uint64 => 64,
        }
    }

    /// Host storage bytes for `elements` values (always `ceil(elements * bits / 8)`).
    pub fn storage_byte_length(self, elements: usize) -> Option<usize> {
        let total_bits = elements.checked_mul(self.bits_per_element())?;
        Some(total_bits.div_ceil(8))
    }

    /// Byte size of one element when the type is whole-byte aligned (`bits % 8 == 0`).
    pub fn bytes_per_element(self) -> usize {
        let bits = self.bits_per_element();
        debug_assert!(
            bits.is_multiple_of(8),
            "bytes_per_element is only defined for byte-aligned types; use storage_byte_length for int4/uint4"
        );
        bits / 8
    }
}

/// Packed int4/uint4 layout: even logical indices in the low nibble, odd in the high (ONNX/WebNN
/// convention, i.e. element `2*i` occupies the least-significant 4 bits of byte `i`).
pub fn unpack_int4(data: &[u8], element_count: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(element_count);
    for i in 0..element_count {
        let byte = data[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        out.push(if nibble >= 8 {
            nibble as i32 - 16
        } else {
            nibble as i32
        });
    }
    out
}

pub fn pack_int4(values: &[i32]) -> Vec<u8> {
    let byte_len = DataType::Int4
        .storage_byte_length(values.len())
        .unwrap_or(0);
    let mut out = vec![0u8; byte_len];
    for (i, &v) in values.iter().enumerate() {
        let nibble = ((v.clamp(-8, 7) as i8) as u8) & 0x0F;
        if i % 2 == 0 {
            out[i / 2] = nibble;
        } else {
            out[i / 2] |= nibble << 4;
        }
    }
    out
}

pub fn unpack_uint4(data: &[u8], element_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(element_count);
    for i in 0..element_count {
        let byte = data[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        out.push(nibble);
    }
    out
}

pub fn pack_uint4(values: &[u8]) -> Vec<u8> {
    let byte_len = DataType::Uint4
        .storage_byte_length(values.len())
        .unwrap_or(0);
    let mut out = vec![0u8; byte_len];
    for (i, &v) in values.iter().enumerate() {
        let nibble = v & 0x0F;
        if i % 2 == 0 {
            out[i / 2] = nibble;
        } else {
            out[i / 2] |= nibble << 4;
        }
    }
    out
}

pub fn pack_uint4_from_i32(values: &[i32]) -> Vec<u8> {
    pack_uint4(
        &values
            .iter()
            .map(|&v| v.clamp(0, 15) as u8)
            .collect::<Vec<_>>(),
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
pub struct OperandDescriptor {
    pub data_type: DataType,
    #[serde(default)]
    pub shape: Vec<Dimension>,
    #[serde(default)]
    pub pending_permutation: Vec<u32>,
}

impl OperandDescriptor {
    pub fn has_dynamic_dimensions(&self) -> bool {
        self.shape
            .iter()
            .any(|dim| matches!(dim, Dimension::Dynamic(_)))
    }

    pub fn static_shape(&self) -> Option<Vec<u32>> {
        let mut shape = Vec::with_capacity(self.shape.len());
        for dim in &self.shape {
            match dim {
                Dimension::Static(v) => shape.push(*v),
                Dimension::Dynamic(_) => return None,
            }
        }
        Some(shape)
    }

    pub fn static_or_max_shape(&self) -> Vec<u32> {
        self.shape.iter().map(get_static_or_max_size).collect()
    }

    pub fn element_count(&self) -> Option<usize> {
        if self.shape.is_empty() {
            return Some(1);
        }
        let mut count = 1usize;
        for dim in &self.shape {
            let size = get_static_or_max_size(dim) as usize;
            count = count.checked_mul(size)?;
        }
        Some(count)
    }

    pub fn byte_length(&self) -> Option<usize> {
        let elements = self.element_count()?;
        self.data_type.storage_byte_length(elements)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum OperandKind {
    Input,
    Constant,
    Output,
    // optional operand type, at the moment not required in graphs, but useful for validation and
    // incremental shape inference
    Intermediate,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
pub struct Operand {
    pub kind: OperandKind,
    pub descriptor: OperandDescriptor,
    #[serde(default)]
    pub name: Option<String>,
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
pub struct ConstantData {
    #[serde_as(as = "Base64")]
    pub data: Vec<u8>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphInfo {
    pub operands: Vec<Operand>,
    #[serde(default)]
    pub input_operands: Vec<u32>,
    #[serde(default)]
    pub output_operands: Vec<u32>,
    #[serde(default)]
    pub operations: Vec<Operation>,
    #[serde(default)]
    pub constant_operand_ids_to_handles: HashMap<u32, ConstantData>,
    #[serde(default)]
    pub id_to_constant_tensor_operand_map: HashMap<u32, String>,
    #[serde(default)]
    pub quantized: bool,
}

impl GraphInfo {
    pub fn operand(&self, id: u32) -> Option<&Operand> {
        self.operands.get(id as usize)
    }

    pub fn has_dynamic_dimensions(&self) -> bool {
        self.operands
            .iter()
            .any(|operand| operand.descriptor.has_dynamic_dimensions())
    }

    /// Ensures `input_operands` / `output_operands` match operands tagged as graph I/O.
    pub fn validate_io_operand_lists(&self) -> Result<(), GraphError> {
        let mut derived_inputs = Vec::new();
        let mut derived_outputs = Vec::new();

        for (idx, operand) in self.operands.iter().enumerate() {
            match operand.kind {
                OperandKind::Input => derived_inputs.push(idx as u32),
                OperandKind::Output => derived_outputs.push(idx as u32),
                OperandKind::Constant | OperandKind::Intermediate => {}
            }
        }

        if derived_inputs != self.input_operands {
            return Err(GraphError::InputIdListMismatch {
                input_ids: self.input_operands.clone(),
                input_ids_in_operands: derived_inputs,
            });
        }
        if derived_outputs != self.output_operands {
            return Err(GraphError::OutputIdListMismatch {
                output_ids: self.output_operands.clone(),
                output_ids_in_operands: derived_outputs,
            });
        }

        Ok(())
    }
}

pub enum WeightsToHash<'a> {
    None,
    All,
    Some(&'a HashSet<u32>),
}

/// Named input/output operands for `MLGraph` dispatch.
pub type IoBindingMaps = (
    HashMap<String, OperandDescriptor>,
    HashMap<String, OperandDescriptor>,
);

impl GraphInfo {
    /// Named input/output operands for `MLGraph` dispatch, after list consistency checks.
    #[allow(clippy::type_complexity)]
    pub fn io_binding_maps(
        &self,
    ) -> Result<
        (
            HashMap<String, OperandDescriptor>,
            HashMap<String, OperandDescriptor>,
        ),
        GraphError,
    > {
        self.validate_io_operand_lists()?;

        let mut inputs = HashMap::new();
        for &id in &self.input_operands {
            let operand = self.operand(id).expect("validated above");
            let name = operand
                .name
                .as_ref()
                .ok_or(GraphError::MissingInputName { operand: id })?;
            if name.is_empty() {
                return Err(GraphError::MissingInputName { operand: id });
            }
            if inputs
                .insert(name.clone(), operand.descriptor.clone())
                .is_some()
            {
                return Err(GraphError::DuplicateInputName { name: name.clone() });
            }
        }

        let mut outputs = HashMap::new();
        for &id in &self.output_operands {
            let operand = self.operand(id).expect("validated above");
            let name = operand
                .name
                .as_ref()
                .ok_or(GraphError::MissingOutputName { operand: id })?;
            if name.is_empty() {
                return Err(GraphError::MissingOutputName { operand: id });
            }
            if outputs
                .insert(name.clone(), operand.descriptor.clone())
                .is_some()
            {
                return Err(GraphError::DuplicateOutputName { name: name.clone() });
            }
        }

        Ok((inputs, outputs))
    }

    pub fn hash_identifier<'a>(&self, suffix: &str, weights_to_hash: WeightsToHash<'a>) -> String {
        let mut hasher = seahash::SeaHasher::new();
        self.input_operands.hash(&mut hasher);
        self.output_operands.hash(&mut hasher);
        self.operands.hash(&mut hasher);
        self.operations.hash(&mut hasher);
        match weights_to_hash {
            WeightsToHash::None => (),
            WeightsToHash::All => {
                for (constant_id, _) in self
                    .operands
                    .iter()
                    .enumerate()
                    .filter(|(_id, op)| op.kind == OperandKind::Constant)
                {
                    self.constant_operand_ids_to_handles
                        .get(&(constant_id as u32))
                        .hash(&mut hasher);
                }
            }
            WeightsToHash::Some(items) => {
                let mut items: Vec<u32> = items.iter().copied().collect();
                items.sort_unstable();
                for constant_id in items.iter() {
                    self.constant_operand_ids_to_handles
                        .get(constant_id)
                        .hash(&mut hasher);
                }
            }
        }
        let reproduciable_hash_64bit = hasher.finish();
        format!(
            "{reproduciable_hash_64bit:x}_{}_{suffix}",
            self.operands.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::GraphError;

    #[test]
    fn test_data_type_bits_per_element() {
        assert_eq!(DataType::Int4.bits_per_element(), 4);
        assert_eq!(DataType::Uint4.bits_per_element(), 4);
        assert_eq!(DataType::Float16.bits_per_element(), 16);
        assert_eq!(DataType::Float32.bits_per_element(), 32);
        assert_eq!(DataType::Int32.bits_per_element(), 32);
        assert_eq!(DataType::Uint32.bits_per_element(), 32);
        assert_eq!(DataType::Int8.bits_per_element(), 8);
        assert_eq!(DataType::Uint8.bits_per_element(), 8);
        assert_eq!(DataType::Int64.bits_per_element(), 64);
        assert_eq!(DataType::Uint64.bits_per_element(), 64);
    }

    #[test]
    fn test_data_type_bytes_per_element() {
        assert_eq!(DataType::Float16.bytes_per_element(), 2);
        assert_eq!(DataType::Float32.bytes_per_element(), 4);
        assert_eq!(DataType::Int32.bytes_per_element(), 4);
        assert_eq!(DataType::Uint32.bytes_per_element(), 4);
        assert_eq!(DataType::Int8.bytes_per_element(), 1);
        assert_eq!(DataType::Uint8.bytes_per_element(), 1);
        assert_eq!(DataType::Int64.bytes_per_element(), 8);
        assert_eq!(DataType::Uint64.bytes_per_element(), 8);
    }

    #[test]
    fn test_data_type_storage_byte_length_int4() {
        assert_eq!(DataType::Int4.storage_byte_length(0), Some(0));
        assert_eq!(DataType::Int4.storage_byte_length(1), Some(1));
        assert_eq!(DataType::Int4.storage_byte_length(2), Some(1));
        assert_eq!(DataType::Int4.storage_byte_length(3), Some(2));
        assert_eq!(DataType::Int4.storage_byte_length(100), Some(50));
    }

    #[test]
    fn test_pack_unpack_int4() {
        let values = vec![-8_i32, 7, 0, -1];
        let packed = pack_int4(&values);
        // Low-nibble-first: byte0 = 0x8 | (0x7 << 4), byte1 = 0x0 | (0xF << 4).
        assert_eq!(packed, vec![0x78, 0xF0]);
        assert_eq!(unpack_int4(&packed, values.len()), values);
    }

    #[test]
    fn test_pack_unpack_uint4() {
        let values = vec![0_u8, 15, 7, 1];
        let packed = pack_uint4(&values);
        // Low-nibble-first: byte0 = 0x0 | (0xF << 4), byte1 = 0x7 | (0x1 << 4).
        assert_eq!(packed, vec![0xF0, 0x17]);
        assert_eq!(unpack_uint4(&packed, values.len()), values);
    }

    #[test]
    fn test_data_type_serialization() {
        assert_eq!(serde_json::to_string(&DataType::Int4).unwrap(), "\"int4\"");
        assert_eq!(
            serde_json::to_string(&DataType::Uint4).unwrap(),
            "\"uint4\""
        );
        assert_eq!(
            serde_json::to_string(&DataType::Float32).unwrap(),
            "\"float32\""
        );
    }

    #[test]
    fn test_data_type_deserialization() {
        assert_eq!(
            serde_json::from_str::<DataType>("\"int4\"").unwrap(),
            DataType::Int4
        );
        assert_eq!(
            serde_json::from_str::<DataType>("\"uint4\"").unwrap(),
            DataType::Uint4
        );
        assert_eq!(
            serde_json::from_str::<DataType>("\"float32\"").unwrap(),
            DataType::Float32
        );
    }

    #[test]
    fn test_operand_descriptor_element_count() {
        let desc = OperandDescriptor {
            data_type: DataType::Int4,
            shape: to_dimension_vector(&[2, 3, 4]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.element_count(), Some(24));
    }

    #[test]
    fn test_operand_descriptor_byte_length_int4() {
        let desc = OperandDescriptor {
            data_type: DataType::Int4,
            shape: to_dimension_vector(&[10, 10]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.byte_length(), Some(50));
    }

    #[test]
    fn test_operand_descriptor_byte_length_uint4() {
        let desc = OperandDescriptor {
            data_type: DataType::Uint4,
            shape: to_dimension_vector(&[8, 16]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.byte_length(), Some(64));
    }

    #[test]
    fn test_operand_descriptor_byte_length_float32() {
        let desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: to_dimension_vector(&[4, 4]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.byte_length(), Some(64));
    }

    #[test]
    fn test_graph_info_quantized_field_default() {
        let json =
            r#"{"operands": [], "input_operands": [], "output_operands": [], "operations": []}"#;
        let graph: GraphInfo = serde_json::from_str(json).unwrap();
        assert!(!graph.quantized);
    }

    #[test]
    fn test_graph_info_quantized_field_true() {
        let json = r#"{"operands": [], "input_operands": [], "output_operands": [], "operations": [], "quantized": true}"#;
        let graph: GraphInfo = serde_json::from_str(json).unwrap();
        assert!(graph.quantized);
    }

    #[test]
    fn test_graph_info_quantized_field_serialization() {
        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };
        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"quantized\":true"));
    }

    #[test]
    fn test_graph_info_with_int4_operand() {
        let operand = Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: to_dimension_vector(&[1, 3, 224, 224]),
                pending_permutation: vec![],
            },
            name: Some("input".to_string()),
        };

        let graph = GraphInfo {
            operands: vec![operand],
            input_operands: vec![0],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"int4\""));
        assert!(json.contains("\"quantized\":true"));

        let deserialized: GraphInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.operands[0].descriptor.data_type,
            DataType::Int4
        );
        assert!(deserialized.quantized);
    }

    #[test]
    fn test_graph_info_with_uint4_operand() {
        let operand = Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Uint4,
                shape: to_dimension_vector(&[64, 64]),
                pending_permutation: vec![],
            },
            name: Some("weight".to_string()),
        };

        let graph = GraphInfo {
            operands: vec![operand],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: true,
        };

        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"uint4\""));

        let deserialized: GraphInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.operands[0].descriptor.data_type,
            DataType::Uint4
        );
    }

    fn sample_io_graph() -> GraphInfo {
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[2, 2]),
                        pending_permutation: vec![],
                    },
                    name: Some("x".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[2, 2]),
                        pending_permutation: vec![],
                    },
                    name: Some("y".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            id_to_constant_tensor_operand_map: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn validate_io_operand_lists_accepts_consistent_graph() {
        sample_io_graph().validate_io_operand_lists().unwrap();
        sample_io_graph().io_binding_maps().unwrap();
    }

    #[test]
    fn validate_io_operand_lists_rejects_extra_input_operand_id() {
        let mut graph = sample_io_graph();
        graph.input_operands.push(99);
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }

    #[test]
    fn validate_io_operand_lists_rejects_invalid_operand_id_in_list() {
        let mut graph = sample_io_graph();
        graph.input_operands = vec![0, 99];
        graph.operands.push(Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1]),
                pending_permutation: vec![],
            },
            name: Some("z".to_string()),
        });
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }

    #[test]
    fn validate_io_operand_lists_rejects_missing_input_in_list() {
        let mut graph = sample_io_graph();
        graph.operands.push(Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1]),
                pending_permutation: vec![],
            },
            name: Some("z".to_string()),
        });
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }

    #[test]
    fn validate_io_operand_lists_rejects_wrong_kind_in_input_list() {
        let mut graph = sample_io_graph();
        graph.input_operands = vec![1];
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }

    #[test]
    fn validate_io_operand_lists_rejects_output_kind_not_in_list() {
        let mut graph = sample_io_graph();
        graph.operands[0].kind = OperandKind::Output;
        graph.output_operands.push(0);
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }
}
