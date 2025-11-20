use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_with::{base64::Base64, serde_as};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Float16,
    Float32,
    Int32,
    Uint32,
    Int8,
    Uint8,
}

impl DataType {
    pub fn bytes_per_element(self) -> usize {
        match self {
            DataType::Float16 => 2,
            DataType::Float32 => 4,
            DataType::Int32 => 4,
            DataType::Uint32 => 4,
            DataType::Int8 => 1,
            DataType::Uint8 => 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperandDescriptor {
    pub data_type: DataType,
    #[serde(default)]
    pub shape: Vec<u32>,
    #[serde(default)]
    pub pending_permutation: Vec<u32>,
}

impl OperandDescriptor {
    pub fn element_count(&self) -> Option<usize> {
        if self.shape.is_empty() {
            return Some(1);
        }
        let mut count = 1usize;
        for dim in &self.shape {
            count = count.checked_mul(*dim as usize)?;
        }
        Some(count)
    }

    pub fn byte_length(&self) -> Option<usize> {
        let elements = self.element_count()?;
        elements.checked_mul(self.data_type.bytes_per_element())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperandKind {
    Input,
    Constant,
    Output,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operand {
    pub kind: OperandKind,
    pub descriptor: OperandDescriptor,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    #[serde(rename = "type")]
    pub op_type: String,
    #[serde(default)]
    pub input_operands: Vec<u32>,
    pub output_operand: u32,
    #[serde(default)]
    pub attributes: serde_json::Value,
    #[serde(default)]
    pub label: Option<String>,
}

impl Operation {
    pub fn display_name(&self) -> String {
        self.label.clone().unwrap_or_else(|| self.op_type.clone())
    }
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantData {
    #[serde_as(as = "Base64")]
    pub data: Vec<u8>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

impl GraphInfo {
    pub fn operand(&self, id: u32) -> Option<&Operand> {
        self.operands.get(id as usize)
    }
}
