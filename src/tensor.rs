//! Tensor metadata and data handling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported data types for tensors.
///
/// Matches the `safetensors` specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[non_exhaustive]
pub enum DType {
    /// 64-bit floating point.
    F64,
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point (half-precision).
    F16,
    /// Brain Floating Point (bfloat16).
    BF16,
    /// 64-bit signed integer.
    I64,
    /// 32-bit signed integer.
    I32,
    /// 16-bit signed integer.
    I16,
    /// 8-bit signed integer.
    I8,
    /// 64-bit unsigned integer.
    U64,
    /// 32-bit unsigned integer.
    U32,
    /// 16-bit unsigned integer.
    U16,
    /// 8-bit unsigned integer.
    U8,
    /// Boolean.
    // `BOOL` matches the safetensors specification spelling; renaming the
    // variant would change the on-disk JSON format, so the acronym lint is
    // silenced here instead.
    #[allow(clippy::upper_case_acronyms)]
    BOOL,
}

impl DType {
    /// Returns the number of bytes per element for the given data type.
    #[must_use]
    #[inline]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::BOOL => 1,
        }
    }
}

/// Metadata for a single tensor within a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// The data type of the tensor.
    pub dtype: DType,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
    /// The offset of the tensor's data in the data block (inclusive, exclusive).
    pub data_offsets: [usize; 2],
    /// Optional CRC32 checksum of the tensor's data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<u32>,
}

impl TensorMetadata {
    /// Validates the tensor metadata.
    ///
    /// # Errors
    /// Returns an error if the metadata is invalid.
    pub fn validate(&self) -> Result<(), crate::error::Error> {
        if self.data_offsets[0] > self.data_offsets[1] {
            return Err(crate::error::Error::InvalidFormat {
                offset: self.data_offsets[0],
                message: "data_offsets start is greater than end".into(),
            });
        }

        let expected_data_len = self
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .and_then(|elements| elements.checked_mul(self.dtype.size_in_bytes()))
            .ok_or_else(|| crate::error::Error::InvalidFormat {
                offset: self.data_offsets[0],
                message: "tensor shape size computation overflowed".into(),
            })?;

        let actual_data_len = self.data_offsets[1] - self.data_offsets[0];
        if expected_data_len != actual_data_len {
            return Err(crate::error::Error::SizeMismatch {
                tensor_name: "(validation)".to_string(),
                expected: expected_data_len,
                actual: actual_data_len,
            });
        }

        Ok(())
    }
}

/// A tensor with data and metadata.
#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    /// The metadata for the tensor.
    pub metadata: TensorMetadata,
    /// The actual data of the tensor.
    pub data: &'a [u8],
}

/// The current schema version for checkpoint headers.
pub const CURRENT_SCHEMA_VERSION: u64 = 1;

/// The header of a checkpoint, containing metadata for all tensors.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointHeader {
    /// Schema version of the checkpoint format.
    #[serde(default = "default_schema_version")]
    pub schema_version: u64,

    /// Metadata for each tensor, indexed by name.
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorMetadata>,

    /// Optional metadata about the checkpoint.
    #[serde(
        rename = "__metadata__",
        skip_serializing_if = "HashMap::is_empty",
        default
    )]
    pub metadata: HashMap<String, String>,
}

fn default_schema_version() -> u64 {
    CURRENT_SCHEMA_VERSION
}
