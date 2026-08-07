//! Zero-copy mmap-based checkpoint reader.

use crate::error::{Error, Result};
use crate::tensor::{CheckpointHeader, Tensor, CURRENT_SCHEMA_VERSION};
use fs4::fs_std::FileExt;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use tracing::{error, instrument};

/// A reader for `safetensors` compatible checkpoints.
pub struct Reader {
    /// The memory-mapped file content.
    mmap: Mmap,
    /// The parsed header.
    header: CheckpointHeader,
    /// The offset where the data block starts.
    data_offset: usize,
    /// The file handle, kept open to maintain the shared lock.
    _file: File,
    /// The original file size, recorded at open time for integrity checks.
    file_size: usize,
}

impl Reader {
    /// Opens a checkpoint file from the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, memory-mapped,
    /// or if the header is invalid or missing.
    #[instrument(level = "debug")]
    pub fn open<P: AsRef<Path> + std::fmt::Debug>(path: P) -> Result<Self> {
        let file = File::open(&path).map_err(|e| {
            error!(?path, error = %e, "failed to open checkpoint file");
            e
        })?;

        // Acquire a shared lock to prevent concurrent writes.
        // This is advisory-only; non-compliant processes can still modify the file.
        FileExt::lock_shared(&file).map_err(|e| {
            error!(?path, error = %e, "failed to acquire shared lock on checkpoint file");
            e
        })?;

        let file_size = file
            .metadata()
            .map_err(|e| {
                error!(?path, error = %e, "failed to read file metadata");
                e
            })?
            .len();
        let file_size = usize::try_from(file_size).map_err(|_| Error::InvalidFormat {
            offset: 0,
            message: "file size does not fit in this platform's address space".into(),
        })?;
        if file_size < 8 {
            return Err(Error::InvalidFormat {
                offset: 0,
                message: "file too short to contain header length".into(),
            });
        }

        #[allow(unsafe_code)]
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(Error::InvalidFormat {
                offset: 0,
                message: "file too short to contain header length".into(),
            });
        }

        let mut header_len_bytes = [0u8; 8];
        header_len_bytes.copy_from_slice(&mmap[0..8]);
        let header_len = usize::try_from(u64::from_le_bytes(header_len_bytes)).map_err(|_| {
            Error::InvalidFormat {
                offset: 0,
                message: "header length too large for this platform".into(),
            }
        })?;

        // Prevent OOM from maliciously large header lengths.
        // safetensors spec doesn't strictly limit, but 100MB is generous for any real model.
        if header_len > 100 * 1024 * 1024 {
            return Err(Error::InvalidFormat {
                offset: 0,
                message: "header length exceeds 100MB limit (potential DOS)".into(),
            });
        }

        // [fixed 2026-04-23] CRITICAL: header checksum validation before JSON parse
        if mmap.len() < 8 + header_len + 32 {
            return Err(Error::InvalidFormat {
                offset: 8,
                message: "file too short to contain full header and checksum".into(),
            });
        }

        let header_json_range = 8..8 + header_len;
        let checksum_range = 8 + header_len..8 + header_len + 32;

        let mut hasher = blake3::Hasher::new();
        hasher.update(&header_len_bytes);
        hasher.update(&mmap[header_json_range.clone()]);
        let expected_checksum = hasher.finalize();
        let actual_checksum = &mmap[checksum_range];

        if expected_checksum.as_bytes() != actual_checksum {
            // The slice is exactly 32 bytes by construction (checksum_range),
            // so the conversion cannot fail; `unwrap_or` keeps this deny-
            // lint-clean without panicking on an unreachable branch.
            let actual_bytes: [u8; 32] = actual_checksum.try_into().unwrap_or([0u8; 32]);
            return Err(Error::HeaderChecksum {
                expected: expected_checksum.to_string(),
                actual: blake3::Hash::from_bytes(actual_bytes).to_string(),
            });
        }

        let header: CheckpointHeader = serde_json::from_slice(&mmap[header_json_range])?;

        // [fixed 2026-04-23] CRITICAL: reject unknown schema versions
        if header.schema_version > CURRENT_SCHEMA_VERSION {
            return Err(Error::UnsupportedVersion {
                found: header.schema_version,
                supported: CURRENT_SCHEMA_VERSION,
            });
        }

        let data_offset = 8 + header_len + 32;

        Ok(Self {
            mmap,
            header,
            data_offset,
            _file: file,
            file_size,
        })
    }

    /// Returns the metadata for all tensors in the checkpoint.
    #[must_use]
    pub fn header(&self) -> &CheckpointHeader {
        &self.header
    }

    /// Returns a list of all tensor names in the checkpoint.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<String> {
        self.header.tensors.keys().cloned().collect()
    }

    /// Returns a single tensor by name.
    ///
    /// This is a zero-copy operation that returns a reference to the data
    /// within the memory-mapped file.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not found or if its data block
    /// exceeds the file boundaries, or if the checksum is invalid.
    #[instrument(skip(self), level = "debug")]
    pub fn get_tensor(&self, name: &str) -> Result<Tensor<'_>> {
        let metadata = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;

        metadata.validate()?;

        let start = self
            .data_offset
            .checked_add(metadata.data_offsets[0])
            .ok_or_else(|| Error::OffsetOverflow {
                tensor_name: name.to_string(),
                offset: metadata.data_offsets[0],
            })?;
        let end = self
            .data_offset
            .checked_add(metadata.data_offsets[1])
            .ok_or_else(|| Error::OffsetOverflow {
                tensor_name: name.to_string(),
                offset: metadata.data_offsets[1],
            })?;

        // [fixed 2026-04-23] CRITICAL: validate offsets against recorded file size
        if end > self.file_size {
            error!(
                tensor = name,
                expected_end = end,
                file_size = self.file_size,
                "tensor data extends beyond file boundary"
            );
            return Err(Error::SizeMismatch {
                tensor_name: name.to_string(),
                expected: metadata.data_offsets[1] - metadata.data_offsets[0],
                actual: self.file_size.saturating_sub(start),
            });
        }

        if end > self.mmap.len() {
            error!(
                tensor = name,
                expected_end = end,
                mmap_len = self.mmap.len(),
                "tensor data extends beyond mmap boundary"
            );
            return Err(Error::SizeMismatch {
                tensor_name: name.to_string(),
                expected: metadata.data_offsets[1] - metadata.data_offsets[0],
                actual: self.mmap.len().saturating_sub(start),
            });
        }

        let data = &self.mmap[start..end];

        if let Some(expected_checksum) = metadata.checksum {
            let actual_checksum = crc32fast::hash(data);
            if actual_checksum != expected_checksum {
                return Err(Error::Checksum {
                    tensor_name: name.to_string(),
                    expected: expected_checksum,
                    actual: actual_checksum,
                });
            }
        }

        Ok(Tensor {
            metadata: metadata.clone(),
            data,
        })
    }

    /// Returns the global metadata for the checkpoint.
    #[must_use]
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.header.metadata
    }
}
