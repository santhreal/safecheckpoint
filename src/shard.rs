//! Sharding logic for large model checkpoints.

use crate::error::{Error, Result};
use crate::writer::Writer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// A type alias for a tensor in the writer.
type WriterTensor<'a> = (&'a String, &'a (crate::tensor::DType, Vec<usize>, Vec<u8>));

/// The index file for sharded checkpoints.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ShardIndex {
    /// Mapping of tensor names to their respective shard files.
    pub weight_map: HashMap<String, String>,
    /// Optional metadata about the sharded checkpoint.
    #[serde(rename = "__metadata__", skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

use crate::reader::Reader;
use rayon::prelude::*;

impl Writer {
    /// Saves the tensors into multiple shards, supporting resumption.
    ///
    /// This method splits the tensors across multiple files and creates
    /// a central index file (`.index.json`). If a shard file already exists
    /// and is a valid `safetensors` file, it will be skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard fails to write or if the index file
    /// cannot be created.
    pub fn save_sharded<P: AsRef<Path>>(
        &self,
        directory: P,
        prefix: &str,
        num_shards: usize,
    ) -> Result<()> {
        let directory = directory.as_ref();

        Self::validate_shard_args(directory, prefix, num_shards)?;

        if !directory.exists() {
            std::fs::create_dir_all(directory)?;
        }

        // [fixed 2026-04-23] HIGH: directory-level lockfile for save_sharded
        let lock_path = directory.join(".safecheckpoint.lock");
        let lock_file = File::create(&lock_path).map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to create lock file: {e}"),
            ))
        })?;
        fs4::fs_std::FileExt::lock_exclusive(&lock_file).map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to acquire exclusive lock: {e}"),
            ))
        })?;

        let mut sorted_tensors: Vec<WriterTensor> = self.tensors.iter().collect();
        sorted_tensors.sort_by_key(|(name, _)| *name);

        let tensors_per_shard = sorted_tensors.len().div_ceil(num_shards);

        // Write shards in parallel.
        (0..num_shards)
            .into_par_iter()
            .map(|i| {
                let start = i * tensors_per_shard;
                let end = std::cmp::min(start + tensors_per_shard, sorted_tensors.len());
                if start >= end {
                    return Ok(());
                }
                let shard_name = format!("{prefix}-{:05}-of-{:05}.safetensors", i + 1, num_shards);
                let shard_path = directory.join(&shard_name);

                if Self::is_shard_valid(&shard_path, &sorted_tensors[start..end]) {
                    return Ok(());
                }

                // Borrow tensor views straight out of the parent writer's map;
                // do not clone the (possibly multi-gigabyte) tensor bytes into a
                // throwaway per-shard Writer.
                let views: Vec<crate::writer::TensorView<'_>> = sorted_tensors[start..end]
                    .iter()
                    .map(|(name, (dtype, shape, data))| {
                        (name.as_str(), *dtype, shape.as_slice(), data.as_slice())
                    })
                    .collect();
                Writer::save_borrowed(&shard_path, &self.metadata, &views)?;
                Ok(())
            })
            .collect::<Result<Vec<()>>>()?;

        // Verify shards weren't modified concurrently before writing index.
        let weight_map =
            Self::verify_shards(directory, prefix, num_shards, tensors_per_shard, &sorted_tensors)?;

        let index = ShardIndex {
            weight_map,
            metadata: Some(self.metadata.clone()),
        };
        Self::write_index_atomic(directory, prefix, &index)
    }

    /// Atomically write the shard index file.
    ///
    /// The temp file is created through `tempfile` with a randomized name
    /// inside `directory`, exactly like the shard writer: a predictable temp
    /// name plus `File::create` would follow a pre-planted symlink and
    /// overwrite an unrelated file in a shared directory. `persist` performs
    /// the atomic rename; the directory fsync makes it durable.
    fn write_index_atomic(directory: &Path, prefix: &str, index: &ShardIndex) -> Result<()> {
        let index_path = directory.join(format!("{prefix}.safetensors.index.json"));
        let index_json = serde_json::to_string_pretty(index)?;

        let mut temp_file = tempfile::Builder::new()
            .prefix(".safecheckpoint-index-tmp-")
            .suffix(".tmp")
            .tempfile_in(directory)
            .map_err(|e| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to create temp index file: {e}"),
                ))
            })?;
        temp_file.write_all(index_json.as_bytes()).map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to write temp index file: {e}"),
            ))
        })?;
        temp_file.as_file().sync_all().map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to fsync temp index file: {e}"),
            ))
        })?;
        temp_file.persist(&index_path).map_err(|e| {
            Error::AtomicRename(e.error)
        })?;

        // sync the directory so the rename is durable; the error propagates.
        let dir = std::fs::File::open(directory).map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to open directory for sync: {e}"),
            ))
        })?;
        dir.sync_all().map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to sync directory: {e}"),
            ))
        })?;

        Ok(())
    }

    /// Validate the sharded-save arguments: `num_shards` must be non-zero and
    /// neither `directory` nor `prefix` may traverse out of the target.
    fn validate_shard_args(directory: &Path, prefix: &str, num_shards: usize) -> Result<()> {
        if num_shards == 0 {
            return Err(Error::InvalidArgument {
                parameter: "num_shards".into(),
                reason: "must be at least 1; got 0 (division by zero when partitioning tensors)"
                    .into(),
            });
        }

        // Path traversal defense using path components
        for component in directory.components() {
            if component == std::path::Component::ParentDir {
                return Err(Error::PathTraversal {
                    path: directory.to_string_lossy().into_owned(),
                    reason: "directory contains '..'".into(),
                });
            }
        }

        let prefix_path = Path::new(prefix);
        for component in prefix_path.components() {
            if component == std::path::Component::ParentDir {
                return Err(Error::PathTraversal {
                    path: prefix.to_string(),
                    reason: "prefix contains '..'".into(),
                });
            }
        }
        if prefix.contains('/') || prefix.contains('\\') {
            return Err(Error::PathTraversal {
                path: prefix.to_string(),
                reason: "prefix contains directory separators".into(),
            });
        }
        Ok(())
    }

    /// Re-read every shard after the parallel write and build the tensor-name
    /// to shard-file map, failing if any shard was modified concurrently.
    fn verify_shards(
        directory: &Path,
        prefix: &str,
        num_shards: usize,
        tensors_per_shard: usize,
        sorted_tensors: &[WriterTensor<'_>],
    ) -> Result<HashMap<String, String>> {
        let mut weight_map = HashMap::new();
        for i in 0..num_shards {
            let start = i * tensors_per_shard;
            let end = std::cmp::min(start + tensors_per_shard, sorted_tensors.len());

            if start >= end {
                continue;
            }

            let shard_name = format!("{prefix}-{:05}-of-{:05}.safetensors", i + 1, num_shards);
            let shard_path = directory.join(&shard_name);

            let reader = Reader::open(&shard_path).map_err(|e| {
                Error::ShardVerification {
                    shard: shard_name.clone(),
                    source: Box::new(e),
                }
            })?;

            for (name, (_, _, expected_data)) in &sorted_tensors[start..end] {
                let tensor = reader.get_tensor(name.as_str()).map_err(|e| {
                    Error::ShardVerification {
                        shard: shard_name.clone(),
                        source: Box::new(e),
                    }
                })?;

                let expected_checksum = crc32fast::hash(expected_data);
                if tensor.metadata.checksum != Some(expected_checksum) {
                    return Err(Error::ShardVerification {
                        shard: shard_name.clone(),
                        source: Box::new(Error::Checksum {
                            tensor_name: (*name).clone(),
                            expected: expected_checksum,
                            actual: tensor.metadata.checksum.unwrap_or(0),
                        }),
                    });
                }

                weight_map.insert((*name).clone(), shard_name.clone());
            }
        }
        Ok(weight_map)
    }

    /// Checks if a shard exists and is valid.
    ///
    /// [fixed 2026-04-23] HIGH: exact tensor set match (no extra tensors)
    fn is_shard_valid(shard_path: &Path, expected_tensors: &[WriterTensor<'_>]) -> bool {
        if !shard_path.exists() {
            return false;
        }
        let Ok(reader) = Reader::open(shard_path) else {
            return false;
        };

        let expected_names: std::collections::HashSet<&String> =
            expected_tensors.iter().map(|(name, _)| *name).collect();
        let actual_names: std::collections::HashSet<&String> =
            reader.header().tensors.keys().collect();

        if expected_names != actual_names {
            return false;
        }

        for (name, (_, _, expected_data)) in expected_tensors {
            let Ok(tensor) = reader.get_tensor(name.as_str()) else {
                return false;
            };
            if let Some(file_checksum) = tensor.metadata.checksum {
                let expected_checksum = crc32fast::hash(expected_data);
                if file_checksum != expected_checksum {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}
