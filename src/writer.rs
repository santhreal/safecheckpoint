//! Atomic mmap-based checkpoint writer.

use crate::error::{Error, Result};
use crate::tensor::{CheckpointHeader, DType, TensorMetadata, CURRENT_SCHEMA_VERSION};
use byteorder::{LittleEndian, WriteBytesExt};
use memmap2::MmapMut;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;
use tracing::{error, instrument};

/// A borrowed view of one tensor for serialization: `(name, dtype, shape,
/// data)`. Serialization operates over slices of these so a single primitive
/// can write a whole-file checkpoint or an individual shard without ever
/// copying tensor bytes into an intermediate owning `Writer`.
pub(crate) type TensorView<'a> = (&'a str, DType, &'a [usize], &'a [u8]);

/// A writer for `safetensors` compatible checkpoints.
#[derive(Debug, Default)]
pub struct Writer {
    /// Tensors to be written.
    pub(crate) tensors: HashMap<String, (DType, Vec<usize>, Vec<u8>)>,
    /// Optional metadata about the checkpoint.
    pub(crate) metadata: HashMap<String, String>,
}

impl Writer {
    /// Creates a new, empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a tensor to the writer.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor name is reserved (`__metadata__`).
    pub fn add_tensor(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> Result<&mut Self> {
        if name == "__metadata__" {
            return Err(Error::ReservedTensorName {
                name: name.to_string(),
            });
        }
        self.tensors.insert(name.to_string(), (dtype, shape, data));
        Ok(self)
    }

    /// Adds multiple tensors to the writer.
    pub fn add_tensors(
        &mut self,
        tensors: HashMap<String, (DType, Vec<usize>, Vec<u8>)>,
    ) -> &mut Self {
        self.tensors.extend(tensors);
        self
    }

    /// Adds metadata to the writer.
    pub fn add_metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Validates that a write target path does not traverse outside its intended
    /// directory and is not a symlink.
    ///
    /// [fixed 2026-04-23] CRITICAL: path traversal defense (canonicalize + base dir check, symlink reject)
    fn validate_write_path(path: &Path, base: &Path) -> Result<()> {
        // Reject `..` anywhere in the target path (absolute paths are allowed).
        for component in path.components() {
            if matches!(component, std::path::Component::ParentDir) {
                return Err(Error::PathTraversal {
                    path: path.to_string_lossy().into_owned(),
                    reason: "path contains parent-dir components".into(),
                });
            }
        }

        let canonical_base = base.canonicalize().map_err(|e| {
            Error::PathTraversal {
                path: base.to_string_lossy().into_owned(),
                reason: format!("cannot canonicalize base directory: {e}"),
            }
        })?;

        let full = canonical_base.join(path);
        let canonical_full = full.canonicalize().or_else(|_| {
            // If the file doesn't exist yet, canonicalize its parent.
            if let Some(parent) = full.parent() {
                let canonical_parent = parent.canonicalize().map_err(|e| {
                    Error::PathTraversal {
                        path: parent.to_string_lossy().into_owned(),
                        reason: format!("cannot canonicalize parent directory: {e}"),
                    }
                })?;
                Ok(canonical_parent.join(full.file_name().unwrap_or_default()))
            } else {
                Err(Error::PathTraversal {
                    path: full.to_string_lossy().into_owned(),
                    reason: "no parent directory".into(),
                })
            }
        })?;

        if !canonical_full.starts_with(&canonical_base) {
            return Err(Error::PathTraversal {
                path: path.to_string_lossy().into_owned(),
                reason: "resolved path escapes base directory".into(),
            });
        }

        // Reject symlinks anywhere in the path. Fail CLOSED on any stat error
        // other than NotFound - the old `if let Ok(meta)` silently skipped the
        // symlink check for a component whose metadata could not be read.
        reject_symlink_components(&canonical_full, |p| std::fs::symlink_metadata(p))
    }

    /// Writes the checkpoint to the specified path atomically.
    ///
    /// The writing process is atomic: it writes to a temporary file,
    /// performs an fsync to ensure data is on disk, and then renames it
    /// to the final destination.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created, written to,
    /// or if the atomic rename fails.
    #[instrument(level = "info")]
    pub fn save<P: AsRef<Path> + std::fmt::Debug>(&self, path: P) -> Result<()> {
        // Build borrowed, name-sorted views over the owned tensors and delegate
        // to the shared serialization primitive. No tensor bytes are copied.
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        let views: Vec<TensorView<'_>> = names
            .iter()
            .map(|name| {
                let (dtype, shape, data) = &self.tensors[*name];
                (name.as_str(), *dtype, shape.as_slice(), data.as_slice())
            })
            .collect();
        Self::save_borrowed(path, &self.metadata, &views)
    }

    /// Atomically serialize a borrowed, name-sorted tensor set to `path`.
    ///
    /// This is the single owner of the temp-file + fsync + atomic-rename dance.
    /// Both [`Writer::save`] and the sharded writer call it, the latter passing
    /// views borrowed straight out of the parent writer's tensor map so a shard
    /// never clones its (potentially multi-gigabyte) tensor bytes.
    pub(crate) fn save_borrowed<P: AsRef<Path> + std::fmt::Debug>(
        path: P,
        metadata: &HashMap<String, String>,
        tensors: &[TensorView<'_>],
    ) -> Result<()> {
        let path = path.as_ref();

        // For a bare relative filename, `path.parent()` is `Some("")` (an empty
        // path), not `None`. Opening "" would fail, so normalize the empty
        // parent to the current directory. This is the one place the target
        // directory is derived; the atomic-rename fsync below reuses it.
        let base = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        Self::validate_write_path(path, base)?;

        // [fixed 2026-04-23] CRITICAL: use tempfile::NamedTempFile for unique temp paths
        let temp_file = tempfile::Builder::new()
            .prefix(".safecheckpoint-tmp-")
            .suffix(".tmp")
            .tempfile_in(base)
            .map_err(|e| {
                error!(?path, error = %e, "failed to create temporary file");
                Error::Io(e)
            })?;
        let temp_path = temp_file.path().to_path_buf();

        if let Err(e) = Self::write_tensors_to_file(temp_file.as_file(), metadata, tensors) {
            // Attempt cleanup
            if let Err(cleanup_err) = temp_file.close() {
                error!(?temp_path, error = %cleanup_err, "failed to cleanup temp file");
                return Err(Error::AtomicCleanup(cleanup_err));
            }
            return Err(e);
        }

        // Persist the temp file to the final destination.
        temp_file.persist(path).map_err(|e| {
            error!(?path, ?temp_path, error = %e.error, "atomic rename failed");
            Error::AtomicRename(e.error)
        })?;

        // Fsync the directory so the rename is durable. Use `base` (never an
        // empty path) rather than re-deriving `path.parent()`, which is
        // `Some("")` for a bare filename and would fail to open, silently
        // skipping the durability barrier for relative paths.
        let dir = std::fs::File::open(base).map_err(|e| {
            error!(?base, error = %e, "failed to open directory for sync");
            Error::Io(e)
        })?;
        dir.sync_all().map_err(|e| {
            error!(?base, error = %e, "failed to sync directory");
            Error::Io(e)
        })?;

        Ok(())
    }

    /// Prepares the header and calculates total offset sizes over a borrowed,
    /// name-sorted tensor set.
    fn build_header(
        metadata: &HashMap<String, String>,
        tensors: &[TensorView<'_>],
    ) -> Result<(CheckpointHeader, usize)> {
        let mut header = CheckpointHeader {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata: metadata.clone(),
            ..Default::default()
        };

        let mut current_offset: usize = 0;

        for &(name, dtype, shape, data) in tensors {
            let start = current_offset;
            let end = current_offset
                .checked_add(data.len())
                .ok_or_else(|| Error::OffsetOverflow {
                    tensor_name: name.to_string(),
                    offset: current_offset,
                })?;
            let checksum = crc32fast::hash(data);

            // [fixed 2026-04-23] MEDIUM: validate tensor data size against shape * dtype.size_in_bytes()
            let expected_data_len = shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
                .ok_or_else(|| Error::InvalidFormat {
                    offset: start,
                    message: format!("tensor '{name}' shape size computation overflowed"),
                })?;
            if expected_data_len != data.len() {
                return Err(Error::SizeMismatch {
                    tensor_name: name.to_string(),
                    expected: expected_data_len,
                    actual: data.len(),
                });
            }

            header.tensors.insert(
                name.to_string(),
                TensorMetadata {
                    dtype,
                    shape: shape.to_vec(),
                    data_offsets: [start, end],
                    checksum: Some(checksum),
                },
            );

            current_offset = end;
        }

        Ok((header, current_offset))
    }

    /// Serialize a borrowed, name-sorted tensor set to an open file handle.
    fn write_tensors_to_file(
        file: &std::fs::File,
        metadata: &HashMap<String, String>,
        tensors: &[TensorView<'_>],
    ) -> Result<()> {
        let (header, current_offset) = Self::build_header(metadata, tensors)?;

        let header_json = serde_json::to_vec(&header)?;
        let header_len = header_json.len() as u64;

        // Total file size: 8 bytes (header length) + header + data + 32 bytes (blake3 checksum).
        let total_size = 8u64
            .checked_add(header_len)
            .and_then(|s| s.checked_add(current_offset as u64))
            .and_then(|s| s.checked_add(32))
            .ok_or_else(|| Error::OffsetOverflow {
                tensor_name: "<file total>".into(),
                offset: current_offset,
            })?;

        // Acquire an exclusive lock to prevent concurrent writes/reads during modification.
        fs4::fs_std::FileExt::lock_exclusive(file).map_err(|e| {
            error!(error = %e, "failed to acquire exclusive lock");
            e
        })?;

        file.set_len(total_size)?;

        // Write header length.
        let mut writer = BufWriter::new(file);
        writer.write_u64::<LittleEndian>(header_len)?;
        writer.write_all(&header_json)?;

        // Header checksum immediately follows JSON (reader validates before parsing).
        let mut hasher = blake3::Hasher::new();
        hasher.update(&header_len.to_le_bytes());
        hasher.update(&header_json);
        let checksum = hasher.finalize();
        writer.write_all(checksum.as_bytes())?;
        writer.flush()?;

        // Use mmap to write the tensor data for zero-copy efficiency.
        // SAFETY: We have exclusive access to the file (flock) and have set the correct length.
        // The advisory lock does not prevent non-compliant processes from modifying the file;
        // this is a best-effort coordination mechanism.
        #[allow(unsafe_code)]
        let mut mmap = unsafe { MmapMut::map_mut(file)? };
        let mut data_start = 8
            + usize::try_from(header_len).map_err(|_| Error::InvalidFormat {
                offset: 0,
                message: "header length too large for this platform".into(),
            })?
            + 32;

        for &(_, _, _, data) in tensors {
            let data_end = data_start + data.len();
            mmap[data_start..data_end].copy_from_slice(data);
            data_start = data_end;
        }

        mmap.flush()?;
        drop(mmap);

        // Ensure everything is synced to disk.
        file.sync_all()?;

        fs4::fs_std::FileExt::unlock(file).map_err(|e| {
            error!(error = %e, "failed to unlock file");
            e
        })?;

        Ok(())
    }
}

/// Walk `leaf` and every ancestor, rejecting any symlink component. `stat`
/// returns each component's symlink metadata (normally `fs::symlink_metadata`,
/// injectable for tests).
///
/// A `NotFound` is tolerated (a component may be absent in a TOCTOU race), but
/// ANY other stat error fails CLOSED with `PathTraversal`: a security check
/// that cannot read a component's metadata must refuse, never silently skip it
/// (Law 10 / fail-closed for security controls).
fn reject_symlink_components<F>(leaf: &Path, mut stat: F) -> Result<()>
where
    F: FnMut(&Path) -> std::io::Result<std::fs::Metadata>,
{
    let mut current = Some(leaf);
    while let Some(p) = current {
        match stat(p) {
            Ok(meta) => {
                if meta.file_type().is_symlink() {
                    return Err(Error::PathTraversal {
                        path: p.to_string_lossy().into_owned(),
                        reason: "symlinks are not permitted".into(),
                    });
                }
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(Error::PathTraversal {
                    path: p.to_string_lossy().into_owned(),
                    reason: format!(
                        "could not verify path component is not a symlink: {source}"
                    ),
                });
            }
        }
        current = p.parent();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error as IoError, ErrorKind};

    #[test]
    fn reject_symlink_components_fails_closed_on_permission_denied() {
        // A non-NotFound stat error (e.g. permission denied) must fail closed
        // with PathTraversal, never silently skip the symlink check.
        let result = reject_symlink_components(Path::new("/a/b/c"), |_p| {
            Err(IoError::from(ErrorKind::PermissionDenied))
        });
        assert!(
            matches!(result, Err(Error::PathTraversal { .. })),
            "permission-denied on a component must surface, got {result:?}"
        );
    }

    #[test]
    fn reject_symlink_components_tolerates_not_found() {
        // NotFound (a racing-absent component) must be tolerated so a valid
        // not-yet-created path is not spuriously rejected.
        let result = reject_symlink_components(Path::new("/a/b/c"), |_p| {
            Err(IoError::from(ErrorKind::NotFound))
        });
        assert!(result.is_ok(), "NotFound components must be tolerated, got {result:?}");
    }
}
