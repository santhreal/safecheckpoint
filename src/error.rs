//! Custom error types for safecheckpoint.
//!
//! Every error is designed to be actionable and informative.

/// The primary error type for all safecheckpoint operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Error during I/O operations.
    Io(std::io::Error),

    /// Error during JSON serialization or deserialization.
    Json(serde_json::Error),

    /// Error when a tensor is not found.
    TensorNotFound(String),

    /// Error when the checkpoint format is invalid.
    InvalidFormat {
        /// Offset where the error occurred, or 0 if unknown.
        offset: usize,
        /// Contextual message.
        message: String,
    },

    /// Error when a tensor's metadata and data are inconsistent.
    SizeMismatch {
        /// The name of the tensor.
        tensor_name: String,
        /// The expected size in bytes.
        expected: usize,
        /// The actual size found in the data block.
        actual: usize,
    },

    /// Error when an offset computation overflows.
    OffsetOverflow {
        /// The name of the tensor.
        tensor_name: String,
        /// The offending offset value.
        offset: usize,
    },

    /// Error when sharding or shard verification fails.
    ShardVerification {
        /// The shard file path.
        shard: String,
        /// The underlying error.
        source: Box<Error>,
    },

    /// Error when a checksum validation fails.
    Checksum {
        /// The tensor name.
        tensor_name: String,
        /// The expected checksum.
        expected: u32,
        /// The actual checksum.
        actual: u32,
    },

    /// Error when an atomic save (rename) fails.
    AtomicRename(std::io::Error),

    /// Error when an atomic save (cleanup) fails.
    AtomicCleanup(std::io::Error),

    /// Error when a checkpoint schema version is unsupported.
    UnsupportedVersion {
        /// The version found in the file.
        found: u64,
        /// The maximum supported version.
        supported: u64,
    },

    /// Error when a path traversal attempt is detected.
    PathTraversal {
        /// The path that was rejected.
        path: String,
        /// The reason for rejection.
        reason: String,
    },

    /// Error when a reserved tensor name is used.
    ReservedTensorName {
        /// The reserved name that was used.
        name: String,
    },

    /// Error when a header checksum validation fails.
    HeaderChecksum {
        /// The expected checksum.
        expected: String,
        /// The actual checksum.
        actual: String,
    },

    /// Error when a caller-supplied argument is invalid.
    InvalidArgument {
        /// The name of the offending parameter.
        parameter: String,
        /// Why the value was rejected and how to fix it.
        reason: String,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "i/o error: {e}"),
            Self::Json(e) => write!(f, "json error: {e}"),
            Self::TensorNotFound(name) => write!(f, "tensor '{name}' not found in checkpoint"),
            Self::InvalidFormat { offset, message } => {
                write!(f, "invalid checkpoint format at offset {offset}: {message}")
            }
            Self::SizeMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                f,
                "tensor '{tensor_name}' size mismatch: expected {expected}, found {actual}"
            ),
            Self::OffsetOverflow {
                tensor_name,
                offset,
            } => write!(
                f,
                "tensor '{tensor_name}' offset overflow: offset {offset} exceeds addressable range"
            ),
            Self::ShardVerification { shard, source } => {
                write!(f, "shard verification failed for '{shard}': {source}")
            }
            Self::Checksum {
                tensor_name,
                expected,
                actual,
            } => write!(
                f,
                "checksum validation failed for '{tensor_name}': expected {expected}, found {actual}"
            ),
            Self::AtomicRename(e) => write!(f, "atomic save (rename) failed: {e}"),
            Self::AtomicCleanup(e) => write!(f, "atomic save (cleanup) failed: {e}"),
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "unsupported checkpoint schema version {found} (supported: {supported})"
            ),
            Self::PathTraversal { path, reason } => {
                write!(f, "path traversal blocked for '{path}': {reason}")
            }
            Self::ReservedTensorName { name } => {
                write!(f, "reserved tensor name '{name}' is not allowed")
            }
            Self::HeaderChecksum { expected, actual } => write!(
                f,
                "header checksum mismatch: expected {expected}, found {actual}"
            ),
            Self::InvalidArgument { parameter, reason } => {
                write!(f, "invalid argument '{parameter}': {reason}")
            }
        }
    }
}

impl std::error::Error for Error {
    // Several variants legitimately share the `Some(e)` body: merging them
    // would hurt readability, so the identical-arms lint is silenced here.
    #[allow(clippy::match_same_arms)]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::ShardVerification { source, .. } => Some(source),
            Self::AtomicRename(e) | Self::AtomicCleanup(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

/// A specialized Result type for safecheckpoint.
pub type Result<T> = std::result::Result<T, Error>;
