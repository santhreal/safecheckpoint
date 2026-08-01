//#![forbid(unsafe_code)] // mmap crate exception: safe mmap abstractions in reader.rs / writer.rs
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::todo,
        clippy::unimplemented,
        clippy::panic
    )
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
)]
//! `safecheckpoint`: A Rust-native ML model checkpoint library.
//!
//! Replaces `torch.save()` (pickle) with `safetensors`-compatible, mmap-based
//! serialization. Features include atomic saves, parallel sharding, and zero-copy reads.
//!
//! ## Safe defaults
//!
//! **Input size:** Header sizes are bounded during deserialization. Maximum memory-mapped allocation is caller-controlled by file selection.
//!
//! **Recursion depth:** None. Header parsing and shard validation use flat iterative loops over structure definitions.
//!
//! **Outbound network:** None. The crate operates on local file paths and `mmap` buffers with zero network I/O.
//!
//! **Process spawning:** None. The crate does not execute external processes or call `std::process::Command`.
//!
//! **Filesystem writes:** Atomic file writes take place within caller-specified target paths or temporary directory guards. Symlink traversal is rejected.
//!
//! **Credential exposure:** None. Model parameters and array byte buffers do not handle or log secret key materials.
//!
//! # Example: Simple Save and Load
//!
//! ```rust
//! use safecheckpoint::{Writer, Reader, DType};
//! # use std::fs;
//! # use tempfile::tempdir;
//!
//! # let dir = tempdir().unwrap();
//! # let path = dir.path().join("model.safetensors");
//!
//! // Save a tensor
//! let mut writer = Writer::new();
//! writer.add_tensor("w1", DType::F32, vec![2, 2], vec![0; 16]);
//! writer.save(&path).unwrap();
//!
//! // Load the tensor
//! let reader = Reader::open(&path).unwrap();
//! let tensor = reader.get_tensor("w1").unwrap();
//! assert_eq!(tensor.data.len(), 16);
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod error;
pub mod reader;
pub mod shard;
pub mod tensor;
pub mod writer;

pub use error::{Error, Result};
pub use reader::Reader;
pub use shard::ShardIndex;
pub use tensor::{CheckpointHeader, DType, Tensor, TensorMetadata, CURRENT_SCHEMA_VERSION};
pub use writer::Writer;
