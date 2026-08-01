# safecheckpoint  -  Technical Spec

## Overview

`safecheckpoint`: A Rust-native ML model checkpoint library.  Replaces `torch.save()` (pickle) with `safetensors`-compatible, mmap-based serialization. Features include atomic saves, parallel sharding, and zero-copy reads.  # Example: Simple Save and Load  ```rust use safecheckpoint::{Writer, Reader, DType}; # use std::fs; # use tempfile::tempdir;  # let dir = tempdir().unwrap(); # let path = dir.path().join("model.safetensors");  // Save a tensor let mut writer = Writer::new(); writer.add_tensor("w1", DType::F32, vec![2, 2], vec![0; 16]); writer.save(&path).unwrap();  // Load the tensor let reader = Reader::open(&path).unwrap(); let tensor = reader.get_tensor("w1").unwrap(); assert_eq!(tensor.data.len(), 16); ```

## Architecture

The crate is organized into the following public modules:

- `error`
- `reader`
- `shard`
- `tensor`
- `writer`

## Guarantees

- `#![forbid(unsafe_code)]` where applicable; see `src/lib.rs` for the exact lint preamble.
- All public types have doc comments.
- Error messages are actionable where applicable.

## Public API Summary

Key entry points are exported from `src/lib.rs` via `pub mod` and `pub use` re-exports.
Consult the module-level documentation in each source file for function signatures and usage examples.

## Error Handling

- `Error`
