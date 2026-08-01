//! Gap test suite for safecheckpoint.
//!
//! Probes edge cases and boundaries for safecheckpoint Reader and Writer primitives.

use safecheckpoint::{Reader, Writer};
use tempfile::tempdir;

/// Probes reading a non-existent checkpoint path and asserts path-not-found error.
#[test]
fn test_gap_non_existent_checkpoint_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("missing.safetensors");
    let result = Reader::open(&path);
    assert!(result.is_err(), "opening missing checkpoint path must return Error");
}

/// Probes saving an empty writer with no tensors and verifies header creation.
#[test]
fn test_gap_empty_checkpoint_write_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.safetensors");
    let writer = Writer::new();
    writer.save(&path).expect("saving empty writer should succeed");

    let reader = Reader::open(&path).expect("opening empty checkpoint should succeed");
    assert!(reader.tensor_names().is_empty(), "empty checkpoint must contain no tensor names");
}
