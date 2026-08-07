//! Truncation / partial-write adversarial suite for safecheckpoint.
//!
//! A checkpoint file that was cut short by a crashed writer, a full disk, or
//! an interrupted copy is the most common real-world corruption. The reader
//! contract under truncation:
//!
//! 1. `Reader::open` must NEVER panic at any truncation point, including
//!    lengths 0..8 (no length prefix), a partial header, and a partial
//!    checksum.
//! 2. If `open` succeeds the header is provably intact (length prefix, JSON,
//!    and blake3 checksum all present and verified), so the only remaining
//!    damage is missing tensor data, and `get_tensor` must then fail with a
//!    bounds or checksum error rather than returning short slices.

use safecheckpoint::{DType, Reader, Writer};
use tempfile::tempdir;

/// Save a small checkpoint and return its path and bytes.
fn saved_checkpoint(dir: &tempfile::TempDir) -> (std::path::PathBuf, Vec<u8>) {
    let path = dir.path().join("model.safetensors");
    let mut writer = Writer::new();
    writer
        .add_tensor("w", DType::F32, vec![2, 2], vec![7; 16])
        .unwrap();
    writer.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    (path, bytes)
}

/// Why: truncation can happen at ANY byte offset, and every offset must obey
/// the reader contract. Exhaustively truncating at every prefix length proves
/// there is no special length that panics (e.g. an off-by-one in the
/// `8 + header_len + 32` guard) and no length that yields silently short
/// tensor data.
#[test]
fn test_every_truncation_point_never_panics_and_never_lies() {
    let dir = tempdir().unwrap();
    let (path, original) = saved_checkpoint(&dir);
    let header_len = u64::from_le_bytes(original[0..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len + 32; // first byte of tensor data

    for cut in 0..original.len() {
        std::fs::write(&path, &original[..cut]).unwrap();

        match Reader::open(&path) {
            Err(_) => {
                // Expected for every cut before the file is complete: the
                // length prefix, header JSON, checksum, or data is missing.
            }
            Ok(reader) => {
                // Open succeeded, so cut >= header_end: header is intact.
                assert!(
                    cut >= header_end,
                    "open succeeded at cut {cut} (< header end {header_end}); \
                     a truncated header must never validate"
                );
                // Data is necessarily truncated (cut < original.len()), so the
                // tensor read must fail loudly, not return a short slice.
                let res = reader.get_tensor("w");
                assert!(
                    res.is_err(),
                    "cut {cut}: truncated tensor data must error, got Ok({:?})",
                    res.map(|t| t.data.len())
                );
            }
        }
    }

    // Boundary: the untruncated file must still open and read cleanly,
    // proving the loop above tested truncation and not a broken fixture.
    std::fs::write(&path, &original).unwrap();
    let reader = Reader::open(&path).unwrap();
    assert_eq!(reader.get_tensor("w").unwrap().data.len(), 16);
}

/// Why: files shorter than 8 bytes cannot even contain the length prefix.
/// This is the boundary where naive `mmap[0..8]` slicing would panic; the
/// reader must return InvalidFormat instead. Each length 0..8 is a distinct
/// boundary case (including the empty file, which mmap may refuse).
#[test]
fn test_tiny_files_below_length_prefix() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("tiny.safetensors");

    for len in 0..8usize {
        std::fs::write(&path, vec![0x55; len]).unwrap();
        let res = Reader::open(&path);
        assert!(
            res.is_err(),
            "a {len}-byte file must be rejected (no length prefix possible)"
        );
    }
}

/// Why: the exact boundary where the header is complete but the 32-byte
/// checksum is partially or wholly missing is the sharpest off-by-one risk
/// in the `mmap.len() < 8 + header_len + 32` guard. Test each checksum
/// prefix length explicitly.
#[test]
fn test_partial_checksum_boundary() {
    let dir = tempdir().unwrap();
    let (path, original) = saved_checkpoint(&dir);
    let header_len = u64::from_le_bytes(original[0..8].try_into().unwrap()) as usize;
    let checksum_start = 8 + header_len;

    for present in [0usize, 1, 16, 31] {
        let cut = checksum_start + present;
        std::fs::write(&path, &original[..cut]).unwrap();
        let res = Reader::open(&path);
        assert!(
            matches!(res, Err(safecheckpoint::Error::InvalidFormat { .. })),
            "{present}/32 checksum bytes present must fail InvalidFormat, got {:?}", res.err()
        );
    }
}

/// Why: a file that ends exactly after the checksum (zero tensor bytes) is
/// structurally openable, but every tensor read must then fail the
/// file-size bounds check. This separates "header valid" from "data valid".
#[test]
fn test_truncated_exactly_after_checksum() {
    let dir = tempdir().unwrap();
    let (path, original) = saved_checkpoint(&dir);
    let header_len = u64::from_le_bytes(original[0..8].try_into().unwrap()) as usize;
    let cut = 8 + header_len + 32;

    std::fs::write(&path, &original[..cut]).unwrap();
    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("w");
    assert!(
        matches!(res, Err(safecheckpoint::Error::SizeMismatch { .. })),
        "zero data bytes must surface as SizeMismatch, got {:?}", res.err()
    );
}

/// Why: losing exactly the LAST byte of tensor data is the minimal data
/// truncation. The CRC covers the truncated slice bounds, so the bounds
/// check or the CRC must catch it; a silent 15-of-16-byte read would be a
/// correctness hole downstream.
#[test]
fn test_last_data_byte_missing_detected() {
    let dir = tempdir().unwrap();
    let (path, original) = saved_checkpoint(&dir);
    let cut = original.len() - 1;

    std::fs::write(&path, &original[..cut]).unwrap();
    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("w");
    assert!(
        res.is_err(),
        "missing final data byte must be detected, got Ok({:?})",
        res.map(|t| t.data.len())
    );
}

/// Why: concurrent readers must not observe a partial file from an
/// interrupted overwrite: the writer's temp-file + rename dance guarantees
/// the target path is only ever complete or absent. This test exercises the
/// failure half of that contract: saving into a directory that does not
/// exist must fail cleanly and must NOT leave a partial target file behind.
#[test]
fn test_failed_save_leaves_no_partial_target() {
    let dir = tempdir().unwrap();
    let missing_dir = dir.path().join("does-not-exist");
    let path = missing_dir.join("model.safetensors");

    let mut writer = Writer::new();
    writer
        .add_tensor("w", DType::F32, vec![1], vec![0; 4])
        .unwrap();
    let res = writer.save(&path);
    assert!(res.is_err(), "save into a missing directory must fail");
    assert!(
        !path.exists(),
        "failed save must not leave a partial target file"
    );
}
