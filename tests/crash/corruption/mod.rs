//! Corruption adversarial suite for safecheckpoint.
//!
//! Every test here simulates a checkpoint file that was damaged after a
//! successful save: bit rot on disk, a hostile actor editing bytes, or a
//! crashed process leaving torn pages behind. The reader must detect each
//! corruption class with a specific error and must never panic or hand back
//! silently wrong tensor bytes.

use safecheckpoint::{DType, Error, Reader, Writer};
use tempfile::tempdir;

/// Save a small two-tensor checkpoint and return its path plus raw bytes.
fn saved_checkpoint(dir: &tempfile::TempDir) -> (std::path::PathBuf, Vec<u8>) {
    let path = dir.path().join("model.safetensors");
    let mut writer = Writer::new();
    writer
        .add_tensor("alpha", DType::F32, vec![2, 2], vec![0xAA; 16])
        .unwrap();
    writer
        .add_tensor("beta", DType::I64, vec![4], vec![0xBB; 32])
        .unwrap();
    writer.add_metadata("creator", "corruption-suite");
    writer.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    (path, bytes)
}

/// Overwrite `path` with `bytes`.
fn rewrite(path: &std::path::Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap();
}

/// Parse the on-disk layout boundaries: (header_json_range, checksum_range, data_offset).
fn layout(bytes: &[u8]) -> (std::ops::Range<usize>, std::ops::Range<usize>, usize) {
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let json = 8..8 + header_len;
    let checksum = 8 + header_len..8 + header_len + 32;
    let data_offset = 8 + header_len + 32;
    (json, checksum, data_offset)
}

/// Why: a single flipped bit anywhere in the file is the canonical disk
/// corruption. Because the header is blake3-protected and every tensor block
/// is CRC32-protected, EVERY byte position must be detectable: positions in
/// the length prefix, JSON, or stored checksum must fail `Reader::open`;
/// positions in the data block must fail the affected tensor's CRC check.
/// A flip that reads back as valid data would be a silent-integrity hole.
#[test]
fn test_every_single_byte_flip_is_detected() {
    let dir = tempdir().unwrap();
    let (path, original) = saved_checkpoint(&dir);

    for pos in 0..original.len() {
        let mut mutated = original.clone();
        mutated[pos] ^= 0x01;
        rewrite(&path, &mutated);

        let outcome = Reader::open(&path).map(|reader| {
            // Open succeeded, so the header survived; the flip must be in the
            // data block and at least one tensor read must catch it.
            let results: Vec<_> = ["alpha", "beta"]
                .iter()
                .map(|name| reader.get_tensor(name))
                .collect();
            assert!(
                results.iter().any(|r| matches!(r, Err(Error::Checksum { .. }))),
                "byte flip at data offset {pos} was not detected by any tensor CRC"
            );
        });

        // Neither open nor the reads above may panic; this assert fires only
        // if the closure above did not (i.e. detection succeeded or open
        // failed, both acceptable). The inner assert enforces detection.
        let _ = outcome;
    }
}

/// Why: the 8-byte length prefix is the first thing an attacker controls.
/// Forging it to claim the header is shorter than written must still fail,
/// because the checksum is computed over the length prefix itself.
#[test]
fn test_corrupted_header_length_prefix_rejected() {
    let dir = tempdir().unwrap();
    let (path, original) = saved_checkpoint(&dir);

    let mut mutated = original.clone();
    mutated[0..8].copy_from_slice(&16u64.to_le_bytes());
    rewrite(&path, &mutated);

    let res = Reader::open(&path);
    assert!(
        matches!(
            res,
            Err(Error::HeaderChecksum { .. }) | Err(Error::InvalidFormat { .. })
        ),
        "forged header length must fail checksum or format validation, got {:?}", res.err()
    );
}

/// Why: a one-byte edit inside the header JSON (e.g. changing a shape digit)
/// must be caught by the blake3 header checksum before any JSON parsing
/// happens, so tampered metadata can never reach shape validation.
#[test]
fn test_corrupted_header_json_caught_by_checksum() {
    let dir = tempdir().unwrap();
    let (path, original) = saved_checkpoint(&dir);
    let (json, _, _) = layout(&original);

    let mut mutated = original.clone();
    mutated[json.start] = if mutated[json.start] == b'{' { b'}' } else { b'{' };
    rewrite(&path, &mutated);

    let res = Reader::open(&path);
    assert!(
        matches!(res, Err(Error::HeaderChecksum { .. })),
        "header JSON corruption must surface as HeaderChecksum, got {:?}", res.err()
    );
}

/// Why: if the stored checksum bytes themselves rot, the reader must report
/// HeaderChecksum rather than panic on a slice conversion or accept the file.
#[test]
fn test_corrupted_stored_checksum_rejected() {
    let dir = tempdir().unwrap();
    let (path, original) = saved_checkpoint(&dir);
    let (_, checksum, _) = layout(&original);

    let mut mutated = original.clone();
    for i in checksum.clone() {
        mutated[i] = 0xFF;
    }
    rewrite(&path, &mutated);

    let res = Reader::open(&path);
    assert!(
        matches!(res, Err(Error::HeaderChecksum { .. })),
        "rotted stored checksum must surface as HeaderChecksum, got {:?}", res.err()
    );
}

/// Why: a forged header that passes the checksum gate but lies about shapes
/// (shape says 4 elements, offsets say 16 bytes... but dtype says F32 while
/// data claims 8 elements) must be rejected by metadata validation, proving
/// the deep validation layer works even when the checksum layer is bypassed
/// by an attacker who recomputes it.
#[test]
fn test_shape_lie_with_recomputed_checksum_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("lie.safetensors");

    // Header claims shape [4] (16 bytes) but offsets span only 4 bytes.
    // safetensors layout: tensor entries are flattened at the top level.
    let header = serde_json::json!({
        "schema_version": 1,
        "w": {
            "dtype": "F32",
            "shape": [4],
            "data_offsets": [0, 4]
        }
    });
    let header_json = serde_json::to_vec(&header).unwrap();
    crate::adversarial::write_checkpoint_with_checksum(&path, &header_json, &[0; 4]);

    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("w");
    assert!(
        matches!(res, Err(Error::SizeMismatch { .. })),
        "shape/offset lie must surface as SizeMismatch, got {:?}", res.err()
    );
}

/// Why: a checkpoint written by a newer, incompatible schema must be refused
/// with UnsupportedVersion instead of being parsed under wrong assumptions.
/// The checksum is recomputed so version gating itself is what is exercised.
#[test]
fn test_future_schema_version_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("future.safetensors");

    let header = serde_json::json!({
        "schema_version": 9999
    });
    let header_json = serde_json::to_vec(&header).unwrap();
    crate::adversarial::write_checkpoint_with_checksum(&path, &header_json, &[]);

    let res = Reader::open(&path);
    assert!(
        matches!(res, Err(Error::UnsupportedVersion { .. })),
        "future schema version must surface as UnsupportedVersion, got {:?}", res.err()
    );
}

/// Why: inverted offsets (start > end) would underflow the length subtraction
/// `data_offsets[1] - data_offsets[0]`; validation must reject the tensor
/// before any arithmetic on attacker-controlled offsets.
#[test]
fn test_inverted_offsets_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("inverted.safetensors");

    let header = serde_json::json!({
        "schema_version": 1,
        "w": {
            "dtype": "U8",
            "shape": [4],
            "data_offsets": [8, 4]
        }
    });
    let header_json = serde_json::to_vec(&header).unwrap();
    crate::adversarial::write_checkpoint_with_checksum(&path, &header_json, &[0; 12]);

    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("w");
    assert!(
        matches!(res, Err(Error::InvalidFormat { .. })),
        "inverted offsets must surface as InvalidFormat, got {:?}", res.err()
    );
}

/// Why: corrupting one tensor's data must not poison reads of a sibling
/// tensor. Isolation matters because partial corruption of a sharded model
/// should still let callers salvage intact shards.
#[test]
fn test_corruption_isolated_to_affected_tensor() {
    let dir = tempdir().unwrap();
    let (path, original) = saved_checkpoint(&dir);
    let (_, _, data_offset) = layout(&original);

    // "alpha" sorts first, so its 16 bytes start at data_offset.
    let mut mutated = original.clone();
    mutated[data_offset] ^= 0xFF;
    rewrite(&path, &mutated);

    let reader = Reader::open(&path).unwrap();
    assert!(
        matches!(reader.get_tensor("alpha"), Err(Error::Checksum { .. })),
        "corrupted tensor must fail its CRC"
    );
    let beta = reader.get_tensor("beta").unwrap();
    assert!(
        beta.data.iter().all(|&b| b == 0xBB),
        "sibling tensor bytes must survive corruption of a neighbor"
    );
}
