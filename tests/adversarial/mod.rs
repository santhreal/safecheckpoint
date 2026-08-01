mod test_depth_adversarial;
mod test_depth_adversarial_boundaries;
mod test_depth_fault_injection;
mod test_depth_gap;
mod test_depth_integer_overflow;
mod test_malformed;

/// Write a checkpoint file with an arbitrary (possibly hostile) header and data
/// body, computing the real 32-byte blake3 header checksum the reader requires.
///
/// The on-disk layout is: 8-byte LE header length, header JSON, 32-byte
/// blake3(header_len_bytes ‖ header_json), then the data body. Crafted-file
/// adversarial tests use this so `Reader::open` passes the checksum gate and the
/// test can exercise the deeper parse/bounds error it actually targets, rather
/// than tripping the missing-checksum guard first.
#[allow(dead_code)]
pub fn write_checkpoint_with_checksum(
    path: &std::path::Path,
    header_json: &[u8],
    data: &[u8],
) {
    use std::io::Write;
    let header_len = header_json.len() as u64;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&header_len.to_le_bytes());
    hasher.update(header_json);
    let checksum = hasher.finalize();

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(header_json).unwrap();
    f.write_all(checksum.as_bytes()).unwrap();
    f.write_all(data).unwrap();
}
