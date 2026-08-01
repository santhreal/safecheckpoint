use safecheckpoint::{DType, Error, Reader, Writer};
use std::fs::{self, OpenOptions};
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_negative_length_simulation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("negative_len.safetensors");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .unwrap();

    // Write i64::MIN as header length
    file.write_all(&(i64::MIN as u64).to_le_bytes()).unwrap();
    file.write_all(&[0; 8]).unwrap();

    let res = Reader::open(&path);
    assert!(
        res.is_err(),
        "Negative length (interpreted as huge u64) should be blocked by 100MB limit"
    );

    let err = res.err().unwrap();
    match err {
        Error::InvalidFormat { message, .. } => {
            assert!(message.contains("100MB limit") || message.contains("too large"));
        }
        _ => panic!("Expected InvalidFormat error"),
    }
}

#[test]
fn test_invalid_json_header() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("invalid_json.safetensors");
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .unwrap();

    // Valid header checksum over invalid JSON, so Reader::open passes the
    // checksum gate and fails at JSON parsing (the behavior under test).
    drop(file);
    super::write_checkpoint_with_checksum(&path, b"{invalid_json_here}", &[]);

    let res = Reader::open(&path);
    assert!(res.is_err());
    assert!(
        matches!(res.as_ref().err().unwrap(), Error::Json(_)),
        "invalid JSON header must surface as Error::Json, got {:?}",
        res.err().unwrap()
    );
}

#[test]
fn test_data_offset_overflow() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("overflow.safetensors");
    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![0; 4]).unwrap();
    writer.save(&path).unwrap();

    // Manually manipulate the header json to have offsets close to usize::MAX
    let content = fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(content[0..8].try_into().unwrap()) as usize;
    let header_json = &content[8..8 + header_len];
    let mut header: serde_json::Value = serde_json::from_slice(header_json).unwrap();

    header["w1"]["data_offsets"] = serde_json::json!([usize::MAX - 10, usize::MAX - 5]);
    let new_header_json = serde_json::to_vec(&header).unwrap();

    // Rewrite with a valid header checksum so Reader::open reaches the offset
    // bounds check instead of failing on the missing 32-byte checksum.
    super::write_checkpoint_with_checksum(&path, &new_header_json, &[0; 4]);

    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("w1");
    assert!(
        matches!(res, Err(Error::OffsetOverflow { .. }) | Err(Error::SizeMismatch { .. })),
        "near-usize::MAX offsets should be caught by checked_add/bounds checks, got {res:?}"
    );
}
