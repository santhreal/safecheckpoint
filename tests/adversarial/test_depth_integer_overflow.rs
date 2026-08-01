use safecheckpoint::{DType, Error, Reader, Writer};
use std::fs::{self, OpenOptions};
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_integer_overflow_header_length() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("overflow_header.safetensors");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .unwrap();

    // Use a value that would truncate to 0 when cast to u32, or similar bounds
    // u32::MAX + 1 = 4294967296
    let huge_len: u64 = (u32::MAX as u64) + 1;
    file.write_all(&huge_len.to_le_bytes()).unwrap();
    file.write_all(&[0; 8]).unwrap();

    let res = Reader::open(&path);
    assert!(
        res.is_err(),
        "Reader must reject unreasonably large header sizes to prevent allocation overflow"
    );

    match res.err().unwrap() {
        Error::InvalidFormat { message, .. } => {
            assert!(
                message.contains("100MB limit") || message.contains("too large"),
                "Expected size limit error, got: {}", message
            );
        }
        _ => panic!("Expected InvalidFormat error on large header"),
    }
}

#[test]
fn test_integer_overflow_offsets() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("overflow_offsets.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![0; 4]).unwrap();
    writer.save(&path).unwrap();

    let content = fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(content[0..8].try_into().unwrap()) as usize;
    let header_json = &content[8..8 + header_len];
    let mut header: serde_json::Value = serde_json::from_slice(header_json).unwrap();

    // Set offsets that cause u64 to overflow when added or compared
    let max_usize = usize::MAX;
    header["w1"]["data_offsets"] = serde_json::json!([max_usize, max_usize]);
    
    let new_header_json = serde_json::to_vec(&header).unwrap();
    // Rewrite with a valid header checksum so Reader::open reaches the offset
    // bounds check instead of failing on the missing 32-byte checksum.
    super::write_checkpoint_with_checksum(&path, &new_header_json, &[0; 4]);

    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("w1");
    assert!(
        matches!(res, Err(Error::OffsetOverflow { .. }) | Err(Error::SizeMismatch { .. })),
        "usize::MAX data_offsets must be caught as OffsetOverflow/SizeMismatch, got {res:?}"
    );
}

#[test]
fn test_integer_overflow_shape_size() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("overflow_shape.safetensors");

    let mut writer = Writer::new();
    // Shape limits that exceed u32 capabilities or normal memory bounds.
    writer.add_tensor("w1", DType::F32, vec![1024, 1024, 1024, 1024], vec![0; 4]).unwrap();
    let res = writer.save(&path);

    if let Err(_e) = res {
        // Engine safely rejected it early.
        return;
    }
    
    let reader = Reader::open(&path);
    match reader {
        Ok(r) => {
            let t = r.get_tensor("w1");
            assert!(t.is_err(), "Engine failed to catch shape size mismatch");
        }
        Err(_e) => {
            // Engine caught the mismatch during read/open, which is also valid defense
        }
    }
}
