use safecheckpoint::{DType, Error, Reader, Writer};
use std::fs;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_path_traversal_in_tensor_name() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("traversal.safetensors");

    let mut writer = Writer::new();
    // Path traversal in tensor name (should just be treated as a weird string)
    writer.add_tensor("../w1", DType::F32, vec![1], vec![0; 4])?;
    writer.save(&path)?;

    let reader = Reader::open(&path)?;
    let tensor = reader.get_tensor("../w1")?;
    assert_eq!(tensor.metadata.dtype, DType::F32);
    Ok(())
}

#[test]
fn test_path_traversal_in_save() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir
        .path()
        .join("legit_dir")
        .join("../malicious.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![0; 4])?;

    let res = writer.save(&path);
    assert!(res.is_err());
    match res.unwrap_err() {
        Error::PathTraversal { reason, .. } => assert!(!reason.is_empty()),
        other => panic!("Expected PathTraversal error, got {other:?}"),
    }

    // Also test save_sharded
    let res = writer.save_sharded(dir.path().join("../legit"), "model", 2);
    assert!(res.is_err());
    match res.unwrap_err() {
        Error::PathTraversal { reason, .. } => assert!(reason.contains("'..'")),
        other => panic!("Expected PathTraversal error, got {other:?}"),
    }

    let res = writer.save_sharded(dir.path(), "../model", 2);
    assert!(res.is_err());
    match res.unwrap_err() {
        Error::PathTraversal { reason, .. } => assert!(!reason.is_empty()),
        other => panic!("Expected PathTraversal error, got {other:?}"),
    }

    Ok(())
}

#[test]
fn test_null_bytes_and_unicode() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("weird_strings.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("w1\0null", DType::F32, vec![1], vec![0; 4])?;
    writer.add_tensor("w2🌟", DType::F32, vec![1], vec![0; 4])?;
    writer.add_metadata("key\0", "val\0");
    writer.add_metadata("key🌟", "val🌟");

    writer.save(&path)?;

    let reader = Reader::open(&path)?;
    assert!(reader.get_tensor("w1\0null").is_ok());
    assert!(reader.get_tensor("w2🌟").is_ok());

    let meta = reader.metadata();
    assert_eq!(meta.get("key\0").map(|s| s.as_str()), Some("val\0"));
    assert_eq!(meta.get("key🌟").map(|s| s.as_str()), Some("val🌟"));

    Ok(())
}

#[test]
fn test_empty_and_huge_inputs() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("sizes.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("empty", DType::U8, vec![0], vec![])?;
    writer.add_tensor("0xff", DType::U8, vec![1024], vec![0xFF; 1024])?;

    // 2MB huge tensor to test memory mapping boundaries
    let huge_data = vec![0x42; 2 * 1024 * 1024];
    writer.add_tensor("huge", DType::U8, vec![2 * 1024 * 1024], huge_data.clone())?;

    writer.save(&path)?;

    let reader = Reader::open(&path)?;
    let t_empty = reader.get_tensor("empty")?;
    assert_eq!(t_empty.data.len(), 0);

    let t_ff = reader.get_tensor("0xff")?;
    assert_eq!(t_ff.data, vec![0xFF; 1024].as_slice());

    let t_huge = reader.get_tensor("huge")?;
    assert_eq!(t_huge.data.len(), 2 * 1024 * 1024);
    assert_eq!(t_huge.data[0], 0x42);

    Ok(())
}

#[test]
fn test_truncated_file() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("truncated.safetensors");

    let mut writer = Writer::new();
    // 256 F32 elements = 1024 bytes (shape must match data length).
    writer.add_tensor("w1", DType::F32, vec![256], vec![0; 1024])?;
    writer.save(&path)?;

    // Truncate the file slightly
    let metadata = fs::metadata(&path)?;
    let file = fs::OpenOptions::new().write(true).open(&path)?;
    file.set_len(metadata.len() - 500)?;

    let reader = Reader::open(&path)?;
    // The header might be intact, but getting the tensor should fail due to size mismatch
    let res = reader.get_tensor("w1");
    assert!(res.is_err());
    match res.unwrap_err() {
        Error::SizeMismatch { .. } => {}
        _ => panic!("Expected SizeMismatch error"),
    }

    Ok(())
}

#[test]
fn test_header_dos_limit() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("dos.safetensors");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)?;

    // Write 101MB as header length
    let huge_len: u64 = 101 * 1024 * 1024;
    file.write_all(&huge_len.to_le_bytes())?;
    file.write_all(&[0; 8])?;

    let res = Reader::open(&path);
    assert!(res.is_err());
    match res.err().unwrap() {
        Error::InvalidFormat { message, .. } => {
            assert!(message.contains("100MB limit"));
        }
        _ => panic!("Expected InvalidFormat error"),
    }

    Ok(())
}
