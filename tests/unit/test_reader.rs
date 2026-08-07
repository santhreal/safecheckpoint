use safecheckpoint::{DType, Error, Reader, Writer};
use tempfile::tempdir;

#[test]
fn test_reader_empty_checkpoint() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.safetensors");
    let writer = Writer::new();
    writer.save(&path).unwrap();

    let reader = Reader::open(&path).unwrap();
    assert!(reader.tensor_names().is_empty());
    assert!(reader.metadata().is_empty());
}

#[test]
fn test_reader_invalid_tensor_name() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("basic.safetensors");
    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![0, 0, 0, 0]).unwrap();
    writer.save(&path).unwrap();

    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("w2");
    assert!(res.is_err());
    match res.unwrap_err() {
        Error::TensorNotFound(name) => assert_eq!(name, "w2"),
        other => panic!("Expected TensorNotFound error, got: {:?}", other),
    }
}

#[test]
fn test_reader_checksum_mismatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupt.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![0, 0, 0, 0]).unwrap();
    writer.save(&path).unwrap();

    // Corrupt the data byte after the header
    let mut content = std::fs::read(&path).unwrap();
    // The data starts after header_len (8) + header_json + checksum (32)
    let header_len = u64::from_le_bytes(content[0..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_len + 32;
    content[data_start] = 0xFF;
    std::fs::write(&path, &content).unwrap();

    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("w1");
    assert!(
        matches!(
            res,
            Err(Error::Checksum {
                ref tensor_name,
                ..
            }) if tensor_name == "w1"
        ),
        "Expected Checksum error, got: {:?}",
        res
    );
}
#[test]
fn test_reader_open_zero_byte_file_returns_invalid_format() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("zero_byte.safetensors");
    std::fs::write(&path, &[]).unwrap();

    match Reader::open(&path) {
        Err(Error::InvalidFormat { offset: 0, .. }) => {}
        Err(err) => panic!("expected InvalidFormat at offset 0, got Err({err:?})"),
        Ok(_) => panic!("expected 0-byte file open to fail, got Ok"),
    }
}
