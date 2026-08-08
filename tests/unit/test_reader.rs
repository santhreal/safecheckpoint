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
#[test]
fn test_reader_get_tensor_size_mismatch_preserves_tensor_name() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mismatch_name.safetensors");

    let header_json = br#"{"schema_version":1,"layer_1_weights":{"dtype":"F32","shape":[10],"data_offsets":[0,16]}}"#;
    let mut file_bytes = Vec::new();
    let header_len = header_json.len() as u64;
    file_bytes.extend_from_slice(&header_len.to_le_bytes());
    file_bytes.extend_from_slice(header_json);

    let mut hasher = blake3::Hasher::new();
    hasher.update(&header_len.to_le_bytes());
    hasher.update(header_json);
    let checksum = hasher.finalize();
    file_bytes.extend_from_slice(checksum.as_bytes());

    // 16 bytes of dummy tensor data
    file_bytes.extend_from_slice(&[0u8; 16]);
    std::fs::write(&path, &file_bytes).unwrap();

    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("layer_1_weights");
    match res {
        Err(Error::SizeMismatch { tensor_name, expected, actual }) => {
            assert_eq!(tensor_name, "layer_1_weights", "Error must retain exact tensor name");
            assert_eq!(expected, 40);
            assert_eq!(actual, 16);
        }
        other => panic!("expected SizeMismatch error with tensor name, got {:?}", other),
    }
}
