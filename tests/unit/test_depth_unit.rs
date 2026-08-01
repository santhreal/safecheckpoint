use safecheckpoint::{DType, Error, Reader, TensorMetadata, Writer};
use std::collections::HashMap;
use std::fs;
use tempfile::tempdir;

#[test]
fn test_dtype_size_in_bytes() -> std::result::Result<(), Box<dyn std::error::Error>> {
    assert_eq!(DType::F64.size_in_bytes(), 8);
    assert_eq!(DType::F32.size_in_bytes(), 4);
    assert_eq!(DType::F16.size_in_bytes(), 2);
    assert_eq!(DType::BF16.size_in_bytes(), 2);
    assert_eq!(DType::I64.size_in_bytes(), 8);
    assert_eq!(DType::I32.size_in_bytes(), 4);
    assert_eq!(DType::I16.size_in_bytes(), 2);
    assert_eq!(DType::I8.size_in_bytes(), 1);
    assert_eq!(DType::U64.size_in_bytes(), 8);
    assert_eq!(DType::U32.size_in_bytes(), 4);
    assert_eq!(DType::U16.size_in_bytes(), 2);
    assert_eq!(DType::U8.size_in_bytes(), 1);
    assert_eq!(DType::BOOL.size_in_bytes(), 1);
    Ok(())
}

#[test]
fn test_tensor_metadata_validate() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Valid cases
    let valid_meta = TensorMetadata {
        dtype: DType::F32,
        shape: vec![1, 2],
        data_offsets: [0, 8],
        checksum: None,
    };
    valid_meta.validate()?;

    let empty_meta = TensorMetadata {
        dtype: DType::F32,
        shape: vec![0],
        data_offsets: [0, 0],
        checksum: None,
    };
    empty_meta.validate()?;

    // Invalid cases
    let invalid_meta = TensorMetadata {
        dtype: DType::F32,
        shape: vec![1, 2],
        data_offsets: [8, 0], // start > end
        checksum: None,
    };
    let res = invalid_meta.validate();
    assert!(res.is_err());
    match res.unwrap_err() {
        Error::InvalidFormat { offset, .. } => assert_eq!(offset, 8),
        _ => panic!("Expected InvalidFormat error"),
    }

    Ok(())
}

#[test]
fn test_writer_new_empty_save_and_load() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("empty.safetensors");

    let writer = Writer::new();
    writer.save(&path)?;

    assert!(path.exists());

    let reader = Reader::open(&path)?;
    assert_eq!(reader.tensor_names().len(), 0);
    assert_eq!(reader.metadata().len(), 0);

    Ok(())
}

#[test]
fn test_writer_add_tensor_mix_and_order() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("mixed.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("z_tensor", DType::F64, vec![1], vec![1; 8])?;
    writer.add_tensor("a_tensor", DType::U8, vec![2], vec![2, 3])?;
    writer.save(&path)?;

    let reader = Reader::open(&path)?;
    let mut names = reader.tensor_names();
    names.sort();
    assert_eq!(names, vec!["a_tensor", "z_tensor"]);

    let t_a = reader.get_tensor("a_tensor")?;
    assert_eq!(t_a.data, &[2, 3]);
    assert_eq!(t_a.metadata.dtype, DType::U8);

    let t_z = reader.get_tensor("z_tensor")?;
    assert_eq!(t_z.data, &[1; 8]);
    assert_eq!(t_z.metadata.dtype, DType::F64);

    Ok(())
}

#[test]
fn test_writer_add_tensors_and_metadata() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("bulk.safetensors");

    let mut writer = Writer::new();
    let mut bulk = HashMap::new();
    bulk.insert("bulk1".to_string(), (DType::BOOL, vec![1], vec![0]));
    bulk.insert("bulk2".to_string(), (DType::I32, vec![1], vec![0, 0, 0, 0]));
    writer.add_tensors(bulk);

    writer.add_metadata("key1", "val1");
    writer.add_metadata("", ""); // Empty key and value

    writer.save(&path)?;

    let reader = Reader::open(&path)?;
    assert_eq!(reader.tensor_names().len(), 2);

    let meta = reader.metadata();
    assert_eq!(meta.get("key1").map(|s| s.as_str()), Some("val1"));
    assert_eq!(meta.get("").map(|s| s.as_str()), Some(""));

    Ok(())
}

#[test]
fn test_reader_edge_cases() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("missing.safetensors");

    // Read non-existent file
    let res = Reader::open(&path);
    assert!(res.is_err());

    // Read directory
    let res = Reader::open(dir.path());
    assert!(res.is_err());

    let file_path = dir.path().join("valid.safetensors");
    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![0; 4])?;
    writer.save(&file_path)?;

    let reader = Reader::open(&file_path)?;
    // Read non-existent tensor
    let res = reader.get_tensor("missing_tensor");
    assert!(res.is_err());
    match res.unwrap_err() {
        Error::TensorNotFound(name) => assert_eq!(name, "missing_tensor"),
        _ => panic!("Expected TensorNotFound error"),
    }

    Ok(())
}

#[test]
fn test_writer_save_sharded_basic() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![1; 4])?;
    writer.add_tensor("w2", DType::F32, vec![1], vec![2; 4])?;
    writer.add_tensor("w3", DType::F32, vec![1], vec![3; 4])?;
    writer.add_tensor("w4", DType::F32, vec![1], vec![4; 4])?;

    writer.save_sharded(dir.path(), "model", 2)?;

    // Verify files
    let index_path = dir.path().join("model.safetensors.index.json");
    assert!(index_path.exists());
    let shard1_path = dir.path().join("model-00001-of-00002.safetensors");
    assert!(shard1_path.exists());
    let shard2_path = dir.path().join("model-00002-of-00002.safetensors");
    assert!(shard2_path.exists());

    // Load and check index
    let index_str = fs::read_to_string(&index_path)?;
    let index: safecheckpoint::shard::ShardIndex = serde_json::from_str(&index_str)?;
    assert_eq!(index.weight_map.len(), 4);

    Ok(())
}
