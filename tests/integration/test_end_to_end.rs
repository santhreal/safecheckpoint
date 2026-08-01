use safecheckpoint::{DType, Reader, Writer};
use tempfile::tempdir;

#[test]
fn test_integration_full_workflow() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("full.safetensors");

    // 1. Write phase
    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![2, 2], vec![0; 16]).unwrap();
    writer.add_tensor("w2", DType::I32, vec![1, 1], vec![1; 4]).unwrap();
    writer.add_metadata("type", "test_integration");
    writer.save(&path).unwrap();

    // 2. Verification phase
    let reader = Reader::open(&path).unwrap();
    assert_eq!(reader.metadata().get("type").unwrap(), "test_integration");

    let t1 = reader.get_tensor("w1").unwrap();
    assert_eq!(t1.metadata.dtype, DType::F32);
    assert_eq!(t1.data.len(), 16);

    let t2 = reader.get_tensor("w2").unwrap();
    assert_eq!(t2.metadata.dtype, DType::I32);
    assert_eq!(t2.data.len(), 4);
    assert_eq!(t2.data[0], 1);
}
