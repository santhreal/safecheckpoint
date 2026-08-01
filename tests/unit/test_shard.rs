use safecheckpoint::shard::ShardIndex;
use safecheckpoint::{DType, Reader, Writer};
use std::fs;
use tempfile::tempdir;

#[test]
fn test_shard_basic() {
    let dir = tempdir().unwrap();
    let mut writer = Writer::new();

    // Add 4 tensors
    writer.add_tensor("t1", DType::F32, vec![1], vec![1, 0, 0, 0]).unwrap();
    writer.add_tensor("t2", DType::F32, vec![1], vec![2, 0, 0, 0]).unwrap();
    writer.add_tensor("t3", DType::F32, vec![1], vec![3, 0, 0, 0]).unwrap();
    writer.add_tensor("t4", DType::F32, vec![1], vec![4, 0, 0, 0]).unwrap();
    writer.add_metadata("model", "sharded");

    writer.save_sharded(dir.path(), "model", 2).unwrap();

    let index_path = dir.path().join("model.safetensors.index.json");
    assert!(index_path.exists());

    let index_data = fs::read_to_string(&index_path).unwrap();
    let index: ShardIndex = serde_json::from_str(&index_data).unwrap();

    assert_eq!(
        index.metadata.as_ref().unwrap().get("model").unwrap(),
        "sharded"
    );
    assert_eq!(index.weight_map.len(), 4);

    let s1 = dir.path().join("model-00001-of-00002.safetensors");
    let s2 = dir.path().join("model-00002-of-00002.safetensors");
    assert!(s1.exists());
    assert!(s2.exists());

    let r1 = Reader::open(&s1).unwrap();
    let r2 = Reader::open(&s2).unwrap();

    assert_eq!(r1.tensor_names().len(), 2);
    assert_eq!(r2.tensor_names().len(), 2);
}

#[test]
fn test_save_sharded_zero_shards_errors_not_panics() {
    let dir = tempdir().unwrap();
    let mut writer = Writer::new();
    writer
        .add_tensor("t1", DType::F32, vec![1], vec![1, 0, 0, 0])
        .unwrap();

    // num_shards == 0 previously panicked via div_ceil(0) (divide by zero).
    // It must now fail closed with an actionable InvalidArgument error.
    let err = writer
        .save_sharded(dir.path(), "model", 0)
        .expect_err("num_shards=0 must be rejected, not panic");
    match err {
        safecheckpoint::Error::InvalidArgument { parameter, .. } => {
            assert_eq!(parameter, "num_shards");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }

    // No shard/index files should have been created.
    assert!(!dir
        .path()
        .join("model.safetensors.index.json")
        .exists());
}
