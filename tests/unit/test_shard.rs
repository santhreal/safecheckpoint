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
#[test]
fn test_save_sharded_rejects_symlink_directory() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let dir = tempdir().unwrap();
        let target_dir = dir.path().join("real_shard_dir");
        fs::create_dir(&target_dir).unwrap();

        let link_dir = dir.path().join("link_shard_dir");
        symlink(&target_dir, &link_dir).unwrap();

        let mut writer = Writer::new();
        writer
            .add_tensor("t1", DType::F32, vec![1], vec![1, 0, 0, 0])
            .unwrap();

        let err = writer
            .save_sharded(&link_dir, "model", 1)
            .expect_err("save_sharded into symlink directory must fail closed");
        assert!(
            matches!(err, safecheckpoint::Error::PathTraversal { .. }),
            "expected PathTraversal error, got {err:?}"
        );
    }
}

#[test]
fn test_save_sharded_rejects_symlink_lock_file() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let dir = tempdir().unwrap();
        let victim = dir.path().join("victim.txt");
        fs::write(&victim, b"secret data").unwrap();

        let lock_path = dir.path().join(".safecheckpoint.lock");
        symlink(&victim, &lock_path).unwrap();

        let mut writer = Writer::new();
        writer
            .add_tensor("t1", DType::F32, vec![1], vec![1, 0, 0, 0])
            .unwrap();

        let err = writer
            .save_sharded(dir.path(), "model", 1)
            .expect_err("save_sharded with symlinked lockfile must fail closed");
        assert!(
            matches!(err, safecheckpoint::Error::PathTraversal { .. }),
            "expected PathTraversal error, got {err:?}"
        );
        assert_eq!(
            fs::read_to_string(&victim).unwrap(),
            "secret data",
            "symlink target victim file must not be overwritten or truncated"
        );
    }
}

#[test]
fn test_is_shard_valid_invalidated_by_metadata_or_shape_mismatch() {
    let dir = tempdir().unwrap();
    let mut writer1 = Writer::new();
    writer1
        .add_tensor("t1", DType::F32, vec![1], vec![1, 0, 0, 0])
        .unwrap();
    writer1.add_metadata("version", "1");
    writer1.save_sharded(dir.path(), "model", 1).unwrap();

    // Now try to save_sharded with different metadata or tensor shape under same prefix
    let mut writer2 = Writer::new();
    writer2
        .add_tensor("t1", DType::F32, vec![1], vec![2, 0, 0, 0])
        .unwrap();
    writer2.add_metadata("version", "2");

    // Saving writer2 should detect metadata difference, overwrite existing shard rather than skipping
    writer2.save_sharded(dir.path(), "model", 1).unwrap();

    let shard_path = dir.path().join("model-00001-of-00001.safetensors");
    let reader = Reader::open(&shard_path).unwrap();
    assert_eq!(
        reader.metadata().get("version").unwrap(),
        "2",
        "shard must have been rewritten with updated metadata"
    );
    let tensor = reader.get_tensor("t1").unwrap();
    assert_eq!(tensor.data, vec![2, 0, 0, 0]);
}
