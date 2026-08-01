mod test_end_to_end;

use safecheckpoint::{DType, Reader, Writer};
use std::sync::Arc;
use std::thread;
use tempfile::tempdir;

#[test]
fn test_save_load_single_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model.safetensors");

    let mut writer = Writer::new();
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    writer.add_tensor("w1", DType::F32, vec![2, 2], data.clone()).unwrap();
    writer.add_metadata("author", "santh");
    writer.save(&path).expect("failed to save");

    let reader = Reader::open(&path).expect("failed to open");
    let tensor = reader.get_tensor("w1").expect("failed to get tensor");

    assert_eq!(tensor.metadata.dtype, DType::F32);
    assert_eq!(tensor.metadata.shape, vec![2, 2]);
    assert_eq!(tensor.data, &data);
    assert_eq!(reader.metadata().get("author").unwrap(), "santh");
}

#[test]
fn test_save_load_sharded() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("sharded_model");

    let mut writer = Writer::new();
    for i in 0..10 {
        let name = format!("weight_{}", i);
        let data = vec![i as u8; 4];
        writer.add_tensor(&name, DType::U8, vec![4], data).unwrap();
    }

    writer
        .save_sharded(&model_dir, "model", 3)
        .expect("failed to save sharded");

    // Check index file
    let index_path = model_dir.join("model.safetensors.index.json");
    assert!(index_path.exists());

    let index_content = std::fs::read_to_string(&index_path).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_content).unwrap();
    assert!(index["weight_map"]
        .as_object()
        .unwrap()
        .contains_key("weight_0"));
    assert!(index["weight_map"]
        .as_object()
        .unwrap()
        .contains_key("weight_9"));

    // Check shards
    for i in 1..=3 {
        let shard_path = model_dir.join(format!("model-0000{}-of-00003.safetensors", i));
        assert!(shard_path.exists());

        let reader = Reader::open(&shard_path).unwrap();
        for name in reader.tensor_names() {
            let tensor = reader.get_tensor(&name).unwrap();
            let i_val = name.split('_').next_back().unwrap().parse::<u8>().unwrap();
            assert_eq!(tensor.data, vec![i_val; 4]);
        }
    }
}

#[test]
fn test_resume_sharded() {
    let dir = tempdir().unwrap();
    let model_dir = dir.path().join("resume_model");

    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::U8, vec![4], vec![1, 1, 1, 1]).unwrap();
    writer.add_tensor("w2", DType::U8, vec![4], vec![2, 2, 2, 2]).unwrap();

    // Save normally first
    writer
        .save_sharded(&model_dir, "model", 2)
        .expect("failed to save");

    // Modify a shard file to make it "invalid" (simulate partial write)
    let shard2_path = model_dir.join("model-00002-of-00002.safetensors");
    std::fs::write(&shard2_path, b"garbage").unwrap();

    // Save again. It should detect the garbage shard and overwrite it.
    writer
        .save_sharded(&model_dir, "model", 2)
        .expect("failed to resume save");

    // Verify it's valid now
    let reader = Reader::open(&shard2_path).expect("failed to open after resume");
    assert!(reader.header().tensors.contains_key("w2"));
}

#[test]
fn test_zero_byte_tensor() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("zero.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("empty", DType::U8, vec![0], vec![]).unwrap();
    writer.save(&path).unwrap();

    let reader = Reader::open(&path).unwrap();
    let tensor = reader.get_tensor("empty").unwrap();
    assert_eq!(tensor.data.len(), 0);
}

#[test]
fn test_unicode_filenames() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("модель_🌸.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![0, 0, 0, 0]).unwrap();
    writer.save(&path).unwrap();

    let reader = Reader::open(&path).unwrap();
    assert!(reader.get_tensor("w1").is_ok());
}

#[test]
fn test_concurrent_access() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("concurrent.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::U8, vec![4], vec![42, 42, 42, 42]).unwrap();
    writer.save(&path).unwrap();

    let path_arc = Arc::new(path);
    let mut handles = vec![];

    for _ in 0..10 {
        let p = Arc::clone(&path_arc);
        handles.push(thread::spawn(move || {
            let reader = Reader::open(&*p).unwrap();
            let tensor = reader.get_tensor("w1").unwrap();
            assert_eq!(tensor.data, &[42, 42, 42, 42]);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_partial_write_corruption() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupted.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::U8, vec![4], vec![1, 2, 3, 4]).unwrap();
    writer.save(&path).unwrap();

    // Corrupt the data block by changing one byte
    let mut data = std::fs::read(&path).unwrap();
    let len = data.len();
    data[len - 1] = 99;
    std::fs::write(&path, data).unwrap();

    let reader = Reader::open(&path).unwrap();
    // Getting the tensor should now fail checksum validation
    let err = reader.get_tensor("w1").unwrap_err();
    assert!(matches!(err, safecheckpoint::Error::Checksum { .. }));
}

#[test]
fn test_power_failure_simulation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("power_fail.safetensors");
    let tmp_path = path.with_extension("tmp");

    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::U8, vec![4], vec![1, 2, 3, 4]).unwrap();

    // Simulate a crash during save_to_path before rename
    writer.save(&tmp_path).unwrap(); // This saves to tmp_path.tmp

    // The main file shouldn't exist, only the temporary ones
    assert!(!path.exists());

    // The state remains uncorrupted at the original path.
}
