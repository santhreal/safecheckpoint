use safecheckpoint::{DType, Error, Reader, Writer};
use tempfile::tempdir;

#[test]
fn test_writer_basic_save_and_load() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model.safetensors");

    let mut writer = Writer::new();
    writer
        .add_tensor(
            "w1",
            DType::F32,
            vec![2, 2],
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        )
        .unwrap();
    writer.add_metadata("format", "pt");

    writer.save(&path).unwrap();

    let reader = Reader::open(&path).unwrap();
    assert_eq!(reader.tensor_names(), vec!["w1".to_string()]);

    let t1 = reader.get_tensor("w1").unwrap();
    assert_eq!(t1.metadata.shape, vec![2, 2]);
    assert_eq!(t1.metadata.dtype, DType::F32);
    assert_eq!(
        t1.data,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    );

    assert_eq!(reader.metadata().get("format").unwrap(), "pt");
}

#[test]
fn test_writer_rejects_reserved_metadata_tensor_name() {
    let mut writer = Writer::new();
    let result = writer.add_tensor("__metadata__", DType::F32, vec![1], vec![0; 4]);
    assert!(
        matches!(result, Err(Error::ReservedTensorName { ref name }) if name == "__metadata__"),
        "Expected ReservedTensorName error for '__metadata__', got: {:?}",
        result
    );
}

#[test]
fn test_writer_rejects_shape_size_mismatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mismatch.safetensors");

    let mut writer = Writer::new();
    // Claim shape [2,2] (16 bytes for F32) but provide only 4 bytes
    writer.add_tensor("w1", DType::F32, vec![2, 2], vec![0; 4]).unwrap();
    let result = writer.save(&path);
    assert!(
        matches!(
            result,
            Err(Error::SizeMismatch { ref tensor_name, expected: 16, actual: 4 }) if tensor_name == "w1"
        ),
        "Expected SizeMismatch error, got: {:?}",
        result
    );
}

#[test]
fn test_writer_rejects_path_traversal() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("..").join("escape.safetensors");

    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![0; 4]).unwrap();
    let result = writer.save(&path);
    assert!(
        matches!(result, Err(Error::PathTraversal { .. })),
        "Expected PathTraversal error, got: {:?}",
        result
    );
}

#[test]
fn test_writer_save_bare_relative_filename_is_durable() {
    // A bare filename has `path.parent() == Some("")` (an empty path), not
    // `None`. Before the fix, the directory-sync opened "" and failed, and the
    // tempfile base was empty too. Saving to a bare filename must work and the
    // file must be readable back.
    let dir = tempdir().unwrap();
    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(dir.path()).unwrap();

    let result = (|| {
        let mut writer = Writer::new();
        writer.add_tensor("w1", DType::F32, vec![1], vec![7, 0, 0, 0])?;
        // Bare relative filename: no directory component.
        writer.save("bare_model.safetensors")?;

        let reader = Reader::open("bare_model.safetensors")?;
        let tensor = reader.get_tensor("w1")?;
        assert_eq!(tensor.data, vec![7, 0, 0, 0]);
        Ok::<(), Error>(())
    })();

    // Always restore cwd before asserting so a failure does not poison other tests.
    std::env::set_current_dir(&prev).unwrap();
    result.expect("bare relative filename save+load must succeed");
}
