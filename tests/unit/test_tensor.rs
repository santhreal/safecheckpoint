use safecheckpoint::error::Error;
use safecheckpoint::tensor::{DType, TensorMetadata};

#[test]
fn test_dtype_size_in_bytes() {
    assert_eq!(DType::F64.size_in_bytes(), 8);
    assert_eq!(DType::I64.size_in_bytes(), 8);
    assert_eq!(DType::U64.size_in_bytes(), 8);
    assert_eq!(DType::F32.size_in_bytes(), 4);
    assert_eq!(DType::I32.size_in_bytes(), 4);
    assert_eq!(DType::U32.size_in_bytes(), 4);
    assert_eq!(DType::F16.size_in_bytes(), 2);
    assert_eq!(DType::BF16.size_in_bytes(), 2);
    assert_eq!(DType::I16.size_in_bytes(), 2);
    assert_eq!(DType::U16.size_in_bytes(), 2);
    assert_eq!(DType::I8.size_in_bytes(), 1);
    assert_eq!(DType::U8.size_in_bytes(), 1);
    assert_eq!(DType::BOOL.size_in_bytes(), 1);
}

#[test]
fn test_tensor_metadata_validate_valid() {
    let metadata = TensorMetadata {
        dtype: DType::F32,
        shape: vec![2, 2],
        data_offsets: [0, 16],
        checksum: None,
    };
    assert!(metadata.validate().is_ok());
}

#[test]
fn test_tensor_metadata_validate_invalid_offsets() {
    let metadata = TensorMetadata {
        dtype: DType::F32,
        shape: vec![2, 2],
        data_offsets: [16, 0],
        checksum: None,
    };
    let result = metadata.validate();
    assert!(result.is_err());
    match result {
        Err(Error::InvalidFormat { offset, message }) => {
            assert_eq!(offset, 16);
            assert_eq!(message, "data_offsets start is greater than end");
        }
        other => panic!("Expected InvalidFormat error, got: {:?}", other),
    }
}

#[test]
fn test_tensor_metadata_validate_size_mismatch() {
    let metadata = TensorMetadata {
        dtype: DType::F32,
        shape: vec![2, 2],
        data_offsets: [0, 4],
        checksum: None,
    };
    let result = metadata.validate();
    assert!(
        matches!(
            result,
            Err(Error::SizeMismatch { expected: 16, actual: 4, .. })
        ),
        "Expected SizeMismatch for data length mismatch, got: {:?}",
        result
    );
}
