use proptest::prelude::*;
use safecheckpoint::{CheckpointHeader, DType, TensorMetadata, CURRENT_SCHEMA_VERSION};

// Strategy to generate DType
fn dtype_strategy() -> impl Strategy<Value = DType> {
    prop_oneof![
        Just(DType::F64),
        Just(DType::F32),
        Just(DType::F16),
        Just(DType::BF16),
        Just(DType::I64),
        Just(DType::I32),
        Just(DType::I16),
        Just(DType::I8),
        Just(DType::U64),
        Just(DType::U32),
        Just(DType::U16),
        Just(DType::U8),
        Just(DType::BOOL),
    ]
}

// Strategy to generate valid TensorMetadata. The offset span must equal
// shape_product * dtype.size_in_bytes() or `validate()` rejects it, so derive
// the end offset from the shape and dtype rather than generating it freely.
fn tensor_metadata_strategy() -> impl Strategy<Value = TensorMetadata> {
    (
        dtype_strategy(),
        prop::collection::vec(1usize..100, 1..4), // shape
        0usize..1000,                             // start offset
        prop::option::of(any::<u32>()),           // checksum
    )
        .prop_map(|(dtype, shape, start, checksum)| {
            let elements: usize = shape.iter().product();
            let expected_len = elements * dtype.size_in_bytes();
            TensorMetadata {
                dtype,
                shape,
                data_offsets: [start, start + expected_len],
                checksum,
            }
        })
}

// Strategy to generate CheckpointHeader
fn checkpoint_header_strategy() -> impl Strategy<Value = CheckpointHeader> {
    (
        prop::collection::hash_map("[a-zA-Z0-9_-]{1,20}", tensor_metadata_strategy(), 0..10),
        prop::collection::hash_map("[a-zA-Z0-9_-]{1,20}", ".{0,50}", 0..5),
    )
        .prop_map(|(tensors, metadata)| CheckpointHeader {
            schema_version: CURRENT_SCHEMA_VERSION,
            tensors,
            metadata,
        })
}

proptest! {
    #[test]
    fn test_dtype_size_in_bytes_never_panics_and_returns_valid_sizes(dtype in dtype_strategy()) {
        let size = dtype.size_in_bytes();
        prop_assert!(size == 1 || size == 2 || size == 4 || size == 8);
    }

    #[test]
    fn test_checkpoint_header_roundtrip(header in checkpoint_header_strategy()) {
        let serialized = serde_json::to_string(&header).unwrap();
        let deserialized: CheckpointHeader = serde_json::from_str(&serialized).unwrap();

        prop_assert_eq!(header.tensors.len(), deserialized.tensors.len());
        prop_assert_eq!(header.metadata.len(), deserialized.metadata.len());

        for (name, meta) in &header.tensors {
            let des_meta = deserialized.tensors.get(name).unwrap();
            prop_assert_eq!(meta.dtype, des_meta.dtype);
            prop_assert_eq!(&meta.shape, &des_meta.shape);
            prop_assert_eq!(&meta.data_offsets, &des_meta.data_offsets);
            prop_assert_eq!(meta.checksum, des_meta.checksum);
        }

        for (name, value) in &header.metadata {
            let des_value = deserialized.metadata.get(name).unwrap();
            prop_assert_eq!(value, des_value);
        }
    }

    #[test]
    fn test_tensor_metadata_validation(meta in tensor_metadata_strategy()) {
        prop_assert!(meta.validate().is_ok());
    }
}
