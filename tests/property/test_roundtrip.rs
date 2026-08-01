use proptest::prelude::*;
use safecheckpoint::{DType, Reader, Writer};
use std::collections::HashMap;
use tempfile::tempdir;

fn arb_dtype() -> impl Strategy<Value = DType> {
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

proptest! {
    #[test]
    fn test_property_roundtrip_proptest(
        metadata in proptest::collection::hash_map("[a-zA-Z0-9_]+", "[a-zA-Z0-9_\\-]+", 0..5),
        tensors in proptest::collection::vec(
            ("[a-zA-Z0-9_]+", arb_dtype(), proptest::collection::vec(1usize..10usize, 1..3))
                .prop_flat_map(|(name, dtype, shape)| {
                    // Data length must equal shape_product * dtype size, or save()
                    // rejects the tensor with SizeMismatch.
                    let len = shape.iter().product::<usize>() * dtype.size_in_bytes();
                    (Just(name), Just(dtype), Just(shape), proptest::collection::vec(any::<u8>(), len..=len))
                }),
            1..20
        )
    ) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("prop.safetensors");
        let mut writer = Writer::new();

        // Data length matches shape * dtype_size (enforced by the strategy), so
        // save() accepts every tensor; we verify what goes in comes out.

        // Deduplicate tensor names to avoid writer overwriting causing reader mismatch
        let mut expected_tensors = HashMap::new();

        for (name, dtype, shape, data) in tensors {
            expected_tensors.insert(name.clone(), (dtype, shape.clone(), data.clone()));
            writer.add_tensor(&name, dtype, shape, data).unwrap();
        }

        for (k, v) in &metadata {
            writer.add_metadata(k, v);
        }

        writer.save(&path).unwrap();

        let reader = Reader::open(&path).unwrap();

        for (k, v) in &metadata {
            assert_eq!(reader.metadata().get(k).unwrap(), v);
        }

        for (name, (expected_dtype, expected_shape, expected_data)) in &expected_tensors {
            let t = reader.get_tensor(name).unwrap();
            assert_eq!(t.data, expected_data.as_slice());
            assert_eq!(t.metadata.shape, *expected_shape);
            assert_eq!(t.metadata.dtype, *expected_dtype);
        }
    }
}
