use safecheckpoint::{DType, Reader, Writer};
use tempfile::tempdir;

#[test]
fn test_extreme_boundary_tensors() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("extreme.safetensors");
    let mut writer = Writer::new();

    // 1. Empty input
    writer.add_tensor("t_empty", DType::U8, vec![0], vec![]).unwrap();

    // 2. Single byte
    writer.add_tensor("t_single", DType::U8, vec![1], vec![0x42]).unwrap();

    // 3. All zeros vs All 0xFF
    writer.add_tensor("t_zeros", DType::U8, vec![1024], vec![0x00; 1024]).unwrap();
    writer.add_tensor("t_ffs", DType::U8, vec![1024], vec![0xFF; 1024]).unwrap();

    // 4. Alternating pattern
    let mut alternating = vec![0; 1024];
    for (i, byte) in alternating.iter_mut().enumerate() {
        *byte = if i % 2 == 0 { 0xAA } else { 0x55 };
    }
    writer.add_tensor("t_alt", DType::U8, vec![1024], alternating.clone()).unwrap();

    // 5. Very large size (we can't allocate u32::MAX bytes in memory easily for tests, 
    // but we can allocate a reasonably large tensor or use one that pretends to be huge)
    // We will do a 1MB one to represent a large allocation block boundary
    writer.add_tensor("t_1mb", DType::U8, vec![1024 * 1024], vec![0x11; 1024 * 1024]).unwrap();

    writer.save(&path).unwrap();

    let reader = Reader::open(&path).expect("Engine failed to open boundaries file");

    let empty_data: &[u8] = &[];
    let single_data: &[u8] = &[0x42];
    assert_eq!(reader.get_tensor("t_empty").expect("Failed to get t_empty").data, empty_data, "Empty tensor data mismatch");
    assert_eq!(reader.get_tensor("t_single").expect("Failed to get t_single").data, single_data, "Single byte tensor data mismatch");
    assert_eq!(reader.get_tensor("t_zeros").expect("Failed to get t_zeros").data, vec![0x00; 1024].as_slice(), "All zeros tensor data mismatch");
    assert_eq!(reader.get_tensor("t_ffs").expect("Failed to get t_ffs").data, vec![0xFF; 1024].as_slice(), "All 0xFF tensor data mismatch");
    assert_eq!(reader.get_tensor("t_alt").expect("Failed to get t_alt").data, alternating.as_slice(), "Alternating pattern tensor data mismatch");
    assert_eq!(reader.get_tensor("t_1mb").expect("Failed to get t_1mb").data.len(), 1024 * 1024, "1MB tensor size mismatch");
}

#[test]
fn test_hash_collision_maximization() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("collisions.safetensors");
    let mut writer = Writer::new();

    // Hash collision testing for JSON maps or internal BTrees (SipHash / Rust default)
    // To generate collisions reliably we'd need SipHash seeds, but we can generate a huge 
    // number of keys with similar prefixes or pathological lengths to stress the map.
    // If it uses BTreeMap, this tests tree depth/balancing under load.
    for i in 0..10_000 {
        let key = format!("tensor_{:08x}_{}", i, "A".repeat(i % 100));
        writer.add_tensor(&key, DType::U8, vec![1], vec![0]).unwrap();
    }

    let res = writer.save(&path);
    assert!(res.is_ok(), "Engine failed to handle large number of keys");

    let reader = Reader::open(&path).expect("Engine failed to open collisions file");
    // Reconstruct the exact keys with the same formula the writer used, rather
    // than hand-counting the 'A' suffix (i % 100 for the last index 9999 = 99).
    let first_key = format!("tensor_{:08x}_{}", 0, "A".repeat(0));
    let last_key = format!("tensor_{:08x}_{}", 9999, "A".repeat(9999 % 100));
    assert!(reader.get_tensor(&first_key).is_ok(), "Engine failed to retrieve first tensor from large map");
    assert!(reader.get_tensor(&last_key).is_ok(), "Engine failed to retrieve last tensor from large map");
}

#[test]
fn test_metadata_boundaries() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("metadata.safetensors");
    let mut writer = Writer::new();

    writer.add_tensor("t1", DType::U8, vec![1], vec![0]).unwrap();

    // Max metadata values
    writer.add_metadata("empty_key", "");
    writer.add_metadata("", "empty_val");
    writer.add_metadata(&"A".repeat(100_000), &"B".repeat(100_000));

    // Try to trigger json escaping edge cases
    writer.add_metadata("escape\"\n\t\r\\", "val\"\n\t\r\\");
    writer.add_metadata("unicode_\u{1F980}", "val_\u{1F980}");

    writer.save(&path).expect("Engine failed to save metadata boundaries");

    let reader = Reader::open(&path).expect("Engine failed to open metadata boundaries file");
    let meta = reader.metadata();
    
    assert_eq!(meta.get("empty_key").map(|s| s.as_str()), Some(""), "Engine failed to store empty metadata value");
    assert_eq!(meta.get("").map(|s| s.as_str()), Some("empty_val"), "Engine failed to store empty metadata key");
    assert_eq!(meta.get("escape\"\n\t\r\\").map(|s| s.as_str()), Some("val\"\n\t\r\\"), "Engine failed to store escaped metadata strings");
    assert_eq!(meta.get("unicode_\u{1F980}").map(|s| s.as_str()), Some("val_\u{1F980}"), "Engine failed to store unicode metadata");
    assert_eq!(meta.get(&"A".repeat(100_000)).expect("Failed to get large metadata").len(), 100_000, "Engine failed to store large metadata length");
}
