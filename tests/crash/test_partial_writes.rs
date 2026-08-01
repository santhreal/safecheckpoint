use safecheckpoint::{DType, Reader, Writer};
use std::fs;
use std::io::{Seek, Write};
use tempfile::tempdir;

#[test]
fn test_crash_corrupted_data_checksum_mismatch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupted.safetensors");
    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![0, 0, 0, 0]).unwrap();
    writer.save(&path).unwrap();

    // Maliciously modify the data block directly bypassing checks
    let _metadata = fs::metadata(&path).unwrap();
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    // Assuming data block is at the very end
    file.seek(std::io::SeekFrom::End(-1)).unwrap();
    file.write_all(&[1]).unwrap(); // Corrupt a byte

    let reader = Reader::open(&path).unwrap();
    let res = reader.get_tensor("w1");
    assert!(res.is_err(), "Checksum mismatch should be caught");
    match res.unwrap_err() {
        safecheckpoint::Error::Checksum { .. } => (),
        _ => panic!("Expected Checksum error"),
    }
}
