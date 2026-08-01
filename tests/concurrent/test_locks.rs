use safecheckpoint::{DType, Reader, Writer};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::tempdir;

#[test]
fn test_concurrent_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("concurrent.safetensors");
    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![100], vec![0; 400]).unwrap();
    writer.save(&path).unwrap();

    let num_threads = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    let path_arc = Arc::new(path);

    for _ in 0..num_threads {
        let b = Arc::clone(&barrier);
        let p = Arc::clone(&path_arc);
        handles.push(thread::spawn(move || {
            b.wait();
            let reader = Reader::open(p.as_path()).unwrap();
            let tensor = reader.get_tensor("w1").unwrap();
            assert_eq!(tensor.data.len(), 400);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}
