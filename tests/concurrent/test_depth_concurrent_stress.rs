use safecheckpoint::{DType, Error, Reader, Writer};
use std::sync::{Arc, Barrier, RwLock};
use std::thread;
use tempfile::tempdir;

#[test]
fn test_concurrent_stress_read_write() {
    let dir = tempdir().expect("Failed to create temp directory");
    let path = dir.path().join("stress.safetensors");
    let path_arc = Arc::new(path);

    // Initial write
    {
        let mut writer = Writer::new();
        writer.add_tensor("w1", DType::F32, vec![100], vec![0; 400]).unwrap();
        writer.save(path_arc.as_path()).expect("Initial save failed");
    }

    let num_threads = 32;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];
    let file_lock = Arc::new(RwLock::new(()));

    for i in 0..num_threads {
        let b = Arc::clone(&barrier);
        let p = Arc::clone(&path_arc);
        let lock = Arc::clone(&file_lock);

        handles.push(thread::spawn(move || {
            b.wait();
            
            for j in 0..50 {
                // Randomly decide to read or write
                if j % 2 == 0 {
                    // Read
                    let _read_lock = lock.read().unwrap();
                    let res = Reader::open(p.as_path());
                    if let Ok(reader) = res {
                        let t = reader.get_tensor("w1");
                        if let Ok(tensor) = t {
                            assert_eq!(tensor.data.len(), 400, "Thread {} read mismatch on iter {}", i, j);
                        } else {
                            panic!("Thread {} failed to get tensor on iter {}", i, j);
                        }
                    } else if let Err(e) = res {
                        // Accept errors that might arise from extreme concurrency if locks fail,
                        // but with RwLock this should always succeed.
                        panic!("Thread {} failed to read on iter {}: {:?}", i, j, e);
                    }
                } else {
                    // Write
                    let _write_lock = lock.write().unwrap();
                    let mut writer = Writer::new();
                    writer.add_tensor("w1", DType::F32, vec![100], vec![1; 400]).unwrap();
                    let res = writer.save(p.as_path());
                    if let Err(e) = res {
                        panic!("Thread {} failed to write on iter {}: {:?}", i, j, e);
                    }
                }
            }
        }));
    }

    for h in handles {
        let res = h.join();
        assert!(res.is_ok(), "Thread panicked during stress test");
    }
}

#[test]
fn test_concurrent_stress_sharded() {
    let dir = tempdir().expect("Failed to create tempdir");
    let path_arc = Arc::new(dir.path().to_path_buf());

    let num_threads = 32;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = vec![];

    // Have 32 threads hammering save_sharded simultaneously into the same directory, 
    // potentially causing race conditions in shard validation and temporary file creation.
    // It should handle this safely without panicking.
    for i in 0..num_threads {
        let b = Arc::clone(&barrier);
        let p = Arc::clone(&path_arc);
        
        handles.push(thread::spawn(move || {
            b.wait();
            
            let mut writer = Writer::new();
            writer.add_tensor(&format!("w_{}", i), DType::F32, vec![10], vec![0; 40]).unwrap();
            
            let res = writer.save_sharded(p.as_path(), "stress_model", 4);
            match res {
                Ok(_) => (),
                Err(e) => {
                    // It's possible for concurrent sharded saves to conflict, but they should return errors, not panic.
                    // We check that the error is gracefully returned.
                    assert!(matches!(e, Error::Io(_) | Error::Json(_) | Error::SizeMismatch{..} | Error::InvalidFormat{..} | Error::Checksum { .. }), "Unexpected error type: {:?}", e);
                }
            }
        }));
    }

    for h in handles {
        let res = h.join();
        assert!(res.is_ok(), "Thread panicked during sharded stress test");
    }
}
