//! Resilience tests driving REAL, deterministic failure conditions.
//!
//! These previously used `faultkit`'s cooperative injector, but faultkit is a
//! dev-only dependency and its `should_fail_*` hooks are never called from the
//! production library, so the injected faults were never observed and the
//! `assert!(res.is_err())` checks could never pass. They are rewritten here to
//! provoke genuine OS/format errors that the code actually surfaces, so the
//! error paths are exercised for real and the tests are portable and root-safe.

use safecheckpoint::{DType, Error, Reader, Writer};
use std::fs;
use std::io::Write;
use tempfile::tempdir;

/// A save whose target directory is actually a regular file must fail with a
/// real I/O error (ENOTDIR when the temp file is created), not silently
/// succeed. Works regardless of privilege.
#[test]
fn test_save_into_non_directory_parent_errors() {
    let dir = tempdir().unwrap();
    let not_a_dir = dir.path().join("i_am_a_file");
    fs::write(&not_a_dir, b"x").unwrap();
    // Parent component is a file, so the directory does not exist as a dir.
    let path = not_a_dir.join("child.safetensors");

    let mut writer = Writer::new();
    writer
        .add_tensor("w1", DType::F32, vec![100], vec![0; 400])
        .unwrap();

    let res = writer.save(&path);
    match res {
        Err(Error::Io(_)) | Err(Error::PathTraversal { .. }) => {}
        Ok(()) => panic!("save into a non-directory parent should not succeed"),
        Err(other) => panic!("expected Io/PathTraversal error, got {other:?}"),
    }
}

/// Opening a truncated file (fewer than the 8 header-length bytes) must return a
/// format error rather than panicking on the mmap slice.
#[test]
fn test_open_truncated_file_errors_not_panics() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("truncated.safetensors");

    let mut writer = Writer::new();
    writer
        .add_tensor("w1", DType::F32, vec![100], vec![0; 400])
        .unwrap();
    writer.save(&path).unwrap();

    // Truncate below the 8-byte header-length prefix.
    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(4).unwrap();
    drop(file);

    match Reader::open(&path) {
        Err(Error::InvalidFormat { .. }) => {}
        Ok(_) => panic!("opening a 4-byte file should fail"),
        Err(other) => panic!("expected InvalidFormat, got {other:?}"),
    }
}

/// Opening a path that is a directory must fail (the mmap of a directory cannot
/// succeed) instead of panicking. Works regardless of privilege.
#[test]
fn test_open_directory_path_errors_not_panics() {
    let dir = tempdir().unwrap();
    let subdir = dir.path().join("a_directory.safetensors");
    fs::create_dir(&subdir).unwrap();

    let res = Reader::open(&subdir);
    assert!(
        res.is_err(),
        "opening a directory as a checkpoint must fail, not succeed"
    );
}

/// A crafted header claiming a length beyond the reader's DoS ceiling must be
/// rejected before any large allocation, protecting against OOM from hostile
/// files. This is the real, deterministic form of the old OOM-injection test.
#[test]
fn test_open_oversized_header_rejected_before_alloc() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("oversized_header.safetensors");

    // 8-byte little-endian header length far above the 100MB reader ceiling.
    let bogus_header_len: u64 = 200 * 1024 * 1024;
    let mut f = fs::File::create(&path).unwrap();
    f.write_all(&bogus_header_len.to_le_bytes()).unwrap();
    // A little padding so the file is longer than 8 bytes but nowhere near the
    // claimed header size.
    f.write_all(&[0u8; 64]).unwrap();
    drop(f);

    match Reader::open(&path) {
        Err(Error::InvalidFormat { message, .. }) => {
            assert!(
                message.contains("100MB") || message.contains("too short"),
                "unexpected message: {message}"
            );
        }
        Ok(_) => panic!("oversized header must be rejected"),
        Err(other) => panic!("expected InvalidFormat, got {other:?}"),
    }
}
