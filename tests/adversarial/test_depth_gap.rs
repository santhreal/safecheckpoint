use safecheckpoint::{DType, Reader, Writer};
use std::fs;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_gap_toctou_shard_modification() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Gap Test: If a shard file exists and is valid initially during save_sharded
    // and then gets modified maliciously, what happens?
    // The implementation writes shards in parallel, checking if they exist and are valid.
    // If we write 2 shards, and we simulate the condition where shard 1 is pre-existing.

    let dir = tempdir()?;
    let path = dir.path();

    // Create a writer with enough data for 2 shards.
    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![1], vec![1; 4])?; // goes to shard 1
    writer.add_tensor("w2", DType::F32, vec![1], vec![2; 4])?; // goes to shard 2

    // Run an initial save_sharded to create valid shards
    writer.save_sharded(path, "model", 2)?;

    // Now we modify shard 1. In reality TOCTOU would mean it modifies it *between* the
    // `is_shard_valid` check and the final verification loop.
    // But let's see if the final verification catches it.
    let shard1_path = path.join("model-00001-of-00002.safetensors");
    let mut shard1_file = fs::OpenOptions::new().write(true).open(&shard1_path)?;

    // Corrupt the file data (e.g., set the data byte to 0xFF)
    // First, find the offset of the tensor. It's safe to just append or modify.
    // Let's just append some junk which changes the file size, or overwrite the whole file
    // so `is_shard_valid` and the verification both fail if called.
    shard1_file.write_all(&[0xFF; 100])?;
    shard1_file.sync_all()?;

    // Now if we run save_sharded again.
    // If it relies on previous validity or doesn't check properly, it might succeed.
    // Based on the code, `is_shard_valid` returns false, so it regenerates it!
    // But what if it's valid during `is_shard_valid`, and then we modify it before the index check?
    // We can't easily hook the timing in Rust without patching.
    // However, if we corrupt it, it should regenerate it and succeed. Let's verify that.

    let res = writer.save_sharded(path, "model", 2);
    assert!(
        res.is_ok(),
        "save_sharded should regenerate corrupted shards"
    );

    // Let's create a *fake* valid shard that is NOT valid for this model.
    let mut bad_writer = Writer::new();
    bad_writer.add_tensor("w1", DType::F32, vec![1], vec![99; 4])?; // Wrong data!
    bad_writer.save(&shard1_path)?;

    let res2 = writer.save_sharded(path, "model", 2);
    assert!(
        res2.is_ok(),
        "save_sharded should regenerate shards with mismatched checksums"
    );

    // After regeneration, it should be the correct data again.
    let reader = Reader::open(&shard1_path)?;
    let t = reader.get_tensor("w1")?;
    assert_eq!(t.data, &[1, 1, 1, 1]);

    Ok(())
}

#[test]
fn test_gap_symlink_attack() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Unix only symlink test
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let dir = tempdir()?;

        let target_dir = dir.path().join("secret_target");
        fs::create_dir(&target_dir)?;
        let target_file = target_dir.join("secret.txt");
        fs::write(&target_file, "secret data")?;

        let symlink_path = dir.path().join("link.safetensors");
        symlink(&target_file, &symlink_path)?;

        let mut writer = Writer::new();
        writer.add_tensor("w1", DType::F32, vec![1], vec![0; 4])?;

        // This will write to `link.safetensors.tmp` and then rename it to `link.safetensors`.
        // The rename will overwrite the symlink, rather than following it.
        writer.save(&symlink_path)?;

        // The original secret file should be untouched
        let content = fs::read_to_string(&target_file)?;
        assert_eq!(content, "secret data");

        // The symlink is now a regular file (safetensors)
        assert!(fs::symlink_metadata(&symlink_path)?.is_file());
    }

    Ok(())
}

#[test]
fn test_index_temp_symlink_is_not_followed() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Regression lock (Unix only): the sharded index writer used a PREDICTABLE
    // temp name (`<prefix>.safetensors.index.json.tmp`) with `File::create`,
    // which follows symlinks. In a directory an attacker can write to, a
    // pre-planted symlink at that name made save_sharded overwrite an
    // unrelated victim file with index JSON. The temp file now comes from
    // `tempfile` with a randomized name, so the planted symlink is ignored.
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let dir = tempdir()?;
        let shard_dir = dir.path().join("shards");
        fs::create_dir(&shard_dir)?;

        let victim = dir.path().join("victim.txt");
        fs::write(&victim, "do not overwrite")?;

        // Attacker plants a symlink at the old predictable temp path.
        let planted = shard_dir.join("model.safetensors.index.json.tmp");
        symlink(&victim, &planted)?;

        let mut writer = Writer::new();
        writer.add_tensor("w1", DType::F32, vec![1], vec![7; 4])?;
        writer.save_sharded(&shard_dir, "model", 1)?;

        // The victim must be byte-identical: the write never went through the link.
        let content = fs::read_to_string(&victim)?;
        assert_eq!(content, "do not overwrite");

        // The planted symlink must still be a symlink (never created through).
        assert!(fs::symlink_metadata(&planted)?.file_type().is_symlink());

        // And the real index file must exist and be valid JSON.
        let index = shard_dir.join("model.safetensors.index.json");
        let index_text = fs::read_to_string(&index)?;
        assert!(index_text.contains("w1"), "index must map the tensor: {index_text}");
    }

    Ok(())
}
