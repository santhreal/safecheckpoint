# Changelog

## [0.1.3] - 2026-08-07

### Security
- Fixed path validation in `Writer::validate_write_path` to run symlink component checks on input path ancestors *before* `canonicalize()`, preventing symlink directory traversal bypasses where `canonicalize()` resolved symlinks prior to verification.
- Enforced symlink metadata checks in `Writer::is_shard_valid` and `save_sharded` to reject planted symlinks as valid shard files fail-closed.

### Fixed
- Fixed directory duplication bug in `Writer::validate_write_path` when saving to relative paths containing subdirectories (e.g. `sub/model.safetensors`).
- Hardened `Reader::open` size check to return `Error::InvalidFormat { offset: 0, .. }` for 0-byte files before `mmap` allocation, unifying truncated file error handling.
- Hardened buffer index calculations in `write_tensors_to_file` with `checked_add` to prevent overflow.


## [0.1.2] - 2026-08-07

### Security
- Shard index writes use randomized tempfile + atomic persist (no predictable temp name symlink clobber).

### Changed
- Crate `authors` set to `Santh <64453045+santhreal@users.noreply.github.com>`.

## [0.1.1] - 2026-07-31

### Security
- Sharded index writes now use a randomized `tempfile` name plus atomic
  `persist`, matching the shard writer. The previous predictable temp name
  (`<prefix>.safetensors.index.json.tmp`) combined with `File::create` would
  follow a pre-planted symlink and overwrite an unrelated file in a shared
  directory.

### Fixed
- `Reader::open` rejects files whose size does not fit the platform address
  space instead of truncating the `u64` length to `usize`.
- Removed an `expect` on the header-checksum error path (deny-lint
  violation); the 32-byte slice conversion now has a non-panicking fallback.
- `DType::size_in_bytes` takes `self` by value (`DType` is `Copy`).
- `save_sharded` is decomposed into `validate_shard_args`, `verify_shards`,
  and `write_index_atomic` helpers; no behavior change.
- Test suite cleaned of ~60 pre-existing lint warnings (unused `Result`
  fixtures now use `?` or `unwrap`, explicit truncate flags, dead imports).

## [0.1.0] - 2026-04-12

### Added
- Initial release of safecheckpoint.
