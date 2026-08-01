# safecheckpoint

![Status: alpha](https://img.shields.io/badge/status-alpha-blue.svg)

A Rust-native ML model checkpoint library replacing `torch.save()` with zero-copy mmap-based serialization and parallel shard writes.

## What it does

`safecheckpoint` provides zero-copy, `safetensors`-compatible model checkpoint reading and writing. It replaces insecure Python `pickle` serialization (`torch.save()`) with deterministic memory-mapped binary formats, atomic file persistence, CRC32 integrity validation, and parallel sharding across model parameters.

Key features:
- **Zero-copy reads**: Memory-maps model checkpoints directly into process memory using `memmap2`.
- **Atomic saves**: Writes temporary files first before committing to target file paths, preventing partial state corruption.
- **Parallel sharded writes**: Distributes large tensor shards across Rayon threads.
- **Symlink rejection**: Enforces path traversal checks and rejects symlink targets to prevent arbitrary file overwrites.

## Quick start

```rust
use safecheckpoint::{Writer, Reader, DType};

let mut writer = Writer::new();
writer.add_tensor("w1", DType::F32, vec![2, 2], vec![0; 16]);
writer.save("model.safetensors").expect("failed to save checkpoint");

let reader = Reader::open("model.safetensors").expect("failed to open checkpoint");
let tensor = reader.get_tensor("w1").expect("missing tensor w1");
assert_eq!(tensor.data.len(), 16);
```

## When to use / when not

### When to use
- Saving and loading machine learning model weights safely without arbitrary code execution risks.
- High-throughput zero-copy loading of multi-gigabyte neural network checkpoints.
- Parallel model weight sharded serialization.

### When not to use
- Serializing non-tensor arbitrary Python object graphs (use standard structured data formats like JSON/BSON/Protobuf).

## Compared to alternatives

- **`torch.save()` / Python pickle**: Exposes remote code execution (RCE) risks during deserialization. `safecheckpoint` parses fixed header metadata and raw byte buffers safely.
- **Naïve binary file dumps**: Vulnerable to partial write corruption on process crashes. `safecheckpoint` uses atomic file writes and checksum verifications.

## How it fits in Santh

`safecheckpoint` is located in `libs/general/` and serves as the primary secure checkpoint storage utility across Santh ML tools, model runners, and analysis engines.

## Contributing

Contributions require unit, adversarial, property, and gap test coverage. Run `cargo test -p safecheckpoint` to verify changes.

## License

Licensed under the MIT License ([LICENSE](LICENSE)) or Apache License 2.0.
