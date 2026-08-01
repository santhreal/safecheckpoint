//! Basic example of saving and loading tensors using `safecheckpoint`.

use safecheckpoint::{DType, Reader, Writer};
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("model.safetensors");

    // Save a tensor
    let mut writer = Writer::new();
    writer.add_tensor("w1", DType::F32, vec![2, 2], vec![0; 16])?;
    writer.save(&path)?;
    println!("Saved tensor to: {:?}", path);

    // Load the tensor
    let reader = Reader::open(&path)?;
    let tensor = reader.get_tensor("w1")?;
    println!("Loaded tensor 'w1' with data length: {}", tensor.data.len());

    Ok(())
}
