mod model;
mod utils;
use burn::data::dataloader::Dataset;
use burn::{
    backend::{Autodiff, Cuda},
    prelude::*,
};
fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::cuda::CudaDevice::default();

    let test_path = "dataset/test";
    let train_dataset = match model::data::BirdDataset::train() {
        Ok(dataset) => dataset,
        Err(e) => {
            eprintln!("Error loading training dataset: {}", e);
            return;
        }
    };
    let sample_item = train_dataset.get(42);
    if let Some(item) = sample_item {
        utils::float_vec_to_image(
            &item.image,
            model::constants::HEIGHT,
            model::constants::WIDTH,
        )
        .map(|img| {
            img.save("sample_image.png")
                .expect("Failed to save sample image");
            println!("Sample image saved as 'sample_image.png'");
        })
        .unwrap_or_else(|| {
            eprintln!("Failed to convert item to image.");
        });
    } else {
        eprintln!("No item found at index 0 in the training dataset.");
    }
}
