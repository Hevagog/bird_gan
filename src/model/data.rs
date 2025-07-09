use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use image::ImageReader;
use std::path::{Path, PathBuf};

use crate::model::constants::{CHANNELS, HEIGHT, WIDTH};

#[derive(Debug, Clone)]
pub struct BirdItem {
    pub image: Vec<f32>,
}

pub struct BirdDataset {
    pub image_paths: Vec<PathBuf>,
}

impl BirdDataset {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let path = path.as_ref();
        let mut image_paths = Vec::new();

        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    match ext.to_lowercase().as_str() {
                        "jpg" | "jpeg" | "png" | "bmp" | "tiff" => {
                            image_paths.push(path);
                        }
                        _ => {}
                    }
                }
            }
        }
        if image_paths.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No valid image files found in the directory",
            ));
        }
        Ok(Self { image_paths })
    }
    pub fn train() -> Result<Self, std::io::Error> {
        Self::new("dataset/train")
    }
    pub fn test() -> Result<Self, std::io::Error> {
        Self::new("dataset/test")
    }
}

impl Dataset<BirdItem> for BirdDataset {
    fn len(&self) -> usize {
        self.image_paths.len()
    }

    fn get(&self, index: usize) -> Option<BirdItem> {
        let path = &self.image_paths[index];
        let image = ImageReader::open(path).ok()?.decode().ok()?.to_rgb8();
        let mut image_data: Vec<f32> = Vec::with_capacity(CHANNELS * HEIGHT * WIDTH);
        for pixel in image.pixels() {
            // Normalize to [-1, 1]
            image_data.push((pixel[0] as f32 / 127.5) - 1.0); // R
            image_data.push((pixel[1] as f32 / 127.5) - 1.0); // G
            image_data.push((pixel[2] as f32 / 127.5) - 1.0); // B
        }

        Some(BirdItem { image: image_data })
    }
}

#[derive(Clone, Default)]
pub struct BirdBatcher {}

#[derive(Clone, Debug)]
pub struct BirdBatch<B: Backend> {
    pub images: Tensor<B, 4>, // Shape: [batch_size, channels, height, width]
}

impl<B: Backend> Batcher<B, BirdItem, BirdBatch<B>> for BirdBatcher {
    fn batch(&self, items: Vec<BirdItem>, device: &B::Device) -> BirdBatch<B> {
        let image_tensors: Vec<Tensor<B, 4>> = items
            .into_iter()
            .map(|item| {
                Tensor::<B, 3>::from_data(
                    TensorData::new(item.image, [CHANNELS, HEIGHT, WIDTH])
                        .convert::<B::FloatElem>(),
                    device,
                )
                .reshape([1, CHANNELS, HEIGHT, WIDTH])
            })
            .collect();
        let images = Tensor::cat(image_tensors, 0);
        BirdBatch { images }
    }
}
