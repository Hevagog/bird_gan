mod model;
mod utils;
use burn::{
    backend::{Autodiff, Cuda},
    optim::AdamConfig,
};

use model::architecture::{EnDiscriminatorConfig, GeneratorModelConfig, ModelConfig};

fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::cuda::CudaDevice::default();

    let artifact_dir = "/tmp/gan_bird";
    model::training::train::<MyAutodiffBackend>(
        artifact_dir,
        model::training::TrainingConfig::new(
            ModelConfig::new(GeneratorModelConfig::new(), EnDiscriminatorConfig::new()),
            AdamConfig::new(),
            AdamConfig::new(),
        ),
        device.clone(),
    );
}
