use crate::model::{
    architecture::{Model, ModelConfig},
    constants::{HEIGHT, LATENT_DIM, WIDTH},
    data::{BirdBatcher, BirdDataset},
};

use crate::utils::{chw_vec_to_image, float_vec_to_image};

use burn::{
    data::dataloader::{self, DataLoaderBuilder},
    nn::loss::{BinaryCrossEntropyLossConfig, Reduction::Mean},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::Distribution,
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer_g: AdamConfig,
    pub optimizer_d: AdamConfig,

    #[config(default = 40)]
    pub num_epochs: usize,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 10)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher = BirdBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(BirdDataset::train().expect("Training dataset should be created successfully"));

    let mut generator = config.model.generator.init(&device);
    let mut discriminator = config.model.discriminator.init(&device);

    // Initialize optimizers
    let mut optim_g = config.optimizer_g.init();
    let mut optim_d = config.optimizer_d.init();

    let loss_criterion = BinaryCrossEntropyLossConfig::new().init(&device);

    for epoch in 1..=config.num_epochs {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let real_images = batch.images.to_device(&device);
            let batch_size = real_images.dims()[0];

            // --- 1. Train the Discriminator --- //
            // Create labels for real (1) and fake (0) images
            let real_labels = Tensor::<B, 1, Int>::ones([batch_size], &device);
            let fake_labels = Tensor::<B, 1, Int>::zeros([batch_size], &device);

            // -- Loss on real images --
            let real_output = discriminator.forward(real_images);
            let real_output = real_output.reshape([batch_size]);

            let loss_real = loss_criterion.forward(real_output, real_labels.clone());

            // -- Loss on fake images --
            let noise = Tensor::<B, 2>::random(
                [batch_size, LATENT_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let fake_images = generator.forward(noise);

            let fake_output = discriminator.forward(fake_images.detach());
            let fake_output = fake_output.reshape([batch_size]); // Ensure shape matches labels
            let loss_fake = loss_criterion.forward(fake_output, fake_labels.clone());

            let loss_d = (loss_real + loss_fake) / 2;

            let grads_d = loss_d.backward();
            let grads_d = GradientsParams::from_grads(grads_d, &discriminator);

            discriminator = optim_d.step(config.learning_rate, discriminator, grads_d);

            // --- 2. Train the Generator --- //
            // Generate a new batch of fake images
            let noise = Tensor::<B, 2>::random(
                [batch_size, LATENT_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let fake_images_for_g = generator.forward(noise);
            let output_for_g = discriminator.forward(fake_images_for_g);
            let output_for_g = output_for_g.reshape([batch_size]);

            let loss_g = loss_criterion.forward(output_for_g, real_labels.clone());

            let grads_g = loss_g.backward();
            let grads_g = GradientsParams::from_grads(grads_g, &generator);
            generator = optim_g.step(config.learning_rate, generator, grads_g);

            if iteration % 100 == 0 {
                println!(
                    "[Epoch {}/{} Iter {}] D Loss: {:.4}, G Loss: {:.4}",
                    epoch,
                    config.num_epochs,
                    iteration,
                    loss_d.clone().into_scalar(),
                    loss_g.clone().into_scalar()
                );
            }
        }

        let recorder = CompactRecorder::new();
        generator
            .clone()
            .save_file(
                format!("{artifact_dir}/generator-epoch-{}", epoch),
                &recorder,
            )
            .expect("Failed to save generator");
        discriminator
            .clone()
            .save_file(
                format!("{artifact_dir}/discriminator-epoch-{}", epoch),
                &recorder,
            )
            .expect("Failed to save discriminator");

        let noise_sample =
            Tensor::<B, 2>::random([1, LATENT_DIM], Distribution::Normal(0.0, 1.0), &device);
        let sample_image_tensor = generator.forward(noise_sample);
        let sample_image_tensor: burn::tensor::Tensor<B, 3> = sample_image_tensor.squeeze(0);

        let image_data: Vec<f32> = sample_image_tensor.into_data().to_vec().unwrap();

        if let Some(img) = chw_vec_to_image(&image_data, HEIGHT, WIDTH) {
            img.save(format!("train_generated/sample-epoch-{}.png", epoch))
                .expect("Failed to save sample image");
        }
    }
}
