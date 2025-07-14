use crate::model::{
    architecture::{Model, ModelConfig},
    constants::{HEIGHT, LATENT_DIM, WIDTH},
    data::{BirdBatcher, BirdDataset},
};

use crate::utils::{chw_vec_to_image, float_vec_to_image};

use burn::{
    data::dataloader::DataLoaderBuilder,
    grad_clipping::{GradientClipping, GradientClippingConfig},
    module::list_param_ids,
    nn::loss::{MseLoss, Reduction::Mean},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::Distribution,
    tensor::backend::AutodiffBackend,
    tensor::*,
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

    #[config(default = 540)]
    pub num_epochs: usize,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 10)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 8e-5)]
    pub gen_learning_rate: f64,

    #[config(default = 2e-5)]
    pub disc_learning_rate: f64,

    #[config(default = 10.0)]
    pub margin: f64,

    #[config(default = 0.2)]
    pub pt_weight: f64,

    #[config(default = 1)]
    pub k_discrimator_updates: usize,
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
    let gradient_clipping_g = GradientClippingConfig::init(&GradientClippingConfig::Norm(1.0));
    let gradient_clipping_d = GradientClippingConfig::init(&GradientClippingConfig::Norm(1.0));

    let mut optim_g = config
        .optimizer_g
        .with_beta_1(0.5)
        .with_beta_2(0.999)
        .init()
        .with_grad_clipping(gradient_clipping_g);
    let mut optim_d = config
        .optimizer_d
        .with_beta_1(0.5)
        .with_beta_2(0.999)
        .init()
        .with_grad_clipping(gradient_clipping_d);

    for epoch in 1..=config.num_epochs {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let real_images = batch.images.to_device(&device);
            let batch_size = real_images.dims()[0];

            let mut loss_d: Tensor<B, 1> = Tensor::<B, 1>::zeros([1], &device);
            let mut noise = Tensor::<B, 2>::random(
                [batch_size, LATENT_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            );

            // --- 1. Train the Discriminator --- //
            for _ in 0..config.k_discrimator_updates {
                noise = Tensor::<B, 2>::random(
                    [batch_size, LATENT_DIM],
                    Distribution::Normal(0.0, 1.0),
                    &device,
                );

                // Detach the fake images from the generator's computation graph
                let fake_images_detached = generator.forward(noise.clone()).detach();

                let (reconstructed_fake, _) = discriminator.forward(fake_images_detached.clone());
                let (reconstructed_real, _) = discriminator.forward(real_images.clone());

                // Loss for real images (aims for low reconstruction error)
                let loss_d_real =
                    MseLoss::new().forward(reconstructed_real, real_images.clone(), Mean);

                // Loss for fake images (aims for high reconstruction error)
                let loss_d_fake =
                    MseLoss::new().forward(reconstructed_fake, fake_images_detached, Mean);

                let margin_tensor =
                    Tensor::<B, 1>::from_data(TensorData::from([config.margin as f32]), &device);

                // Hinge loss: punish the discriminator if the reconstruction error for fake images is below the margin
                let loss_d_hinge = (margin_tensor - loss_d_fake).clamp_min(0.0);

                // Total discriminator loss
                loss_d = loss_d_real + loss_d_hinge;

                let grads_d = loss_d.backward();

                let grads_d = GradientsParams::from_grads(grads_d, &discriminator);

                discriminator = optim_d.step(config.disc_learning_rate, discriminator, grads_d);
            }

            // --- 2. Train the Generator --- //
            let noise = Tensor::<B, 2>::random(
                [batch_size, LATENT_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            );

            let fake_images = generator.forward(noise);

            let (reconstructed_fake, fake_embeddings) = discriminator.forward(fake_images.clone());

            let loss_g_reconstruct = MseLoss::new().forward(reconstructed_fake, fake_images, Mean);

            let loss_pt = pulling_away_loss(fake_embeddings);

            // The generator's total loss
            let loss_g = loss_g_reconstruct + loss_pt.clone() * (config.pt_weight as f32);

            let grads_g = loss_g.backward();
            let grads_g = GradientsParams::from_grads(grads_g, &generator);
            generator = optim_g.step(config.gen_learning_rate, generator, grads_g);

            if iteration % 100 == 0 {
                println!(
                    "[Epoch {}/{} Iter {}] D Loss: {:.4}, G Loss: {:.4}, PT Loss: {:.4}",
                    epoch,
                    config.num_epochs,
                    iteration,
                    loss_d.clone().into_scalar(),
                    loss_g.clone().into_scalar(),
                    loss_pt.clone().into_scalar(),
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

        if let Some(img) = chw_vec_to_image(&image_data, HEIGHT, WIDTH, true) {
            img.save(format!("train_generated/sample-epoch-{}.png", epoch))
                .expect("Failed to save sample image");
        }
    }
}

/// Pulling-away loss for EBGAN
fn pulling_away_loss<B: AutodiffBackend>(embeddings: Tensor<B, 4>) -> Tensor<B, 1> {
    let [batch_size, d1, d2, d3] = embeddings.dims();
    let embedding_len = d1 * d2 * d3;
    let embeddings_flat = embeddings.reshape([batch_size, embedding_len]);

    // Normalize embeddings
    let norm = embeddings_flat.clone().powf_scalar(2.0).sum_dim(1).sqrt();
    let norm = norm.reshape([batch_size, 1]);
    let normalized_embeddings = embeddings_flat.div(norm);

    // Calculate cosine similarity matrix
    let similarity = normalized_embeddings
        .clone()
        .matmul(normalized_embeddings.transpose());

    // PT loss
    let pt_loss = (similarity.powf_scalar(2.0).sum() - batch_size as f32)
        / (batch_size * (batch_size - 1)) as f32;
    pt_loss
}

// -- Label Smoothing Functions --
// Smooth labels for the discriminator to improve training stability
/// Smooth positive labels: range 0.8 - 1.2
pub fn smooth_positive_labels<B: Backend>(
    labels: Tensor<B, 1, burn::tensor::Int>,
) -> Tensor<B, 1, burn::tensor::Float> {
    let labels = labels.float();
    let shape = labels.dims();
    let noise =
        Tensor::<B, 1>::random(shape, Distribution::Uniform(0.0, 1.0), &labels.device()) * 0.4;
    labels - 0.2 + noise
}

/// Smooth negative labels: range 0.0 - 0.3
pub fn smooth_negative_labels<B: Backend>(
    labels: Tensor<B, 1, burn::tensor::Int>,
) -> Tensor<B, 1, burn::tensor::Float> {
    let labels = labels.float();
    let shape = labels.dims();
    let noise =
        Tensor::<B, 1>::random(shape, Distribution::Uniform(0.0, 1.0), &labels.device()) * 0.3;
    labels + noise
}

// Manual BCE implementation
fn binary_cross_entropy_with_continuous_targets<B: Backend>(
    predictions: Tensor<B, 1>,
    targets: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let eps = 1e-7; //to avoid log(0)
    let predictions = predictions.clamp(eps, 1.0 - eps);

    let loss = targets.clone() * predictions.clone().log()
        + (Tensor::<B, 1>::ones_like(&predictions) - targets)
            * (Tensor::<B, 1>::ones_like(&predictions) - predictions).log();
    -loss.mean()
}
