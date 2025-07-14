use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, LeakyRelu, LeakyReluConfig, Linear,
        LinearConfig, PaddingConfig2d, Relu, Sigmoid, Tanh,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

use crate::model::constants::LATENT_DIM;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub generator: GeneratorModel<B>,
    pub discriminator: EnDiscriminatorModel<B>,
}

#[derive(Module, Debug)]
pub struct GeneratorModel<B: Backend> {
    conv1: ConvTranspose2d<B>,
    conv2: ConvTranspose2d<B>,
    conv3: ConvTranspose2d<B>,
    conv4: ConvTranspose2d<B>,
    bn1: BatchNorm<B, 2>,
    bn2: BatchNorm<B, 2>,
    bn3: BatchNorm<B, 2>,
    bn4: BatchNorm<B, 2>,
    linear1: Linear<B>,
    activation: Relu,
    activation2: Tanh,
}

#[derive(Config, Debug)]
pub struct GeneratorModelConfig {
    #[config(default = "0.7")]
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct EnDiscriminatorModel<B: Backend> {
    // Encoder
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    bn2: BatchNorm<B, 2>,
    bn3: BatchNorm<B, 2>,
    bn4: BatchNorm<B, 2>,

    // Decoder
    deconv1: ConvTranspose2d<B>,
    deconv2: ConvTranspose2d<B>,
    deconv3: ConvTranspose2d<B>,
    deconv4: ConvTranspose2d<B>,
    dbn1: BatchNorm<B, 2>,
    dbn2: BatchNorm<B, 2>,
    dbn3: BatchNorm<B, 2>,

    activation: LeakyRelu,
    activation_final: Tanh,
}

#[derive(Config, Debug)]
pub struct EnDiscriminatorConfig {
    #[config(default = "0.2")]
    leaky_relu_slope: f64,
}

impl GeneratorModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GeneratorModel<B> {
        GeneratorModel {
            linear1: LinearConfig::new(LATENT_DIM, 512 * 4 * 4).init(device),
            conv1: ConvTranspose2dConfig::new([512, 256], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            conv2: ConvTranspose2dConfig::new([256, 128], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            conv3: ConvTranspose2dConfig::new([128, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            conv4: ConvTranspose2dConfig::new([64, 3], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            bn1: BatchNormConfig::new(512).init(device),
            bn2: BatchNormConfig::new(256).init(device),
            bn3: BatchNormConfig::new(128).init(device),
            bn4: BatchNormConfig::new(64).init(device),
            activation: Relu::new(),
            activation2: Tanh::new(),
        }
    }
}

impl EnDiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EnDiscriminatorModel<B> {
        EnDiscriminatorModel {
            // Encoder
            conv1: Conv2dConfig::new([3, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device), // 64 -> 32
            conv2: Conv2dConfig::new([64, 128], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device), // 32 -> 16
            conv3: Conv2dConfig::new([128, 256], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device), // 16 -> 8
            conv4: Conv2dConfig::new([256, 512], [4, 4])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device), // 8 -> 4

            bn1: BatchNormConfig::new(64).init(device),
            bn2: BatchNormConfig::new(128).init(device),
            bn3: BatchNormConfig::new(256).init(device),
            bn4: BatchNormConfig::new(512).init(device),

            // Decoder
            deconv1: ConvTranspose2dConfig::new([512, 256], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device), // 4 -> 8
            deconv2: ConvTranspose2dConfig::new([256, 128], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device), // 8 -> 16
            deconv3: ConvTranspose2dConfig::new([128, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device), // 16 -> 32
            deconv4: ConvTranspose2dConfig::new([64, 3], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device), // 32 -> 64

            dbn1: BatchNormConfig::new(256).init(device),
            dbn2: BatchNormConfig::new(128).init(device),
            dbn3: BatchNormConfig::new(64).init(device),

            activation: LeakyReluConfig::new()
                .with_negative_slope(self.leaky_relu_slope)
                .init(),
            activation_final: Tanh::new(),
        }
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub generator: GeneratorModelConfig,
    pub discriminator: EnDiscriminatorConfig,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            generator: self.generator.init(device),
            discriminator: self.discriminator.init(device),
        }
    }
}

impl<B: Backend> GeneratorModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 4> {
        let mut x = input;
        x = self.linear1.forward(x);
        let mut x = x.reshape([-1, 512, 4, 4]);
        x = self.bn1.forward(x);
        x = self.activation.forward(x);
        x = self.conv1.forward(x);
        x = self.bn2.forward(x);
        x = self.activation.forward(x);
        x = self.conv2.forward(x);
        x = self.bn3.forward(x);
        x = self.activation.forward(x);
        x = self.conv3.forward(x);
        x = self.bn4.forward(x);
        x = self.activation.forward(x);
        x = self.conv4.forward(x);
        self.activation2.forward(x)
    }
}

impl<B: Backend> EnDiscriminatorModel<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // Encoder
        let mut x = self.conv1.forward(input.clone());
        x = self.bn1.forward(x);
        x = self.activation.forward(x);

        x = self.conv2.forward(x);
        x = self.bn2.forward(x);
        x = self.activation.forward(x);

        x = self.conv3.forward(x);
        x = self.bn3.forward(x);
        x = self.activation.forward(x);

        x = self.conv4.forward(x);
        x = self.bn4.forward(x);
        let embedding = self.activation.forward(x);

        // Decoder
        let mut x = self.deconv1.forward(embedding.clone());
        x = self.dbn1.forward(x);
        x = self.activation.forward(x);

        x = self.deconv2.forward(x);
        x = self.dbn2.forward(x);
        x = self.activation.forward(x);

        x = self.deconv3.forward(x);
        x = self.dbn3.forward(x);
        x = self.activation.forward(x);

        x = self.deconv4.forward(x);
        let reconstruction = self.activation_final.forward(x);

        (reconstruction, embedding)
    }
}
