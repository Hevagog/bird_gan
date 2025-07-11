use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, LeakyRelu, LeakyReluConfig, Linear,
        LinearConfig, PaddingConfig2d, Relu, Sigmoid, Tanh,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub generator: GeneratorModel<B>,
    pub discriminator: DiscriminatorModel<B>,
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
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct GeneratorModelConfig {
    #[config(default = "0.5")]
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct DiscriminatorModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    bn2: BatchNorm<B, 2>,
    bn3: BatchNorm<B, 2>,
    bn4: BatchNorm<B, 2>,
    pool: AdaptiveAvgPool2d,
    activation: Relu,
    activation2: LeakyRelu,
    activation_final: Sigmoid,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct DiscriminatorConfig {
    #[config(default = "0.5")]
    dropout: f64,
    #[config(default = "0.2")]
    leaky_relu_slope: f64,
}

impl GeneratorModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GeneratorModel<B> {
        GeneratorModel {
            linear1: LinearConfig::new(100, 512 * 8 * 8).init(device),
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
            conv4: ConvTranspose2dConfig::new([64, 3], [3, 3])
                .with_stride([1, 1])
                .with_padding([1, 1])
                .init(device),
            bn1: BatchNormConfig::new(512).init(device),
            bn2: BatchNormConfig::new(256).init(device),
            bn3: BatchNormConfig::new(128).init(device),
            bn4: BatchNormConfig::new(64).init(device),
            activation: Relu::new(),
            activation2: Tanh::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl DiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiscriminatorModel<B> {
        DiscriminatorModel {
            linear1: LinearConfig::new(4, 64).init(device),
            linear2: LinearConfig::new(64, 1).init(device),
            conv1: Conv2dConfig::new([3, 16], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv3: Conv2dConfig::new([32, 16], [2, 2])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv4: Conv2dConfig::new([16, 4], [2, 2])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn1: BatchNormConfig::new(16).init(device),
            bn2: BatchNormConfig::new(32).init(device),
            bn3: BatchNormConfig::new(16).init(device),
            bn4: BatchNormConfig::new(4).init(device),
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            activation: Relu::new(),
            activation2: LeakyReluConfig::new()
                .with_negative_slope(self.leaky_relu_slope)
                .init(),
            activation_final: Sigmoid::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub generator: GeneratorModelConfig,
    pub discriminator: DiscriminatorConfig,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            generator: self.generator.init(device),
            discriminator: self.discriminator.init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn new(
        generator_config: GeneratorModelConfig,
        discriminator_config: DiscriminatorConfig,
        device: &B::Device,
    ) -> Self {
        Model {
            generator: generator_config.init(device),
            discriminator: discriminator_config.init(device),
        }
    }
    pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 4>, Tensor<B, 2>) {
        let generated = self.generator.forward(input);
        let discriminator_output = self.discriminator.forward(generated.clone());
        (generated, discriminator_output)
    }
}

impl<B: Backend> GeneratorModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 4> {
        let mut x = input;
        x = self.linear1.forward(x);
        let mut x = x.reshape([-1, 512, 8, 8]);
        x = self.bn1.forward(x);
        x = self.activation.forward(x);
        x = self.conv1.forward(x);
        x = self.bn2.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);
        x = self.conv2.forward(x);
        x = self.bn3.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);
        x = self.conv3.forward(x);
        x = self.bn4.forward(x);
        x = self.activation.forward(x);
        x = self.conv4.forward(x);
        self.activation2.forward(x)
    }
}

impl<B: Backend> DiscriminatorModel<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _, _, _] = input.dims();
        let mut x = input;
        x = self.conv1.forward(x);
        x = self.bn1.forward(x);
        x = self.activation2.forward(x);
        x = self.dropout.forward(x);

        x = self.conv2.forward(x);
        x = self.bn2.forward(x);
        x = self.activation2.forward(x);
        x = self.dropout.forward(x);

        x = self.conv3.forward(x);
        x = self.bn3.forward(x);
        x = self.activation2.forward(x);
        x = self.dropout.forward(x);

        x = self.conv4.forward(x);
        x = self.bn4.forward(x);
        x = self.activation2.forward(x);
        x = self.pool.forward(x); // [batch, 4, 1, 1]
        let mut x = x.reshape([batch_size, 4]);
        x = self.linear1.forward(x);
        x = self.activation.forward(x);
        x = self.dropout.forward(x);
        x = self.linear2.forward(x);
        self.activation_final.forward(x)
    }
}
