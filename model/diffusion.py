from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, Conv2D, GlobalMaxPool2D, Activation, DepthwiseConv2D
from keras.layers import MaxPooling2D, Dropout, GlobalAveragePooling2D, AveragePooling2D, UpSampling2D, Concatenate
from keras.layers import Layer, Input, concatenate, Normalization, Add, Subtract, Resizing, Lambda, BatchNormalization
from keras.models import Sequential
from keras.models import Model, clone_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
from keras import regularizers
import math
from keras.applications.inception_v3 import preprocess_input, InceptionV3
import tensorflow as tf
min_signal_rate = 0.02
max_signal_rate = 0.95

kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20
ema = 0.999

embedding_dims = 32
embedding_max_frequency = 1000.0

# class KID(tf.keras.metrics.Metric):
#     def __init__(self, name, **kwargs):
#         super().__init__(name=name, **kwargs)
#
#         # KID is estimated per batch and is averaged across batches
#         self.kid_tracker = tf.keras.metrics.Mean(name="kid_tracker")
#
#         # a pretrained InceptionV3 is used without its classification layer
#         # transform the pixel values to the 0-255 range, then use the same
#         # preprocessing as during pretraining
#         self.encoder = Sequential(
#             [
#                 Input(shape=(image_size, image_size, 3)),
#                 Rescaling(255.0),
#                 Resizing(height=kid_image_size, width=kid_image_size),
#                 Lambda(preprocess_input),
#                 InceptionV3(
#                     include_top=False,
#                     input_shape=(kid_image_size, kid_image_size, 3),
#                     weights="imagenet",
#                 ),
#                 GlobalAveragePooling2D(),
#             ],
#             name="inception_encoder",
#         )
#
#     def polynomial_kernel(self, features_1, features_2):
#         feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
#         return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0
#
#     def update_state(self, real_images, generated_images, sample_weight=None):
#         real_features = self.encoder(real_images, training=False)
#         generated_features = self.encoder(generated_images, training=False)
#
#         # compute polynomial kernels using the two sets of features
#         kernel_real = self.polynomial_kernel(real_features, real_features)
#         kernel_generated = self.polynomial_kernel(
#             generated_features, generated_features
#         )
#         kernel_cross = self.polynomial_kernel(real_features, generated_features)
#
#         # estimate the squared maximum mean discrepancy using the average kernel values
#         batch_size = tf.shape(real_features)[0]
#         batch_size_f = tf.cast(batch_size, dtype=tf.float32)
#         mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
#             batch_size_f * (batch_size_f - 1.0)
#         )
#         mean_kernel_generated = tf.reduce_sum(
#             kernel_generated * (1.0 - tf.eye(batch_size))
#         ) / (batch_size_f * (batch_size_f - 1.0))
#         mean_kernel_cross = tf.reduce_mean(kernel_cross)
#         kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
#
#         # update the average KID estimate
#         self.kid_tracker.update_state(kid)
#
#     def result(self):
#         return self.kid_tracker.result()
#
#     def reset_state(self):
#         self.kid_tracker.reset_state()

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = Conv2D(width, kernel_size=1)(x)
        x = BatchNormalization(center=False, scale=False)(x)
        x = Conv2D(
            width, kernel_size=3, padding="same", activation='swish'
        )(x)
        x = Conv2D(width, kernel_size=3, padding="same")(x)
        x = Add()([x, residual])  # filter=width
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = AveragePooling2D(pool_size=2)(x)  # valid无填充，奇数维减一除以二
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply

def get_network(image_size, widths, block_depth, channel):
    noisy_images = Input(shape=(image_size[0], image_size[1], channel))
    noise_variances = Input(shape=(1, 1, 1))

    e = Lambda(sinusoidal_embedding)(noise_variances)
    e = UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = Concatenate()([x, e])  # [?, 89, 6, widths[0]*2]

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = Conv2D(channel, kernel_size=1, kernel_initializer="zeros")(x)  # original image size
    model = Model([noisy_images, noise_variances], x, name="residual_unet")
    plot_model(model, to_file='model1.png', show_shapes=True)
    return model

class DiffusionModel(Model):
    def __init__(self, image_size, widths, block_depth, batch_size=1):  # image_size=[x, x, x]
        super().__init__()
        self.normalizer = Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = clone_model(self.network)
        self.batch_size = batch_size

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = tf.keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="i_loss")
        # self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, self.image_size[0], self.image_size[1], self.image_size[2]))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # # measure KID between real and generated images
        # # this is computationally demanding, kid_diffusion_steps has to be small
        # images = self.denormalize(images)
        # generated_images = self.generate(
        #     num_images=self.batch_size, diffusion_steps=kid_diffusion_steps
        # )
        # self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    # def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
    #     # plot random generated images for visual evaluation of generation quality
    #     generated_images = self.generate(
    #         num_images=num_rows * num_cols,
    #         diffusion_steps=plot_diffusion_steps,
    #     )
    #
    #     plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
    #     for row in range(num_rows):
    #         for col in range(num_cols):
    #             index = row * num_cols + col
    #             plt.subplot(num_rows, num_cols, index + 1)
    #             plt.imshow(generated_images[index])
    #             plt.axis("off")
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()
