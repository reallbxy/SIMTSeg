from keras.layers import Dense, Flatten, Conv2D, GlobalMaxPool2D, Activation, DepthwiseConv2D, Permute
from keras.layers import GlobalMaxPooling2D, Dropout, GlobalAveragePooling2D, AveragePooling2D, UpSampling2D, Concatenate
from keras.layers import Layer, Input, multiply, Normalization, Add, Subtract, Reshape, Lambda, BatchNormalization
# from keras.models import Sequential
from keras.models import Model, clone_model
# from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from keras.utils import plot_model
from keras.optimizers import Adam, SGD
# from keras import regularizers
import math
import tensorflow as tf
import keras_tuner as kt
from keras_tuner import HyperModel
import keras.backend as K
import cupy as cp
# tf.enable_eager_execution()


min_signal_rate = 0.02
max_signal_rate = 0.95

kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 100
ema = 0.999

embedding_dims = 32
embedding_max_frequency = 1000.0


class Encoder_CNN(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='conv1')
        self.conv2 = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu', name='conv2')
        self.conv3 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu', name='conv3')

    def __call__(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Decoder_CNN(Layer):
    def __init__(self,output_shape, **kwargs):
        super().__init__(**kwargs)
        self.conv4 = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu', name='conv4')
        self.conv5 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='conv5')
        self.conv6 = Conv2D(filters=output_shape, kernel_size=(3, 3), padding='same', activation='relu', name='conv6')

    def __call__(self, inputss):
        x = self.conv4(inputss)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class segnet_ae(object):
    model_name = 'segnet'
    VERBOSE = 1

    def __init__(self, encoder, decoder):
        self.model = None
        self.encoder = encoder
        self.decoder = decoder
        self.metric = None
        self.config = None
        self.history =None

    def create_model(self, metric, input_shape, lr):
        input1 = Input(shape=input_shape)
        z = self.encoder(input1)
        reconstruct = self.decoder(z)
        Optimizer = Adam(lr=lr)
        model = Model(input1, reconstruct)
        encoder = Model(input1, z)
        model.compile(optimizer=Optimizer, loss='mae', metrics=[metric])
        # plot_model(model, to_file='model2.png', show_shapes=True)
        print(model.summary())
        return model, encoder

    def fit(self, train_x, val_x, input_shape, epochs=10, metric='accuracy', lr=0.01):
        self.input_shape = input_shape
        # self.output_dim = train_y.shape[1]
        self.metric = metric

        self.model, self.encoder = self.create_model(metric=self.metric, input_shape=self.input_shape, lr=lr)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=100,
                                                    verbose=0, factor=0.5, min_lr=0.00001)
        earlystop = EarlyStopping(monitor='loss', patience=200)

        history = self.model.fit(x=train_x, y=train_x,
                                 epochs=epochs, validation_data=(val_x, val_x),
                                 verbose=self.VERBOSE, callbacks=[learning_rate_reduction, earlystop]).history
        return history

    def encoder_predict(self, test_data):
        return self.encoder.predict(test_data, verbose=1)

    def predict(self, test_data):
        return self.model.predict(test_data, verbose=1)


# diffusion-based
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


def channel_attention(input_feature, ratio=8):
    channel_axis = -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def DenoiseResNet(image_size, channel):
    noisy_images = Input(shape=(image_size[0], image_size[1], channel))

    noise_variances = Input(shape=(1, 1, 1))
    e = Lambda(sinusoidal_embedding)(noise_variances)
    e = UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = channel_attention(noisy_images)
    x = Conv2D(32, kernel_size=1)(x)
    x = Concatenate()([x, e])

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('elu')(x)
    for i in range(5):
        skip = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        skip = BatchNormalization()(skip)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Subtract()([skip, x])  # input - noise
        x = Activation('elu')(x)
        # last layer, Conv
    x = Conv2D(filters=channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)  # noise
    # x = Subtract()([noisy_images, x])  # input - noise
    return Model(inputs=[noisy_images, noise_variances], outputs=x)
    # return Model(inputs=noisy_images, outputs=x)


class ReverseDiffusion(Model):
    def __init__(self, image_size, channel, batch_size=1):  # image_size=[x, x]
        super().__init__()
        self.normalizer = Normalization()
        # self.network = DnCNN(image_size, channel)
        self.network = DenoiseResNet(image_size, channel)
        self.ema_network = clone_model(self.network)
        self.batch_size = batch_size
        self.image_size = image_size
        self.channel = channel

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = tf.keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="i_loss")
        # self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]
        # return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return images  # tf.clip_by_value(images, 0.0, 1.0)

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
        pred_noises = network([noisy_images, noise_rates**2], training=training)  # [x, x, x, channel]
        # pred_noises = network(noisy_images, training=training)  # [x, x, x, channel]
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates  # [x, x, x, channel]
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps, optimize, batch_size=1):
        if optimize == False:
            num_images = initial_noise.shape[0]
        else:
            h = initial_noise.shape[1]
            l = initial_noise.shape[2]
            w = initial_noise.shape[3]
            initial_noise = initial_noise[0, :, :, :]
            initial_noise = tf.reshape(initial_noise, [batch_size, h, l, w])
            num_images = batch_size

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

    def generate(self, images, diffusion_steps, optimize, batch_size=1):
        # noise -> images -> denormalized images
        # images = self.normalizer(images, training=False)
        generated_images = self.reverse_diffusion(images, diffusion_steps, optimize, batch_size)
        # generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):  # add noise to the image to train the denoising network
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size[0], self.image_size[1], self.channel))
        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)  # train the model with noise loss
        # gradients = tape.gradient(image_loss, self.network.trainable_weights)  # train the model with image loss
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size[0], self.image_size[1], self.channel))

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

        return {m.name: m.result() for m in self.metrics}


class hpmodel(HyperModel):
    # decoder for optimizing Diffusion steps
    def __init__(self, rd_encoder, input_size, output_shape):
        super().__init__()
        self.encoder = rd_encoder
        self.input_size = input_size
        self.output_shape = output_shape
        self.metric = None
        self.config = None
        self.history = None

    def build(self, hp):
        diffusion_times = hp.Int(name="diffusion_times", min_value=100, max_value=2000, step=100)
        images = Input(shape=(self.input_size[0], self.input_size[1], self.input_size[2]))
        z = self.encoder.generate(images, diffusion_times)
        x = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(z)
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=self.output_shape, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        model = Model(inputs=images, outputs=x)
        model.compile(loss='mse')
        return model


