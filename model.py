from constants import *
import tensorflow as tf
from tensorflow.keras import layers, models


def define_generator(image_shape=(64, 64, 4)):
    inputs = layers.Input(shape=image_shape)

    # Initial convolution
    x1 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.LeakyReLU(alpha=0.2)(x1)

    # Downsample
    x2 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(alpha=0.2)(x2)

    x3 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.LeakyReLU(alpha=0.2)(x3)

    # Upscale + skip connections
    x4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.LeakyReLU(alpha=0.2)(x4)
    x4 = layers.Add()([x4, x2])  # Skip connection from x2

    x5 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x4)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.LeakyReLU(alpha=0.2)(x5)
    x5 = layers.Add()([x5, x1])  # Skip connection from x1

    # tanh is best for wgan (from the wgan paper)
    outputs = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x5)

    model = models.Model(inputs, outputs)
    return model

def define_discriminator(image_shape=(64, 64, 3)):
    filters = [64, 64, 128, 256, 512]

    in_image = layers.Input(shape=image_shape)

    d = layers.Conv2D(filters[0], (4, 4), strides=(2, 2), padding='same')(in_image)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Dropout(0.3)(d)  # Dropout for regularization

    d = layers.Conv2D(filters[1], (4, 4), strides=(2, 2), padding='same')(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Dropout(0.3)(d)  # Dropout for regularization

    d = layers.Conv2D(filters[2], (4, 4), strides=(2, 2), padding='same')(d)
    d = layers.LeakyReLU(alpha=0.2)(d)

    d = layers.Conv2D(filters[3], (4, 4), strides=(2, 2), padding='same')(d)
    d = layers.LeakyReLU(alpha=0.2)(d)

    d = layers.Conv2D(filters[4], (4, 4), strides=(2, 2), padding='same')(d)
    d = layers.LeakyReLU(alpha=0.2)(d)

    out = layers.Flatten()(d)
    out = layers.Dense(1)(out)

    model = models.Model(in_image, out)
    return model
    
def discriminator_loss(real_output, fake_output, gradient_penalty):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + LAMBDA * gradient_penalty

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


# UNet model for image segmentation
def UNet(image_shape=(64, 64, 3)):
    inputs = keras.Input(shape=image_shape)

    # Initial convolution
    x1 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.LeakyReLU(alpha=0.2)(x1)

    # Downsample
    x2 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(alpha=0.2)(x2)

    x3 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.LeakyReLU(alpha=0.2)(x3)

    # Upscale + skip connections
    x4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.LeakyReLU(alpha=0.2)(x4)
    x4 = layers.Add()([x4, x2])  # Skip connection from x2

    x5 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x4)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.LeakyReLU(alpha=0.2)(x5)
    x5 = layers.Add()([x5, x1])  # Skip connection from x1

    # Sigmoid output because we want binary masks
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x5)

    model = keras.models.Model(inputs, outputs)
    return model