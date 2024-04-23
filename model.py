import tensorflow as tf
from constants import *
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, LayerNormalization, UpSampling2D, LeakyReLU, Dropout, Activation

def squeeze_excite_block(input_tensor, ratio=16):
    init = input_tensor
    channel_axis = -1  # assuming channels-last
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation=LeakyReLU(alpha=0.2), use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid')(se)

    x = layers.multiply([init, se])
    return x

def atrous_conv_block(input_tensor, filters):
    x = layers.Conv2D(filters, 3, activation=LeakyReLU(alpha=0.2), padding='same', dilation_rate=1)(input_tensor)
    x = layers.Conv2D(filters, 3, activation=LeakyReLU(alpha=0.2), padding='same', dilation_rate=2)(x)
    x = layers.Conv2D(filters, 3, activation=LeakyReLU(alpha=0.2), padding='same', dilation_rate=4)(x)
    x = layers.Conv2D(filters, 3, activation=LeakyReLU(alpha=0.2), padding='same')(x)
    return x

def upsample_and_concatenate(x, skip, filters):
    """Upsamples x and concatenates with the skip layer."""
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, skip])
    return x

def define_generator(image_shape=(256, 256, 4)):
    inputs = Input(image_shape)

    filters = [64, 128, 256, 512, 1024]  # Updated filters list to include one more layer

    # Encoder with SE blocks after first three blocks
    c1 = Conv2D(filters[0], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(inputs)
    c1 = squeeze_excite_block(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(filters[1], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(p1)
    c2 = squeeze_excite_block(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(filters[2], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(p2)
    c3 = squeeze_excite_block(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Added an additional convolutional layer here
    #c4 = Conv2D(filters[3], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(p3)
    #c4 = squeeze_excite_block(c4)
    #p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck with atrous convolutions
    bn = atrous_conv_block(p3, filters[3])

    # Decoder with SE blocks after first three blocks and added fourth block
    #d4 = upsample_and_concatenate(bn, c4, filters[3])
    #d4 = Conv2D(filters[3], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(d4)
    #d4 = Conv2D(filters[3], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(d4)

    d3 = upsample_and_concatenate(bn, c3, filters[2])
    d3 = Conv2D(filters[2], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(d3)
    d3 = Conv2D(filters[2], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(d3)

    d2 = upsample_and_concatenate(d3, c2, filters[1])
    d2 = Conv2D(filters[1], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(d2)
    d2 = Conv2D(filters[1], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(d2)

    d1 = upsample_and_concatenate(d2, c1, filters[0])
    d1 = Conv2D(filters[0], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(d1)
    d1 = Conv2D(filters[0], (3, 3), activation=LeakyReLU(alpha=0.2), padding='same')(d1)

    # Final convolution
    outputs = Conv2D(3, (1, 1), activation='tanh')(d1)  # Assuming a 3-channel RGB output
    model = models.Model(inputs, outputs)

    return model

def define_discriminator(image_shape=(256, 256, 3)):
    # Filters
    filters = [32, 64, 128]

    # Input: Image
    in_image = layers.Input(shape=image_shape)

    # First Convolution Block
    d = layers.Conv2D(filters[0], (4, 4), strides=(2, 2), padding='same')(in_image)
    d = layers.LayerNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = Dropout(0.3)(d)

    # Second Convolution Block
    d = layers.Conv2D(filters[1], (4, 4), strides=(2, 2), padding='same')(d)
    d = layers.LayerNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = Dropout(0.3)(d)

    # Third Convolution Block
    d = layers.Conv2D(filters[2], (4, 4), strides=(2, 2), padding='same')(d)
    d = layers.LayerNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = Dropout(0.3)(d)

    out = layers.Flatten()(d)
    out = layers.Dense(1)(out)

    # Define model
    model = models.Model(in_image, out)

    return model

def discriminator_loss(real_output, fake_output, gradient_penalty):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + LAMBDA * gradient_penalty
  
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)
