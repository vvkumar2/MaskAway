# %pip install tensorflow-addons
# %load_ext tensorboard
# import os
# import gc
# import sys
# import glob
# import random
# from random import shuffle
# import cv2
# import zipfile
# import math
# import time
# import datetime
# import numpy as np
# from PIL import Image
# import keras
# from keras.utils import plot_model
# from keras.models import Model
# import tensorflow as tf
# import tensorflow_addons as tfa
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
# from tensorflow.keras import layers, models
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, LayerNormalization, UpSampling2D, LeakyReLU, Dropout, Activation
# from tensorflow.keras.optimizers import Adam, SGD
# sys.path.append("/content/drive/MyDrive/MaskAway")
# import DiffAugment_tf

import os
import zipfile
import random
import gc
import time
import matplotlib.pyplot as plt
import DiffAugment_tf
from constants import *
from helper import *
from preprocessing import *
from model import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras import backend as K



# Create directory if needed
os.makedirs(DATASET_PATH, exist_ok=True)

# Unzip dataset - contains three folders (orig, mask, binary)
with zipfile.ZipFile(BASE_PATH + '/dataset-1500.zip', 'r') as zip_ref:
    zip_ref.extractall(DATASET_PATH)

# Create datasets for training and testing
gan_original_images_paths = get_file_paths(ORIGINAL_IMAGES_PATH)[:DATASET_SIZE]
gan_mask_images_paths = get_file_paths(MASK_IMAGES_PATH)[:DATASET_SIZE]
gan_binary_images_paths = get_file_paths(BINARY_IMAGES_PATH)[:DATASET_SIZE]

# Split into 80/20 training/test sets containing real images, masked images, and binary masks
train_orig_paths, test_orig_paths, train_mask_paths, test_mask_paths, train_binary_paths, test_binary_paths = train_test_split(
    gan_original_images_paths, gan_mask_images_paths, gan_binary_images_paths, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Create datasets for each
train_dataset_gan_len = len(train_orig_paths)
test_dataset_gan_len = len(test_orig_paths)
print(train_dataset_gan_len, test_dataset_gan_len)
train_dataset_gan = create_dataset_gan(train_orig_paths, train_mask_paths, train_binary_paths, GAN_BATCH_SIZE, shuffle=True)
test_dataset_gan = create_dataset_gan(test_orig_paths, test_mask_paths, test_binary_paths, GAN_BATCH_SIZE)

# Define models
IMG_SHAPE = (256, 256, 3)
generator = define_generator()
discriminator = define_discriminator()

# Define optimizers
generator_optimizer = Adam(learning_rate=INIT_LR, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = Adam(learning_rate=INIT_LR, beta_1=0.5, beta_2=0.9)

# For Loss Aggregation
gen_loss_total = []
disc_loss_total = []
val_losses = []

# Checkpoint
ckpt = tf.train.Checkpoint(generator=generator,
                           discriminator=discriminator,
                           G_optimizer=generator_optimizer,
                           D_optimizer=discriminator_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, CKPT_PATH, max_to_keep=5)


def update_learning_rate(current_lr, decay_factor=DECAY_FACTOR):
    '''
        Calculate new learning rate using decay factor
    '''
    new_lr = max(current_lr / decay_factor, MIN_LR)
    generator_optimizer.lr = new_lr
    discriminator_optimizer.lr = new_lr
    return new_lr

def generate_and_save_images(generator, epoch, test_dataset, num_samples=20, save_dir='content/generated_images'):
    # Ensure the output directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize lists to hold input images for generation
    gen_input_list = []

    # Gather the first num_samples from the test dataset
    for real_images, masked_images, binary_masks in test_dataset.take(num_samples):
        gen_input = tf.concat([masked_images, binary_masks], axis=-1)
        gen_input_list.append(gen_input)
    
    # Concatenate all gathered inputs
    gen_input_array = np.vstack(gen_input_list)

    # Generate images
    predictions = generator.predict(gen_input_array)
    
    # Plot the results
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        if i < predictions.shape[0]:  # Check to avoid index error
            img = (predictions[i] + 1) / 2  # Rescale images from [-1, 1] to [0, 1]
            ax.imshow(img)
            ax.axis('off')
    plt.suptitle(f'Generated Images at Epoch {epoch}')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'generated_images_epoch_{epoch}.png'))

def generate_random_policy(color_prob=0.7, translation_prob=0.5, cutout_prob=0.35):
    policy_parts = []
    if random.random() < color_prob:
        policy_parts.append('color')
    if random.random() < translation_prob:
        policy_parts.append('translation')
    if random.random() < cutout_prob:
        policy_parts.append('cutout')
    policy = ','.join(policy_parts)
    return policy

def add_noise_to_inputs(inputs, std_dev=0.15):
    noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=std_dev, dtype=tf.float32)
    return inputs + noise


@tf.function
def WGAN_GP_train_d_step(real_images, masked_images, binary_masks, batch_size, step):
    '''
    One discriminator training step for WGAN-GP
    '''
    # Initialize necessary variables
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    
    # Concatenate masked_images and binary_masks as generator input
    generator_input = tf.concat([masked_images, binary_masks], axis=-1)

    with tf.GradientTape(persistent=True) as d_tape:
        # Generate fake images
        fake_images = generator(generator_input, training=True)

        # Apply augmentation and noise to real and fake images
        policy = generate_random_policy()
        aug_real_images = DiffAugment_tf.DiffAugment(real_images, policy=policy)
        aug_fake_images = DiffAugment_tf.DiffAugment(fake_images, policy=policy)
        aug_real_images = add_noise_to_inputs(aug_real_images)
        aug_fake_images = add_noise_to_inputs(aug_fake_images)

        with tf.GradientTape() as gp_tape:
            interpolated_images = epsilon * aug_real_images + (1 - epsilon) * aug_fake_images
            gp_tape.watch(interpolated_images)
            pred_interpolated = discriminator(interpolated_images, training=True)
        
        # Gradient penalty calculation
        gradients = gp_tape.gradient(pred_interpolated, [interpolated_images])[0]
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0))

        fake_pred = discriminator(aug_fake_images, training=True)
        real_pred = discriminator(aug_real_images, training=True)

        d_loss = discriminator_loss(real_pred, fake_pred, gradient_penalty)

    # Calculate and apply gradients
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    if step % 10 == 0:
      with TRAIN_WRITER.as_default():
        for grad, var in zip(d_gradients, discriminator.trainable_variables):
          tf.summary.histogram(var.name, grad, step=step)
        tf.summary.scalar('d_loss', d_loss, step=step)
      TRAIN_WRITER.flush()

    return d_loss

@tf.function
def WGAN_GP_train_g_step(masked_images, binary_masks, step):
    '''
    One generator training step for WGAN-GP
    '''
    # Concatenate masked_images and binary_masks as generator input
    generator_input = tf.concat([masked_images, binary_masks], axis=-1)

    with tf.GradientTape() as g_tape:
        fake_images = generator(generator_input, training=True)
        fake_pred = discriminator(fake_images, training=True)

        g_loss = generator_loss(fake_pred)

    # Calculate and apply gradients
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    if step % 10 == 0:
      with TRAIN_WRITER.as_default():
        for grad, var in zip(g_gradients, generator.trainable_variables):
          tf.summary.histogram(var.name, grad, step=step)
        tf.summary.scalar('g_loss', g_loss, step=step)
      TRAIN_WRITER.flush()
    
    return g_loss


current_learning_rate = INIT_LR
trace = True
n_critic_count = 0

# Assuming `train_dataset` yields a tuple of (masked_images, binary_masks, real_images_without_mask)
for epoch in range(CURRENT_EPOCH, EPOCHS + 1):
    start = time.time()
    print(f'Start of epoch {epoch}')

    # Reset the loss metrics at the start of the epoch
    epoch_gen_loss_avg = tf.metrics.Mean()
    epoch_disc_loss_avg = tf.metrics.Mean()

    # Using learning rate decay
    current_learning_rate = update_learning_rate(current_learning_rate)
    print(f'current_learning_rate {current_learning_rate}')

    for step, (real_images, masked_images, binary_masks) in enumerate(train_dataset_gan):
        current_batch_size = real_images.shape[0]
        # Train discriminator (critic)
        disc_loss = WGAN_GP_train_d_step(real_images, masked_images, binary_masks, batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))
        epoch_disc_loss_avg.update_state(disc_loss)
        n_critic_count += 1
        
        if n_critic_count >= N_CRITIC:
            # Train generator
            gen_loss = WGAN_GP_train_g_step(masked_images, binary_masks, step=tf.constant(step, dtype=tf.int64))
            epoch_gen_loss_avg.update_state(gen_loss)
            n_critic_count = 0

        if step % 100 == 0:
            print('.', end='')

    # Generate and save images at the end of the epoch
    generate_and_save_images(generator, epoch, test_dataset_gan)

    if epoch % SAVE_EVERY_N_EPOCH == 0:
        # Assuming `ckpt_manager` is set up for checkpointing
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch} at {ckpt_save_path}')

    print(f'\nEpoch {epoch}: Avg Generator Loss: {epoch_gen_loss_avg.result()}, Avg Discriminator Loss: {epoch_disc_loss_avg.result()}')
    print(f'Time taken for epoch {epoch} is {time.time() - start} sec\n')
    gc.collect()

# Save at the final epoch
ckpt_save_path = ckpt_manager.save()
print(f'Saving checkpoint for epoch {EPOCHS} at {ckpt_save_path}')

generator.save('content/gan_models/generator_wgangp')  # SavedModel format
discriminator.save('content/gan_models/discriminator_wgangp')  # SavedModel format
