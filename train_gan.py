import os
import zipfile
import random
import gc
import time
import matplotlib.pyplot as plt
import DiffAugment_tf
from utils.constants import *
from utils.helper import *
from utils.preprocessing import *
from models import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras import backend as K


# Check if dataset is already extracted
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH, exist_ok=True)
    with zipfile.ZipFile(BASE_PATH + '/dataset-200k.zip', 'r') as zip_ref:
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
train_dataset_gan = create_dataset_gan(train_orig_paths, train_mask_paths, train_binary_paths, GAN_BATCH_SIZE, train=True, shuffle=True)
test_dataset_gan = create_dataset_gan(test_orig_paths, test_mask_paths, test_binary_paths, GAN_BATCH_SIZE)

# Define models
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

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    latest_epoch = int(ckpt_manager.latest_checkpoint.split('-')[1])
    CURRENT_EPOCH = latest_epoch * SAVE_EVERY_N_EPOCH + 1
    print ('Latest checkpoint of epoch {} restored!!'.format(CURRENT_EPOCH))


def update_learning_rate(current_lr, decay_factor=DECAY_FACTOR):
    '''
        Calculate new learning rate using decay factor
    '''
    new_lr = max(current_lr / decay_factor, MIN_LR)
    generator_optimizer.lr = new_lr
    discriminator_optimizer.lr = new_lr
    return new_lr

def generate_random_policy(color_prob=0.5, translation_prob=0.35, cutout_prob=0.25):
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
def WGAN_GP_train_d_step(real_images, masked_images, binary_masks, batch_size, step, epoch):
    '''
    One discriminator training step for WGAN-GP
    '''
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 0.1)

    # Concatenate masked_images and binary_masks as generator input
    generator_input = tf.concat([masked_images, binary_masks], axis=-1)

    with tf.GradientTape(persistent=True) as d_tape:
        fake_images = generator(generator_input, training=True)

        # Apply augmentation and noise to real and fake images
        policy = generate_random_policy()
        aug_real_images = DiffAugment_tf.DiffAugment(real_images, policy=policy)
        aug_fake_images = DiffAugment_tf.DiffAugment(fake_images, policy=policy)
        aug_real_images = add_noise_to_inputs(aug_real_images, epoch)
        aug_fake_images = add_noise_to_inputs(aug_fake_images, epoch)

        with tf.GradientTape() as gp_tape:
            interpolated_images = epsilon * aug_real_images + (1 - epsilon) * aug_fake_images
            gp_tape.watch(interpolated_images)
            pred_interpolated = discriminator(interpolated_images, training=True)

        # Gradient penalty calculation
        gradients = gp_tape.gradient(pred_interpolated, [interpolated_images])[0]
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        # Using updated gradient penalty formula
        gradient_penalty = tf.reduce_mean(tf.square(tf.clip_by_value(gradients_norm - 1.0, 0., np.infty)))


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


# Initialize training variables
current_learning_rate = INIT_LR
n_critic_count = 0

# `train_dataset` yields a tuple of (masked_images, binary_masks, real_images_without_mask)
for epoch in range(CURRENT_EPOCH, EPOCHS + 1):
    start = time.time()
    print(f'Start of epoch {epoch}')

    # Reset the loss metrics at the start of the epoch
    epoch_gen_loss_avg = tf.metrics.Mean()
    epoch_disc_loss_avg = tf.metrics.Mean()

    # Learning rate decay
    current_learning_rate = update_learning_rate(current_learning_rate)
    print(f'current_learning_rate {current_learning_rate}')

    for step, (real_images, masked_images, binary_masks) in enumerate(train_dataset_gan):
        current_batch_size = real_images.shape[0]
        # Train critic
        disc_loss = WGAN_GP_train_d_step(real_images, masked_images, binary_masks, batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64), epoch=epoch)
        epoch_disc_loss_avg.update_state(disc_loss)
        n_critic_count += 1

        if n_critic_count >= N_CRITIC:
            # Train generator
            gen_loss = WGAN_GP_train_g_step(masked_images, binary_masks, step=tf.constant(step, dtype=tf.int64))
            epoch_gen_loss_avg.update_state(gen_loss)
            n_critic_count = 0

        if step % 100 == 0:
            print('.', end='')

    # Write summaries for tensorboard
    with TRAIN_WRITER.as_default():
        tf.summary.scalar('epoch_generator_loss', epoch_gen_loss_avg.result(), step=epoch)
        tf.summary.scalar('epoch_discriminator_loss', epoch_disc_loss_avg.result(), step=epoch)
        TRAIN_WRITER.flush()

    # Generate and save images at the end of the epoch
    generate_and_save_images(generator, epoch, test_dataset_gan)

    if epoch % SAVE_EVERY_N_EPOCH == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch} at {ckpt_save_path}')

    print(f'\nEpoch {epoch}: Avg Generator Loss: {epoch_gen_loss_avg.result()}, Avg Discriminator Loss: {epoch_disc_loss_avg.result()}')
    print(f'Time taken for epoch {epoch} is {time.time() - start} sec\n')

    gc.collect()


# Save at the very end
ckpt_save_path = ckpt_manager.save()
print(f'Saving checkpoint for epoch {EPOCHS} at {ckpt_save_path}')

generator.save('content/drive/MyDrive/MaskAway/generator_wgangp')
discriminator.save('content/drive/MyDrive/MaskAway/discriminator_wgangp')
