import os
import cv2
import numpy as np
from constants import *

def learning_rate_decay(current_lr, decay_factor=DECAY_FACTOR):
    '''
        Calculate new learning rate using decay factor
    '''
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr):
    '''
        Set new learning rate to optimizers
    '''
    K.set_value(discriminator_optimizer.lr, new_lr)
    K.set_value(generator_optimizer.lr, new_lr)

def generate_and_save_images(generator, epoch, test_dataset, num_samples=20, save_dir='/content/generated_images'):
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