import os
import gc
import random
import numpy as np
from constants import *
import matplotlib.pyplot as plt

def generate_and_save_images(generator, epoch, test_dataset, num_samples=20, save_dir='content/generated_images'):
    # Ensure the output directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
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
        if i < predictions.shape[0]:
            # Rescale images from [-1, 1] to [0, 1]
            img = (predictions[i] + 1) / 2  
            ax.imshow(img)
            ax.axis('off')
    plt.suptitle(f'Generated Images at Epoch {epoch}')
    
    plt.savefig(os.path.join(save_dir, f'generated_images_epoch_{epoch}.png'))

def generate_random_policy(color_prob=0.30, translation_prob=0, cutout_prob=0):
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

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()