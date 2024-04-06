import tensorflow as tf
from helper import *
from constants import *

def get_file_paths(directory):
    # Get all file paths
    paths = [os.path.join(directory, file) for file in os.listdir(directory) if file != '.DS_Store']
    return sorted(paths, key=lambda x: x.split('/')[-1])

def preprocess_image(image_path, is_binary=False, size=IMAGE_SIZE):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if is_binary else cv2.IMREAD_COLOR)
    if not is_binary:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image
    image = cv2.resize(image, size)

    # Normalize the image
    image = image / 255.0

    # If binary, add channel dimension
    if is_binary:
        image = np.expand_dims(image, axis=-1)

    return image

def random_flip(mask_img, binary_img):
    return tf.image.flip_left_right(mask_img), tf.image.flip_left_right(binary_img)

def apply_with_probability(prob, func, *args):
    should_apply = tf.random.uniform((), maxval=1) < prob
    return tf.cond(
        should_apply,
        lambda: func(*args),
        lambda: args if len(args) > 1 else args[0]
    )

def augment_images(mask_img, binary_img):
    # Augmentations that apply to both images
    mask_img, binary_img = apply_with_probability(0.3, random_flip, mask_img, binary_img)

    # Augmentations that apply only to the mask image
    mask_img = apply_with_probability(0.25, lambda x: tf.image.random_brightness(x, max_delta=0.1), mask_img)
    mask_img = apply_with_probability(0.25, lambda x: tf.image.random_contrast(x, lower=0.9, upper=1.1), mask_img)
    mask_img = apply_with_probability(0.2, lambda x: tf.image.random_hue(x, max_delta=0.1), mask_img)
    mask_img = apply_with_probability(0.2, lambda x: tf.image.random_saturation(x, lower=0.8, upper=1.2), mask_img)
    mask_img = apply_with_probability(0.1, lambda x: x + tf.cast(tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1), x.dtype), mask_img)

    return mask_img, binary_img

def image_generator(mask_paths, binary_paths):
    for mask_path, binary_path in zip(mask_paths, binary_paths):
        mask_img = preprocess_image(mask_path)
        binary_img = preprocess_image(binary_path, is_binary=True)

        # Apply consistent augmentation to both
        mask_img, binary_img = augment_images(mask_img, binary_img)

        yield (mask_img, binary_img)

def create_dataset(mask_paths, binary_paths, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(mask_paths, binary_paths),
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), dtype=tf.float32)
        )
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(mask_paths))
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

def image_generator_gan(orig_paths, mask_paths, binary_paths):
    for orig_path, mask_path, binary_path in zip(orig_paths, mask_paths, binary_paths):
        mask_img = preprocess_image(mask_path)
        binary_img = preprocess_image(binary_path, is_binary=True)
        orig_img = preprocess_image(orig_path)

        # Apply consistent augmentation to both
        # mask_img, binary_img = augment_images(mask_img, binary_img)

        yield (orig_img, mask_img, binary_img)

def create_dataset_gan(orig_paths, mask_paths, binary_paths, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator_gan(orig_paths, mask_paths, binary_paths),
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), dtype=tf.float32)
        )
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(mask_paths))
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)