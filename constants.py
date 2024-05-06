import os
import datetime
import tensorflow as tf

# Directory Paths
BASE_PATH = 'data'
DATASET_PATH = 'content/dataset'
ORIGINAL_IMAGES_PATH = DATASET_PATH + '/orig'
MASK_IMAGES_PATH = DATASET_PATH + '/mask'
BINARY_IMAGES_PATH = DATASET_PATH + '/binary'

# Constants
IMAGE_SIZE = (64, 64)
IMAGE_HEIGHT = IMAGE_SIZE[0]
IMAGE_WIDTH = IMAGE_SIZE[1]

RANDOM_STATE = 42
GAN_BATCH_SIZE = 16
TEST_SIZE = 0.2
DATASET_SIZE = 25000

LAMBDA = 10
NOISE_DIM = 100

MODEL_NAME="DCGAN"
INIT_LR = 0.0001
MIN_LR = 0.000001
DECAY_FACTOR=1.00004
EPOCHS = 5000
N_CRITIC = 5
N_CRITIC_UPDATE = 75
SAVE_EVERY_N_EPOCH = 1
CURRENT_EPOCH = 1
CKPT_PATH = os.path.join("checkpoints", "tensorflow", MODEL_NAME)
TRAIN_LOGDIR = os.path.join("logs", "tensorflow", MODEL_NAME, 'train_data') # Sets up a log directory.

CUR_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TRAIN_LOGDIR = 'logs/' + CUR_TIME + '/train'
TRAIN_WRITER = tf.summary.create_file_writer(TRAIN_LOGDIR)
