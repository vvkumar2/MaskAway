import os
import zipfile
from models import UNet
from utils.constants import *
from utils.preprocessing import get_file_paths, create_dataset, GarbageCollectorCallback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split

# Check if dataset is already extracted
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH, exist_ok=True)
    with zipfile.ZipFile(BASE_PATH + '/dataset-200k.zip', 'r') as zip_ref:
        zip_ref.extractall(DATASET_PATH)

# Get all file paths
original_images_paths = get_file_paths(ORIGINAL_IMAGES_PATH)[0:IMSEG_DATASET_SIZE]
mask_images_paths = get_file_paths(MASK_IMAGES_PATH)[0:IMSEG_DATASET_SIZE]
binary_images_paths = get_file_paths(BINARY_IMAGES_PATH)[0:IMSEG_DATASET_SIZE]

print(len(original_images_paths), len(mask_images_paths), len(binary_images_paths))

# Split into train and temp (which will be further split into validation and test)
train_mask_paths, temp_mask_paths, train_binary_paths, temp_binary_paths = train_test_split(
    mask_images_paths, binary_images_paths, test_size=0.2, random_state=42)

# Split the temp into validation and test
val_mask_paths, test_mask_paths, val_binary_paths, test_binary_paths = train_test_split(
    temp_mask_paths, temp_binary_paths, test_size=0.5, random_state=42)

# Create datasets
train_dataset = create_dataset(train_mask_paths, train_binary_paths, IMSEG_BATCH_SIZE, shuffle=True)
val_dataset = create_dataset(val_mask_paths, val_binary_paths, IMSEG_BATCH_SIZE)
test_dataset = create_dataset(test_mask_paths, test_binary_paths, IMSEG_BATCH_SIZE)

# Create UNet model
model = UNet((IMSEG_IMAGE_HEIGHT, IMSEG_IMAGE_WIDTH, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint_cb = ModelCheckpoint("unet_model_50k_bs16_extraconv.keras", save_best_only=True)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr_cb = ReduceLROnPlateau(factor=0.1, patience=5)
tensorboard_cb = TensorBoard(log_dir=os.path.join("logs", "unet"))
garbage_collector_cb = GarbageCollectorCallback()
callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb, garbage_collector_cb]

# Train the model
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=callbacks,
    batch_size=IMSEG_BATCH_SIZE
)