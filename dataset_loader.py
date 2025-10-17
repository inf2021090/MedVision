import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_segmentation_dataset(
    images_dir,
    masks_dir,
    image_size=(256, 256),
    test_size=0.2,
    to_rgb=False,
    shuffle=True,
):
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))
    assert len(image_files) == len(mask_files), "Images and masks count mismatch."

    images = []
    masks = []

    for img_name, mask_name in zip(image_files, mask_files):
        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, mask_name)

        # Image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_size)
        if to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image.astype(np.float32) / 255.0

        # Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size)
        mask = (mask > 127).astype(np.float32)  # δυαδική μάσκα

        images.append(image)
        masks.append(mask)

    X = np.array(images, dtype=np.float32)
    y = np.array(masks, dtype=np.float32)

    if not to_rgb:
        X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)

    # Train/Test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=shuffle
    )

    # Convert to tf
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = train_ds.shuffle(100).batch(8).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(8).prefetch(tf.data.AUTOTUNE)

    print(f"✅ Loaded {len(X)} samples")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}, Mask shape: {y_train[0].shape}")

    return train_ds, test_ds

