from dataset_loader import load_segmentation_dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Ορισμός φακέλων
    IMAGES_DIR = "~/Projects/Left/COVID/images"
    MASKS_DIR = "~/Projects/Left/COVID/masks"

    # Φόρτωση dataset
    train_ds, test_ds = load_segmentation_dataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        image_size=(256, 256),
        to_rgb=False
    )

    # Δείξε ένα batch για έλεγχο
    for images, masks in train_ds.take(1):
        img = images[0].numpy().squeeze()
        msk = masks[0].numpy().squeeze()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Image")

        plt.subplot(1, 2, 2)
        plt.imshow(msk, cmap="gray")
        plt.title("Mask")

        plt.show()
        break

