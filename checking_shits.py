import numpy as np
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r"E:\datasets\MT\all\segs\makeup\52f7351bb817d8b45fa6aa4007b31181.png"
mask = Image.open(path).convert("L")  # grayscale mode
mask_np = np.array(mask)

unique_values = np.unique(mask_np)
print("Unique class labels in mask:", unique_values)


def visualize_mask_simple(mask_path):
    # Load grayscale mask (0–255 class IDs)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Assign a random color to each unique label
    unique_labels = np.unique(mask)
    color_map = {label: np.random.randint(0, 255, size=3) for label in unique_labels}
    # color_map = {label: np.random.randint(0, 255, size=3) for label in [9]}

    # Create a color image to visualize the mask
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_mask[mask == label] = color

    # Display with matplotlib
    plt.figure(figsize=(6,6))
    plt.imshow(color_mask)
    plt.title(f"Mask with labels: {list(unique_labels)}")
    plt.axis('off')
    plt.show()

visualize_mask_simple(path)