import cv2
import numpy as np

# Load the original image
original_image = cv2.imread('e_noise.jpg', cv2.IMREAD_GRAYSCALE)

# Apply a 3x3 Gaussian filter
filtered_image = cv2.GaussianBlur(original_image, (3, 3), 0)

# Threshold the filtered image to create a binary image
_, binary_image = cv2.threshold(filtered_image, 128, 255, cv2.THRESH_BINARY)

# Find connected components and label them
num_labels, labeled_image = cv2.connectedComponents(binary_image)

# Create a blank image with three channels to draw colored labels
colored_labels = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)

# Assign colors to different labels
for label in range(1, num_labels):
    color = (0, 0, 0)  # Initialize with black
    if label % 2 == 0:
        color = (0, 0, 255)  # Blue for even labels
    else:
        color = (0, 255, 0)  # Green for odd labels

    # Create a mask for the current label and set the color
    mask = labeled_image == label
    colored_labels[mask] = color

# Display the original image, filtered image, and the colored labeled image
cv2.imshow('Original Image', original_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.imshow('Colored Labeled Image', colored_labels)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()




