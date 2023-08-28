import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('e_noise.jpg')

# Apply a 3x3 Gaussian filter
filtered_image = cv2.GaussianBlur(image, (3, 3), 0)

# histogram
plt.hist(image.ravel(),256,[0,256])
plt.show()

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
