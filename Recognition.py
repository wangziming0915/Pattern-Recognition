import cv2
import numpy as np

# Load the original image
original_image = cv2.imread('e_noise.jpg', cv2.IMREAD_COLOR)

# Apply a 3x3 Gaussian filter
filtered_image = cv2.GaussianBlur(original_image, (3, 3), 0)

# Convert the filtered image to grayscale for contour detection
filtered_gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

# Threshold the filtered grayscale image to create a binary image
_, binary_image = cv2.threshold(filtered_gray, 128, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image with three channels to draw colored labels
colored_labels = np.zeros_like(original_image)

# Initialize a counter for different shapes
shape_counter = 0

# Process each contour
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Define colors and labels for different shapes
    if len(approx) == 3:
        color = (0, 0, 255)  # Red for triangles
        label = "Triangle"
    elif len(approx) == 4:
        color = (0, 255, 0)  # Green for rectangles
        label = "Square"
    elif len(approx) >= 5:
        color = (255, 0, 0)  # Blue for circles and other shapes
        label = "Circle"
    else:
        color = (0, 0, 0)    # Black (undefined shape)
        label = "Undefined"

    # Draw the contour filled with the corresponding color
    cv2.drawContours(colored_labels, [contour], -1, color, -1)

    # Add the label text next to the shape
    x, y, w, h = cv2.boundingRect(contour)
    cv2.putText(colored_labels, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the original image with recognized shapes and labels
cv2.imshow('Original Image with Recognized Shapes', colored_labels)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()



