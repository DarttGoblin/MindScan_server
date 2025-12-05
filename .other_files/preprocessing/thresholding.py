import cv2
import numpy as np
from matplotlib import pyplot as plt

# Path to your image
img = cv2.imread('data_sample.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to a fixed size (optional)
gray = cv2.resize(gray, (224, 224))

# threshold_value = 100
# _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# Apply Otsu thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Normalize (0–1 range)
norm = thresh / 255.0

# Show results
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(norm, cmap='gray')
plt.title('Thresholded')
plt.axis('off')

plt.show()