import cv2
import numpy as np
from matplotlib import pyplot as plt

# Path to your image
img = cv2.imread('data_sample.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to consistent size (optional)
gray = cv2.resize(gray, (224, 224))

# Apply median blur (removes salt-and-pepper noise)
denoised = cv2.medianBlur(gray, 5)

# Enhance contrast using CLAHE (very effective for MRI)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(denoised)

# Normalize pixel values between 0–1
normalized = enhanced / 255.0

# Show the results
plt.figure(figsize=(10,4))
plt.subplot(1,4,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(denoised, cmap='gray')
plt.title('Denoised')
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(enhanced, cmap='gray')
plt.title('CLAHE Enhanced')
plt.axis('off')

plt.show()
