import cv2
from matplotlib import pyplot as plt

img_path = "data_sample.jpg"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(100, 100))
clahe_img = clahe.apply(img)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(clahe_img, cmap='gray')
plt.title("After CLAHE")
plt.axis("off")

plt.show()
