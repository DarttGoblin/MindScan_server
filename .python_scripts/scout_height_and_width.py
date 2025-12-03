import os
from PIL import Image
import matplotlib.pyplot as plt

folder_path = '../brain_mri_dataset/notumor'

widths, heights = [], []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(os.path.join(folder_path, filename))
        w, h = img.size
        widths.append(w)
        heights.append(h)

plt.scatter(widths, heights)
plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Image Sizes in Folder')
plt.show()
