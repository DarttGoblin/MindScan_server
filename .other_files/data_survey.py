import os
import matplotlib.pyplot as plt

data_dir = "brain_mri_dataset"

classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
class_counts = [len(os.listdir(os.path.join(data_dir, c))) for c in classes]

plt.figure(figsize=(8, 6))
plt.bar(classes, class_counts)
plt.title("Distribution of MRI Image Classes")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.show()