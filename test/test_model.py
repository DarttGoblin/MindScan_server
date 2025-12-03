from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load models
binary_model = load_model("models/mobilenetv2/binary_classifier/binary_brain_mri_model.keras")
multi_model = load_model("models/mobilenetv2/multi_classifier/multi_brain_mri_model.keras")

# Preprocessing function
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Path to your image
img_path = "abnormal_pituitary.jpg"
input_array = preprocess_image(img_path)

# Binary prediction
binary_preds = binary_model.predict(input_array)
binary_confidence = float(np.max(binary_preds))
binary_label = "abnormal" if np.argmax(binary_preds) == 1 else "normal"

print(f"Binary Prediction: {binary_label}, Confidence: {binary_confidence:.4f}")

# If abnormal, pass to multi-class model
if binary_label == "abnormal":
    multi_preds = multi_model.predict(input_array)
    multi_confidence = float(np.max(multi_preds))
    tumor_classes = ['glioma', 'meningioma', 'pituitary']
    multi_label = tumor_classes[np.argmax(multi_preds)]
    print(f"Multi-class Prediction: {multi_label}, Confidence: {multi_confidence:.4f}")