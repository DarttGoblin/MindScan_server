from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model("models/mobilenetv2/binary_classifier/binary_brain_mri_model.keras")

# Preprocessing function
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Path to your image
img_path = "abnormal.jpg"
input_array = preprocess_image(img_path)

# Make prediction
preds = model.predict(input_array)
confidence = float(np.max(preds))
label = "abnormal" if np.argmax(preds) == 1 else "normal"

print(f"Prediction: {label}, Confidence: {confidence:.4f}")