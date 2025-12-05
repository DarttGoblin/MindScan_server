from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load models
binary_model = load_model("models/mobilenetv2/binary_classifier/binary_brain_mri_model.keras")
multi_model = load_model("models/mobilenetv2/multi_classifier/multi_brain_mri_model.keras")
tumor_classes = ['glioma', 'meningioma', 'pituitary']

# Preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    img_file = request.files['image']
    img_bytes = img_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    input_array = preprocess_image(img)

    # Binary prediction
    binary_preds = binary_model.predict(input_array)
    binary_conf = float(np.max(binary_preds))
    binary_label = "tumor" if np.argmax(binary_preds) == 1 else "normal"

    response = {"binary_prediction": binary_label, "binary_confidence": binary_conf}

    # Multi-class prediction if tumor
    if binary_label == "tumor":
        multi_preds = multi_model.predict(input_array)
        multi_conf = float(np.max(multi_preds))
        multi_label = tumor_classes[np.argmax(multi_preds)]
        response.update({"multi_prediction": multi_label, "multi_confidence": multi_conf})

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
