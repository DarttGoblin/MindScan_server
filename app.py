from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the model
model = load_model("models/mobilenetv2/binary_classifier/binary_brain_mri_model.keras")

# Define preprocessing
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
    
    image_file = request.files['image']
    img_bytes = image_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    input_array = preprocess_image(img)

    preds = model.predict(input_array)
    confidence = float(np.max(preds))
    label = "tumor" if np.argmax(preds) == 1 else "normal"
    
    return jsonify({"prediction": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)


# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    img_bytes = image_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
    
    label = "tumor" if pred_class.item() == 1 else "normal"
    return jsonify({"prediction": label, "confidence": confidence.item()})

if __name__ == "__main__":
    app.run(debug=True)
