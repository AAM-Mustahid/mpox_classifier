import os
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

# Create an upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('M:/Thesis/monkeypox_classifier/models/EfficientNetB3_monkeypox.keras')

# Class names
class_names = ['Chickenpox', 'Cowpox', 'Healthy', 'Measles', 'Monkeypox', 'Smallpox']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        # Save the file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load and process the image
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.array(img_array, dtype=np.uint8)  # Ensure original data type

        # Get the prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]  # Get the predicted class

        # Convert image to base64 for frontend preview
        with open(file_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({"prediction": predicted_class, "image_base64": image_base64})

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
