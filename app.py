from flask import Flask, request, jsonify, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from io import BytesIO
from PIL import Image
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model

import os

# Check if the model file exists in the expected path
model_file_path = 'newrecycle.h5'
if os.path.exists(model_file_path):
    print("Model file exists.")
else:
    print("Model file does not exist. Please check the path.")

model = load_model('newrecycle.h5')

# Define image dimensions expected by the model
IMG_WIDTH, IMG_HEIGHT = 150, 150  # Adjust as per your model's input shape




# Route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Process the uploaded image without saving it
        img = Image.open(BytesIO(file.read()))

        # Convert image to RGB mode if it's in RGBA or other mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Preprocess the image for the model
        image = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to match model input shape
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map prediction to the respective class
        class_labels = {0: 'Organic', 1: 'Recyclable'}  # Adjust according to your model
        result = class_labels[predicted_class]

        return jsonify({'prediction': result})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
