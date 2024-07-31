from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import requests
import pickle
import cv2
import re


app = Flask(__name__)
CORS(app)

model_loaded = False
with open('data/processed/vgg19_preds.pkl', 'rb') as file:
    preds = pickle.load(file)  

# Function to preprocess the image
def preprocess_image(img):
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white_bg, img)
    img = img.convert("RGB")
    img = np.array(img)
    img = img[:, :, ::-1]
    img = cv2.resize(img, (200,200))
    img = np.expand_dims(img, axis=0)
    return img

# Function to download image from URL
def download_image(url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    
    return img

@app.route('/wakeup', methods=['GET'])
def wakeup():
    return jsonify({"awake": "true"}), 200
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    url = data['url']
    
    if model_loaded:
        try:
            img = download_image(url)
            img = preprocess_image(img)
            prediction = np.argmax(model.predict(img), axis=1)

            with open('label_encoder.pkl', 'rb') as file:
                encoder = pickle.load(file)   
                
            response = {
                'prediction': encoder.inverse_transform(prediction).tolist()  # Convert prediction to list for JSON serialization
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        match = re.search(r'/(\d+)\.png$', url)
        idx = int(match.group(1))
        response = {
            'prediction': preds[idx]
        }
        return jsonify(response)


if __name__ == '__main__':
    app.run()
