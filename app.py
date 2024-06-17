from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import re

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model_bad_sentence_detector.h5')

def preprocess(text, verbose=0):
    # Example preprocessing: convert to lower case and remove non-word characters
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    # Convert characters to ASCII integer values as a simple form of vectorization
    # Adjust the vectorization according to your model's input requirements
    vector = [ord(char) for char in text]
    return vector

# Define the dictionary for class mapping
dict_classes = {
    0: 'VERY NEGATIVE. Hate Speech and Abusive Tweet',
    1: 'NEGATIVE. Hate Speech but NOT Abusive Tweet',
    2: 'NETRAL. NOT Hate Speech but Abusive Tweet',
    3: 'POSITIVE. NOT Hate Speech and NOT Abusive Tweet'
}

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': True,
        'message': 'API is running!'
    }), 200

@app.route('/predict', methods=['POST'])
def predict_text():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({
                'status': False,
                'error': 'No text provided'
            }), 400

        preprocessed_text = preprocess(text, verbose=0)

        # Ensure the vector is of the same length as expected by the model
        max_length = 1000  # Adjust based on your model's expected input size
        if len(preprocessed_text) < max_length:
            preprocessed_text += [0] * (max_length - len(preprocessed_text))
        elif len(preprocessed_text) > max_length:
            preprocessed_text = preprocessed_text[:max_length]

        the_tweet = np.array([preprocessed_text])

        prediction = model.predict(the_tweet, verbose=0)
        classes = np.argmax(prediction, axis=1)

        result = dict_classes[classes[0]]

        return jsonify({
            'status': True,
            'result': result
        }), 200

    except Exception as e:
        return jsonify({
            'status': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
