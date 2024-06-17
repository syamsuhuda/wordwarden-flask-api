from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re

app = Flask(__name__)

model = load_model('model_bad_sentence_detector.h5')

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "keras.preprocessing.text":
            module = "tensorflow.keras.preprocessing.text"
        return super().find_class(module, name)

def load_tokenizer(path):
    with open(path, 'rb') as handle:
        tokenizer = CustomUnpickler(handle).load()
    return tokenizer

tokenizer = load_tokenizer('tokenizer.pickle')

def preprocess(text, verbose=0):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

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

        seq_tweet = tokenizer.texts_to_sequences([preprocessed_text])
        max_length = 1000  
        the_tweet = pad_sequences(seq_tweet, padding='post', maxlen=max_length, truncating='post')

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
