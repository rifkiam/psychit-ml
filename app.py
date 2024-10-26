from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import json

app = Flask(__name__)
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = tf.keras.models.load_model("./model.h5", custom_objects={"TFBertModel": TFBertModel})

def preprocess_text(text, model):
    tok = tokenizer.batch_encode_plus(
        [text],
        max_length = 128,
        pad_to_max_length=True,
        truncation=True
    )

    tok = {
        "text" : tf.convert_to_tensor(tok["input_ids"]),
        "text_mask" : tf.convert_to_tensor(tok["attention_mask"])
    }

    hasil = model.predict(tok)

    return hasil

@app.route("/", methods=["GET"])
def handle():
	return jsonify({"msg": "Halo! Selamat datang di PsychIT"})


@app.route('/predict', methods=['GET'])
def predict():
    try:
        text = request.args.get('text')
        text = preprocess_text(text, model)
        prediction = text
        print("Prediction:", prediction)

        return jsonify({'prediction': json.dumps(str(prediction[0][0]))})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)

