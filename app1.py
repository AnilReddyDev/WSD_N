from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    return text  # You can add more preprocessing steps here

# Function to get BERT embedding for the target word
def get_target_word_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    target_word_index = next(i for i, token in enumerate(tokens) if target_word in token)
    with torch.no_grad():
        outputs = model(**inputs)
    target_word_embedding = outputs.last_hidden_state[0, target_word_index]
    return target_word_embedding.numpy()

# API endpoint to predict word sense
@app.route('/predict', methods=['POST'])
def predict_sense():
    data = request.json
    sentence = data['sentence']
    target_word = data['target_word']

    # Preprocess and get embedding
    processed_sentence = preprocess_text(sentence)
    embedding = get_target_word_embedding(processed_sentence, target_word)
    
    # Here you would use your trained model to predict the sense
    # For simplicity, let's return the embedding as a placeholder
    response = {"embedding": embedding.tolist()}  # Convert numpy array to list for JSON response
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
