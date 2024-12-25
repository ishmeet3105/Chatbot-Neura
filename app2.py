from flask import Flask, render_template, request, jsonify
  # Import your chatbot logic from cb.py


import nltk
from keras.src.legacy.saving.legacy_h5_format import load_model_from_hdf5
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import tensorflow as tf
import json
import random

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = tf.keras.models.load_model("my_model.h5")

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # filter out predictions below a threshold
    p = bow(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag']== tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text)
    res = getResponse(ints, intents)
    return res

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page


@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get('message')  # Get the user's message from the request
    if user_message:
        try:
            bot_response = chatbot_response(user_message)  # Get the chatbot's response
        except Exception as e:
            bot_response = "Sorry, I couldn't process your request. Please try again."
    else:
        bot_response = "Please enter a message."

    return jsonify({'response': bot_response})  # Return the bot's response as JSON


