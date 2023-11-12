
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

#initialize lemmatizer, load intents, words, classes, and model
lemmatizer = WordNetLemmatizer()
intents = json.loads (open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

#tokenize and lemmatize the sentence
def clean_up_sentence (sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize (word) for word in sentence_words]
    return sentence_words

#find a word in the sentence that we know and mark it in the bag
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#predict the intent with at least a 25% error threshold, sort by strength of the prediction and return the list of predictions
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = .25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

#check for if there are intents predicted, if so get the response
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "I'm sorry, but I'm not sure how to respond to that."


print('Bot is running')
#get input from the user
while True:
    message = input('')
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)