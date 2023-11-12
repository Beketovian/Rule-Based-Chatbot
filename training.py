import random
import json
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer #This is for lemmatizing words. For example: eat, eaten, eating would all be considered as the same word
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer() #initialize lemmatizer

intents = json.loads(open('intents.json').read()) #load intents that I made from json file

#initialization of lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

#go through each intent and each pattern within that intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #tokenize each word. For example in the sentence "I like to eat" it would tokenize each word: "I", "like", "to", "eat"
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
# print(documents)
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters] #lemmatize and filter out ignored letters

#remove duplicates and sort classes
words = sorted(set(words)) 
classes = sorted(set(classes))

#save words and classes into picle files to be used later
pickle.dump(words, open('words.pkl', 'wb')) #save the words into a pkl file, write as binary
pickle.dump(classes, open('classes.pkl', 'wb'))
# print(words)

#initialize training set and output
training = []
output_empty = [0] * len(classes)

#prepare the training set
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

#shuffle training data and convert to array
random.shuffle(training)
training = np.array(training, dtype=object)

#split training set into input (patterns) and output (intents)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

#define the basic sequential model from Keras
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True) #define sgd w/ .1 learning rate. Values can be fine tuned
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics='accuracy') #compile the model using categorical crossentropy for multi-class classification

#train and save the model.
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Done")