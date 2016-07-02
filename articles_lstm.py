from __future__ import print_function
import zipfile

from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import base_filter
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import tensorflow as tf
import numpy as np
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical

from sklearn.cross_validation import train_test_split

np.random.seed(1337)  # for reproducibility
max_sentence_length = 6
max_lines = 10000
batch_size = 16
nb_epoch = 5

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    lines = []
    with open(filename) as f:
        for line in f:
            line_processed = tf.compat.as_str(line).decode('utf-8').encode("utf-8")
            lines.append(line_processed)
            if len(lines) == max_lines:
                break
    return lines

tokenizer = Tokenizer(nb_words=None, filters=base_filter(),
        lower=True, split=" ")
# tokenizer = Tokenizer(nb_words=None)

filename = "data/europarl-v7.de-en.en"
lines = read_data(filename)
print('Lines:', len(lines))

X = []
Y = []
for line in lines:
    words = line.lower().strip().split(" ")
    if len(words) < max_sentence_length:
        continue
    sentence = []
    y = None
    for index, word in enumerate(words):
        word = word.strip()
        if len(word) == 0:
            continue
        if word in ["a", "an", "the"]:
            if not y:
                y = word
            else:
                sentence.append(word)
        else:
            sentence.append(word)

        if len(sentence) == max_sentence_length:
            X.append(" ".join(sentence))
            if not y:
                y = "none"
            Y.append(y)
            break
    if len(X) == max_lines:
        break


tokenizer.fit_on_texts(X + Y)

print("size X:", len(X))
print("size Y:", len(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print(len(X_train), 'train sentences')
print(len(X_test), 'test sentences')
print(len(Y_train), 'train classes')
print(len(Y_test), 'test classes')

print('Vectorizing sequence data...')
Y_train = tokenizer.texts_to_sequences(Y_train)
Y_test = tokenizer.texts_to_sequences(Y_test)

nb_classes = np.max(Y_train) + 1

Y_train_new = []
Y_test_new = []
for y in Y_train:
    Y_train_new.append(y[0])
for y in Y_test:
    Y_test_new.append(y[0])
Y_train = np.array(Y_train_new)
Y_test = np.array(Y_test_new)

X_train = tokenizer.texts_to_matrix(X_train, mode='binary')
X_test = tokenizer.texts_to_matrix(X_test, mode='binary')

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print(X_train[0:10])
print(Y_train[0:10])

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)


print('Building model...')

model = Sequential()

model.add(Embedding(6, 256, input_length=X_train.shape[1]))
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True,
                  validation_split=0.1)

score = model.evaluate(X_test, Y_test, batch_size=batch_size)

print(score)
