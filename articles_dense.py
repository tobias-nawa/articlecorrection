from __future__ import print_function

import os
import tarfile
import urllib
import zipfile
from collections import Counter

from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.text import base_filter
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import tensorflow as tf
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics.classification import classification_report

seed = 1337
np.random.seed(seed)  # for reproducibility
input_dim = 6
max_lines = 1000
batch_size = 256
nb_epoch = 4
nb_classes = 4 # a, an, the, none
validation_split = 0.2
optim = 'adam'
loss = 'categorical_crossentropy'


def read_data():
    if 'europarl-v7.de-en.en' not in os.listdir('data'):
        print("Downloading corpus (189 MB)...")
        url = 'http://www.statmt.org/europarl/v7/de-en.tgz'
        filename = "data/de-en.tgz"
        urllib.urlretrieve(url, filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall("data/")
        tar.close()
    lines = []
    with open("data/europarl-v7.de-en.en") as f:
        for line in f:
            line_processed = tf.compat.as_str(line).decode('utf-8').encode("utf-8")
            lines.append(line_processed)
            if len(lines) == max_lines:
                break
    return lines


tokenizer = Tokenizer(nb_words=None, filters=base_filter(),
        lower=True, split=" ")
# tokenizer = Tokenizer(nb_words=None)

lines = read_data()
print('Lines:', len(lines))

X = []
Y = []
counter = Counter()
for line in lines:
    words = line.lower().strip().split(" ")
    if len(words) < input_dim:
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

        if len(sentence) == input_dim:
            X.append(" ".join(sentence))
            if not y:
                y = "none"
            Y.append(y)
            break
    if len(X) == max_lines:
        break

tokenizer.fit_on_texts(Y + X)

print("size X:", len(X))
print("size Y:", len(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print(len(X_train), 'train sentences')
print(len(X_test), 'test sentences')
print(len(Y_train), 'train classes')
print(len(Y_test), 'test classes')

c = Counter(Y_train)
print(c.items())

X_train = tokenizer.texts_to_matrix(X_train, mode='binary')
X_test = tokenizer.texts_to_matrix(X_test, mode='binary')

Y_train = tokenizer.texts_to_sequences(Y_train)
Y_test = tokenizer.texts_to_sequences(Y_test)

Y_train_new = []
Y_test_new = []
for y in Y_train:
    Y_train_new.append(y[0])
for y in Y_test:
    Y_test_new.append(y[0])

Y_train = Y_train_new
Y_test = Y_test_new

c = Counter(Y_train)
print(c.items())

nb_classes = np.max(Y_train) + 1

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
Y_train = np_utils.to_categorical(Y_train, nb_classes=nb_classes)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)


print('Building model...')


def create_model():
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1]))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(20000))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optim, metrics=['accuracy'])
    return model


classifier = KerasClassifier(build_fn=create_model, nb_epoch=nb_epoch, batch_size=batch_size)
history = classifier.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
Y_pred = classifier.predict(X_test, batch_size=batch_size)

print(classification_report(y_true=Y_test, y_pred=Y_pred))

plt.figure()
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("data/acc.png")

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("data/loss.png")
