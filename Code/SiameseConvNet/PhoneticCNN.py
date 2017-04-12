"""
Takes two inputs and train a CNN on each and then apply a merge layer and then two dense layers and
classifies them
"""
from collections import defaultdict
from keras.layers import *
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras import backend as K
import itertools as it
import numpy as np
import pickle
np.random.seed(1337)  # for reproducibility
import codecs, sys
from sklearn import metrics
from keras.regularizers import l2

max_word_len = 10
nb_filter = 16
filter_length = 2
nb_epoch = 20
batch_size = 128
tr_threshold = 0.7

def wrd_to_1d(w):
    onehot = []
    for x in w:
        temp = len(unique_chars)*[0]
        if x == "0":
            onehot += temp
        else:
            idx = unique_chars.index(x)
            temp[idx] = 1
            onehot += temp
    return np.array(onehot)

def wrd_to_2d(w):
    w2d = []
    for x in w:
        temp = len(unique_chars)*[0]
        if x == "0":
            w2d.append(temp)
        else:
            idx = unique_chars.index(x)
            temp[idx] = 1
            w2d.append(temp)
    return np.array(w2d).T

def wrd_to_phonetic(w, d):
    w2d = []
    for x in w:
        temp = 16*[0]
        if x not in d:
            w2d.append(temp)
        else:
            temp = d[x]
            w2d.append(temp)
    return np.array(w2d).T

def pad_word(w):
    global max_word_len
    return w.center(max_word_len,"0")[:max_word_len]

############################################
## Prep data

phonetic_features = defaultdict()

f = open("my_phonetic_features_without_vowel.txt.csv", "r")
header = f.readline()
for line in f:
    line = line.replace("\n","")
    arr = line.split("\t")
    character, bin_str = arr[0], arr[1:]
    bin_vec = [int(x) for x in arr[1:]]
    phonetic_features[character] = bin_vec
f.close()

n_dim = len(header.split("\t")) - 1
n_dim = 16

# data_file = 'data/abvd2-part2.tsv.asjp'
# data_file = 'data/Mayan.tsv'
data_file = 'data/IELex-2016.tsv.asjp'
data_file = '../DataPickles/CrossLanguage/Austro/LangInfo_DataFold1.pkl'
# data_file = '../DataPickles/CrossLanguage/Mayan/LangInfo_DataFold1.pkl'
# data_file = '../DataPickles/CrossLanguage/IELex_ASJP/LangInfo_DataFold1.pkl'
use_lang_feat = False

train_pairs, train_labels, test_pairs, test_labels, train_lang_pairs, test_lang_pairs = pickle.load(open(data_file))

languages = set([])
for lang_pair in train_lang_pairs:
    languages.update(lang_pair)
for lang_pair in test_lang_pairs:
    languages.update(lang_pair)
languages = list(languages)
unique_chars = set([])
for word_a, word_b in train_pairs:
    unique_chars.update(list(word_a))
    unique_chars.update(list(word_b))
for word_a, word_b in test_pairs:
    unique_chars.update(list(word_a))
    unique_chars.update(list(word_b))
unique_chars = list(unique_chars)

print len(unique_chars), " CHARACTERS"
print unique_chars

print len(languages), " LANGUAGES"
print languages

train_1 = []
train_2 = []
test_1 = []
test_2 = []
train_lang = []
test_lang = []

for p1, p2 in train_pairs:
    onehotp1, onehotp2 = wrd_to_phonetic(pad_word(p1), phonetic_features), wrd_to_phonetic(pad_word(p2), phonetic_features)
    train_1.append(onehotp1)
    train_2.append(onehotp2)

for p1, p2 in test_pairs:
    onehotp1, onehotp2 = wrd_to_phonetic(pad_word(p1), phonetic_features), wrd_to_phonetic(pad_word(p2), phonetic_features)
    test_1.append(onehotp1)
    test_2.append(onehotp2)

for p1, p2 in train_lang_pairs:
    y = [0]*len(languages)
    idx1 = languages.index(p1)
    idx2 = languages.index(p2)
    y[idx1] = 1
    y[idx2] = 1
    train_lang.append(y)

for p1, p2 in test_lang_pairs:
    y = [0]*len(languages)
    idx1 = languages.index(p1)
    idx2 = languages.index(p2)
    y[idx1] = 1
    y[idx2] = 1
    test_lang.append(y)

train_1 = np.array(train_1)
train_2 = np.array(train_2)

test_1 = np.array(test_1)
test_2 = np.array(test_2)

train_lang = np.array(train_lang)
test_lang = np.array(test_lang)

in_dim = max_word_len*len(unique_chars)
n_langs = len(languages)

print train_1.shape, train_2.shape

print "Random labeling training accuracy %f" %(1.0-np.mean(train_labels))
print "Random labeling test accuracy %f" %(1.0-np.mean(test_labels))
print train_1.shape, train_2.shape
print test_1.shape, test_2.shape

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

train_1 = train_1.reshape(train_1.shape[0], 1, n_dim, max_word_len)
train_2 = train_2.reshape(train_2.shape[0], 1, n_dim, max_word_len)
test_1 = test_1.reshape(test_1.shape[0], 1, n_dim, max_word_len)
test_2 = test_2.reshape(test_2.shape[0], 1, n_dim, max_word_len)

#train_lang = train_lang.reshape(train_lang.shape[0],1,n_langs)
#test_lang = test_lang.reshape(test_lang.shape[0],1,n_langs)

print train_lang.shape

train_1 = train_1.astype('float32')
train_2 = train_2.astype('float32')
test_1 = test_1.astype('float32')
test_2 = test_2.astype('float32')

############################################
## Model

word_1 = Input(shape=(1, n_dim, max_word_len))
word_2 = Input(shape=(1, n_dim, max_word_len))

word_input = Input(shape=(1, n_dim, max_word_len))
word_input_r = Reshape((n_dim, max_word_len, 1))(word_input)
x = Convolution2D(10, 2, 3, input_shape = (1, n_dim, max_word_len))(word_input_r)
x = Convolution2D(10, 2, 3)(x)
x = MaxPooling2D(pool_size=(1, 2))(x)
out = Flatten()(x)
word_model = Model(word_input, out)
word_model.summary()

encoded_1 = word_model(word_1)
encoded_2 = word_model(word_2)
print(word_model.get_output_shape_at(0))

merged_vector = merge([encoded_1, encoded_2],  mode=lambda x: abs(x[0]-x[1]), output_shape=lambda x: x[0])

lang_input = Input(shape=(n_langs, ), name="lang_in")
#lang_dense = Dense(8)(lang_input)
if use_lang_feat:
    y = merge([merged_vector, lang_input], mode = "concat", concat_axis=1)
else:
    y = merged_vector

predictions = Dense(8, activation='relu')(y)
predictions = Dropout(.5)(predictions) 
predictions = Dense(1, activation='sigmoid')(predictions) 

if use_lang_feat:
    model = Model(input=[word_1, word_2, lang_input], output=predictions)
else:
    model = Model(input=[word_1, word_2], output=predictions)
model.summary()

model.compile(optimizer="adadelta", loss='binary_crossentropy', metrics=['accuracy'])
if use_lang_feat:
    model.fit([train_1, train_2, train_lang], train_labels, epochs=nb_epoch, batch_size=batch_size, validation_data=([test_1, test_2, test_lang], test_labels), callbacks=[early_stopping])
else:
    model.fit([train_1, train_2], train_labels, epochs=nb_epoch, batch_size=batch_size, validation_data=([test_1, test_2], test_labels), callbacks=[early_stopping])


#####################################
## Results

if use_lang_feat:
    tr_score = model.predict([train_1, train_2, train_lang], verbose=1)
    te_score = model.predict([test_1, test_2, test_lang], verbose=1)
else:
    tr_score = model.predict([train_1, train_2], verbose=1)
    te_score = model.predict([test_1, test_2], verbose=1)

print("\n\nAverage Precision Score %s " %(metrics.average_precision_score(test_labels, te_score, average="micro")))
c = tr_score > 0.5
b = te_score > 0.5
tr_pred = c.astype('int')
te_pred = b.astype('int')

print("Training")
print(metrics.classification_report(train_labels, tr_pred, digits=3))
print("Testing")
print(metrics.classification_report(test_labels, te_pred, digits=3))
print("Testing Accuracy")
print(metrics.accuracy_score(test_labels, te_pred))

