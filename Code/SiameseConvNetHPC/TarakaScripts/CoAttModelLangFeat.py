#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from collections import defaultdict
from keras.layers import *
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn import metrics
from keras.regularizers import l2
import itertools as it
import numpy as np
import codecs, sys
import argparse
import logging
import numpy as np
import os
import pdb
import pickle
import theano
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.callbacks import *
from keras.constraints import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.normalization import BatchNormalization
from keras.layers import *
from sklearn.metrics import *
from Utils import *
from datetime import datetime
from matplotlib import cm
from AttentionLayer import *

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

np.random.seed(1337)

unique_chars = []
languages = []
max_word_len = 10
nb_filter = 16
filter_length = 2
nb_epoch = 20
batch_size = 128
tr_threshold = 0.7

def make_pairs(d):
    tr_labels = []
    tr_pairs = []
    tr_lang_pairs = []
    te_labels = []
    te_pairs = []
    te_lang_pairs = []
    concepts = d.keys()
    n_concepts = len(concepts)
    print "No. of concepts %d" %(n_concepts)
    # len_concepts = int(sys.argv[1])
    len_concepts = int(n_concepts * 0.7)
    tr_concepts = concepts[:len_concepts]
    te_concepts = concepts[len_concepts:]
    print "No. of training concepts %d testing concepts %d" %(len(tr_concepts),len(te_concepts))
    for concept in tr_concepts:
        for i1, i2 in it.combinations(d[concept], r=2):
            w1, g1, l1 = i1
            w2, g2, l2 = i2
            if l1 != l2:
                tr_pairs.append((w1,w2))
                tr_lang_pairs.append((l1, l2))
                if g1 == g2:
                    tr_labels.append(1)
                else:
                    tr_labels.append(0)
    for concept in te_concepts:
        for i1, i2 in it.combinations(d[concept], r=2):
            w1, g1, l1 = i1
            w2, g2, l2 = i2
            if l1 != l2:
                te_pairs.append((w1,w2))
                te_lang_pairs.append((l1, l2))
                if g1 == g2:
                    te_labels.append(1)
                else:
                    te_labels.append(0)

    return (tr_pairs, tr_labels, te_pairs, te_labels, tr_lang_pairs, te_lang_pairs)

data_file = sys.argv[1]
#data_file = 'data/abvd2-part2.tsv.asjp'
#data_file = 'data/Mayan.tsv'
#data_file = 'data/IELex-2016.tsv.asjp'
d = defaultdict(list)

f = codecs.open(data_file,"r", encoding="utf-8")
f.readline()
for line in f:
    line = line.strip()
    arr = line.split("\t")
    lang = arr[0]
    if lang not in languages:
        languages.append(lang)
    concept = arr[3]
    cogid = arr[6]
    cogid = cogid.replace("-","")
    cogid = cogid.replace("?","")
    asjp_word = arr[5].split(",")[0]
    asjp_word = asjp_word.replace(" ", "")
    #tokenized_word = ipa2tokens(asjp_word)
    #asjp_word = "".join(tokens2class(tokenized_word, 'asjp'))
    asjp_word = asjp_word.replace("%","")
    asjp_word = asjp_word.replace("~","")
    asjp_word = asjp_word.replace("*","")
    asjp_word = asjp_word.replace("$","")
    asjp_word = asjp_word.replace("K","k")
    asjp_word = asjp_word.replace("\"","")
    if len(asjp_word) < 1:
        continue
    for x in asjp_word:
        if x not in unique_chars:
            unique_chars.append(x)
    # if len(asjp_word) > max_word_len:
    #     #print "Exceeded maximum word length %s ",word 
    #     asjp_word = asjp_word[:max_word_len]
    # else:
    #     asjp_word = asjp_word.center(max_word_len,"0")
    d[concept].append((asjp_word, cogid, lang))
    
f.close()

print len(unique_chars), " CHARACTERS"
print unique_chars

print len(languages), " LANGUAGES"
print languages

############################################
## Prep data

def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=30, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=128, dest="batch_size", type=int)
    parser.add_argument('-xmaxlen', action="store", default=12, dest="xmaxlen", type=int)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-l2', action="store", default=0.02, dest="l2", type=float)
    parser.add_argument('-dropout', action="store", default=0.1, dest="dropout", type=float)
    parser.add_argument('-local', action="store", default=False, dest="local", type=bool)
    parser.add_argument('-embd', action="store", default=12, dest='embd_size', type=int)
    parser.add_argument('-tkn_simple', action="store", default=False, dest='tokenize_simple', type=bool)
    parser.add_argument('-concept', action="store", default=False, dest='concept', type=bool)
    opts = parser.parse_args(sys.argv[2:])
    print "lstm_units", opts.lstm_units
    print "epochs", opts.epochs
    print "batch_size", opts.batch_size
    print "xmaxlen", opts.xmaxlen
    print "regularization factor", opts.l2
    print "dropout", opts.dropout
    print "LR", opts.lr
    print "Embedding Size", opts.embd_size
    print "Tokenize Simple", opts.tokenize_simple
    print "Using Concept Fold Data", opts.concept
    return opts

options = get_params()

train_pairs, train_labels, test_pairs, test_labels, train_lang_pairs, test_lang_pairs = make_pairs(d)

train = []
test = []
for i in range(len(train_pairs)):
    train.append([train_pairs[i][0], train_pairs[i][1], train_labels[i]])
for i in range(len(test_pairs)):
    test.append([test_pairs[i][0], test_pairs[i][1], test_labels[i]])
vocab = get_vocab(train, options.tokenize_simple)

def load_data(train, vocab, labels = {'0':0,'1':1,0:0,1:1}, tokenize_simple = False):
    X,Y,Z = [],[],[]
    for p,h,l in train:
        p = map_to_idx(tokenize(p, tokenize_simple), vocab)
        h = map_to_idx(tokenize(h, tokenize_simple), vocab)
        if l in labels:         
            X += [p]
            Y += [h]
            Z += [labels[l]]
    return X,Y,Z

X_train, Y_train, labels_train = load_data(train, vocab, tokenize_simple = True)
X_test,  Y_test,  labels_test  = load_data(test,  vocab, tokenize_simple = True)

## Lang Feat
num_lang = len(languages)
lang_vocab = {j:i for i,j in enumerate(languages)}
lang_train = np.zeros((len(train_lang_pairs), num_lang))
for i in range(len(train_lang_pairs)):
    lang_train[i, lang_vocab[train_lang_pairs[i][0]]] = 1
    lang_train[i, lang_vocab[train_lang_pairs[i][1]]] = 1
lang_test = np.zeros((len(test_lang_pairs), num_lang))
for i in range(len(test_lang_pairs)):
    lang_test[i, lang_vocab[test_lang_pairs[i][0]]] = 1
    lang_test[i, lang_vocab[test_lang_pairs[i][1]]] = 1

XMAXLEN = options.xmaxlen
X_train = pad_sequences(X_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
X_test  = pad_sequences(X_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
Y_train = pad_sequences(Y_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
Y_test  = pad_sequences(Y_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')

options.vocab_size = len(vocab)
options.num_lang = len(languages)
print "Vocab Size : ", len(vocab)

############################################
## Model

def get_last(X):
    # get last element from time dimension
    return X[:, -1, :]

def build_model(opts, verbose=False):

    input_word_a = Input(shape=(opts.xmaxlen,), dtype='int32', name="Input Word A")
    input_word_b = Input(shape=(opts.xmaxlen,), dtype='int32', name="Input Word B")

    emb_layer = Embedding(opts.vocab_size, 
                            opts.embd_size,
##                            W_constraint = unitnorm(),
                            input_length = opts.xmaxlen,
                            mask_zero = True,
                            name = "Embedding Layer")    
    emb_dropout_layer = SpatialDropout1D(opts.dropout)

    lstm_layer = Bidirectional(LSTM(opts.lstm_units,
                                    return_sequences = True, 
                                    name="LSTM Layer"), name="Bidir LSTM Layer")
    lstm_dropout_layer = SpatialDropout1D(opts.dropout, name="LSTM Dropout Layer")

    word_a = lstm_dropout_layer (lstm_layer (emb_dropout_layer (emb_layer (input_word_a))))
    word_b = lstm_dropout_layer (lstm_layer (emb_dropout_layer (emb_layer (input_word_b))))

    attention_layer = WbwAttentionLayer(return_att=True, name="Attention Layer")

    r_a, alpha_a_b = attention_layer ([word_a, word_b])
    r_b, alpha_b_a = attention_layer ([word_b, word_a])

    k = 2 * opts.lstm_units
    r_a_n = Lambda(get_last, output_shape=(k,), name="r_a_n")(r_a)
    r_b_n = Lambda(get_last, output_shape=(k,), name="r_b_n")(r_b)

    h_star = Activation('tanh')(concatenate([r_a_n, r_b_n], axis=1))
    input_lang_feat = Input(shape=(opts.num_lang,), dtype='int32', name="Input Lang Feat")

    h_star = concatenate([h_star, input_lang_feat], axis=1)
    h_star = Dense(20, activation='tanh', kernel_regularizer=l2(opts.l2), name="Hidden Layer")(h_star)
    output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l2(opts.l2), name="Output Layer")(h_star)

    model = Model(inputs = [input_word_a, input_word_b, input_lang_feat], outputs = output_layer)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(opts.lr))
    print "Model Compiled"

    att_model = Model(inputs = [input_word_a, input_word_b], outputs = alpha_a_b)

    return model, att_model
    # return model

def getConfig(opts):
    conf = [opts.lstm_units, opts.embd_size, opts.vocab_size, opts.lr, opts.l2, opts.xmaxlen]
    if opts.concept:
        concept = '_Concept'
    else:
        concept = ''
    return "_".join(map(lambda x: str(x), conf)) + concept

def compute_acc(X, Y, model, filename=None):
    scores = model.predict(X)
    plabels = np.round(scores)
    tlabels = np.matrix(Y).transpose()
    p, r, f, _ = precision_recall_fscore_support(tlabels, plabels)
    return p[1], r[1], f[1]

class Metrics(Callback):
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x 
        self.test_y = test_y

    def on_epoch_end(self, epochs, logs={}):
        train_pre, train_rec, train_f = compute_acc(self.train_x, self.train_y, self.model)
        test_pre, test_rec, test_f  = compute_acc(self.test_x, self.test_y, self.model)
        print "\n\nTraining -> Precision: ", train_pre, "\t Recall: ", train_rec, "\t F-Score: ", train_f
        print "Testing  -> Precision: ", test_pre,  "\t Recall: ", test_rec,  "\t F-Score: ", test_f, "\n"


class WeightSave(Callback):
    def __init__(self, path, config_str):
        self.path = path
        self.config_str = config_str

    def on_epoch_end(self,epochs, logs={}):
        self.model.save_weights(self.path + self.config_str +"_"+ str(epochs) +  ".weights") 

print 'Building model'
model, attention_model = build_model(options)

print 'Training New Model'
ModelSaveDir = "./Models/MAYAN_CoAtt_Model_"
# save_weights = WeightSave(ModelSaveDir, getConfig(options))
metrics_callback = Metrics([X_train, Y_train, lang_train], labels_train, [X_test, Y_test, lang_test], labels_test)

history = model.fit(x = [X_train, Y_train, lang_train], 
                    y = labels_train,
                    batch_size = options.batch_size,
                    epochs = options.epochs,
                    class_weight = {1:2.0, 0:1.0},
                    callbacks = [metrics_callback])

#####################################
## Results

tr_score = model.predict([X_train, Y_train, lang_train], verbose=1)
te_score = model.predict([X_test, Y_test, lang_test], verbose=1)

print("\n\nAverage Precision Score %s " %(metrics.average_precision_score(labels_test, te_score, average="micro")))
c = tr_score > 0.5
b = te_score > 0.5
tr_pred = c.astype('int')
te_pred = b.astype('int')

print("Training")
print(metrics.classification_report(labels_train, tr_pred, digits=3))
print("Testing")
print(metrics.classification_report(labels_test, te_pred, digits=3))
print("Testing Accuracy")
print(metrics.accuracy_score(labels_test, te_pred))

#####################################
#####################################
