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

############################################

# data_file = 'data/abvd2-part2.tsv.asjp'
# data_file = 'data/Mayan.tsv'
data_file = 'data/IELex-2016.tsv.asjp'
data_file = '../DataPickles/CrossLanguage/Austro/LangInfo_DataFold1.pkl'
data_file = '../DataPickles/CrossLanguage/Mayan/LangInfo_DataFold1.pkl'
# data_file = '../DataPickles/CrossLanguage/IELex_ASJP/LangInfo_DataFold1.pkl'
data_file = sys.argv[1]
pret_file = sys.argv[2]
use_lang_feat = False

print "Pretraining on ", pret_file
print "Training on ", data_file

train_pairs, train_labels, test_pairs, test_labels, train_lang_pairs, test_lang_pairs = pickle.load(open(data_file))
pret_train_pairs, pret_train_labels, pret_test_pairs, pret_test_labels, pret_train_langpairs, pret_test_langpairs = pickle.load(open(pret_file))

languages = set([])
for llist in [train_lang_pairs, test_lang_pairs, pret_train_langpairs, pret_test_langpairs]:
    for lang_a, lang_b in llist:
        languages.add(lang_a)
        languages.add(lang_b)
languages = list(languages)
unique_chars = set([])
for llist in [train_pairs, test_pairs, pret_train_pairs, pret_test_pairs]:
    for word_a, word_b in llist:
        unique_chars.update(list(word_a))
        unique_chars.update(list(word_b))
unique_chars = list(unique_chars)

print len(unique_chars), " CHARACTERS"
print unique_chars
n_dim = len(unique_chars)+1

print len(languages), " LANGUAGES"
print languages
num_lang = len(languages)

############################################
## Prep data

def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=40, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=128, dest="batch_size", type=int)
    parser.add_argument('-xmaxlen', action="store", default=12, dest="xmaxlen", type=int)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-l2', action="store", default=0.01, dest="l2", type=float)
    parser.add_argument('-dropout', action="store", default=0.1, dest="dropout", type=float)
    parser.add_argument('-local', action="store", default=False, dest="local", type=bool)
    parser.add_argument('-embd', action="store", default=10, dest='embd_size', type=int)
    parser.add_argument('-tkn_simple', action="store", default=False, dest='tokenize_simple', type=bool)
    parser.add_argument('-concept', action="store", default=False, dest='concept', type=bool)
    parser.add_argument('-init_taraka', action="store", default=False, dest='use_init_taraka_embeddings', type=bool)
    parser.add_argument('-ptamount', action="store", default=1.0, dest='pret_amount', type=float)
    opts = parser.parse_args(sys.argv[3:])
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
    print "Initit Embed with Taraka", opts.use_init_taraka_embeddings
    return opts

options = get_params()
options.use_lang_feat = use_lang_feat

train = []
test = []
pret_train = []
pret_test = []
for i in range(len(train_pairs)):
    train.append([train_pairs[i][0], train_pairs[i][1], train_labels[i]])
for i in range(len(test_pairs)):
    test.append([test_pairs[i][0], test_pairs[i][1], test_labels[i]])
print "Fraction of Pretraining Data used : ", options.pret_amount
for i in range((int)(len(pret_train_pairs)*options.pret_amount)):
    pret_train.append([pret_train_pairs[i][0], pret_train_pairs[i][1], pret_train_labels[i]])
for i in range(len(pret_test_pairs)):
    pret_test.append([pret_test_pairs[i][0], pret_test_pairs[i][1], pret_test_labels[i]])
vocab = get_vocab(train+pret_train, options.tokenize_simple)

## Taraka Embeddings
phonetic_features = defaultdict()
n_dim = 16
f = open("TarakaScripts/my_phonetic_features_without_vowel.txt.csv", "r")
header = f.readline()
for line in f:
    line = line.replace("\n","")
    arr = line.split("\t")
    character, bin_str = arr[0], arr[1:]
    bin_vec = [int(x) for x in arr[1:]]
    phonetic_features[character] = bin_vec
f.close()
if options.use_init_taraka_embeddings:
    options.embd_size = n_dim
for item in vocab:
    if item not in phonetic_features:
        phonetic_features[item] = [0 for i in range(n_dim)]
init_embedding_matrix = []
inv_vocab = {vocab[x]:x for x in vocab}
for i in range(len(vocab)):
    init_embedding_matrix.append(phonetic_features[inv_vocab[i]])
init_embedding_matrix = np.array(init_embedding_matrix)


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
pret_X_train, pret_Y_train, pret_labels_train = load_data(pret_train, vocab, tokenize_simple = True)
pret_X_test,  pret_Y_test,  pret_labels_test  = load_data(pret_test,  vocab, tokenize_simple = True)

##SYM
X_train.extend(Y_train)
Y_train.extend(X_train[:len(Y_train)])
labels_train.extend(labels_train)
pret_X_train.extend(pret_Y_train)
pret_Y_train.extend(pret_X_train[:len(pret_Y_train)])
pret_labels_train.extend(pret_labels_train)

## Lang Feat
#lang_vocab = {j:i for i,j in enumerate(languages)}
#lang_train = np.zeros((len(train_lang_pairs), num_lang))
#for i in range(len(train_lang_pairs)):
#    lang_train[i, lang_vocab[train_lang_pairs[i][0]]] = 1
#    lang_train[i, lang_vocab[train_lang_pairs[i][1]]] = 1
#lang_test = np.zeros((len(test_lang_pairs), num_lang))
#for i in range(len(test_lang_pairs)):
#    lang_test[i, lang_vocab[test_lang_pairs[i][0]]] = 1
#    lang_test[i, lang_vocab[test_lang_pairs[i][1]]] = 1

XMAXLEN = options.xmaxlen
X_train = pad_sequences(X_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
X_test  = pad_sequences(X_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
Y_train = pad_sequences(Y_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
Y_test  = pad_sequences(Y_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
pret_X_train = pad_sequences(pret_X_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
pret_X_test  = pad_sequences(pret_X_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
pret_Y_train = pad_sequences(pret_Y_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
pret_Y_test  = pad_sequences(pret_Y_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')

options.vocab_size = len(vocab)
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
                            mask_zero = False,
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
    
    if opts.use_lang_feat:
        input_lang_feat = Input(shape=(opts.num_lang,), dtype='int32', name="Input Lang Feat")
        h_star = concatenate([h_star, input_lang_feat], axis=1)
    
    h_star = Dense(20, activation='tanh', kernel_regularizer=l2(opts.l2), name="Hidden Layer")(h_star)
    output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l2(opts.l2), name="Output Layer")(h_star)

    if opts.use_lang_feat:
        model = Model(inputs = [input_word_a, input_word_b, input_lang_feat], outputs = output_layer)
    else:
        model = Model(inputs = [input_word_a, input_word_b], outputs = output_layer)

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(opts.lr))
    print "Model Compiled"

    att_model = Model(inputs = [input_word_a, input_word_b], outputs = alpha_a_b)

    return model, att_model
    # return model

def getConfig(opts):
    conf = [opts.lstm_units, opts.embd_size, opts.vocab_size, opts.lr, opts.l2, opts.xmaxlen, opts.pret_amount]
    conf = "_".join(map(lambda x: str(x), conf))
    if opts.use_lang_feat:
        conf += '_LangFeat'
    if opts.use_init_taraka_embeddings:
        conf += '_TarakaInit'
    return conf

def compute_acc(X, Y, model, filename=None):
    scores = model.predict(X)
    plabels = np.round(scores)
    tlabels = np.matrix(Y).transpose()
    p, r, f, _ = precision_recall_fscore_support(tlabels, plabels)
    precision, recall, thresholds = precision_recall_curve(tlabels, scores)

    return p[1], r[1], f[1], auc(recall, precision)

class Metrics(Callback):
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x 
        self.test_y = test_y

    def on_epoch_end(self, epochs, logs={}):
        train_pre, train_rec, train_f, train_auc = compute_acc(self.train_x, self.train_y, self.model)
        test_pre, test_rec, test_f, test_auc  = compute_acc(self.test_x, self.test_y, self.model)
        print "\n\nTraining -> Precision: ", train_pre, "\t Recall: ", train_rec, "\t F-Score: ", train_f, "\t AUC: ", train_auc
        print "Testing  -> Precision: ", test_pre,  "\t Recall: ", test_rec,  "\t F-Score: ", test_f, "\t AUC: ", test_auc, "\n"

class WeightSave(Callback):
    def __init__(self, path, config_str):
        self.path = path
        self.config_str = config_str

    def on_epoch_end(self,epochs, logs={}):
        Weights = self.model.get_weights()
        print "Saving To : ", self.path + self.config_str +"_"+ str(epochs) +  ".weights"
        pickle.dump(Weights, open(self.path + self.config_str +"_"+ str(epochs) +  ".weights",'w'))
        # self.model.save_weights(self.path + self.config_str +"_"+ str(epochs) +  ".weights") 

print 'Building model'
model, attention_model = build_model(options)

if options.use_init_taraka_embeddings:
    model.layers[2].set_weights([init_embedding_matrix])

print 'Training New Model'
# ModelSaveDir = "./Models/MAYAN_PretCoAtt_Model_"
ModelSaveDir = "./Models/RE_SYM_" + data_file.split('/')[-1].split('.')[0] + pret_file.split('/')[-1].split('.')[0] + "_PretCoAtt_Model_"
save_weights = WeightSave(ModelSaveDir, getConfig(options))
if use_lang_feat:
    metrics_callback = Metrics([X_train, Y_train, lang_train], labels_train, [X_test, Y_test, lang_test], labels_test)
    train_data = [X_train, Y_train, lang_train]
else:
    metrics_callback = Metrics([X_train, Y_train], labels_train, [X_test, Y_test], labels_test)
    pret_metrics = Metrics([pret_X_train, pret_Y_train], pret_labels_train, [pret_X_test, pret_Y_test], pret_labels_test)
    train_data = [X_train, Y_train]

print "Starting Pretraining..."
print "Training data shape = ", pret_X_train.shape
history = model.fit(x = [pret_X_train, pret_Y_train], y = pret_labels_train, batch_size = options.batch_size, epochs = options.epochs, class_weight = {1:2.0, 0:1.0}, callbacks = [pret_metrics])


print "Starting Training..."
history = model.fit(x = train_data, 
                    y = labels_train,
                    batch_size = options.batch_size,
                    epochs = options.epochs,
                    class_weight = {1:2.0, 0:1.0},
                    callbacks = [metrics_callback, save_weights])

#####################################
## Results

if use_lang_feat:
    tr_score = model.predict([X_train, Y_train, lang_train], verbose=1)
    te_score = model.predict([X_test, Y_test, lang_test], verbose=1)
else:
    tr_score = model.predict([X_train, Y_train], verbose=1)
    te_score = model.predict([X_test, Y_test], verbose=1)

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
