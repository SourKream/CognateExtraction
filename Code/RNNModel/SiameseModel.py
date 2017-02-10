#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import argparse
import codecs
import logging
import numpy as np
import os
import pdb
import pickle
import sys
import theano

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

np.random.seed(1337)
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.wrappers import *
from keras.layers import *
from sklearn.metrics import *
from Utils import *
from datetime import datetime
from matplotlib import cm

def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-conv', action="store", default=10, dest="conv_dim", type=int)
    parser.add_argument('-epochs', action="store", default=10, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=128, dest="batch_size", type=int)
    parser.add_argument('-xmaxlen', action="store", default=10, dest="xmaxlen", type=int)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-local', action="store", default=False, dest="local", type=bool)
    parser.add_argument('-embd', action="store", default=16, dest='embd_size', type=int)
    parser.add_argument('-tkn_simple', action="store", default=False, dest='tokenize_simple', type=bool)
    opts = parser.parse_args(sys.argv[1:])
    print "conv_dim", opts.conv_dim
    print "epochs", opts.epochs
    print "batch_size", opts.batch_size
    print "xmaxlen", opts.xmaxlen
    print "LR", opts.lr
    print "Embedding Size", opts.embd_size
    print "Tokenize Simple", opts.tokenize_simple
    return opts

def abs_diff(X):
    return K.abs(X[0] - X[1])

def return_first(X):
    return X[0]

def build_model(opts, verbose=False):

    # Conv Output Dimension
    k = opts.conv_dim

    # Word Length
    L = opts.xmaxlen

    ## Conv Filter Size
    m = 3
    h = 2

    ## Max Pooling Filter Size
    s = 3
    t = 2

    print "Word Length : ", L

    input_word_a = Input(shape=(L,), dtype='int32', name="Input Word 1")
    input_word_b = Input(shape=(L,), dtype='int32', name="Input Word 2")

    emb_layer = Embedding(opts.vocab_size, opts.embd_size, input_length = L, name = "Embedding Layer")

    word_a = Reshape((L, opts.embd_size, 1))(Dropout(0.1)(emb_layer(input_word_a)))
    word_b = Reshape((L, opts.embd_size, 1))(Dropout(0.1)(emb_layer(input_word_b)))    
    
    conv_layer = Convolution2D(k, m, h, input_shape=(1, L, opts.embd_size))

    word_a = Flatten()(MaxPooling2D(pool_size=(s, t), strides=(1,1))(conv_layer(word_a)))
    word_b = Flatten()(MaxPooling2D(pool_size=(s, t), strides=(1,1))(conv_layer(word_b)))

    merged_vector = merge([word_a, word_b], mode=abs_diff, output_shape=return_first)

    predictions = Dense(1, activation='sigmoid')(merged_vector)

    model = Model(input=[input_word_a, input_word_b], output=predictions)
    model.summary()

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['precision', 'recall', 'fmeasure'])
    print "Model Compiled"
    return model

def compute_acc(X, Y, model, filename=None):
    scores = model.predict(X)
    plabels = np.round(scores)
    tlabels = np.matrix(Y).transpose()
    # print plabels
    # print tlabels
    p, r, f, _ = precision_recall_fscore_support(tlabels, plabels)

    # if filename != None:
    #     with open(filename, 'w') as f:
    #         for i, sample in enumerate(X):
    #             f.write(map_to_txt(sample, vocab)+"\t")
    #             f.write(str(plabels[i])+"\n")

    return p[1], r[1], f[1]

def getConfig(opts):
    conf = [opts.conv_dim, opts.embd_size, opts.vocab_size, opts.lr, opts.xmaxlen]
    return "_".join(map(lambda x: str(x), conf))

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

if __name__ == "__main__":

    options = get_params()

#    options.local = True
    if options.local:
        dataPath = './Data/Dyen/'
    else:
        dataPath = './Data/IELex/'

    train = [line.strip().decode('utf-8').split('\t') for line in open(dataPath + 'Train.txt')]
    test = [line.strip().decode('utf-8').split('\t') for line in open(dataPath + 'Test.txt')]
    vocab = get_vocab(train, options.tokenize_simple)

    options.vocab_size = len(vocab)
    print "Vocab Size : ", len(vocab)

    X_train, Y_train, labels_train = load_data(train, vocab, tokenize_simple = options.tokenize_simple)
    X_test,  Y_test,  labels_test  = load_data(test,  vocab, tokenize_simple = options.tokenize_simple)
   
    params = {'xmaxlen': options.xmaxlen}
    setattr(K, 'params', params)
   
    XMAXLEN = options.xmaxlen
    X_train = pad_sequences(X_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
    X_test  = pad_sequences(X_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
    Y_train = pad_sequences(Y_train, maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
    Y_test  = pad_sequences(Y_test,  maxlen = XMAXLEN, value = vocab["pad_tok"], padding = 'post')
   
    options.load_save = True
    MODEL_WGHT = './Models/SiameseModel_10_16_539_0.001_10_9.weights'

    if options.load_save and os.path.exists(MODEL_WGHT):
        print("Loading pre-trained model from ", MODEL_WGHT)
        model = build_model(options)
        model.load_weights(MODEL_WGHT)

    else:
        print 'Building model'
        model = build_model(options)

        print 'Training New Model'
        ModelSaveDir = "./Models/SiameseModel_"
        save_weights = WeightSave(ModelSaveDir, getConfig(options))
        metrics_callback = Metrics([X_train, Y_train], labels_train, [X_test, Y_test], labels_test)

        history = model.fit(x = [X_train, Y_train], 
                            y = labels_train,
                            batch_size = options.batch_size,
                            nb_epoch = options.epochs,
                            callbacks = [save_weights, metrics_callback])
