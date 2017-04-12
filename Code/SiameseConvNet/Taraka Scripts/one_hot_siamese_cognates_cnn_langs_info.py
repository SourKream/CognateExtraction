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
np.random.seed(1337)  # for reproducibility
import codecs, sys
from sklearn import metrics
from keras.regularizers import l2

unique_chars = []
languages = []
max_word_len = 10
nb_filter = 16
filter_length = 2
nb_epoch = 10
batch_size = 128
tr_threshold = 0.7

def wrd_to_2d(w):
    w2d = []
    for x in w:
        temp = (len(unique_chars)+1)*[0]
        if x == "0":
            w2d.append(temp)
        else:
            if x in unique_chars:
                idx = unique_chars.index(x)+1
                temp[idx] = 1
                w2d.append(temp)
    return np.array(w2d).T

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

data_file = sys.argv[2]
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
    if len(asjp_word) > max_word_len:
        #print "Exceeded maximum word length %s ",word 
        asjp_word = asjp_word[:max_word_len]
    else:
        asjp_word = asjp_word.center(max_word_len,"0")
    d[concept].append((asjp_word, cogid, lang))
    
f.close()

print len(unique_chars), " CHARACTERS"
print unique_chars

print len(languages), " LANGUAGES"
print languages


############################################
## Prep data

n_dim = len(unique_chars)+1
train_pairs, train_labels, test_pairs, test_labels, train_lang_pairs, test_lang_pairs = make_pairs(d)

train_1 = []
train_2 = []
test_1 = []
test_2 = []
train_lang = []
test_lang = []

for p1, p2 in train_pairs:
    onehotp1, onehotp2 = wrd_to_2d(p1), wrd_to_2d(p2)
    train_1.append(onehotp1)
    train_2.append(onehotp2)

for p1, p2 in test_pairs:
    onehotp1, onehotp2 = wrd_to_2d(p1), wrd_to_2d(p2)
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

print train_1.shape, train_2.shapet

print "Random labeling training accuracy %f" %(1.0-np.mean(train_labels))
print "Random labeling test accuracy %f" %(1.0-np.mean(test_labels))
print train_1.shape, train_2.shape
print test_1.shape, test_2.shape

early_stopping = EarlyStopping(monitor='val_loss', patience=1)

train_1 = train_1.reshape(train_1.shape[0], 1, n_dim, max_word_len)
train_2 = train_2.reshape(train_2.shape[0], 1, n_dim, max_word_len)
test_1 = test_1.reshape(test_1.shape[0], 1, n_dim, max_word_len)
test_2 = test_2.reshape(test_2.shape[0], 1, n_dim, max_word_len)

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
x = Convolution2D(10, n_dim, 2, input_shape = (1, n_dim, max_word_len))(word_input_r)##Uncomment for COLING paper
# x = Convolution2D(10, 2, 3)(x)
x = MaxPooling2D(pool_size=(1, 2))(x)##Uncomment for COLING paper
##x = Dropout(0.25)(x)
out = Flatten()(x)
word_model = Model(word_input, out)
word_model.summary()

encoded_1 = word_model(word_1)
encoded_2 = word_model(word_2)
print(word_model.get_output_shape_at(0))

merged_vector = merge([encoded_1, encoded_2],  mode=lambda x: abs(x[0]-x[1]), output_shape=lambda x: x[0])

lang_input = Input(shape=(n_langs, ), name="lang_in")
y = merge([merged_vector, lang_input], mode = "concat", concat_axis=1)

predictions = Dense(92, activation='relu')(y)
predictions = Dropout(.5)(predictions) 
predictions = Dense(1, activation='sigmoid')(predictions) 

model = Model(input=[word_1, word_2, lang_input], output=predictions)
model.summary()

model.compile(optimizer="adadelta", loss='binary_crossentropy', metrics=['accuracy'])
model.fit([train_1, train_2, train_lang], train_labels, epochs=nb_epoch, batch_size=batch_size, validation_data=([test_1, test_2, test_lang], test_labels), callbacks=[early_stopping])

#####################################
## Results

tr_score = model.predict([train_1, train_2, train_lang], verbose=1)
te_score = model.predict([test_1, test_2, test_lang], verbose=1)

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

