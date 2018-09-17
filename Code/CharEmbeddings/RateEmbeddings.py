#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import codecs
import sys
import pickle
import numpy as np
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import LinearSVC
from scipy.spatial.distance import cosine

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

def showResults(labels, predicted):
	p, r, f, _ = precision_recall_fscore_support(labels, predicted)
	print "Precision : ", p[1]*100, "%"
	print "Recall    : ", r[1]*100, "%"
	print "F-Score   : ", f[1]

## Load PanPhon Data
panphon = {}
for line in open('Panphon_all.csv'):
    line = line.strip().split(',')
    panphon[line[0].decode('utf-8')] = line[1:]

for key in panphon:
	for i in range(len(panphon[key])):      
		a = panphon[key][i]
		if a == '+':
			panphon[key][i] = 0
		elif a == '-':
			panphon[key][i] = 1
		else:
			panphon[key][i] = 2
	panphon[key] = np.array(panphon[key])

# ## Extract Set
# temp_set = set([])
# set_index = 17
# set_index_value = 0
# for item in ipa_chars:
# 	if panphon[item][set_index] == set_index_value:
# 		temp_set.add(item)

# Expand Dimension
panphonEmbed = {}
for key in panphon:
	a = panphon[key]
	b = np.zeros((a.shape[0]*3, ))
	for i in range(a.shape[0]):
		b[3*i + a[i]] = 1
	panphonEmbed[key] = b

## Load Embeddings Data
Embed = pickle.load(open('Embeddings_539_80.pkl'))
vocab = pickle.load(open('Vocab_539.pkl'))

my_ipa = set(vocab.keys())
all_ipa = set(panphon.keys())
ipa_chars = set(my_ipa.intersection(all_ipa))
ipa_chars = list(ipa_chars)

for IndexToPredict in range(len(panphon.values()[0])):
	if IndexToPredict not in [19]:
		print '\n',IndexToPredict
		idx = []
		Y = []
		for item in ipa_chars:
			idx.append(vocab[item])
			Y.append(panphon[item][IndexToPredict])
		X = Embed[idx,:]

		counter = {}
		for item in Y:
			counter[item] = counter.setdefault(item, 0) + 1
		print counter

		split = 150
		x_train = X[:split,:]
		y_train = Y[:split]
		x_test = X[split:,:]
		y_test = Y[split:]

		model = LinearSVC().fit(x_train, y_train)
		p_label = model.predict(x_train)
		print cm(y_train, p_label)
		showResults(y_train, p_label)
		p_label = model.predict(x_test)
		print cm(y_test, p_label)
		showResults(y_test, p_label)

## Pairwise Distance
temp_set = list(temp_set)
dist = []
for i in range(len(temp_set)):
    for j in range(i+1, len(temp_set)):
        a = temp_set[i]
        b = temp_set[j]
        dist.append((cosine(panphonEmbed[a], panphonEmbed[b]), cosine(Embed[vocab[a]], Embed[vocab[b]])))


## Diacritic Test
# a - b + c == d?

v = Embed[vocab[a]] - Embed[vocab[b]] + Embed[vocab[c]]
dist = []
for i in range(len(vocab)):
	dist.append(cosine(v, Embed[i]))
D = {}
invV = {vocab[b]:b for b in vocab}
for i in range(len(vocab)):
	D[dist[i]] = invV[i]




