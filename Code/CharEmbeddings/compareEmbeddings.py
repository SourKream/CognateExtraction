#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import codecs
import sys
import pickle
import numpy as np
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import LinearSVC
from scipy.spatial.distance import *
from scipy.stats import pearsonr, spearmanr

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

## Distance Function
distance_func = cosine
# distance_func = euclidean

## Load embeddings
my_vocab = pickle.load(open('Vocab_539.pkl'))
my_embed = pickle.load(open('Embeddings_539_80.pkl'))

pan_vocab = []  
for line in open('Panphon_all.csv'):                           
	line = line.strip().decode('utf-8').split(',')
	pan_vocab.append(line)  
pan_vocab = pan_vocab[1:]
pan_embed = {i[0]:[int(j) for j in i[1:]] for i in pan_vocab}
pan_vocab = {j[0]:i for i,j in enumerate(pan_vocab)}

tar_vocab = []
for line in open('my_phonetic_features_without_vowel.txt.csv'):
	line = line.strip().decode('utf-8').split('\t')
	tar_vocab.append(line)    
tar_vocab = tar_vocab[1:]
tar_embed = {i[0]:[int(j) for j in i[1:]] for i in tar_vocab}
tar_vocab = {j[0]:i for i,j in enumerate(tar_vocab)}

IPA2ASJP = {}
for line in open('IPA2ASJP.txt'):
	line = line.strip().decode('utf-8').split('\t')
	IPA2ASJP[line[0]] = line[1]

## Load data

char_classes = {}
chars = []
all_classes = set([])
for line in open('Chars.txt'):
	line = line.strip().decode('utf-8').split('\t')
	char_classes[line[0]] = line[1]
	chars.append(line[0])
	all_classes.add(line[1])
all_classes = list(all_classes)

class_dist = {}
for item in all_classes:
	class_dist[item] = {}
for line in open('ClassDistances.txt'):
	line = line.strip().split('\t')
	class_dist[line[0].strip()][line[1].strip()] = float(line[2])
	class_dist[line[1].strip()][line[0].strip()] = float(line[2])


X = []
Y = []
for i in range(len(chars)):
	for j in range(i+1, len(chars)):
		X.append([chars[i], chars[j]])
		if char_classes[chars[i]] == char_classes[chars[j]]:
			Y.append(0)
		else:
			Y.append(class_dist[char_classes[chars[i]]][char_classes[chars[j]]])

## My Dist
my_corr_a = []
my_corr_b = []
for pair,dist in zip(X,Y):
	e1 = my_embed[my_vocab[pair[0]]]
	e2 = my_embed[my_vocab[pair[1]]]
	my_corr_a.append(distance_func(e1,e2))
	my_corr_b.append(dist)

## Panphon Dist
pan_corr_a = []
pan_corr_b = []
for pair,dist in zip(X,Y):
	if pair[0] in pan_embed and pair[1] in pan_embed:
		e1 = np.array(pan_embed[pair[0]])
		e2 = np.array(pan_embed[pair[1]])
		pan_corr_a.append(distance_func(e1,e2))
		pan_corr_b.append(dist)

## Taraka Dist
tar_corr_a = []
tar_corr_b = []
for pair,dist in zip(X,Y):
	if IPA2ASJP[pair[0]] in tar_embed and IPA2ASJP[pair[1]] in tar_embed:
		e1 = np.array(tar_embed[IPA2ASJP[pair[0]]])
		e2 = np.array(tar_embed[IPA2ASJP[pair[1]]])
		tar_corr_a.append(distance_func(e1,e2))
		tar_corr_b.append(dist)

print "Pearson Correlation"
print "My Embed      : ", pearsonr(my_corr_a, my_corr_b)
print "Panphon Embed : ", pearsonr(pan_corr_a, pan_corr_b)
print "Taraka Embed  : ", pearsonr(tar_corr_a, tar_corr_b)

print "Spearman Correlation"
print "My Embed      : ", spearmanr(my_corr_a, my_corr_b)
print "Panphon Embed : ", spearmanr(pan_corr_a, pan_corr_b)
print "Taraka Embed  : ", spearmanr(tar_corr_a, tar_corr_b)

