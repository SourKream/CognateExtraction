#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import sys
import codecs
import pickle
import numpy as np
import random
#import matplotlib.pyplot as plt
from math import sqrt
from scipy.sparse import csr_matrix as csr
from sklearn.metrics import confusion_matrix
from Utils import *
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

dataFile = 'DataPickles/Dyen/DataFold1.pkl'
dataFile = 'DataPickles/CrossLanguage/IELex_ASJP/DataFold1.pkl'
#dataFile = 'DataPickles/CrossLanguage/Mayan/DataFold1.pkl'
dataFile = 'DataPickles/CrossLanguage/Austro/DataFold1.pkl'
#dataFile = 'DataPickles/Mayan/DataFold1.pkl'
#dataFile = 'DataPickles/TarakaDF/MAYAN_ConceptDF_Taraka.pkl'

X_train, y_train, X_test, y_test = pickle.load(open(dataFile,'r'))

words = set([])
testWords = set([])
characters = set([])

for dataPoint in X_train:
	words.add(dataPoint[0])
	words.add(dataPoint[1])

for dataPoint in X_test:
	testWords.add(dataPoint[0])
	testWords.add(dataPoint[1])

for word in words:
	for char in word:
		characters.add(char)

V_set = set(['a','e','i','o','u','y'])
C_set = characters.difference(V_set)

CV_words = {}
for word in words.union(testWords):
	CV_word = ""
	for character in word:
		if character in V_set:
			CV_word += "V"
		else:
			CV_word += "C"
	CV_words[word] = CV_word

#########################################################
def featureAdditive(pair):
	feature_add = {}

	for subseq in PHI[pair[0]]:
		feature_add[subseq] = PHI[pair[0]][subseq]
	for subseq in PHI[pair[1]]:
		if subseq in feature_add:
			feature_add[subseq] += PHI[pair[1]][subseq]
		else:
			feature_add[subseq] = PHI[pair[1]][subseq]

	return feature_add

def featureMultiplicative(pair):
	feature_mult = {}

	for subseq in PHI[pair[0]]:
		if subseq in PHI[pair[1]]:
			feature_mult[subseq] = PHI[pair[0]][subseq] + PHI[pair[1]][subseq]

	return feature_mult

def featureHybridNorm(pair, steal_ratio):

	feature = {}

	for subseq in PHI[pair[0]]:
		if subseq in PHI[pair[1]]:
			feature[subseq] = PHI[pair[0]][subseq] + PHI[pair[1]][subseq]

	extra_prob_mass = 0
	for subseq in feature:
		extra_prob_mass += steal_ratio * feature[subseq]
		feature[subseq] *= (1 - steal_ratio)

	weight_sum = 0
	for subseq in PHI[pair[0]]:
		weight_sum += PHI[pair[0]][subseq]
	for subseq in PHI[pair[1]]:
		weight_sum += PHI[pair[1]][subseq]

	for subseq in PHI[pair[0]]:
		feature[subseq] = feature.setdefault(subseq, 0) + PHI[pair[0]][subseq]*extra_prob_mass/weight_sum
	for subseq in PHI[pair[1]]:
		feature[subseq] = feature.setdefault(subseq, 0) + PHI[pair[1]][subseq]*extra_prob_mass/weight_sum

	return feature

def featureHybrid(pair, steal_ratio):

	feature_mult = {}
	feature_add = {}

	for subseq in PHI[pair[0]]:
		if subseq in PHI[pair[1]]:
			feature_mult[subseq] = PHI[pair[0]][subseq] + PHI[pair[1]][subseq]

	for subseq in PHI[pair[0]]:
		feature_add[subseq] = PHI[pair[0]][subseq]
	for subseq in PHI[pair[1]]:
		if subseq in feature_add:
			feature_add[subseq] += PHI[pair[1]][subseq]
		else:
			feature_add[subseq] = PHI[pair[1]][subseq]

	feature = {}
	for subseq in feature_add:
		if subseq in feature_mult:
			feature[subseq] = steal_ratio*feature_add[subseq] + (1-steal_ratio)*feature_mult[subseq]
		else:
			feature[subseq] = steal_ratio*feature_add[subseq]

	return feature

#########################################################
## Parameters
max_subseq_length = 3
lamb = 0.7

# for lamb in [x/10.0 for x in range(1,10)]:
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "LAMBDA : ", lamb, ", P : ", max_subseq_length

subsequences = set([])
for word in words:
	for subseq_length in range(2, max_subseq_length+1):
		for subseq in getSubsequences(word,subseq_length):
			subsequences.add(subseq)

for word in set(CV_words.values()):
	for subseq_length in range(2, max_subseq_length+1):
		for subseq in getSubsequences(word,subseq_length):
			subsequences.add(subseq)

subsequences = list(subsequences)
subseq_id = {y:x for x,y in enumerate(subsequences)}

PHI = {}
counter = 0

for word in words.union(testWords):

	print "Words Processed : ", counter, " / ", len(words.union(testWords)), "\r",
	sys.stdout.flush()
	counter += 1

	feature_vec = {}
	for subseq_length in range(2, max_subseq_length+1):
		for subseq, span in getSubsequencesWithSpan(word, subseq_length):
			if subseq in subseq_id:
				feature_vec[subseq_id[subseq]] = feature_vec.setdefault(subseq_id[subseq],0) + pow(lamb, span[1]-span[0]+1)
		for subseq, span in getSubsequencesWithSpan(CV_words[word], subseq_length):
			if subseq in subseq_id:
				feature_vec[subseq_id[subseq]] = feature_vec.setdefault(subseq_id[subseq],0) + pow(lamb, span[1]-span[0]+1)

	sum_of_squares = 0
	for subseq in feature_vec:
		sum_of_squares += feature_vec[subseq]*feature_vec[subseq]
	sum_of_squares = sqrt(sum_of_squares)	
	for subseq in feature_vec:
		feature_vec[subseq] = feature_vec[subseq]/sum_of_squares

	PHI[word] = feature_vec

print "---------------------------------"
#########################################################

## Get Feature Vectors
x_train = []
j = 0
for dataPoint in X_train:
 	x_train.append(featureMultiplicative(dataPoint))
	j += 1
	print "Progress : ", j, " / ", len(X_train), "\r",
	sys.stdout.flush()

x_test = []
for dataPoint in X_test:
 	x_test.append(featureMultiplicative(dataPoint))


# z = zip(x_train, y_train)
# random.shuffle(z)
# x_train = [a for a,b in z]
# y_train = [b for a,b in z]


v = DictVectorizer(sparse = True)
x_train = v.fit_transform(x_train)

print "Shape of Data     : ", x_train.shape
print "Positive Labels   : ", sum(y_train)

model = LinearSVC().fit(x_train, y_train)
p_label = model.predict(x_train)
showResults(y_train, p_label)

print "Testing set size : ", len(x_test)
print "Positive Labels  : ", sum(y_test)

x_test = v.transform(x_test)
p_label = model.predict(x_test)
showResults(y_test, p_label)
print "PR AUC  : ", getAUC(x_test, y_test, model)

#	getPRCurvePersistance (x_test, y_test, model)
print "---------------------------------"

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.show(block=False)
