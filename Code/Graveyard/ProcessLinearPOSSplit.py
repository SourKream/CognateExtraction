#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import sys
import codecs
import pickle
import numpy as np
import random
from math import sqrt
from scipy.sparse import csr_matrix as csr
from sklearn.metrics import confusion_matrix
from Utils import *
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

# with open('Pickles/DyenDataset.pkl','r') as fil:
# 	Data = pickle.load(fil)

with open('Pickles/ieLex2016.pkl','r') as fil:
	Data = pickle.load(fil)

with open('Pickles/tags.pkl','r') as fil:
	Tags = pickle.load(fil)

X = {}
Y = {}
for meaning in Data:
	X[meaning] = []
	Y[meaning] = []
	for i in range(len(Data[meaning])):
		for j in range(i+1, len(Data[meaning])):

			if Data[meaning][i][4] != '' and Data[meaning][j][4] != '':
				X[meaning].append((Data[meaning][i][2].lower(), Data[meaning][j][2].lower()))
				if Data[meaning][i][4] == Data[meaning][j][4]:
					Y[meaning].append(1)
				else:
					Y[meaning].append(-1)

words = set([])
for meaning in Data:
	for entry in Data[meaning]:
		words.add(entry[2].lower())

characters = set([])
for word in words:
	for char in word:
		characters.add(char)

V_set = set(['a','e','i','o','u','y'])
C_set = characters.difference(V_set)

CV_words = {}
for word in words:
	CV_word = ""
	for character in word:
		if character in V_set:
			CV_word += "V"
		else:
			CV_word += "C"
	CV_words[word] = CV_word

#########################################################
def feature_for_word_pair(pair):
	global subseq_id
	global use_new_model

	feature = {}

	if use_new_model:
		## New Features
		for subseq in PHI[pair[0]]:
			if subseq in PHI[pair[1]]:
				feature[subseq_id[subseq]] = PHI[pair[0]][subseq] + PHI[pair[1]][subseq]
			else:
				feature[subseq_id[subseq]] = PHI[pair[0]][subseq]
		for subseq in PHI[pair[1]]:
			if subseq not in PHI[pair[0]]:
				feature[subseq_id[subseq]] = PHI[pair[1]][subseq]
	else:
		## Original Features
		for subseq in PHI[pair[0]]:
			if subseq in PHI[pair[1]]:
				feature[subseq_id[subseq]] = PHI[pair[0]][subseq] + PHI[pair[1]][subseq]

	return feature

#########################################################
## Parameters
max_subseq_length = 3
lamb = 0.9
use_new_model = False

# for lamb in [x/10.0 for x in range(1,10)]:

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "LAMBDA : ", lamb, ", P : ", max_subseq_length, ", New Model : ", use_new_model

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
for word in words :

	print "Words Processed : ", counter, " / ", len(words), "\r",
	sys.stdout.flush()
	counter += 1

	feature_vec = {}
	for subseq_length in range(2, max_subseq_length+1):
		for subseq, span in getSubsequencesWithSpan(word, subseq_length):
			feature_vec[subseq] = feature_vec.setdefault(subseq,0) + pow(lamb, span[1]-span[0]+1)
		for subseq, span in getSubsequencesWithSpan(CV_words[word], subseq_length):
			feature_vec[subseq] = feature_vec.setdefault(subseq,0) + pow(lamb, span[1]-span[0]+1)

	sum_of_squares = 0
	for subseq in feature_vec:
		sum_of_squares += feature_vec[subseq]*feature_vec[subseq]
	sum_of_squares = sqrt(sum_of_squares)	
	for subseq in feature_vec:
		feature_vec[subseq] = feature_vec[subseq]/sum_of_squares

	PHI[word] = feature_vec

## Remove Duplicates
Combined = {}
for meaning in X:
	Combined[meaning] = set([])
	for i in range(len(X[meaning])):
		Combined[meaning].add((X[meaning][i], Y[meaning][i]))
	Combined[meaning] = list(Combined[meaning])
	random.shuffle(Combined[meaning])
	X[meaning] = []
	Y[meaning] = []
	for itemx, itemy in Combined[meaning]:
		X[meaning].append(itemx)
		Y[meaning].append(itemy)

## Get Feature Vectors
x = {}
j = 0
for meaning in X:
	x[meaning] = []
	for i in range(len(X[meaning])):
	 	x[meaning].append(feature_for_word_pair(X[meaning][i]))
	j += 1
	print "Progress : ", j, " / ", len(X), "\r",
	sys.stdout.flush()

## K Fold Evaluation
K = 5
meanings = sorted(x.keys())

avg_a = {}
avg_p = {}
avg_r = {}
avg_f = {}
for meaning in meanings:
	avg_a[meaning] = 0
	avg_p[meaning] = 0
	avg_r[meaning] = 0
	avg_f[meaning] = 0

for i in range(K):
	X_test = {}
	x_test = {}
	y_test = {}
	for meaning in meanings:
		X_test[meaning] = []
		x_test[meaning] = []
		y_test[meaning] = []
	X_train = []
	x_train = []
	y_train = []

	## Fold On Meanings
	# for j in range(len(meanings)):
	# 	if j > (len(meanings) * i)/K and j < (len(meanings) * (i+1))/K :
	# 		X_test.extend(X[meanings[j]])			
	# 		x_test.extend(x[meanings[j]])
	# 		y_test.extend(Y[meanings[j]])

	# 	else:
	# 		X_train.extend(X[meanings[j]])
	# 		x_train.extend(x[meanings[j]])
	# 		y_train.extend(Y[meanings[j]])

	## Fold within Meanings and Ensure distinct words in test and train
	for j in range(len(meanings)):
		train_words = set([])

		for item in X[meanings[j]]:
			train_words.add(item[0])
			train_words.add(item[1])

		train_words = list(train_words)
		split = len(train_words)/K
		train_words = set(train_words[:-split])

		for k in range(len(X[meanings[j]])):
			if X[meanings[j]][k][0] not in train_words and X[meanings[j]][k][1] not in train_words:				
				X_test[meanings[j]].append(X[meanings[j]][k])			
				x_test[meanings[j]].append(x[meanings[j]][k])
				y_test[meanings[j]].append(Y[meanings[j]][k])
			elif X[meanings[j]][k][0] in train_words and X[meanings[j]][k][1] in train_words:
				X_train.append(X[meanings[j]][k])
				x_train.append(x[meanings[j]][k])
				y_train.append(Y[meanings[j]][k])

	# ## Fold within Meanings
	# for j in range(len(meanings)):
	# 	split = int(len(x[meanings[j]])/K)
	# 	X_test[meanings[j]].extend(X[meanings[j]][i*split:(i+1)*split])			
	# 	x_test[meanings[j]].extend(x[meanings[j]][i*split:(i+1)*split])
	# 	y_test[meanings[j]].extend(Y[meanings[j]][i*split:(i+1)*split])

	# 	X_train.extend(X[meanings[j]][:i*split] + X[meanings[j]][(i+1)*split:])
	# 	x_train.extend(x[meanings[j]][:i*split] + x[meanings[j]][(i+1)*split:])
	# 	y_train.extend(Y[meanings[j]][:i*split] + Y[meanings[j]][(i+1)*split:])

	print "---------------------------------"
	print "FOLD ", i+1, "\n"


	z = [(x_train[ii],y_train[ii]) for ii in range(len(x_train))]
	random.shuffle(z)
	x_train = [a for a,b in z]
	y_train = [b for a,b in z]


	print "Training on fold ", i+1
	print "Training set size : ", len(x_train)
	print "Positive Labels : ", (len(y_train) + sum(y_train)) / 2, "\n"

	v = DictVectorizer(sparse=True)
	x_train = v.fit_transform(x_train)
	print "Shape of Data : ",
	print x_train.shape
	model = LinearSVC(dual=False, max_iter=1000, penalty='l2', C=35).fit(x_train, y_train)
	# model = LinearSVC(dual=False, max_iter=2000).fit(x_train, y_train)
	p_label = model.predict(x_train)
	showResults(y_train, p_label)

	print "\nTesting on fold ", i+1

	for meaning in meanings:
		# print meaning, "  - "
		p_label = model.predict(v.transform(x_test[meaning]))
		# showResults(y_test[meaning], p_label)

		a, p, r, f = getResults(y_test[meaning], p_label)
		avg_a[meaning] += a
		avg_p[meaning] += p
		avg_r[meaning] += r
		avg_f[meaning] += f

	print "---------------------------------"

print "Average Results"
for meaning in meanings:
	print meaning.upper(),"\t",Tags.setdefault(meaning.upper(),"XX"),"\t",len(Y[meaning]),"\t",(len(Y[meaning]) + sum(Y[meaning]))/2,"\t",avg_a[meaning]/K,"\t",avg_p[meaning]/K,"\t",avg_r[meaning]/K,"\t",avg_f[meaning]/K
	print "---------------------------------"
