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

sys.path.append('/Users/Shantanu/Documents/College/SemVII/BTP/Code/liblinear/python/')

from liblinearutil import *

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

with open('DyenDataset.pkl','r') as fil:
	Data = pickle.load(fil)

with open('tags.pkl','r') as fil:
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

## p = 2
subsequences_2 = set([])
for word in words:
	for i in range(len(word)):
		for j in range(i+1, len(word)):
			subsequences_2.add(word[i]+word[j])

for word in set(CV_words.values()):
	for i in range(len(word)):
		for j in range(i+1, len(word)):
			subsequences_2.add(word[i]+word[j])

# ## p = 3
# subsequences_3 = set([])
# for word in words:
# 	for i in range(len(word)):
# 		for j in range(i+1, len(word)):
# 			for k in range(j+1, len(word)):
# 				subsequences_3.add(word[i]+word[j]+word[k])

# ## p = 4
# subsequences_4 = set([])
# for word in words:
# 	for i in range(len(word)):
# 		for j in range(i+1, len(word)):
# 			for k in range(j+1, len(word)):
# 				for l in range(k+1, len(word)):
# 					subsequences_4.add(word[i]+word[j]+word[k]+word[l])

subsequences_2 = list(subsequences_2)
# subsequences_3 = list(subsequences_3)
# subsequences_4 = list(subsequences_4)

subseq_2_id = {y:x for x,y in enumerate(subsequences_2)}
# subseq_3_id = {y:x for x,y in enumerate(subsequences_3)}
# subseq_4_id = {y:x for x,y in enumerate(subsequences_4)}

lamb = 0.7
PHI = {}
counter = 0
for word in words.union(set(CV_words.values())) :

	print "Words Processed : ", counter, " / ", len(words), "\r",
	sys.stdout.flush()
	counter += 1

	feature_vec = {}
	for i in range(len(word)):
		for j in range(i+1, len(word)):
			if (word[i]+word[j]) in feature_vec:
				feature_vec[word[i]+word[j]] += pow(lamb, j-i+1)
			else:
				feature_vec[word[i]+word[j]] = pow(lamb, j-i+1)

	sum_of_squares = 0
	for subseq in feature_vec:
		sum_of_squares += feature_vec[subseq]*feature_vec[subseq]
	sum_of_squares = sqrt(sum_of_squares)	
	for subseq in feature_vec:
		feature_vec[subseq] = feature_vec[subseq]/sum_of_squares

	PHI[word] = feature_vec

def feature_for_word_pair(pair):

	feature = {}
	for subseq in PHI[pair[0]]:
		if subseq in PHI[pair[1]]:
			feature[subseq_2_id[subseq]] = PHI[pair[0]][subseq] + PHI[pair[1]][subseq]

	for subseq in PHI[CV_words[pair[0]]]:
		if subseq in PHI[CV_words[pair[1]]]:
			feature[subseq_2_id[subseq]] = PHI[CV_words[pair[0]]][subseq] + PHI[CV_words[pair[1]]][subseq]

	# 	else:
	# 		feature[subseq_2_id[subseq]] = PHI[pair[0]][subseq]
	# for subseq in PHI[pair[1]]:
	# 	if subseq not in PHI[pair[1]]:
	# 		feature[subseq_2_id[subseq]] = PHI[pair[1]][subseq]


	return feature

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
K = 3
meanings = sorted(x.keys())

for i in range(K):
	X_test = []
	x_test = []
	y_test = []
	X_train = []
	x_train = []
	y_train = []

	# for j in range(len(meanings)):
	# 	if j > (len(meanings) * i)/K and j < (len(meanings) * (i+1))/K :
	# 		X_test.extend(X[meanings[j]])			
	# 		x_test.extend(x[meanings[j]])
	# 		y_test.extend(Y[meanings[j]])

	# 	else:
	# 		X_train.extend(X[meanings[j]])
	# 		x_train.extend(x[meanings[j]])
	# 		y_train.extend(Y[meanings[j]])

	print "Here 1"
	for meaning in meanings:
		X_train.extend(X[meaning])
		x_train.extend(x[meaning])
		y_train.extend(Y[meaning])

	print "Here 1"
	z = [(x_train[ii],y_train[ii]) for ii in range(len(x_train))]
	print "Here 1"
	random.shuffle(z)
	print "Here 1"
	split = int(len(z) * 0.8)
	x_train = [a for a,b in z[:split]]
	y_train = [b for a,b in z[:split]]
	x_test = [a for a,b in z[split:]]
	y_test = [b for a,b in z[split:]]
	print "Here 1"

	print "---------------------------------"
	print "FOLD ", i+1, "\n"


	# z = [(x_train[ii],y_train[ii]) for ii in range(len(x_train))]
	# random.shuffle(z)
	# x_train = [a for a,b in z]
	# y_train = [b for a,b in z]


	print "Training on fold ", i+1
	print "Training set size : ", len(x_train)
	print "Positive Labels : ", (len(y_train) + sum(y_train)) / 2, "\n"

	model = train(y_train, x_train)
	# p_label, p_acc, p_val = predict(y_train, x_train, model)
	# showResults(y_train, p_label)

	print "\nTesting on fold ", i+1
	print "Testing set size : ", len(x_test)
	print "Positive Labels : ", (len(y_test) + sum(y_test)) / 2, "\n"

	p_label, p_acc, p_val = predict(y_test, x_test, model)
	showResults(y_test, p_label)

	print "---------------------------------"

