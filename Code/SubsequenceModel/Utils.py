
import matplotlib.pyplot as plt
from sklearn.metrics import *

def showResults (labels, predicted):
	p, r, f, _ = precision_recall_fscore_support(labels, predicted)
	print "Precision : ", p[1]*100, "%"
	print "Recall    : ", r[1]*100, "%"
	print "F-Score   : ", f[1]

def getResults (labels, predicted):
	p, r, f, _ = precision_recall_fscore_support(labels, predicted)
	return p[1]*100, r[1]*100, f[1]

def getPRCurve (x_test, y_test, model):
	p_proba = model.decision_function(x_test)
	precision, recall, thresholds = precision_recall_curve(y_test, p_proba)

	plt.clf()
	plt.plot(recall, precision, label='Precision-Recall curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('Precision-Recall example: AUC={0:0.2f}'.format(auc(recall, precision)))
	plt.show(block=False)

def getPRCurvePersistance (x_test, y_test, model, label='Model'):
	p_proba = model.decision_function(x_test)
	precision, recall, thresholds = precision_recall_curve(y_test, p_proba)
	plt.plot(recall, precision, label='{0} (AUC = {1:0.2f})'.format(label, auc(recall, precision)))

def getPRCurvePersistanceKeras (x_test, y_test, model, label='DL Model'):
	p_proba = model.predict(x_test)
	p_proba = p_proba[:,1]
	precision, recall, thresholds = precision_recall_curve(y_test, p_proba)
	plt.plot(recall, precision, label='{0} (AUC = {1:0.2f})'.format(label, auc(recall, precision)))

def getAUC (x_test, y_test, model):
	p_proba = model.decision_function(x_test)
	precision, recall, thresholds = precision_recall_curve(y_test, p_proba)
	return auc(recall, precision)

def getSubsequences(word, size):
	if size == 1:
		return list(word)
	subseq = []
	for i in range(len(word) - size + 1):
		subseq.extend([word[i] + seq for seq in getSubsequences(word[i+1:], size-1)])
	return subseq

def getSubsequencesWithSpan(word, size):
	if size == 1:
		return [(word[i], (i,i)) for i in range(len(word))]

	subseq = []
	for i in range(len(word) - size + 1):
		subseq.extend([(word[i] + seq, (i, i+1+span[1])) for (seq, span) in getSubsequencesWithSpan(word[i+1:], size-1)])
	return subseq
