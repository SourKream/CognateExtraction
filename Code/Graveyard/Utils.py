
from sklearn.metrics import confusion_matrix

def showResults (labels, predicted):
	c = confusion_matrix(labels, predicted, [1, -1])
	print "Confusion Matrix : "
	print c
	p = float(c[0][0])/(c[0][0]+c[1][0])
	r = float(c[0][0])/(c[0][0]+c[0][1])
	f = 2*p*r/(p+r)
	a = float(c[0][0] + c[1][1])/(c[0][0]+c[1][0]+c[0][1]+c[1][1])
	print "Accuracy  : ", a*100, "%"
	print "Precision : ", p*100, "%"
	print "Recall    : ", r*100, "%"
	print "F-Score   : ", f

def getResults (labels, predicted):
	c = confusion_matrix(labels, predicted, [1, -1])
	a = float(c[0][0] + c[1][1])/(c[0][0]+c[1][0]+c[0][1]+c[1][1]+1)
	p = float(c[0][0])/(c[0][0]+c[1][0]+0.001)
	r = float(c[0][0])/(c[0][0]+c[0][1]+0.001)
	f = 2*p*r/(p+r+0.001)
	return a*100, p*100, r*100, f

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
