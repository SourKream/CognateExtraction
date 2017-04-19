from sklearn.metrics import *
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def readFile(filename):
	data = []
	for line in open(filename):
		line = float(line.strip())
		data.append(line)
	return data

folder_path = './PRCurveFiles/IELex_CrossLang/'
# file_dict = {'CoAtt' : 'Coatt.txt', 'CoAtt + Concept' : 'CoattConcept.txt', 'CoAtt + PretAustro' : 'CoattPretAustro.txt', 'CoAtt + PretIELex' : 'CoattPretIElex.txt', 'CharCNN' : 'Char.txt', 'PhoneticCNN' : 'Phonetic.txt', 'Subsequence' : 'Subseq.txt'}
file_dict = {'CoAtt' : 'Coatt.txt', 'CoAtt + Concept' : 'CoattConcept.txt', 'CoAtt + Pret' : 'CoattPretAustro.txt', 'CharCNN' : 'Char.txt', 'PhoneticCNN' : 'Phonetic.txt', 'Subsequence' : 'Subseq.txt', 'Oracle': 'Oracle.txt'}
# file_dict = {'CoAtt + Concept' : 'CoattConcept.txt', 'CharCNN' : 'Char.txt', 'PhoneticCNN' : 'Phonetic.txt', 'Subsequence' : 'Subseq.txt'}

prob_dict = {}
for item in file_dict:
	prob_dict[item] = readFile(folder_path + file_dict[item])
labels = readFile(folder_path + 'Labels.txt')

for item in sorted(prob_dict.keys()):
	precision, recall, thresholds = precision_recall_curve(labels, prob_dict[item])
	plt.plot(recall, precision, label='{0} (AUC = {1:0.2f})'.format(item, auc(recall, precision)))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show(block=False)

