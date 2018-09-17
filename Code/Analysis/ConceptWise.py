from sklearn.metrics import *

datafile = 'predictions_mayan.txt'
# datafile = 'predictions_austro.txt'
datafile = 'predictions_ielex.txt'
# datafile = 'preds_ipavsasjp.txt'

numModels = 6
Models = {}
for i in range(numModels):
	Models[i] = {}

Labels = {}
for line in open(datafile):
	line = line.strip().split('\t')
	if line[0] not in Labels:
		Labels[line[0]] = []
	Labels[line[0]].append(int(line[1]))
	for i in range(numModels):
		if line[0] not in Models[i]:
			Models[i][line[0]] = []
		Models[i][line[0]].append(int(line[2+i]))


for concept in Labels:
	print concept, '\t', len(Labels[concept]), '\t', sum(Labels[concept]),'\t',
	for i in range(numModels):
	    p, r, f, _ = precision_recall_fscore_support(Labels[concept], Models[i][concept])
	    if len(f) == 1:
	    	print f[0], '\t',
	    else:
		    print f[1], '\t',
	print ""
