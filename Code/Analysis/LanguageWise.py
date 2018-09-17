from sklearn.metrics import *

datafile = 'lang_predictions_mayan.txt'
datafile = 'lang_predictions_austro.txt'
# datafile = 'lang_predictions_ielex.txt'

numModels = 4
Models = {}
for i in range(numModels):
	Models[i] = {}

Labels = {}
for line in open(datafile):
	line = line.strip().split('\t')
	if line[0] not in Labels:
		Labels[line[0]] = []
	if line[1] not in Labels:
		Labels[line[1]] = []
	Labels[line[0]].append(int(line[2]))
	Labels[line[1]].append(int(line[2]))
	for i in range(numModels):
		if line[0] not in Models[i]:
			Models[i][line[0]] = []
		if line[1] not in Models[i]:
			Models[i][line[1]] = []
		Models[i][line[0]].append(int(line[3+i]))
		Models[i][line[1]].append(int(line[3+i]))


for lang in Labels:
	print lang, '\t', len(Labels[lang]), '\t', sum(Labels[lang]),'\t',
	for i in range(numModels):
	    p, r, f, _ = precision_recall_fscore_support(Labels[lang], Models[i][lang])
	    if len(f) == 1:
	    	print f[0], '\t',
	    else:
		    print f[1], '\t',
	print ""
