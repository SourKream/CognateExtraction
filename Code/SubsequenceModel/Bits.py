##########################################

for steal_ratio in [x/100.0 for x in range(0,110,10)]:
	p_label = model[steal_ratio].predict(x_test[steal_ratio])
	p, r, f, _ = precision_recall_fscore_support(y_test, p_label)
	print p[1]*100, "\t", r[1]*100, "\t", f[1], "\t", getAUC(x_test[steal_ratio], y_test, model[steal_ratio])


##########################################

import matplotlib.pyplot as plt
from sklearn.metrics import *

modelA, x_testA, y_test = pickle.load(open('DataDump/HybridModelDyenDataFold1.pkl')) 
modelN, x_testN, _ = pickle.load(open('DataDump/HybridNormModelDyenDataFold1.pkl'))
# modelA, x_testA, y_test = pickle.load(open('DataDump/HybridModelIELexDataFold1.pkl')) 
# modelN, x_testN, _ = pickle.load(open('DataDump/HybridNormModelIELexDataFold1.pkl'))

model = {}
x_test = {}

model['Additive'] = modelA[1.0]
model['Multiplicative'] = modelA[0.0]
model['Hy-Avg 0.2'] = modelA[0.2]
model['Hy-Avg 0.5'] = modelA[0.5]
model['Hy-Norm 0.2'] = modelN[0.2]
model['Hy-Norm 0.5'] = modelN[0.5]

x_test['Additive'] = x_testA[1.0]
x_test['Multiplicative'] = x_testA[0.0]
x_test['Hy-Avg 0.2'] = x_testA[0.2]
x_test['Hy-Avg 0.5'] = x_testA[0.5]
x_test['Hy-Norm 0.2'] = x_testN[0.2]
x_test['Hy-Norm 0.5'] = x_testN[0.5]


for label in sorted(model.keys()):
	p_proba = model[label].decision_function(x_test[label])
	precision, recall, _ = precision_recall_curve(y_test, p_proba)
	plt.plot(recall, precision, label='{0} (AUC = {1:0.2f})'.format(label, auc(recall, precision)))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="upper right")
plt.show(block=False)

##########################################

p_label = {}
for label in sorted(model.keys()):
	p_label[label] = model[label].predict(x_test[label])

with open('Analysis.txt','w') as f:
    f.write('Word 1 \t Word 2 \t Label \t Additive \t Multiplicative \t Hy-Avg 0.5 \t Hy-Avg 0.7 \t Hy-Norm 0.5 \t Hy-Norm 0.7\n')
    for i, pair in enumerate(X_test):
        i = i-1
        f.write(pair[0]+'\t')
        f.write(pair[1]+'\t')
        f.write(str(y_test[i])+'\t')
        for label in ['Additive', 'Multiplicative', 'Hy-Avg 0.5', 'Hy-Avg 0.7', 'Hy-Norm 0.5', 'Hy-Norm 0.7']:
	        f.write(str(p_label['Additive'][i])+'\t')
        f.write('\n')

##########################################

