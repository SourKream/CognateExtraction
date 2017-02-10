from Utils import *

execfile('WbwAttentionModel.py')

def show_for_word(i):
	w1 = test[i][0]
	w2 = test[i][1]
	l1 = len(tokenize(w1))
	l2 = len(tokenize(w2))

	plot_attention(att[i][-l1:, 1:l2+1], labels=(tokenize(w1), tokenize(w2)))

att = attention_model.predict(net_test)

