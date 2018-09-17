from Utils import *

execfile('CoAtt.py')

def show_for_word(i):
	w1 = test_pairs[i][0]
	w2 = test_pairs[i][1]
	l1 = len(tokenize(w1))
	l2 = len(tokenize(w2))

	plot_attention(att[i][:l2, :l1], labels=(tokenize(w2,True), tokenize(w1,True)))

att = attention_model.predict([X_test, Y_test], verbose=1)
