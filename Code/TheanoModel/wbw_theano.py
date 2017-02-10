import theano
import theano.tensor as T
import numpy
import numpy as np
import pickle as pkl
import cPickle as pickle

from collections import OrderedDict
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = config.floatX

###################################################
##  Model Save/Load

def shared_to_cpu(shared_params, params):
    for k, v in shared_params.iteritems():
        params[k] = v.get_value()

def cpu_to_shared(params, shared_params):
    for k, v in params.iteritems():
        shared_params[k].set_value(v)

def save_model(filename, options, params, shared_params=None):
    if shared_params != None:
        shared_to_cpu(shared_params, params);
    model = OrderedDict()
    model['options'] = options
    model['params'] = params
    pickle.dump(model, open(filename, 'w'))

def load_model(filename):
    model = pickle.load(open(filename, 'r'))
    options = model['options']
    params = model['params']
    shared_params = init_shared_params(params)
    return options, params, shared_params

###################################################
##  Activation Functions

def tanh(x):
    return T.tanh(x)

def relu(x):
    return T.maximum(x, np.float64(0.))

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + T.exp(-x))

###################################################
##  Weight Matrices Initialisations

def init_weight(n, d, options, activation='tanh', ortho = False):
    ''' initialize weight matrix
    options['init_type'] determines
    gaussian or uniform initlizaiton
    '''
    retVal = None
    if options['init_type'] == 'gaussian':
        retVal = (numpy.random.randn(n, d).astype(floatX)) * options['std']
    elif options['init_type'] == 'uniform':
        # [-range, range]
        retVal = ((numpy.random.rand(n, d) * 2 - 1) * \
                options['range']).astype(floatX)
    
    elif options['init_type'] == 'glorot uniform':
        low = -1.0 * np.sqrt(6.0/(n + d))
        high = 1.0 * np.sqrt(6.0/(n + d))
        if activation == 'sigmoid':
            low = low * 4.0
            high = high * 4.0
        retVal = numpy.random.uniform(low,high,(n,d)).astype(floatX)
    if ortho:
        assert n == d
        u, _, _ = numpy.linalg.svd(retVal)
        retVal = u.astype('float64')
    return retVal

###################################################
##  Layer Initialisations

def init_fflayer(params, nin, nout, options, prefix='ff', activation='tanh'):
    ''' initialize ff layer
    '''
    params[prefix + '_w'] = init_weight(nin, nout, options,activation)
    params[prefix + '_b'] = np.zeros(nout, dtype='float64')
    return params

def init_lstm_layer(params, nin, ndim, options, prefix='lstm'):
    ''' initializt lstm layer
    '''
    params[prefix + '_w_x'] = np.concatenate( [init_weight(nin, ndim, options, activation='sigmoid'),
                                              init_weight(nin, ndim, options, activation='sigmoid'),
                                              init_weight(nin, ndim, options, activation='sigmoid'),
                                              init_weight(nin, ndim, options, activation='tanh')], axis = 1)
    # use svd trick to initializ
    if options['init_lstm_svd']:
        params[prefix + '_w_h'] = np.concatenate([init_weight(ndim, ndim, options, activation='sigmoid', ortho = True),
                                                  init_weight(ndim, ndim, options, activation='sigmoid', ortho = True),
                                                  init_weight(ndim, ndim, options, activation='sigmoid', ortho = True),
                                                  init_weight(ndim, ndim, options, activation='tanh', ortho = True)],
                                                 axis=1)
    else:
        params[prefix + '_w_h'] = init_weight(ndim, 4 * ndim, options)
    params[prefix + '_b_h'] = np.zeros(4 * ndim, dtype='float64')
    # set forget bias to be positive
    params[prefix + '_b_h'][ndim : 2*ndim] = np.float64(options.get('forget_bias', 0))
    return params

def init_wbw_att_layer(params, nin, ndim, options, prefix='wbw_attention'):
    '''
        initialize Word by Word layer
    '''
    params[prefix + '_w_y'] = init_weight(ndim,ndim,options)
    params[prefix + '_w_h'] = init_weight(ndim,ndim,options)
    params[prefix + '_w_r'] = init_weight(ndim,ndim,options)
    params[prefix + '_w_alpha'] = init_weight(ndim,1,options)
    params[prefix + '_w_t'] = init_weight(ndim,ndim,options)
    return params

###################################################
##  Parameters Initialisations

def init_params(options):
    ''' Initialize all the parameters
    '''
    params = OrderedDict()
    n_words = options['n_words']
    n_emb = options['n_emb']
    n_dim = options['n_dim']

    ## Embedding weights
    # params['w_emb'] = init_weight(n_words, n_emb, options) ## use the same initialization as BOW
    params['w_emb'] = ((numpy.random.rand(n_words, n_emb) * 2 - 1) * 0.5).astype(floatX)

    ## Independent params
    params['W_p_w'] = init_weight(n_dim, n_dim, options)
    params['W_x_w'] = init_weight(n_dim, n_dim, options)

    ## FF Layers
    params = init_fflayer(params, n_dim, n_dim, options, prefix='word1_Y')
    params = init_fflayer(params, n_dim, 1, options, prefix='final_layer')

    ## Lstm Layers
    params = init_lstm_layer(params, n_emb, n_dim, options, prefix='word1_lstm')
    params = init_lstm_layer(params, n_emb, n_dim, options, prefix='word2_lstm')
    
    ## WBW Attention Layer
    params = init_wbw_att_layer(params, n_dim, n_dim, options)
    
    return params

def init_shared_params(params):
    ''' return a shared version of all parameters
    '''
    shared_params = OrderedDict()
    for k, p in params.iteritems():
        shared_params[k] = theano.shared(params[k], name = k)
    return shared_params

###################################################
##  Layer Definitions

def fflayer(shared_params, x, options, prefix='ff', act_func='tanh'):
    ''' fflayer: multiply weight then add bias
    '''
    return eval(act_func)(T.dot(x, shared_params[prefix + '_w']) + shared_params[prefix + '_b'])

def dropout_layer(x, dropout, trng, drop_ratio=0.5):
    ''' dropout layer
    '''
    x_drop = T.switch(dropout,
                      (x * trng.binomial(x.shape,
                                         p = 1 - drop_ratio,
                                         n = 1,
                                         dtype = x.dtype) \
                       / (numpy.float64(1.0) - drop_ratio)),
                      x)
    return x_drop

def lstm_layer(shared_params, x, mask, h_0, c_0, options, prefix='lstm'):
    ''' lstm layer:
    :param shared_params: shared parameters
    :param x: input, T x batch_size x n_emb
    :param mask: mask for x, T x batch_size
    '''
    # batch_size = optins['batch_size']
    n_dim = options['n_dim']
    # weight matrix for x, n_emb x 4*n_dim (ifoc)
    lstm_w_x = shared_params[prefix + '_w_x']
    # weight matrix for h, n_dim x 4*n_dim
    lstm_w_h = shared_params[prefix + '_w_h']
    lstm_b_h = shared_params[prefix + '_b_h']

    def recurrent(x_t, mask_t, h_tm1, c_tm1):
        ifoc = T.dot(x_t, lstm_w_x) + T.dot(h_tm1, lstm_w_h) + lstm_b_h
        # 0:3*n_dim: input forget and output gate
        i_gate = T.nnet.sigmoid(ifoc[:, 0 : n_dim])
        f_gate = T.nnet.sigmoid(ifoc[:, n_dim : 2*n_dim])
        o_gate = T.nnet.sigmoid(ifoc[:, 2*n_dim : 3*n_dim])
        # 3*n_dim : 4*n_dim c_temp
        c_temp = T.tanh(ifoc[:, 3*n_dim : 4*n_dim])
        # c_t = input_gate * c_temp + forget_gate * c_tm1
        c_t = i_gate * c_temp + f_gate * c_tm1

        if options['use_tanh']:
            h_t = o_gate * T.tanh(c_t)
        else:
            h_t = o_gate * c_t

        # if mask = 0, then keep the previous c and h
        h_t = mask_t[:, None] * h_t + \
              (numpy.float64(1.0) - mask_t[:, None]) * h_tm1
        c_t = mask_t[:, None] * c_t + \
              (numpy.float64(1.0) - mask_t[:, None]) * c_tm1

        return h_t, c_t

    [h, c], updates = theano.scan(fn = recurrent,
                                  sequences = [x, mask],
                                  outputs_info = [h_0[:x.shape[1]],
                                                  c_0[:x.shape[1]]],
                                  n_steps = x.shape[0])
    return h, c

def wbw_attention_layer(shared_params, premise, hypothesis, mask, r_0, options, prefix='wbw_attention',return_final=False):
    ''' wbw attention layer:
    :param shared_params: shared parameters
    :param premise: batch_size x num_regions x n_dim
    :param hypothesis : T x batch_size x n_dim
    :param r_0 : batch_size x n_dim 
    :param mask: mask for x, T x batch_size
    '''
    
    wbw_w_y = shared_params[prefix + '_w_y'] # n_dim x n_dim
    wbw_w_h = shared_params[prefix + '_w_h'] # n_dim x n_dim
    wbw_w_r = shared_params[prefix + '_w_r'] # n_dim x n_dim
    wbw_w_alpha = shared_params[prefix + '_w_alpha'] # n_dim x 1
    wbw_w_t = shared_params[prefix + '_w_t'] # n_dim x n_dim    
    
    def recurrent(h_t, mask_t, r_tm1, Y):
        # h_t : bt_sz x n_dim
        wht = T.dot(h_t, wbw_w_h) # bt_sz x n_dim
        # r_tm1 : bt_sz x n_dim
        wrtm1 = T.dot(r_tm1, wbw_w_r) # bt_sz x n_dim
        tmp = (wht + wrtm1)[:,None,:] # bt_sz x num_regions x n_dim
        WY = T.dot(Y, wbw_w_y) # bt_sz x num_regions x n_dim
        Mt = tanh(WY + tmp) # bt_sz x num_regions x n_dim         
        
        WMt = T.dot(Mt, wbw_w_alpha).flatten(2) # bt_sz x num_regions
        alpha_ret_t = T.nnet.softmax(WMt) # bt_sz x num_region
        alpha_t = alpha_ret_t.dimshuffle((0,'x',1)) # bt_sz x 1 x num_region
        Y_alpha_t = T.batched_dot(alpha_t, Y)[:,0,:] # bt_sz x n_dim
        r_t = Y_alpha_t + T.dot(r_tm1, wbw_w_t) # bt_sz x n_dim        
        
        r_t = mask_t[:, None] * r_t + (numpy.float64(1.0) - mask_t[:, None]) * r_tm1
        return r_t,alpha_ret_t

    [r,alpha], updates = theano.scan(fn = recurrent,
                                  sequences = [hypothesis, mask],
                                  non_sequences=[premise],
                                  outputs_info = [r_0[:hypothesis.shape[1]],None ],
                                  n_steps = hypothesis.shape[0]
                                  )
    if return_final:
        return r[-1], alpha[-1]
    return r,alpha

###################################################
##  Model Definition

def build_model(shared_params, options):
    trng = RandomStreams(1234)
    drop_ratio = options['drop_ratio']
    batch_size = options['batch_size']
    n_dim = options['n_dim']
    dropout = theano.shared(numpy.float64(0.))

    ## Inputs
    word1_idx = T.imatrix('word1_idx')
    word2_idx = T.imatrix('word2_idx')
    word1_mask = T.matrix('word1_mask')
    word2_mask = T.matrix('word2_mask')
    label = T.ivector('label')

    ## Embedding Layer
    word1_emb = shared_params['w_emb'][word1_idx]
    word2_emb = shared_params['w_emb'][word2_idx]

    if options['sent_drop']:
        word1_emb = dropout_layer(word1_emb, dropout, trng, drop_ratio)
        word2_emb = dropout_layer(word2_emb, dropout, trng, drop_ratio)

    ## Encode words with LSTM
    h_0_word1 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float64'))
    c_0_word1 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float64'))

    h_word1, c_word1 = lstm_layer(shared_params, word1_emb, word1_mask,
                                    h_0_word1, c_0_word1, options, prefix='word1_lstm')

    h_0_word2 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float64'))
    c_0_word2 = c_word1[-1]

    h_word2, c_word2 = lstm_layer(shared_params, word2_emb, word2_mask,
                                    h_0_word2, c_0_word2, options, prefix='word2_lstm')
    
    
    ## Separate 1st word
    Y = fflayer(shared_params, h_word1, options, prefix='word1_Y')
    Y = Y.dimshuffle((1,0,2))

    ## Attention Layer
    r_0 = theano.shared(numpy.zeros((batch_size, n_dim), dtype='float64'))
    r, alpha = wbw_attention_layer(shared_params, Y, h_word2, word2_mask, r_0, options, return_final = False)
    # r: T x batch_size x n_dim , alpha : T x batch_size x L

    r = r[-1]    
    h_star = T.tanh( T.dot(r, shared_params['W_p_w']) + T.dot(h_word2[-1], shared_params['W_x_w'] ) )

    ## Output Probability
    prob = fflayer(shared_params, h_star, options, prefix='final_layer', act_func='sigmoid').flatten()
    pred_label = T.round(prob)

    cost = T.nnet.binary_crossentropy(prob, label).mean()
    true_positive = T.sum(pred_label * label)
    pred_positive = T.sum(pred_label) 
    actual_positive = T.sum(label)

    return word1_idx, word2_idx, word1_mask, word2_mask, \
        label, dropout, cost, true_positive, pred_positive, actual_positive, alpha, pred_label, prob
