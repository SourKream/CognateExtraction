from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import theano
import theano.tensor as T

class WbwAttentionLayer(Layer):
    def __init__(self, premise_length, **kwargs):
        self.L = premise_length
        super(WbwAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.ndim = input_shape[2]
        self.wbw_w_y = self.add_weight(shape=(self.ndim, self.ndim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.wbw_w_h = self.add_weight(shape=(self.ndim, self.ndim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.wbw_w_r = self.add_weight(shape=(self.ndim, self.ndim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.wbw_w_alpha = self.add_weight(shape=(self.ndim, 1), initializer='glorot_uniform', trainable=True) # n_dim x 1
        self.wbw_w_t = self.add_weight(shape=(self.ndim, self.ndim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim    
        super(WbwAttentionLayer, self).build(input_shape)

    
    def call(self, x, mask=None):
        ''' wbw attention layer:
        :param shared_params: shared parameters
        :param premise: batch_size x num_regions x n_dim
        :param hypothesis : T x batch_size x n_dim
        :param r_0 : batch_size x n_dim 
        '''

        def myLoop(h_t, r_tm1, Y):
            # h_t : bt_sz x n_dim
            wht = T.dot(h_t, self.wbw_w_h) # bt_sz x n_dim
            # r_tm1 : bt_sz x n_dim
            wrtm1 = T.dot(r_tm1, self.wbw_w_r) # bt_sz x n_dim
            tmp = (wht + wrtm1)[:,None,:] # bt_sz x num_regions x n_dim
            WY = T.dot(Y, self.wbw_w_y) # bt_sz x num_regions x n_dim
            Mt = T.tanh(WY + tmp) # bt_sz x num_regions x n_dim         
            
            WMt = T.dot(Mt, self.wbw_w_alpha).flatten(2) # bt_sz x num_regions
            alpha_ret_t = T.nnet.softmax(WMt) # bt_sz x num_region
            alpha_t = alpha_ret_t.dimshuffle((0,'x',1)) # bt_sz x 1 x num_region
            Y_alpha_t = T.batched_dot(alpha_t, Y)[:,0,:] # bt_sz x n_dim
            r_t = Y_alpha_t + T.dot(r_tm1, self.wbw_w_t) # bt_sz x n_dim        
            
            # return r_t,alpha_ret_t
            return r_t    

        premise = x[:,:self.L,:]
        hypothesis = x[:,self.L:,:].dimshuffle((1, 0, 2))

        # [r, alpha], updates = theano.scan(fn = myLoop,
        r, updates = theano.scan(fn = myLoop,
                                      sequences = [hypothesis],
                                      outputs_info = K.zeros_like(premise[:,0,:]),
                                      # outputs_info = [K.zeros_like(premise[:,0,:]), None],
                                      non_sequences = [premise],
                                      n_steps = hypothesis.shape[0]
                                      )

        # return r.dimshuffle((1, 0, 2)), alpha.dimshuffle((1, 0, 2))
        return r.dimshuffle((1, 0, 2))


