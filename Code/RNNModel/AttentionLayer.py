from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import theano
import theano.tensor as T

class WbwAttentionLayer(Layer):
    def __init__(self, return_att=False, **kwargs):
        self.return_att = return_att
        super(WbwAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ndim = input_shape[0][2]
        self.wbw_w_y = self.add_weight(shape=(self.ndim, self.ndim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.wbw_w_h = self.add_weight(shape=(self.ndim, self.ndim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.wbw_w_r = self.add_weight(shape=(self.ndim, self.ndim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim
        self.wbw_w_alpha = self.add_weight(shape=(self.ndim, 1), initializer='glorot_uniform', trainable=True) # n_dim x 1
        self.wbw_w_t = self.add_weight(shape=(self.ndim, self.ndim), initializer='glorot_uniform', trainable=True) # n_dim x n_dim    
        super(WbwAttentionLayer, self).build(input_shape)

    
    def call(self, input_tensors, mask=None):
        ''' wbw attention layer:
        :param premise (input_tensors[0]) : batch_size x T x n_dim
        :param hypothesis (input_tensors[1]) : T x batch_size x n_dim
        '''

        if mask == None:
            print "NO MASKING"
            def attentionLoop(h_t, r_tm1, Y):
                # h_t : bt_sz x n_dim
                # r_tm1 : bt_sz x n_dim

                wht = T.dot(h_t, self.wbw_w_h) # bt_sz x n_dim
                wrtm1 = T.dot(r_tm1, self.wbw_w_r) # bt_sz x n_dim
                WY = T.dot(Y, self.wbw_w_y) # bt_sz x T x n_dim
                tmp = (wht + wrtm1)[:,None,:] # bt_sz x T x n_dim
                Mt = T.tanh(WY + tmp) # bt_sz x T x n_dim                     
                WMt = T.dot(Mt, self.wbw_w_alpha).flatten(2) # bt_sz x T
                alpha_ret_t = T.nnet.softmax(WMt) # bt_sz x num_region
                alpha_t = alpha_ret_t.dimshuffle((0,'x',1)) # bt_sz x 1 x T
                Y_alpha_t = T.batched_dot(alpha_t, Y)[:,0,:] # bt_sz x n_dim
                r_t = Y_alpha_t + T.dot(r_tm1, self.wbw_w_t) # bt_sz x n_dim        
                
                return r_t, alpha_ret_t

            premise = input_tensors[0]
            hypothesis = input_tensors[1].dimshuffle((1, 0, 2))

            [r, alpha], updates = theano.scan(fn = attentionLoop,
                                          sequences = [hypothesis],
                                          outputs_info = [K.zeros_like(premise[:,0,:]), None],
                                          non_sequences = [premise],
                                          n_steps = hypothesis.shape[0]
                                          )
        else:
            print "MASKING PRESENT"

            def attentionLoop(h_t, h_mask_t, r_tm1, Y, Y_mask):
                # h_t : bt_sz x n_dim
                # r_tm1 : bt_sz x n_dim

                if h_mask_t == 0:
                    return r_tm1, K.zeros_like(Y[:,:,0])

                wht = T.dot(h_t, self.wbw_w_h) # bt_sz x n_dim
                wrtm1 = T.dot(r_tm1, self.wbw_w_r) # bt_sz x n_dim
                WY = T.dot(Y, self.wbw_w_y) # bt_sz x T x n_dim
                tmp = (wht + wrtm1)[:,None,:] # bt_sz x T x n_dim
                Mt = T.tanh(WY + tmp) # bt_sz x T x n_dim                     
                WMt = T.dot(Mt, self.wbw_w_alpha).flatten(2) # bt_sz x T
                WMt_masked = WMt - 1000 * (1-Y_mask)
                alpha_ret_t = T.nnet.softmax(WMt_masked) # bt_sz x T
                alpha_t = alpha_ret_t.dimshuffle((0,'x',1)) # bt_sz x 1 x T
                Y_alpha_t = T.batched_dot(alpha_t, Y)[:,0,:] # bt_sz x n_dim
                r_t = Y_alpha_t + T.dot(r_tm1, self.wbw_w_t) # bt_sz x n_dim        
                
                return r_t, alpha_ret_t

            premise = input_tensors[0]
            premise_mask = mask[0]
            hypothesis = input_tensors[1].dimshuffle((1, 0, 2))
            hypothesis_mask = mask[1]

            [r, alpha], updates = theano.scan(fn = attentionLoop,
                                          sequences = [hypothesis, hypothesis_mask],
                                          outputs_info = [K.zeros_like(premise[:,0,:]), None],
                                          non_sequences = [premise, premise_mask],
                                          n_steps = hypothesis.shape[0]
                                          )

        if self.return_att:
            return [r.dimshuffle((1, 0, 2)), alpha.dimshuffle((1, 0, 2))]
        else:
            return r.dimshuffle((1, 0, 2))

    def compute_output_shape(self, input_shape):
        if self.return_att:
            return [(input_shape[0][0], input_shape[1][1], input_shape[0][2]), (input_shape[0][0], input_shape[1][1], input_shape[0][1])]
        else:
            return (input_shape[0][0], input_shape[1][1], input_shape[0][2])

    def compute_mask(self, input_tensors, input_masks):
        if self.return_att:
            return [input_masks[1], None]
        else:
            return input_masks[1]
