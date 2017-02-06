#!/usr/bin/env python

import datetime
import os
import sys
import log
import logging
import argparse
import math
import numpy as np
import pdb

from optimization_weight import *
from wbw_theano import *
from data_provision import *
from data_processing import *
from configurations import get_configurations

def get_lr(options, curr_epoch):
    if options['optimization'] == 'sgd':
        power = max((curr_epoch - options['step_start']) / options['step'], 0)
        power = math.ceil(power)
        return options['lr'] * (options['gamma'] ** power)  #
    else:
        return options['lr']

def train(options):

    logger = logging.getLogger('root')
    logger.info(options)
    logger.info('Starting Training')

    ###################################################
    ##  Load Data

    DataProvider = DataProvision(options['data_path'])

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']

    ###################################################
    ##  Build Model 
    
    if not options['load_model']:
        params = init_params(options)
        shared_params = init_shared_params(params)
    else:
        options, params, shared_params = load_model(options['model_file'])

    word1_idx, word2_idx, word1_mask, word2_mask, \
        label, dropout, cost, true_positive, pred_positive, actual_positive, alpha, pred_label, prob = build_model(shared_params, options)
    
    logger.info('Finished building model')

    ###################################################
    ## Add Regularisation 

    weight_decay = theano.shared(numpy.float32(options['weight_decay']), name='weight_decay')
    reg_cost = 0

    for k in shared_params.iterkeys():
        if k != 'w_emb':
            reg_cost += (shared_params[k]**2).sum()

    reg_cost *= weight_decay
    reg_cost += cost

    ###################################################
    ## Build Functions

    grads = T.grad(reg_cost, wrt = shared_params.values())
    grad_buf = [theano.shared(p.get_value() * 0, name='%s_grad_buf'%k) for k, p in shared_params.iteritems()]
    # accumulate the gradients within one batch
    update_grad = [(g_b, g) for g_b, g in zip(grad_buf, grads)]

    grad_clip = options['grad_clip']
    grad_norm = [T.sqrt(T.sum(g_b**2)) for g_b in grad_buf]
    update_clip = [(g_b, T.switch(T.gt(g_norm, grad_clip), g_b*grad_clip/g_norm, g_b)) for (g_norm, g_b) in zip(grad_norm, grad_buf)]

    # corresponding update function
    f_grad_clip = theano.function(inputs = [],
                                  updates = update_clip)
    f_output_grad_norm = theano.function(inputs = [],
                                         outputs = grad_norm)
    f_train = theano.function(inputs = [word1_idx, word2_idx, word1_mask, word2_mask, label],
                              outputs = [cost, true_positive, pred_positive, actual_positive],
                              updates = update_grad,
                              on_unused_input='warn')

    # validation function no gradient updates
    f_val = theano.function(inputs = [word1_idx, word2_idx, word1_mask, word2_mask, label],
                            outputs = [cost, true_positive, pred_positive, actual_positive],
                            on_unused_input='warn')

    # the attention function to get the attention and the predicted label
    f_att_pred = theano.function(inputs = [word1_idx, word2_idx, word1_mask, word2_mask],
                                 outputs = [alpha, prob, pred_label],
                                 on_unused_input='warn')

    f_grad_cache_update, f_param_update = eval(options['optimization'])(shared_params, grad_buf, options)
    logger.info('Finished building function')

    ###################################################
    ## Begin Iterations

    # calculate how many iterations we need
    num_iters_one_epoch = DataProvider.get_size(options['train_split']) / batch_size
    max_iters = max_epochs * num_iters_one_epoch
    eval_interval_in_iters = num_iters_one_epoch / 2
    save_interval_in_iters = num_iters_one_epoch
    disp_interval = options['disp_interval']

    best_val_fscore = 0.0
    best_param = dict()

    for itr in xrange(max_iters + 1):

        if itr % num_iters_one_epoch == 0:
            train_true_positive_list = []
            train_pred_positive_list = []
            train_actual_positive_list = []
            train_cost_list = []

        ## Validation Check
        if (itr % eval_interval_in_iters) == 0 or (itr == max_iters):
            val_cost_list = []
            val_true_positive_list = []
            val_pred_positive_list = []
            val_actual_positive_list = []
            val_count = 0
            dropout.set_value(numpy.float32(0.))

            for batch_word1, batch_word2, batch_label in DataProvider.iterate_batch(options['val_split'], batch_size):
                word1_idx, word1_mask = process_batch(batch_word1, reverse=options['reverse'], position='end')
                word2_idx, word2_mask = process_batch(batch_word2, reverse=options['reverse'], position='begin')

                [cost, true_positive, pred_positive, actual_positive] = f_val(word1_idx, word2_idx, word1_mask, word2_mask, batch_label.astype('int32'))

                [_, _,pred_label] = f_att_pred(word1_idx, word2_idx, word1_mask, word2_mask)
            
                val_count += batch_label.shape[0]
                val_cost_list.append(cost * batch_label.shape[0])
                val_true_positive_list.append(true_positive)
                val_pred_positive_list.append(pred_positive)
                val_actual_positive_list.append(actual_positive)
            
            ave_val_cost = sum(val_cost_list) / float(val_count)
            val_precision = sum(val_true_positive_list) / float(sum(val_pred_positive_list) + np.finfo(float).eps)
            val_recall = sum(val_true_positive_list) / float(sum(val_actual_positive_list) + np.finfo(float).eps)
            val_fscore = 2*val_precision*val_recall / (val_precision + val_recall + np.finfo(float).eps)

            if best_val_fscore < val_fscore:
                best_val_fscore = val_fscore
                shared_to_cpu(shared_params, best_param)
            logger.info('Validation Cost: %f Precision: %f Recall: %f FScore: %f' %(ave_val_cost, val_precision, val_recall, val_fscore))

        ## Iterate on training data
        dropout.set_value(numpy.float32(1.))
        batch_word1, batch_word2, batch_label = DataProvider.next_batch(options['train_split'], batch_size)
        word1_idx, word1_mask = process_batch(batch_word1, reverse=options['reverse'], position='end')
        word2_idx, word2_mask = process_batch(batch_word2, reverse=options['reverse'], position='begin')

        [cost, true_positive, pred_positive, actual_positive] = f_train(word1_idx, word2_idx, word1_mask, word2_mask, batch_label.astype('int32'))
        f_grad_clip()
        f_grad_cache_update()
        lr_t = get_lr(options, itr / float(num_iters_one_epoch))
        f_param_update(lr_t)

        train_true_positive_list.append(true_positive)
        train_pred_positive_list.append(pred_positive)
        train_actual_positive_list.append(actual_positive)
        train_cost_list.append(cost)

        if (itr % disp_interval) == 0  or (itr == max_iters):
            train_precision = sum(train_true_positive_list) / float(sum(train_pred_positive_list) + np.finfo(float).eps)
            train_recall = sum(train_true_positive_list) / float(sum(train_actual_positive_list) + np.finfo(float).eps)
            train_fscore = 2*train_precision*train_recall / (train_precision + train_recall + np.finfo(float).eps)
            cost = sum(train_cost_list)/float(len(train_cost_list))
            logger.info('Epoch %d/%d Iterations %d/%d Cost: %f Precision: %f Recall: %f FScore: %f, lr %f' \
                        % (itr / num_iters_one_epoch, max_epochs,
                           itr, num_iters_one_epoch,
                           cost, train_precision, train_recall, train_fscore, lr_t))
            if np.isnan(cost):
                logger.info('nan detected')
                file_name = options['model_name'] + '_nan_debug.model'
                logger.info('saving the debug model to %s' %(file_name))
                save_model(os.path.join(options['expt_folder'], file_name), options,
                           best_param)
                return 0


    logger.info('Best Validation FScore: %f', best_val_fscore)
    file_name = options['model_name'] + '_best_' + '%.3f' %(best_val_fscore) + '.model'
    logger.info('Saving the best model to %s' %(file_name))
    save_model(os.path.join(options['expt_folder'], file_name), options, best_param)

    return best_val_fscore

if __name__ == '__main__':

    theano.config.optimizer = 'fast_compile' 
    logger = log.setup_custom_logger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('changes', nargs='*', help='Changes to default values', default = '')
    args = parser.parse_args()
    for change in args.changes:
        logger.info('dict({%s})'%(change))
        options.update(eval('dict({%s})'%(change)))

    options = get_configurations()
    train(options)
