import numpy as np
from collections import OrderedDict

def get_configurations():

	## INITIALISATIONS
	options = OrderedDict()

	## Data Related
	options['data_path'] = './Data/IELex/'
	options['expt_folder'] = './Models/'
	options['model_name'] = 'cogwbw'
	options['train_split'] = 'Train'
	options['val_split'] = 'Test'		
	options['reverse'] = False		
	options['load_model'] = False
	options['model_file'] = './Models/something.model'

	## Structure Options
	options['combined_num_mlp'] = 1
	options['combined_mlp_drop_0'] = True
	options['combined_mlp_act_0'] = 'linear'
	options['sent_drop'] = False
	options['use_tanh'] = True
	options['use_attention_drop'] = False

	## Dimensions
	options['n_words'] = 539
	options['n_emb'] = 200 
	options['n_dim'] = 75

	## Initialization
	options['init_type'] = 'glorot uniform'
	options['range'] = 0.01
	options['std'] = 0.01
	options['init_lstm_svd'] = True 

	options['forget_bias'] = np.float32(1.0)

	## Learning Parameters
	options['optimization'] = 'adam' 
	options['batch_size'] = 128
	options['lr'] = np.float32(0.0005)
	options['w_emb_lr'] = np.float32(80)
	options['momentum'] = np.float32(0.9)
	options['gamma'] = 1
	options['step'] = 10
	options['step_start'] = 100
	options['max_epochs'] = 10
	options['weight_decay'] = 0.1
	options['decay_rate'] = np.float32(0.999)
	options['drop_ratio'] = np.float32(0.5)
	options['smooth'] = np.float32(1e-8)
	options['grad_clip'] = np.float32(100)

	## Log Params
	options['disp_interval'] = 1

	return options
