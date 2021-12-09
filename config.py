import copy

def load_config(dataset):
    if dataset == "BBBP":
        config = bbbp_cls
    return config

base_config = {}
base_config['data_dir'] = 'data'
base_config['saved_models_dir'] = 'saved_models'
base_config['results_dir'] = 'results'
base_config['fig_dir'] = 'figs'
base_config['reg'] = ['sparsity', 'consistency'] # the regularization to use, choices: ['sparsity', 'consistency']
base_config['exp_method'] = 'GCAM' # ["Grad", "GCAM", "EB", "cEB"]
base_config['human_data_dir'] = 'human_mask' # the location where human labeled explanations are stored
base_config['human_mask'] = True # The flag for whether to use human labeled explanation

bbbp_cls = copy.deepcopy(base_config)
bbbp_cls['data_fn'] = 'BBBP.csv'
bbbp_cls['d'] = 75
bbbp_cls['init_stddev'] = 0.1
bbbp_cls['L1'] = 256
bbbp_cls['L2'] = 128
bbbp_cls['L3'] = 64
bbbp_cls['N'] = None
bbbp_cls['batch_size'] = 1
bbbp_cls['num_epochs'] = 50
bbbp_cls['num_classes'] = 2
bbbp_cls['learning_rate'] = 0.001