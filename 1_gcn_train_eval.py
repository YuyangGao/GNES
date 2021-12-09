import os

import sys;

sys.path.append("../")

from config import load_config
from utils import (load_data, preprocess, run_train, load_human_data,
                   partition_train_val_test, print_eval_avg)

# dataset
dataset = "BBBP"

# whether to train the model or just evaluation. i.e. True = train, False = evaluate
train_flag = True

save_model = True

# Number of train/test/eval replicates to perform
N = 1

config = load_config(dataset)
data_fp = os.path.join(config['data_dir'], config['data_fn'])
raw_data = load_data(data_fp)
data = preprocess(raw_data)

# load human mask for train/val/test
human_data = load_human_data(config, dataset)

eval_results = {}
model_out_fn = "GNES_{}.h5".format(dataset.lower())
save_path = os.path.join(config["saved_models_dir"], model_out_fn)

for i in range(N):
    print("*" * 50)
    print(i)
    inds = partition_train_val_test(raw_data["smiles"], dataset)

    model, eval_metrics = run_train(config, data, inds, save_path, human_data, train=train_flag)
    eval_results[i] = eval_metrics

# Evaluate
print(dataset + "\n")

for metric in ["roc_auc", "avg_precision", "node_mse", "node_mae", "edge_mse", "edge_mae"]:
    print(metric)
    for split in ["train", "val  ", "test "]:
        res = print_eval_avg(eval_results, split.strip(), metric)
        print(split + " " + res)
    print()
