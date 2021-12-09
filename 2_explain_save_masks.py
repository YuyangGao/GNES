import warnings

warnings.simplefilter('ignore')

import os
import numpy as np
import json
import sys

sys.path.append("../")

from config import load_config
from utils import (load_data, preprocess, partition_train_val_test, keras_gcn, load_human_data)
from plot_utils import (draw_chem_activations, plot_image_grid,
                        create_figs, create_im_arrs)

from methods import (CAM, GradCAM, GradCAMAvg, Gradient, EB, cEB)

os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

##Choose the dataset to study
dataset = "BBBP"

# whether to save the explanation data
save_masks = True
# whether to save the explanation visualization as figures
viz = True

### Data
config = load_config(dataset)
data_fp = os.path.join(config['data_dir'], config['data_fn'])
raw_data = load_data(data_fp)
data = preprocess(raw_data)
smiles = raw_data["smiles"]

if dataset == "TOX21":
    dataset_external = dataset + "-NR-ER"
else:
    dataset_external = dataset

label_to_class_name = {0: "Not {}".format(dataset_external),
                       1: "{}".format(dataset_external)}

# Model to explain
model_fn = "GNES_{}.h5".format(dataset.lower())
model_fp = os.path.join(config["saved_models_dir"], model_fn)
model = keras_gcn(config)
model.load_weights(model_fp)
num_classes = data['labels_one_hot'].shape[1]

# load some examples from BBBP to visualize
# viz_smiles = ["FC(F)(F)c1ccc(N2CCNCC2)nc1Cl",
#               "FC(F)(F)C(Cl)Br",
#               "FC(F)(F)c1ccc2c(c1)N(CCCN1CCN(C3CC3)CC1)c1ccccc1S2",
#               "FCOC(C(F)(F)F)C(F)(F)F"]
# viz_data_inds = [np.argwhere(smiles == viz_smile)[0][0] for viz_smile in viz_smiles]

# or explanating whole dataset
# inds = partition_train_val_test(raw_data["smiles"], dataset)
# train_inds = inds["train_inds"]
# sel_train_inds = random.sample(list(train_inds), int(len(train_inds)*0.1))
# val_inds = inds["val_inds"]
# test_inds = inds["test_inds"]

viz_data_inds = [73, 148, 180, 236, 345, 487, 567]
viz_smiles = smiles[viz_data_inds].tolist()

smile2index={}
index2smile={}
for ind, smile in zip(viz_data_inds, viz_smiles):
    smile2index[smile]= int(ind)
    index2smile[int(ind)]=smile

num_to_explain = len(viz_smiles)

# Gather data for viz
viz_data = {}
for k, v in data.items():
    if isinstance(v, np.ndarray):
        vv = v[viz_data_inds]
    elif isinstance(v, list):
        vv = [v[i] for i in viz_data_inds]
    else:
        raise Exception("Data Type Not Supported")
    viz_data[k] = vv

# init explanation methods
gcam = GradCAM(model)
eb = EB(model)

methods = [gcam, eb]
method_names = ["GCAM", "EB"]

N = len(viz_data['norm_adjs'])
results = []
text = []

mask_dict = {}  # {method: smile: {ground_truth, predicted, node_importance}}
print('Generation explanations...')
for i in range(N):
    Adjs = viz_data['adjs'][i][np.newaxis, :, :]
    A_arr = viz_data['norm_adjs'][i][np.newaxis, :, :]
    X_arr = viz_data['node_features'][i][np.newaxis, :, :]
    Y_arr = viz_data['labels_one_hot'][i]
    smile = viz_smiles[i]

    num_nodes = A_arr.shape[1]

    # human masks are not used so just create for place-holder
    M = np.zeros((1, num_nodes, 1))
    E = np.zeros((1, num_nodes, num_nodes))

    prob = model.predict_on_batch(x=[M, E, Adjs, A_arr, A_arr, A_arr, X_arr])
    y_hat = prob.argmax()
    y = Y_arr.argmax()

    # Save prediction info:
    text.append(("%s" % label_to_class_name[y],  # ground truth label
                 "%.2f" % prob.max(),  # probabilistic softmax output
                 "%s" % label_to_class_name[y_hat]  # predicted label
                 ))

    results_ = []
    for name, method in zip(method_names, methods):
        if name not in mask_dict:
            mask_dict[name]={}

        # node importance
        mask = method.getMasks([M, E, Adjs, A_arr, A_arr, A_arr, X_arr])
        # Normalize
        mask[0] /= (mask[0].max() + 1e-6)
        mask[1] /= (mask[1].max() + 1e-6)
        masks_c0, masks_c1 = mask

        # edge importance
        edge_mask = method.getMasks_edge([M, E, Adjs, A_arr, A_arr, A_arr, X_arr])
        # Normalize
        edge_mask[0] *= Adjs[0]
        edge_mask[1] *= Adjs[0]
        edge_mask[0] /= (edge_mask[0].max() + 1e-6)
        edge_mask[1] /= (edge_mask[1].max() + 1e-6)
        masks_edge_c0, masks_edge_c1 = edge_mask

        if y == 0:
            results_.append({'weights': masks_c0,
                             'edge_weights': masks_edge_c0,
                             'smile': smile,
                             'index': smile2index[smile],
                             'method': name,
                             'class': 0})
        elif y == 1:
            results_.append({'weights': masks_c1,
                             'edge_weights': masks_edge_c1,
                             'smile': smile,
                             'index': smile2index[smile],
                             'method': name,
                             'class': 1})

        if smile not in mask_dict[name]:
            mask_dict[name][smile]={}
            mask_dict[name][smile]['index'] = smile2index[smile]
            mask_dict[name][smile]['ground_truth']=label_to_class_name[y]
            mask_dict[name][smile]['predicted'] = label_to_class_name[y_hat]
            if y == 0:
                mask_dict[name][smile]['node_importance'] = masks_c0.tolist()
                mask_dict[name][smile]['edge_importance'] = masks_edge_c0.tolist()
            elif y == 1:
                mask_dict[name][smile]['node_importance'] = masks_c1.tolist()
                mask_dict[name][smile]['edge_importance'] = masks_edge_c1.tolist()
        else:
            print('something wrong, duplication:', smile)

    results.append(results_)

if save_masks:
    print('Saving explanation masks...')
    for name, info in mask_dict.items():
        results_dir = os.path.join(config["results_dir"], "masks")
        out_fn = "mask_{}_{}.json".format(dataset.lower(), name)
        out_fp = os.path.join(results_dir, out_fn)
        with open(out_fp, 'w') as f:
            json.dump(info, f)

# Visualize
if not viz:
    raise Exception("Stopping viz")

print('Saving explanation visualizations...')
figs = create_figs(results)
