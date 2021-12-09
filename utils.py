import os
import time
import pickle
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from keras import metrics
from keras import backend as K

from keras.optimizers import Adam
from keras.models import (Model, load_model, clone_model)
from keras.layers import (Input, Dense, Softmax, Lambda)
from keras.optimizers import Adagrad
from keras.initializers import RandomNormal

from rdkit.Chem import MolFromSmiles
from deepchem.feat import WeaveFeaturizer, ConvMolFeaturizer
from deepchem.splits import RandomSplitter, ScaffoldSplitter
from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser
from chainer_chemistry.dataset.preprocessors.nfp_preprocessor import NFPPreprocessor

from methods import (CAM, GradCAM, GradCAMAvg, Gradient, EB, cEB)

from sklearn.metrics import (accuracy_score, precision_score, roc_auc_score,
                             recall_score, auc, average_precision_score,
                             roc_curve, precision_recall_curve)
import pandas as pd
import json

def load_data(csv_fp, labels_col="p_np", smiles_col="smiles"):
    """
    Load BBBP data
    """
    csvparser = CSVFileParser(NFPPreprocessor(), labels = labels_col, smiles_col = smiles_col)
    data_ = csvparser.parse(csv_fp,return_smiles = True)
    atoms, adjs, labels = data_['dataset'].get_datasets()
    smiles = data_['smiles']
    return {"atoms": atoms,
            "adjs": adjs,
            "labels":labels,
            "smiles": smiles}

def load_human_data(config, dataset):
    """
    Load human annotation data for a dataset

    input:  config contains base folder path to all the human masks,
            dataset name

    output: dict file with format:

            train:
                img_idx:
                    node_importance:
                    edge_importance:
            val:
                img_idx:
                    node_importance:
                    edge_importance:
            test:
                img_idx:
                    node_importance:
                    edge_importance:
    """
    base_fp = config['human_data_dir']

    # only load the human mask for training if 'human_mask' is True
    if config['human_mask']:
        train_fp = os.path.join(base_fp, dataset+'_train.csv')
        train_data = pd.read_csv(train_fp)
        train_dict = read_human_data_from_pd(train_data)
    else:
        train_dict ={}

    val_fp = os.path.join(base_fp, dataset+'_val.csv')
    val_data = pd.read_csv(val_fp)
    val_dict= read_human_data_from_pd (val_data)

    test_fp = os.path.join(base_fp, dataset+'_test.csv')
    test_data = pd.read_csv(test_fp)
    test_dict = read_human_data_from_pd (test_data)

    return {"train": train_dict,
            "val": val_dict,
            "test":test_dict}

def read_human_data_from_pd(data):
    dict = {}
    N = len(data)
    for i in range(N):

        skip_flag = data["status"][i]

        if skip_flag == "labeled":
            index = data["img_idx"][i]

            if index not in dict:
                dict[index]={}
            else:
                print('duplication detected in human mask for img_idx:', index)

            human_mask = json.loads(data["record"][i])
            dict[index]['node_importance'] = human_mask["node_importance"]
            dict[index]['edge_importance'] = human_mask["edge_importance"]

    return dict

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = sp.csr_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj.toarray()


def preprocess(raw_data, feats="convmol"):
    """
    Preprocess molecule data
    """
    labels = raw_data['labels']
    smiles = raw_data['smiles']
    adjs = raw_data['adjs']
    num_classes = np.unique(labels).shape[0]

    #One hot labels
    labels_one_hot = np.eye(num_classes)[labels.reshape(-1)]

    if feats == "weave":
        featurizer = WeaveFeaturizer()
    elif feats == "convmol":
        featurizer =  ConvMolFeaturizer()

    mol_objs = featurizer.featurize([MolFromSmiles(smile) for smile in smiles])

    #Sort feature matrices by node degree
    node_features = []
    for i,feat in enumerate(mol_objs):
        sortind = np.argsort(adjs[i].sum(axis = 1) - 1)
        N = len(sortind)
        sortMatrix = np.eye(N)[sortind,:]
        node_features.append(np.matmul(sortMatrix.T, feat.get_atom_features()))

    #Normalize Adjacency Mats
    norm_adjs = [preprocess_adj(A) for A in adjs]

    return {'labels_one_hot': labels_one_hot,
            'node_features': node_features,
            'adjs': adjs,
            'norm_adjs': norm_adjs}


def dense(n_hidden, activation='relu',
          init_stddev=0.1, init_mean=0.0,
          seed=None):
    """
    Helper function for configuring `keras.layers.Dense`
    """
    kernel_initializer = RandomNormal(mean=init_mean, stddev=init_stddev, seed=seed)
    bias_initializer = RandomNormal(mean=init_mean, stddev=init_stddev, seed=seed)
    return Dense(n_hidden, activation=activation,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 use_bias=True)

def matmul(XY):
    """
    Matrix multiplication for use with `keras.layers.Lambda`
    Compatible with `keras.models.Model`
    """
    X,Y = XY
    return K.tf.matmul(X,Y)


def GAP(X):
    return K.tf.reduce_mean(X, axis=1, keepdims=True)


def keras_gcn(config):
    """
    Keras GCN for graph classification
    """
    d = config['d']
    init_stddev = config['init_stddev']
    L1 = config['L1']
    L2 = config['L2']
    L3 = config['L3']
    N = config['N']
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    reg_list = config['reg']
    exp_method_name = config['exp_method']
    learning_rate = config['learning_rate']

    # currently the implementation of the pipeline only support batch_size = 1
    assert batch_size == 1, "Batch size != 1 Not Implemented!"

    # adjacency matrix, for regularization propose
    M = Input(shape=(batch_size, N, 1), batch_shape=(batch_size, N, 1))
    E = Input(shape=(batch_size, N, N), batch_shape=(batch_size, N, N))
    Adj = Input(shape=(batch_size, N, N), batch_shape=(batch_size, N, N))
    A_batch1 = Input(shape=(batch_size,N,N), batch_shape=(batch_size,N,N))
    A_batch2 = Input(shape=(batch_size, N, N), batch_shape=(batch_size, N, N))
    A_batch3 = Input(shape=(batch_size, N, N), batch_shape=(batch_size, N, N))
    X_batch = Input(shape=(batch_size,N,d), batch_shape=(batch_size,N,d))
    # Y = Input(shape=(batch_size, num_classes), batch_shape=(batch_size, num_classes))

    h1 = dense(L1)(Lambda(matmul)([A_batch1, X_batch]))
    h2 = dense(L2)(Lambda(matmul)([A_batch2, h1]))
    h3 = dense(L3)(Lambda(matmul)([A_batch3, h2]))
    gap = Lambda(GAP)(h3)
    gap=  Lambda(lambda y: K.squeeze(y, 1))(gap)
    logits = dense(num_classes, activation='linear')(gap)
    Y_hat = Softmax()(logits)
    model = Model(inputs=[M, E, Adj, A_batch1, A_batch2, A_batch3, X_batch], outputs=Y_hat)

    if exp_method_name =='GCAM':
        # node mask
        maskh0 = getGradCamMask(model.layers[-2].output[0,0],model.layers[-5].output)
        maskh1 = getGradCamMask(model.layers[-2].output[0,1],model.layers[-5].output)
        node_mask = K.stack([maskh0, maskh1], axis=0)
        # node_mask = [maskh0,maskh1]

        # edge mask
        maskh0_edge = getGradCamMask_edge(model.layers[-2].output[0,0],model.layers[6].input)
        maskh1_edge = getGradCamMask_edge(model.layers[-2].output[0,1],model.layers[6].input)
        edge_mask = K.stack([maskh0_edge, maskh1_edge], axis=0)

    elif exp_method_name =='EB':
        pLamda4=ebDense(K.squeeze(model.layers[-3].output,0),model.layers[-2].weights[0],K.variable(np.array([1,0])))
        pdense3=ebGAP(model.layers[-5].output,pLamda4)
        pLambda3=ebMoleculeDense(model.layers[-6].output,model.layers[-5].weights[0],pdense3)
        pdense2=ebMoleculeAdj(model.layers[-7].output,K.squeeze(model.layers[6].input,0),pLambda3)
        pLambda2=ebMoleculeDense(model.layers[-9].output,model.layers[-7].weights[0],pdense2)
        pdense1=ebMoleculeAdj(model.layers[-10].output,K.squeeze(model.layers[3].input,0),pLambda2)
        pLambda1=ebMoleculeDense(model.layers[-12].output,model.layers[-10].weights[0],pdense1)
        pin=ebMoleculeAdj(model.layers[-13].output,K.squeeze(model.layers[0].input,0),pLambda1)
        mask0=K.squeeze(K.sum(pin,axis=2),0)
        edge_mask0 = ebMoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)

        pLamda4=ebDense(K.squeeze(model.layers[-3].output,0),model.layers[-2].weights[0],K.variable(np.array([0,1])))
        pdense3=ebGAP(model.layers[-5].output,pLamda4)
        pLambda3=ebMoleculeDense(model.layers[-6].output,model.layers[-5].weights[0],pdense3)
        pdense2=ebMoleculeAdj(model.layers[-7].output,K.squeeze(model.layers[6].input,0),pLambda3)
        pLambda2=ebMoleculeDense(model.layers[-9].output,model.layers[-7].weights[0],pdense2)
        pdense1=ebMoleculeAdj(model.layers[-10].output,K.squeeze(model.layers[3].input,0),pLambda2)
        pLambda1=ebMoleculeDense(model.layers[-12].output,model.layers[-10].weights[0],pdense1)
        pin=ebMoleculeAdj(model.layers[-13].output,K.squeeze(model.layers[0].input,0),pLambda1)
        mask1=K.squeeze(K.sum(pin,axis=2),0)
        edge_mask1 = ebMoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)

        # node mask
        node_mask = K.stack([mask0, mask1], axis=0)
        # edge mask
        edge_mask = K.stack([edge_mask0, edge_mask1], axis=0)
    else:
        print('Unknown exp method name. use GCAM as default')
        # node mask
        maskh0 = getGradCamMask(model.layers[-2].output[0,0],model.layers[-5].output)
        maskh1 = getGradCamMask(model.layers[-2].output[0,1],model.layers[-5].output)
        node_mask = K.stack([maskh0, maskh1], axis=0)
        # node_mask = [maskh0,maskh1]

        # edge mask
        maskh0_edge = getGradCamMask_edge(model.layers[-2].output[0,0],model.layers[6].input)
        maskh1_edge = getGradCamMask_edge(model.layers[-2].output[0,1],model.layers[6].input)
        edge_mask = K.stack([maskh0_edge, maskh1_edge], axis=0)
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=custom_loss(reg_list, logits, Adj, node_mask, edge_mask, M, E))

    # print('node_explanation:', node_explanation[0])
    return model


# Define the proposed GNES loss
def custom_loss(reg_list, logits, A, node_mask, edge_mask, M, E):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):

        loss=K.mean(K.binary_crossentropy(y_true, logits, from_logits=True), axis=-1)
        node_s = K.abs(K.gather(node_mask, K.argmax(y_true, axis=-1)))
        edge_s = K.abs(K.gather(edge_mask, K.argmax(y_true, axis=-1)))
        ones = K.ones(K.shape(node_s))
        edge_s = K.squeeze(edge_s * A, axis=0)

        if 'sparsity' in reg_list:
            print('adding sparsity regularization')
            loss += 0.001 * (K.mean(node_s) + K.mean(edge_s))
        if 'consistency' in reg_list:
            print('adding consistency regularization')
            pair_diff = K.tf.matrix_band_part(node_s[..., None] - K.tf.transpose(node_s[..., None]), 0, -1)
            # we use below term to avoid trivial solution when minimizing consistency loss (i.e. edge_s = 0)
            conn = - K.mean(K.log(K.tf.matmul(edge_s, K.transpose(ones)) + 1e-6))
            loss += 0.1 * (K.mean(edge_s * K.square(pair_diff)) + conn)

        # human node importance supervision
        loss += 1 * K.max(M) * K.mean(K.abs(node_s - M))
        # human edge importance supervision
        loss += 1 * K.max(E) * K.mean(K.abs(edge_s - E))
        return loss
    # Return a function
    return loss

def getGradCamMask(output,activation):
    '''
    This function calculates the importance weight reported in GradCam
    Input:
        ouput: The class output
        activation: activation that we will take gradient with respect to
    '''
    temp_grad= K.gradients(output,activation)
    grad=K.gradients(output,activation)[0]
    alpha=K.squeeze(K.mean(grad,axis=1),0)
    mask=K.squeeze(K.relu(K.sum(activation*alpha,axis=2)),0)
    return mask
def getGradCamMask_edge(output,activation):
    '''
    This function calculates the edge importance via gradient w.r.t. adj
    Input:
        ouput: The class output
        activation: activation that we will take gradient with respect to
    '''
    grad_adj=K.gradients(output,activation)[0]
    mask_adj=K.squeeze(K.relu(grad_adj),0)
    # set diagonal to be all 0
    mask_adj = K.tf.linalg.set_diag(mask_adj, K.tf.zeros([K.tf.shape(mask_adj)[0], ]))
    return mask_adj

def ebDense(activations, W, bottomP):
    '''
    This function calculates eb for a dense layer
    Input:
        activations: d-dimensional vector
        W: Weights dxk-dimensional matrix
        bottomP: k-dimensional probability vector
    Output:
        p: the probability of activation d-dimensional vector
    '''
    Wrelu = K.relu(W)
    pcond = K.tf.matmul(K.tf.diag(activations), Wrelu)
    pcond = pcond / (K.sum(pcond, axis=0) + 1e-5)
    return K.transpose(K.tf.matmul(pcond, K.expand_dims(bottomP, 1)))

def ebMoleculeDense(activations, W, bottomP):
    '''
    This function calculates eb for a dense layer
    Input:
        activations: 1x?xK
        W: Weights dxk-dimensional matrix KxL
        bottomP: probability matrix 1x?xL
    Output:
        p: probability matrix 1x?xK
    '''
    k, l = W.shape.as_list()
    Wrelu = K.relu(W)
    pcond = K.tile(K.expand_dims(activations, 3), (1, 1, 1, l)) * Wrelu
    p = K.mean(K.tile(K.expand_dims(bottomP, 2), (1, 1, k, 1)) * pcond, 3)
    return p

def ebGAP(activations, bottomP):
    '''
    This function calculates eb for GAP layer
    Input:
        activations: 1x?xK
        bottomP: probability matrix 1xK
    Output:
        p: probability matrix 1x?xK
    '''
    epsilon = 1e-5
    pcond = activations / (epsilon + K.sum(activations, axis=1))
    p = pcond * K.squeeze(bottomP, 0)
    p = p / (K.sum(p, axis=1) + epsilon)
    return p

def ebMoleculeAdj(activations, A, bottomP):
    '''
    This function calculates eb for a Adj conv layer
    Input:
        activations: 1x?xK
        A: Adjacency ?x?
        bottomP: probability matrix 1x?xK
    Output:
        p: probability matrix 1x?xK
    '''
    pcond = K.expand_dims(K.tf.matmul(A, K.squeeze(activations, 0)), 0)
    p = pcond * bottomP
    return p

def ebMoleculeEdge(activations, A, bottomP):
    P = K.squeeze(K.sum(bottomP, axis=2), 0)
    mask_adj = A * P + A * K.tf.transpose(P)

    # set diagonal to be all 0
    mask_adj = K.tf.linalg.set_diag(mask_adj, K.tf.zeros([K.tf.shape(mask_adj)[0], ]))
    return mask_adj


def gcn_train(model, data, num_epochs, train_inds, val_inds, save_path, human_data, metric='AUC', exp_method='GCAM'):
    Adjs = data['adjs']
    norm_adjs = data['norm_adjs']
    labels_one_hot = data['labels_one_hot']

    node_features = data['node_features']
    total_loss = []
    best = 0
    for epoch in range(num_epochs):
        epoch_loss = []
        #Train
        rand_inds = np.random.permutation(train_inds)
        for ri in rand_inds:

            Adj = Adjs[ri][np.newaxis, :, :]
            A_arr = norm_adjs[ri][np.newaxis, :, :]
            X_arr = node_features[ri][np.newaxis, :, :]
            Y_arr = labels_one_hot[ri][np.newaxis, :]

            if ri in human_data['train']:
                M = np.array(human_data['train'][ri]['node_importance'])[np.newaxis, :, np.newaxis]
                E = np.array(human_data['train'][ri]['edge_importance'])[np.newaxis, :, :]
            else:
                N = Adj.shape[1]
                M = np.zeros((1, N, 1))
                E = np.zeros((1, N, N))

            sample_loss = model.train_on_batch(x=[M, E, Adj, A_arr, A_arr, A_arr, X_arr], y=Y_arr, )
            epoch_loss.append(sample_loss)
            # print(sample_loss)

        #Eval
        val_eval = evaluate(model, data, val_inds, human_data['val'], exp_method= exp_method, human_eval=False)

        mean_train_loss = sum(epoch_loss) / len(epoch_loss)
        val_acc = val_eval['accuracy']
        val_auc = val_eval['roc_auc']
        # node_mse = val_eval["node_mse"]
        # node_mae = val_eval["node_mae"]
        # edge_mse = val_eval["edge_mse"]
        # edge_mae = val_eval["edge_mae"]

        # choose best model base on AUC
        if metric == 'AUC':
            if val_auc>best:
                model.save(save_path)
                best = val_auc
                print('Model saved!')
        elif metric == 'ACC':
            if val_acc>best:
                model.save(save_path)
                best = val_acc
                print('Model saved!')

        print("Epoch: {}, Train Loss: {:.3f}, Val ACC: {:.3f}, AUC: {:.3f}.".format(epoch, mean_train_loss, val_acc, val_auc))
        # print("Human evaluation: node MSE: {:.3f}, node MAE: {:.3f}, edge MSE: {:.3f}, edge MAE: {:.3f}.".format(node_mse, node_mae, edge_mse, edge_mae))
        total_loss.extend(epoch_loss)

    return total_loss, best


class MockDataset:
    """Mock Dataset class for a DeepChem Dataset"""
    def __init__(self, smiles):
        self.ids = smiles

    def __len__(self):
        return len(self.ids)


def partition_train_val_test(smiles, dataset):
    """
    Split a molecule dataset (SMILES) with deepchem built-ins
    """

    ds = MockDataset(smiles)

    if dataset == "BBBP":
        splitter = ScaffoldSplitter()
    elif dataset == "BACE":
        splitter = ScaffoldSplitter()
    elif dataset == "TOX21":
        splitter = RandomSplitter()

    train_inds, val_inds, test_inds = splitter.split(ds)

    return {"train_inds": train_inds,
            "val_inds": val_inds,
            "test_inds": test_inds}


def run_train(config, data, inds, save_path, human_data, metric='AUC', train=True):
    """
    Sets splitter. Partitions train/val/test.
    Loads model from config. Trains and evals.
    Returns model and eval metrics.
    """
    train_inds = inds["train_inds"]
    val_inds = inds["val_inds"]
    test_inds = inds["test_inds"]

    if train:
        model = keras_gcn(config)
        loss, accuracy = gcn_train(model, data, config['num_epochs'], train_inds, val_inds, save_path, human_data, metric=metric, exp_method = config['exp_method'])

    model = keras_gcn(config)
    model.load_weights(save_path)

    train_eval = evaluate(model, data, train_inds, human_data['train'], config['exp_method'], human_eval=True)
    test_eval = evaluate(model, data, test_inds, human_data['test'], config['exp_method'], human_eval=True)
    val_eval = evaluate(model, data, val_inds, human_data['val'], config['exp_method'], human_eval=True)

    return model, {"train": train_eval,
                   "test": test_eval,
                   "val": val_eval}



def print_evals(eval_dict):
    print("Accuracy: {0:.3f}".format(eval_dict["accuracy"]))
    print("Precision: {0:.3f}".format(eval_dict["precision"]))
    print("AUC ROC: {0:.3f}".format(eval_dict["roc_auc"]))
    print("AUC PR: {0:.3f}".format(eval_dict["avg_precision"]))
    print("eval time (s): {0:.3f}".format(eval_dict["eval_time"]))


def human_evaluate(model, data, inds, human_data, exp_method):
    start = time.time()
    if exp_method == 'GCAM':
        method = GradCAM(model)
    elif exp_method == 'EB':
        method = EB(model)
    else:
        print('Unknown exp method name, use Grad-CAM as default instead')
        method = GradCAM(model)
    end = time.time()
    print("Finish loading exp method, time used:", end - start)

    start = time.time()
    node_mse = []
    node_mae = []
    edge_mse = []
    edge_mae = []
    for i in inds:
        if i in human_data:
            Adj = data["adjs"][i][np.newaxis, :, :]
            label = np.argmax(data["labels_one_hot"][i])

            M = np.array(human_data[i]['node_importance'])[np.newaxis, :, np.newaxis]
            E = np.array(human_data[i]['edge_importance'])[np.newaxis, :, :]

            # node importance
            node_mask = method.getMasks([M, E, data["adjs"][i][np.newaxis, :, :] ,data["norm_adjs"][i][np.newaxis, :, :], data["norm_adjs"][i][np.newaxis, :, :], data["norm_adjs"][i][np.newaxis, :, :],
                                      data["node_features"][i][np.newaxis, :, :]])[label]
            # Normalize
            node_mask /= (node_mask.max() + 1e-6)

            mse = K.mean(K.square(node_mask-M))
            mae = K.mean(K.abs(node_mask-M))
            node_mse.append(mse)
            node_mae.append(mae)

            # edge importance
            edge_mask = method.getMasks_edge([M, E, data["adjs"][i][np.newaxis, :, :] ,data["norm_adjs"][i][np.newaxis, :, :], data["norm_adjs"][i][np.newaxis, :, :], data["norm_adjs"][i][np.newaxis, :, :],
                                      data["node_features"][i][np.newaxis, :, :]])[label]
            # Normalize
            edge_mask *= Adj[0]
            edge_mask /= (edge_mask.max() + 1e-6)
            mse = K.mean(K.square(edge_mask-E))
            mae = K.mean(K.abs(edge_mask-E))
            edge_mse.append(mse)
            edge_mae.append(mae)

    end = time.time()
    print("Finish evaluation, time used:", end - start)

    return {"node_mse":np.mean(K.eval(K.stack(node_mse))),
            "node_mae":np.mean(K.eval(K.stack(node_mae))),
            "edge_mae":np.mean(K.eval(K.stack(edge_mae))),
            "edge_mse":np.mean(K.eval(K.stack(edge_mse)))}



def evaluate(model, data, inds, human_data, exp_method = 'GCAM', human_eval=False, thresh=0.5):
    t_test = time.time()
    preds = np.concatenate([model.predict([np.zeros((1, data["adjs"][i].shape[-1], 1)), np.zeros((1, data["adjs"][i].shape[-1], data["adjs"][i].shape[-1])), data["adjs"][i][np.newaxis, :, :] ,data["norm_adjs"][i][np.newaxis, :, :], data["norm_adjs"][i][np.newaxis, :, :], data["norm_adjs"][i][np.newaxis, :, :],
                              data["node_features"][i][np.newaxis, :, :]])
                              for i in inds], axis=0)

    preds = preds[:,1]
    # print(preds)
    labels = np.array([np.argmax(data["labels_one_hot"][i]) for i in inds])
    roc_auc = roc_auc_score(labels, preds)
    roc_curve_ = roc_curve(labels, preds)
    precision = precision_score(labels, (preds > thresh).astype('int'), zero_division=0)
    acc = accuracy_score(labels, (preds > thresh).astype('int'))
    ap = average_precision_score(labels, preds)
    pr_curve_ = precision_recall_curve(labels, preds)
    if human_eval:
        out = human_evaluate(model, data, inds, human_data, exp_method)
        return {"accuracy": acc,
                "roc_auc": roc_auc,
                "precision": precision,
                "avg_precision": precision,
                "eval_time": (time.time() - t_test),
                "roc_curve": roc_curve_,
                "pr_curve": pr_curve_,
                "node_mse": out["node_mse"],
                "edge_mse": out["edge_mse"],
                "node_mae": out["node_mae"],
                "edge_mae": out["edge_mae"]
                }

    return {"accuracy": acc,
                "roc_auc": roc_auc,
                "precision": precision,
                "avg_precision": precision,
                "eval_time": (time.time() - t_test),
                "roc_curve": roc_curve_,
                "pr_curve": pr_curve_}


def print_eval_avg(eval_dict, split, metric):
    N = len(eval_dict.keys())
    vals = [eval_dict[i][split][metric] for i in range(N)]
    return "{0:.3f} +/- {1:.3f}".format(np.mean(vals), np.std(vals))


def occlude_and_predict(Adj, X_arr, A_arr, masks, thresh, model):
    """
    COPIES and mutates input data

    Returns predicted CLASS (not prob.) of occluded data
    """
    #Copy node features. We need to edit it.
    X_arr_occ = X_arr.copy()

    #Occlude activated nodes for each explain method
    #NB: array shape is (batch, N, D)
    # and batches are always of size 1
    X_arr_occ[0, masks > thresh, :] = 0

    #Predict on occluded image. Save prediction
    prob_occ = model.predict_on_batch(x=[Adj, A_arr, A_arr, A_arr, X_arr_occ])

    y_hat_occ = prob_occ.argmax()
    return y_hat_occ


