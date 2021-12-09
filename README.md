# GNES
Code for ICDM2021 paper [GNES: Learning to Explain Graph Neural Networks](https://www.researchgate.net/profile/Yuyang-Gao-4/publication/355259484_GNES_Learning_to_Explain_Graph_Neural_Networks/links/616986a6b90c512662459391/GNES-Learning-to-Explain-Graph-Neural-Networks.pdf)



## Desciption
This codebase proivdes the necessary running environment (including the human explanation label) to train and evaluate the proposed GNES model on the BBBP molecular datasets. 

* main.py : Model training & testing on the full dataset

* main_fewshot.py : Model training & testing with few-shot learning setup

##  Installation

Python pakage requirement:
- python==3.7.9
- keras==2.2.4
- deepchem==2.3.0
- chainer_chemistry

## Data preparation

1. Download the BBBP dataset and place it in 'data/':
* BBBP: [http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/BBBP.csv](http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/BBBP.csv)

2. Download our human explanation labels for BBBP dataset and place them in 'human_mask/':
* human explanation labels: [https://drive.google.com/file/d/1RD-qs3VWZuRC4TFBHocto0kPtv4s1ha2/view?usp=sharing](https://drive.google.com/file/d/1RD-qs3VWZuRC4TFBHocto0kPtv4s1ha2/view?usp=sharing)

## Run 

Run the following scripts to replicate the results:

1. Train and evaluate GNES
```
python 1_gcn_train_eval.py
```
You will get the model performance from the output, and the final model will be saved to 'saved_models/'

2. Explain and visualize:
```
python 2_explain_save_masks.py
```
You will find the following results once done:

- Explanation data (in json format): will be saved to 'results/masks/'

- Explanation visualization (figures): will be saved to 'figs/'

Below are some example visuzliation from GNES:


## More tips to train GNES on your own datasets:

Please refer to the 'config.py' file first, as it stores major settings of the GNES framework, including:

- Backbone GCN model settings
- Whether to enable human explanation label supervision
- Whether to use the proposed explanation regularazation(s): inlcuding \['sparsity', 'consistency'\] 

If any further questions, please reach out to me via email yuyang.gao@emory.edu

##

And if you find this repo useful in your research, please consider cite our paper:


    

    @InProceedings{gao2021gnes,
    title={GNES: Learning to Explain Graph Neural Networks},
    author={Gao, Yuyang and Sun, Tong and Bhatt, Rishab and Yu, Dazhou and Hong, Sungsoo and Zhao, Liang},
    booktitle = {2021 IEEE International Conference on Data Mining (ICDM)},
    month = {December},
    year = {2021},
    organization={IEEE}
    }
