# Master Thesis

repository published as part of masters thesis at FIT CTU in Prague by Anh Tran Viet.
The goal was to provide a fully functional pipeline for edge flow corrections by reconstructing mesh from a primitive with constraints.

## Setup
All required depencies are listed in `environment.txt` and can be installed using `conda create -n <environment-name> --file environment.txt`

Required raw data used for preprocessing can be found [here](https://drive.google.com/file/d/17uD91g4mYJFJVbTcvB5HjdpWmJjyaxPY/view?usp=sharing) and precomputated data for training, with proper hierarchy can be downloaded [here](https://drive.google.com/file/d/1GQAzLRa3GJKYrStXa_jFwOLSNYPcB5Lb/view?usp=sharing).

Each precomputed sample contains:

`deformed.obj` - end result, when offsets are applied to the sphere primitive

`offsets.pt` - ground truth offsets

`target.obj` - target shape

`sphere.obj` - sphere primiteve


## Run
There are three main entry scripts: `data_prep.py`, `train_model.py`, `predict.py`. Each script has its optional arguments with which it can be called. Invoke `--help` to see available options and their defaults

* `data_prep.py` Script used for preprocessing data. Computing deformation offsets by minimizing Chamfer distance. 
* `train_model.py` is used for training and evaluating the model. The training step can be skipped as well as the evaluation process. We can load checkpoint weights and continue training or only evaluating the prowess of the model.
* `predict.py` is fully functional pipeline for predicting optimized mesh. The result `predicted.obj` is saved in the root directory.

!We suggest running scripts as `python -B <script>.py` to avoid python system caching, which can cause to misleading progress status
