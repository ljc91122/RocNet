# RocNet
This is a pytorch implementation of paper  
"RocNet: Recursive Octree Network for Efficient 3D Deep Representation" 3DV 2020
by Juncheng Liu

This code implements an autoencoder using RocNet, which encodes and reconstructs 3D voxels represented by octrees.

```
requirements

pip install torchfold https://github.com/nearai/torchfold
pip install easydict https://pypi.org/project/easydict/

```

The matlab codes for generating training data from 3D voxels are in MATLAB folder.
The generated data should be put in the "data" folder.

The code is written under python 3.5 + pytorch 0.2.0

The main file for training and testing is "train-nb-128-32.ipynb"
Please open it with jupyter notebook.

The model is defined in ROctNetmodel_32.py



