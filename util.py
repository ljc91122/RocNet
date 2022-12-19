import os
from argparse import ArgumentParser
import easydict

def get_args():
#     parser = ArgumentParser(description='grass_pytorch')
#     parser.add_argument('--box_code_size', type=int, default=12)
#     parser.add_argument('--feature_size', type=int, default=80)
#     parser.add_argument('--hidden_size', type=int, default=200)
#     parser.add_argument('--symmetry_size', type=int, default=8)
#     parser.add_argument('--max_box_num', type=int, default=30)
#     parser.add_argument('--max_sym_num', type=int, default=10)

#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--batch_size', type=int, default=123)
#     parser.add_argument('--show_log_every', type=int, default=3)
#     parser.add_argument('--save_log', action='store_true', default=False)
#     parser.add_argument('--save_log_every', type=int, default=3)
#     parser.add_argument('--save_snapshot', action='store_true', default=False)
#     parser.add_argument('--save_snapshot_every', type=int, default=5)
#     parser.add_argument('--no_plot', action='store_true', default=False)
#     parser.add_argument('--lr', type=float, default=.001)
#     parser.add_argument('--lr_decay_by', type=float, default=1)
#     parser.add_argument('--lr_decay_every', type=float, default=1)

#     parser.add_argument('--no_cuda', action='store_true', default=False)
#     parser.add_argument('--gpu', type=int, default=0)
#     parser.add_argument('--data_path', type=str, default='data')
#     parser.add_argument('--save_path', type=str, default='models')
#     parser.add_argument('--resume_snapshot', type=str, default='')
#     args = parser.parse_args()
#     return args

    args = easydict.EasyDict({
        "box_code_size": 512,
        "feature_size": 80,
        "hidden_size": 200,
        "max_box_num": 6126,
        "epochs": 300,
        "batch_size": 10,
        "show_log_every": 1,
        "save_log": True,
        "save_log_every": 3,
        "save_snapshot": True,
        "save_snapshot_every": 5,
        "save_snapshot":'snapshot',
        "no_plot": False,
        "lr":0.001,
        #"lr": 0.1,
        "lr_decay_by":1,
        "lr_decay_every":1,
        "no_cuda": False,
        "gpu":1,
        "data_path":'data',
        "save_path":'models',
        "resume_snapshot":""
    })
    return args

