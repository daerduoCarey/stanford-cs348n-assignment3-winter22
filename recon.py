"""
    a test script for box-shape reconstruction
"""

import os
import sys
import shutil
import numpy as np
import torch
import utils, vis_utils
from data import PartNetDataset, Tree
import model

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

# how many shapes to evaluate (the top-K in test.txt)
num_recon = 100

# load train config
conf = torch.load('exp_vae/conf.pth')

# load object category information
Tree.load_category_info(conf.category)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
out_dir = 'exp_vae/reconstructed_shapes'
if os.path.exists(out_dir):
    response = input('result directory %s exists, overwrite? (y/n) ' % out_dir)
    if response != 'y':
        sys.exit()
    shutil.rmtree(out_dir)

# create a new directory to store eval results
os.mkdir(out_dir)

# create models
# we disable probabilistic because we do not need to jitter the decoded z during inference
encoder = model.RecursiveEncoder(conf, variational=True, probabilistic=False)
decoder = model.RecursiveDecoder(conf)

# load the pretrained models
print('Loading ckpt pretrained_encoder.pth')
data_to_restore = torch.load('pretrained_encoder.pth')
encoder.load_state_dict(data_to_restore, strict=True)
print('DONE\n')
print('Loading ckpt pretrained_decoder.pth')
data_to_restore = torch.load('pretrained_decoder.pth')
decoder.load_state_dict(data_to_restore, strict=True)
print('DONE\n')

# send to device
encoder.to(device)
decoder.to(device)

# set models to evaluation mode
encoder.eval()
decoder.eval()

# read test.txt
with open('./structurenet_chair_dataset/test.txt', 'r') as fin:
    data_list = [l.rstrip() for l in fin.readlines()]

# reconstruct shapes
with torch.no_grad():
    for i in range(num_recon):
        print(f'Reconstructing {i}/{num_recon} ...')

        # load the gt data as the input
        obj = PartNetDataset.load_object('./structurenet_chair_dataset/'+data_list[i]+'.json')
        obj.to(device)

        # STUDENT CODE START
        # feed through the encoder to get a code z

        # infer through the decoder to get the reconstructed output
        # set maximal tree depth to conf.max_tree_depth

        # STUDENT CODE END
                
        # output the hierarchy
        with open(os.path.join(out_dir, 'data-%03d.txt'%i), 'w') as fout:
            fout.write(data_list[i]+'\n\n')
            fout.write(str(obj)+'\n\n')
            fout.write(str(obj_pred)+'\n\n')

        # output the assembled box-shape
        vis_utils.draw_partnet_objects([obj, obj_pred], object_names=['GT', 'PRED'], \
                out_fn=os.path.join(out_dir, 'data-%03d.png'%i), \
                leafs_only=True, sem_colors_filename='part_colors_'+conf.category+'.txt')

