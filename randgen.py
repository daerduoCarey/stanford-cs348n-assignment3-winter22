"""
    a test script for box-shape free generation
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

# number of shapes to generate
num_gen = 100

# load train config
conf = torch.load('exp_vae/conf.pth')

# load object category information
Tree.load_category_info(conf.category)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
out_dir = 'exp_vae/freely_generated_shapes'
if os.path.exists(out_dir):
    response = input('result directory %s exists, overwrite? (y/n) ' % out_dir)
    if response != 'y':
        sys.exit()
    shutil.rmtree(out_dir)

# create a new directory to store eval results
os.mkdir(out_dir)

# create models
decoder = model.RecursiveDecoder(conf)

# load the pretrained models
print('Loading ckpt pretrained_decoder.pth')
data_to_restore = torch.load('pretrained_decoder.pth')
decoder.load_state_dict(data_to_restore, strict=True)
print('DONE\n')

# send to device
decoder.to(device)

# set models to evaluation mode
decoder.eval()

# generate shapes
with torch.no_grad():
    for i in range(num_gen):
        print(f'Generating {i}/{num_gen} ...')

        # STUDENT CODE START
        # get a Gaussian noise

        # infer through the model to get the generated hierarchy
        # set maximal tree depth to conf.max_tree_depth

        # STUDENT CODE END
                
        # output the hierarchy
        with open(os.path.join(out_dir, 'data-%03d.txt'%i), 'w') as fout:
            fout.write(str(obj)+'\n\n')

        # output the assembled box-shape
        vis_utils.draw_partnet_objects([obj], object_names=['data-%03d.txt'%i], \
                out_fn=os.path.join(out_dir, 'data-%03d.png'%i), \
                leafs_only=True, sem_colors_filename='part_colors_'+conf.category+'.txt')

