"""
    a test script for box-shape global interpolation
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

# two shape IDS
shape_id1 = '2368'
shape_id2 = '2821'
num_interp = 5

# load train config
conf = torch.load('exp_vae/conf.pth')

# load object category information
Tree.load_category_info(conf.category)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
out_dir = 'exp_vae/globally_interped_shapes_%s_%s' % (shape_id1, shape_id2)
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

# globally interpolate shapes
with torch.no_grad():

    # load the two shapes as the inputs
    obj1 = PartNetDataset.load_object('./structurenet_chair_dataset/'+shape_id1+'.json')
    obj1.to(device)
    obj2 = PartNetDataset.load_object('./structurenet_chair_dataset/'+shape_id2+'.json')
    obj2.to(device)

    # store interpolated results for visuals
    obj_outs = []

    # STUDENT CODE START
    # feed through the encoder to get two codes z1 and z2
    z1 = encoder.encode_structure(obj1)
    z2 = encoder.encode_structure(obj2)

    # create a forloop looping 0, 1, 2, ..., num_interp - 1, num_interp
    # interpolate the feature so that the first feature is exactly z1 and the last is exactly z2
    for :

        # infer through the decoder to get the interpolate output
        # set maximal tree depth to conf.max_tree_depth
        
        # add to the list obj_outs

    # STUDENT CODE END

    obj_names = []
    for i in range(num_interp+1):
        obj_names.append('interp-%d'%i)

        # output the hierarchy
        with open(os.path.join(out_dir, 'step-%d.txt'%i), 'w') as fout:
            fout.write(str(obj_out)+'\n\n')

    # output the assembled box-shape
    vis_utils.draw_partnet_objects(obj_outs, object_names=obj_names, \
            out_fn=os.path.join(out_dir, 'interp_figs.png'), figsize=(30, 5), \
            leafs_only=True, sem_colors_filename='part_colors_'+conf.category+'.txt')

