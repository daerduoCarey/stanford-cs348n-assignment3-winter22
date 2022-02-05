# Stanford CS348n Homework 3

## Dependencies

This codebase uses Python 3.6 and PyTorch 1.5.1.

Please install the python environments

    pip install -r requirements.txt

## Problem 1

Implement the two incomplete functions in `data.py`, and then use `prob1.ipynb` for visualizing the results.


## Problem 2

First, go to the folder `cd` and follow the README there to compile the GPU implementation of Chamfer distance.

Then, download [the data](http://download.cs.stanford.edu/orion/cs348n/structurenet_chair_dataset.zip) and unzip it under the home directory.

For problem 2 (a), implement the three pieces of code in `model.py` and run

    python train.py

If you implement things correctly, you should be able to see promising reconstruction results at epoch 3-5.
Please train the model for 10 epochs (roughly 2-3 hours) and report your curves and results.

Download [the pretrained models](http://download.cs.stanford.edu/orion/cs348n/structurenet_vae_pretrained_models.zip) and unzip it under the home directory.

For problem 2 (b), implement the missing code in `recon.py` and run

    python recon.py

For problem 2 (c), implement the missing code in `randgen.py` and run

    python randgen.py

For problem 2 (d), implement `interp.py` and run 

    python interp.py


### Codebase Structure and Experiment Log

Several main files

  * `model.py`: defines the NN;
  * `data.py`: contains the dataset and dataloader code;
  * `train.py`: the main trainer code;
  * `recon.py`: for shape reconstruction;
  * `randgen.py`: for shape free generation;
  * `interp.py`: for shape interpolation;
  * `part_semantics_Chair.txt`: the canonical chair hierarchy template.

Each experiment run will output a directory with name `exp-vae` for VAE, and it contains

  * `ckpts`: store the model checkpoint every 1000 steps of training;
  * `train` and `val`: store the tensorboard information, used for training and validation curve visualization;
  * `val_visu`: contains the validation results for one batch every 10 epoches of training;
  * `conf.pth`: stores all parameters for the training;
  * `data.py` and `train_ae.py`: backups the training and data code.
  * `train_log.txt`: stores the console output for the training.

## Questions

Please ask TA or post on Piazza for any question.
Please do not use the Github Issue for Q&As for the class.


