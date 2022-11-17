import sys
import argparse
import copy
import random
import torch, torchvision
from torchvision import transforms
import torch.utils.data as data
from random import shuffle
from collections import OrderedDict
import matplotlib.pyplot as plt
import continual_benchmark.dataloaders.base
import continual_benchmark.dataloaders as dataloaders
from continual_benchmark.dataloaders.datasetGen import data_split
from vae_experiments import classifier_utils
from vae_experiments import multiband_training, classifier_training, replay_training, training_functions
import global_classifier_training
from vae_experiments import vae_utils
from vae_experiments.validation import Validator, CERN_Validator
from vae_experiments import models_definition, classifier
from visualise import *
import time


# curr_global_decoder = torch.load(f'results/dual/class_based/FashionMNIST_example/model1_curr_decoder')
# print(curr_global_decoder.ones_distribution)
# print(curr_global_decoder.class_table)

# class_table = curr_global_decoder.class_table
# batch_size = 64
# n_prev_examples = 100
# recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = vae_utils.generate_previous_data(
#     curr_global_decoder,
#     class_table=class_table,
#     n_tasks=2,
#     n_img=n_prev_examples,
#     num_local=batch_size,
#     return_z=True,
#     translate_noise=True,
#     same_z=False,
#     equal_split=True)
# z_prev, z_bin_prev = z_prev

# fig = plt.figure()
# for i in range(50):
#     plt.subplot(5,10,i+1)
#     plt.tight_layout()
#     plt.imshow(recon_prev[i+50][0].cpu(), cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(classes_prev[i+50]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
# time.sleep(10)
