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
# from vae_experiments import models_definition, classifier
from visualise import *
import time
from continual_benchmark import models


# def get_head_accuracy(y_pred, y_test):
#     _, y_pred_tag = torch.max(y_pred, 1)
#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     return correct_results_sum


# class HeadDataset(data.Dataset):
#     def __init__(self, decoder, class_table, gen_batch_size):
        
#         img, labels = self._create_dataset(decoder=decoder, 
#                                             class_table=class_table, 
#                                             gen_batch_size=gen_batch_size)

#         self.img_data = img
#         self.img_labels = labels
#         self.dataset_len = len(self.img_labels)

#     def __len__(self):
#         return self.dataset_len

#     def __getitem__(self, idx):
#         return self.img_data[idx], self.img_labels[idx]


#     def _create_dataset(self, decoder, class_table, gen_batch_size):
                        
#         x = []
#         y = []

#         print(f'\nCreating HeadDataset')

#         with torch.no_grad():

#             # generated samples from current task
#             batch_size = gen_batch_size
#             n_prev_examples = 15000
#             recon_prev, classes_prev, z_prev, task_ids_prev, embeddings_prev = vae_utils.generate_previous_data(
#                 decoder,
#                 class_table=class_table,
#                 n_tasks=5,
#                 n_img=n_prev_examples,
#                 num_local=batch_size,
#                 return_z=True,
#                 translate_noise=True,
#                 same_z=False,
#                 equal_split=True,
#                 recent_task_only=False)
            
#             # classify and add generated samples to dataset
#             current_task_counter = 0
#             for i in range(n_prev_examples):
#                 current_task_counter += 1
#                 x.append(recon_prev[i])
#                 y.append(classes_prev[i])
#             print(f'Adding {current_task_counter} generated samples from current task...')

#         # t = torch.FloatTensor(y)
#         # print(torch.unique(t, return_counts=True))

#         print(f'Done creating HeadDataset\n')
#         return x, y




# model = models.lenet.LeNet()
# model = model.to("cuda")

# curr_global_decoder = torch.load(f'results/class_based/FashionMNIST_example/model4_curr_decoder')
# train_dataset = HeadDataset(decoder=curr_global_decoder, class_table=curr_global_decoder.class_table, gen_batch_size=32)
# train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# n_epochs = 20
# lr = 0.001
# scheduler_rate = 0.99

# optimizer = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
# criterion = torch.nn.CrossEntropyLoss()

# for epoch in range(n_epochs):
#     losses = []
#     accuracy = 0
#     total = 0
#     start = time.time()

#     for iteration, batch in enumerate(train_dataloader):

#         x = batch[0]
#         y = batch[1].to("cuda")

#         optimizer.zero_grad()

#         out = model(x)
#         loss = criterion(out, y)

#         loss.backward()
#         optimizer.step()

#         with torch.no_grad():
#             acc = get_head_accuracy(out, y)
#             accuracy += acc.item()
#             total += len(y)
#             losses.append(loss.item())

#     scheduler.step()

#     if epoch % 1 == 0:
#         print("Epoch: {}/{}, loss: {}, Acc: {} %, took: {} s".format(epoch, n_epochs,
#                                                             np.round(np.mean(losses), 3),
#                                                             np.round(accuracy * 100 / total, 3),
#                                                             np.round(time.time() - start), 3))



# torch.save(model, f"results/class_based/FashionMNIST_example/lenet_gloabl")

curr_global_decoder = torch.load('results/vae/Omniglot_example/model0_curr_decoder')
batch_size = 32
n_prev_examples = 50
recon_prev, classes_prev, z_prev, embeddings_prev = vae_utils.generate_previous_data(
    curr_global_decoder,
    n_tasks=1,
    n_img=n_prev_examples,
    num_local=batch_size,
    return_z=True,
    translate_noise=True,
    same_z=False,
    equal_split=False)


fig = plt.figure()
for i in range(50):
    plt.subplot(5,10,i+1)
    plt.tight_layout()
    plt.imshow(recon_prev[i][0].cpu(), cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(classes_prev[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
