import torch
import matplotlib.pyplot as plt
from vae_experiments import vae_utils
from gan_experiments import gan_utils
from visualise import *



generator = torch.load(f"results/test/MNIST_example/model1_curr_global_generator", map_location="cuda")


# class_table = curr_global_decoder.class_table
n_prev_examples = 52
recon_prev, z_prev, task_ids_prev = gan_utils.generate_previous_data(
    4,
    n_prev_examples=n_prev_examples,
    curr_global_generator=generator)


fig = plt.figure()
for i in range(50):
    plt.subplot(5,10,i+1)
    plt.tight_layout()
    plt.imshow(recon_prev[i][0].cpu(), cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(task_ids_prev[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()


