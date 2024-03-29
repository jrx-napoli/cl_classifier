import torch
import matplotlib.pyplot as plt
from vae_experiments import vae_utils
from gan_experiments import gan_utils
from visualise import *


task_id = 19
generator = torch.load(f"results/gan/CIFAR100_example/model{task_id}_curr_global_generator", map_location="cuda")
generator.eval()
generator.translator.eval()

# class_table = curr_global_decoder.class_table
n_prev_examples = 400
# n_prev_examples = int(64 * min(task_id, 5))

with torch.no_grad():

    print(f'n_prev_examples: {n_prev_examples}')

    generations, random_noise, classes, translator_emb = gan_utils.generate_previous_data(
        n_prev_tasks=5*(task_id + 1),
        n_prev_examples=n_prev_examples,
        curr_global_generator=generator)

    print(classes)
    print(len(classes))

    fig = plt.figure()
    for i in range(50):
        plt.subplot(5,10,i+1)
        plt.tight_layout()
        plt.imshow(generations[i][0].cpu(), cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(classes[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
