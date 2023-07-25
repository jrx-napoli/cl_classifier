import matplotlib.pyplot as plt
import torch

from gan_experiments import gan_utils

task_id = 4
generator = torch.load(f"models/gan/CIFAR10_example/model{task_id}_curr_global_generator", map_location="cuda")
generator.eval()
generator.translator.eval()

n_prev_examples = 50

with torch.no_grad():
    print(f'n_prev_examples: {n_prev_examples}')

    generations, random_noise, classes = gan_utils.generate_previous_data(
        n_prev_tasks=2 * (task_id + 1),
        n_prev_examples=n_prev_examples,
        curr_global_generator=generator,
        biggan_training=False)

    generations = ((generations + 1) * 255) / 2
    generations = generations.cpu().long().permute((0, 2, 3, 1)).numpy()
    # print(classes)
    # print(len(classes))

    fig = plt.figure()
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.tight_layout()
        plt.imshow(generations[i])
        plt.title("Ground Truth: {}".format(classes[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
