import torch
import numpy as np
import torch.utils.data as data


def calculate_translated_latent_size(args):
    if args.generator_type == "vae":
        return args.gen_d * args.gen_latent_size
    elif args.generator_type == "gan":
        return 100 # NOTE -> fixed latent size for GAN
    else:
        raise NotImplementedError

def prepare_accuracy_data(n_tasks):
    x = []
    accuracy = []
    for task in range(n_tasks):
        accuracy.append([])
        x.append([])
        for i, _ in enumerate(x):
            x[i].append(task)
    return x, accuracy

    # fig = plt.figure()
    # for i in range(50):
    #     plt.subplot(5,10,i+1)
    #     plt.tight_layout()
    #     plt.imshow(local_imgs[i][0].cpu(), cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(local_classes[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()
    # print(f'local_classes: {local_classes}')