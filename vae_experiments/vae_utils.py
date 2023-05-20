from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np
import torch


class BitUnpacker:
    results_map = {}

    @classmethod
    def unpackbits(cls, y, num_bits):
        with torch.no_grad():
            x = y + 1
            if num_bits == 0:
                return torch.Tensor([])

            if num_bits in cls.results_map:
                mask = cls.results_map[num_bits]
            else:
                print("Mask for num_bits={} does not exist, calculating one.".format(num_bits))

                mask = 2 ** (num_bits - 1 - torch.arange(num_bits).view([1, num_bits])).long()
                cls.results_map[num_bits] = mask

            x = x.view(-1, 1).long()

            return (x & mask).bool().float() * 2 - 1


def prepare_class_samplres(task_id, class_table):
    class_samplers = []
    for task_id in range(task_id):
        local_probs = class_table[task_id] * 1.0 / torch.sum(class_table[task_id])
        class_samplers.append(torch.distributions.categorical.Categorical(probs=local_probs))
    return class_samplers


def plot_results(experiment_name, curr_global_decoder, class_table, n_tasks, n_img=5, same_z=False,
                 translate_noise=True, suffix="", starting_point=None):
    curr_global_decoder.eval()

    if starting_point != None:
        task_ids = np.repeat(starting_point, n_img * (n_tasks + 1))
    else:
        task_ids = np.repeat(list(range(n_tasks + 1)), n_img)
    task_ids = torch.from_numpy(task_ids).float()
    class_samplers = prepare_class_samplres(n_tasks + 1, class_table)

    if same_z:
        z = torch.randn([n_img, curr_global_decoder.latent_size])
        z = z.repeat([n_tasks + 1, 1]).to(curr_global_decoder.device)
    else:
        z = torch.randn([n_img * (n_tasks + 1), curr_global_decoder.latent_size]).to(curr_global_decoder.device)

    sampled_classes = []
    for i in range(n_tasks + 1):  ## Including current class
        sampled_classes.append(class_samplers[i].sample([n_img]))
    sampled_classes = torch.cat(sampled_classes)
    example = generate_images(curr_global_decoder, z, task_ids, sampled_classes, translate_noise=translate_noise)
    example = example.cpu().detach().numpy()
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_tasks + 1, n_img),
                     axes_pad=0.5,
                     )

    if same_z:
        info = "random_bin_vector"
    else:
        info = ""
    for ax, im, target in zip(grid, example, task_ids.cpu().detach().numpy()):  # sampled_classes):
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 0, 1)
        ax.imshow(im.squeeze())
        ax.set_title(f'Task id{info}: {int(target)}')

    plt.savefig("results/" + experiment_name + "/generations_task_" + str(n_tasks) + suffix)
    plt.close()


def generate_images(curr_global_decoder, z, y, return_emb=False, translate_noise=True):
    if return_emb:
        example, emb = curr_global_decoder(z, y, y, return_emb=return_emb, translate_noise=translate_noise)
        return example, emb
    else:
        example = curr_global_decoder(z, y, y, return_emb=return_emb, translate_noise=translate_noise)
        return example


def generate_noise_for_previous_data(n_img, latent_size, tasks_dist, device, num_local=0, same_z=False):

    if same_z:
        z_max = torch.randn([max(tasks_dist + torch.tensor([num_local])), latent_size]).to(device)
        z = []
        for task_id, n_img in enumerate(tasks_dist):
            z.append(z_max[:n_img])            
        z = torch.cat(z)
        return z
    else:
        z = torch.randn([n_img, latent_size]).to(device)
        return z


def generate_previous_data(curr_global_decoder, n_tasks, n_img, num_local=0, translate_noise=True,
                           same_z=False, return_z=False, equal_split=False, recent_task_only=False):
    curr_global_decoder.eval()
    with torch.no_grad():
        class_table = curr_global_decoder.class_table

        if equal_split:
            class_table[class_table > 0] = 1

        if recent_task_only:
            curr_class_table = class_table[n_tasks-1:n_tasks]
            n_tasks = 1
        else:    
            curr_class_table = class_table[:n_tasks]
        
        tasks_dist = torch.sum(curr_class_table, dim=1) * n_img // torch.sum(curr_class_table)
        tasks_dist[0:n_img - tasks_dist.sum()] += 1  # To fix the division
        assert sum(tasks_dist) == n_img

        class_samplers = prepare_class_samplres(n_tasks, curr_class_table)

        sampled_classes = []
        for task_id in range(n_tasks):
            if tasks_dist[task_id] > 0:
                sampled_classes.append(class_samplers[task_id].sample(tasks_dist[task_id].view(-1, 1)))
        sampled_classes = torch.cat(sampled_classes)
        assert len(sampled_classes) == n_img

        z_combined = generate_noise_for_previous_data(n_img, 
                                                      curr_global_decoder.latent_size,
                                                      tasks_dist,
                                                      device=curr_global_decoder.device, 
                                                      num_local=num_local,
                                                      same_z=same_z)

        if return_z:
            example, embeddings = generate_images(curr_global_decoder, 
                                                    z_combined, 
                                                    sampled_classes,
                                                    return_emb=True,
                                                    translate_noise=translate_noise)
            return example, sampled_classes, z_combined, embeddings
        else:
            example = generate_images(curr_global_decoder, 
                                        z_combined, 
                                        sampled_classes,
                                        translate_noise=translate_noise)
            return example, sampled_classes
