import torch
import matplotlib.pyplot as plt

import dataset_gen
from gan_experiments import gan_utils

import umap
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Get transformed data
train_dataset, val_dataset = dataset_gen.__dict__["MNIST"]("data")

# Prepare dataloaders
train_loaders, train_datasets, n_tasks = dataset_gen.split_data(args=args, dataset=train_dataset, drop_last=True)
val_loaders, val_datasets, _ = dataset_gen.split_data(args=args, dataset=val_dataset, drop_last=False)
ci_eval_dataloaders = dataset_gen.create_CI_eval_dataloaders(n_tasks=n_tasks, val_dataset_splits=val_datasets,
                                                             args=args)



task_id = 4
generator = torch.load(f"models/gan/MNIST_example/model{task_id}_curr_global_generator", map_location="cuda")
generator.eval()
generator.translator.eval()
fe = torch.load(f"models/gan/MNIST_example/model{task_id}_feature_extractor")
fe.eval()



def embed_imgs(model, data_loader):
    # Encode all images in the data_laoder using model, and return both images and encodings
    img_list, embed_list = [], []
    model.eval()
    labels = []
    for imgs, label in data_loader:
        with torch.no_grad():
            encoding = model(imgs)
        img_list.append(imgs)
        embed_list.append(encoding)
        labels.append(label)
    return torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0), torch.cat(labels, dim=0)


umap_object = umap.UMAP(metric="cosine", n_neighbors=100)
train_img_embeds = embed_imgs(vae, train_loader)
test_img_embeds = embed_imgs(vae, test_loader)
train_embedded = umap_object.fit_transform(train_img_embeds[1][:5000].cpu())


def plot_latent(train_embedded, train_img_embeds, n_data=5000):
    data = pd.DataFrame(train_embedded[:n_data])
    data["label"] = train_img_embeds[2][:n_data].cpu().numpy()
    examples = []
    examples_locations = []
    for i in np.random.randint(0, n_data, 40):
        examples.append(train_img_embeds[0][i].squeeze(0).cpu().numpy())
        examples_locations.append(data.iloc[i])
    fig, ax = plt.subplots(figsize=(12, 10))
    # ax.scatter(noises_to_plot_tsne[0],noises_to_plot_tsne[1],c=noises_to_plot_tsne["batch"],s=3,alpha=0.8)
    sns.scatterplot(
        x=0, y=1,
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.1
    )
    for location, example in zip(examples_locations, examples):
        x, y = location[0], location[1]
        label = int(location["label"])
        ab = AnnotationBbox(OffsetImage(example, cmap=plt.cm.gray_r, zoom=1), (x, y), frameon=True,
                            bboxprops=dict(facecolor=sns.color_palette("hls", 10)[label], boxstyle="round"))
        ax.add_artist(ab)
    plt.show()


plot_latent(train_embedded, train_img_embeds)

# class_table = curr_global_decoder.class_table
# n_prev_examples = 58
#
# with torch.no_grad():
#     print(f'n_prev_examples: {n_prev_examples}')
#
#     generations, random_noise, classes, translator_emb = gan_utils.generate_previous_data(
#         n_prev_tasks=2 * task_id,
#         n_prev_examples=n_prev_examples,
#         curr_global_generator=generator)
#
#     print(classes)
#     print(len(classes))
#
#     fig = plt.figure()
#     for i in range(50):
#         plt.subplot(5, 10, i + 1)
#         plt.tight_layout()
#         plt.imshow(generations[i][0].cpu(), cmap='gray', interpolation='none')
#         plt.title("Ground Truth: {}".format(classes[i]))
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()
