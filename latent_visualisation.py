import torch
import matplotlib.pyplot as plt
import dataset_gen
import umap
import pandas as pd
import seaborn as sns
import numpy as np
from multiband_vae.gan_experiments import gan_utils
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def visualise(args):
    # Get transformed data
    train_dataset, val_dataset = dataset_gen.__dict__[args.dataset](args.dataroot, args.skip_normalization,
                                                                    args.train_aug)

    # Prepare dataloaders
    # TODO add 1 class only eval datasets
    train_loaders, train_datasets, n_tasks = dataset_gen.split_data(args=args, dataset=train_dataset, drop_last=True)
    val_loaders, val_datasets, _ = dataset_gen.split_data(args=args, dataset=val_dataset, drop_last=False)

    task_id = 1
    generator = torch.load(f"models/gan/MNIST_example/model{task_id}_curr_global_generator", map_location="cuda")
    generator.eval()
    generator.translator.eval()
    fe = torch.load(f"models/gan/MNIST_example/model{task_id}_feature_extractor")
    fe.eval()

    # noise_cache = torch.load(f"models/{args.generator_type}/{args.experiment_name}/model{task_id}_noise_cache")
    # train_loader = train_loaders[-1]

    # def embed_imgs_cache(data_loader, noise_cache):
    #     # Encode all images in the data_laoder using model, and return both images and encodings
    #     img_list, embed_list = [], []
    #     labels = []
    #
    #     for i, batch in enumerate(data_loader):
    #         imgs, label = batch
    #
    #         emb_start_point = i * args.batch_size
    #         emb_end_point = min(len(train_loader.dataset), (i + 1) * args.batch_size)
    #         encoding = noise_cache[emb_start_point:emb_end_point]
    #
    #         img_list.append(imgs)
    #         embed_list.append(encoding)
    #         labels.append(label)
    #     return torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0), torch.cat(labels, dim=0)
    #
    # def embed_imgs(model, data_loader):
    #     # Encode all images in the data_laoder using model, and return both images and encodings
    #     img_list, embed_list = [], []
    #     model.eval()
    #     labels = []
    #     for imgs, label in data_loader:
    #         with torch.no_grad():
    #             encoding = model(imgs.to(device))
    #         img_list.append(imgs)
    #         embed_list.append(encoding)
    #         labels.append(label)
    #     return torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0), torch.cat(labels, dim=0)

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
            # palette=sns.color_palette("hls", 10),
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

    def embed_imgs_test(generator, fe):
        # Encode all images in the data_laoder using model, and return both images and encodings
        generator.eval()
        fe.eval()
        task_id = 2

        img_list = []
        embed_list = []
        labels = []
        fe_embed_list = []

        for i in range(100):
            generations, classes, random_noise, translator_emb = gan_utils.generate_previous_data(
                n_prev_tasks=(2 * task_id),
                n_prev_examples=100,
                curr_global_generator=generator)

            img_list.append(generations)
            embed_list.append(translator_emb)
            labels.append(classes)

            fe_encoding = fe(generations)
            fe_encoding = fe_encoding.detach()
            fe_embed_list.append(fe_encoding)

        generator_data = torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0), torch.cat(labels, dim=0)
        fe_data = torch.cat(img_list, dim=0), torch.cat(fe_embed_list, dim=0), torch.cat(labels, dim=0)

        umap_object = umap.UMAP(metric="cosine", n_neighbors=500)
        generator_embedded = umap_object.fit_transform(generator_data[1][:5000].cpu())
        plot_latent(generator_embedded, generator_data)

        # umap_object = umap.UMAP(metric="cosine", n_neighbors=500)
        fe_embedded = umap_object.fit_transform(fe_data[1][:5000].cpu())
        plot_latent(fe_embedded, fe_data)

        return

    embed_imgs_test(generator=generator, fe=fe)

    # umap_object = umap.UMAP(metric="cosine", n_neighbors=100)
    # train_img_embeds = embed_imgs(fe, train_loader)
    # train_img_embeds = embed_imgs_cache(train_loader, noise_cache)
    # train_embedded = umap_object.fit_transform(train_img_embeds[1][:5000].cpu())
    # plot_latent(train_embedded, train_img_embeds)

    return
