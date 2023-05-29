import torch
import torch.utils.data as data
from torch.utils.data import Subset, ConcatDataset


def get_dataloader(args, dataset):
    
    n_tasks = None
    
    if args.dataset.lower() == "cifar100":
        # split cifar into 20 disjoint tasks, each containing 5 new classes
        n_tasks = 20
        
        loaders = []
        datasets = []
        for task_id in range(n_tasks):

            sub_datasets = []
            idx = torch.zeros_like(dataset.labels)

            for class_id in range(5):
                class_idx = torch.tensor(dataset.labels) == ((task_id * 5) + class_id)
                idx = idx | class_idx
                mask = idx.nonzero().reshape(-1)

            train_subset = Subset(dataset, mask)
            sub_datasets.append(train_subset)

            concat_dataset = ConcatDataset(sub_datasets)
            # NOTE -> no shuffeling, because of gan noise cache
            datasets.append(concat_dataset)
            loaders.append(data.DataLoader(dataset=concat_dataset, batch_size=args.gen_batch_size, shuffle=False, drop_last=True))
        
        return loaders, datasets, n_tasks
    

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