import torch
import torch.utils.data as data
from torch.utils.data import Subset, ConcatDataset


def get_dataloader(args, dataset):

    if args.dataset.lower() == "cifar100":
        # split cifar into 20 disjoint tasks, each containing 5 new classes    
        
        loaders = []
        for task_id in range(20):

            sub_datasets = []
            idx = torch.zeros_like(dataset.labels)

            for class_id in range(5):
                class_idx = torch.tensor(dataset.labels) == ((task_id * 5) + class_id)
                idx = idx | class_idx
                mask = idx.nonzero().reshape(-1)

            train_subset = Subset(dataset, mask)
            sub_datasets.append(train_subset)

            concat_dataset = ConcatDataset(sub_datasets)
            loaders.append(data.DataLoader(dataset=concat_dataset, batch_size=args.gen_batch_size, shuffle=True, drop_last=False))
        
        return loaders