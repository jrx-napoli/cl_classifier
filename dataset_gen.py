import torch
import torch.utils.data as data
from torch.utils.data import Subset, ConcatDataset


def create_CI_eval_dataloaders(task_names, val_dataset_splits, args):
    # Eval CI dataset contains data from all previous tasks
    eval_loaders = []
    for task_id in range(task_names):
        datasets = []

        for i in range(task_id + 1):
            datasets.append(val_dataset_splits[i])

        eval_data = data.ConcatDataset(datasets)
        eval_loader = data.DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True)
        eval_loaders.append(eval_loader)
    
    return eval_loaders

def split_data(args, dataset, drop_last):
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
                class_idx = (dataset.labels).clone().detach() == ((task_id * 5) + class_id)
                idx = idx | class_idx
                mask = idx.nonzero().reshape(-1)

            train_subset = Subset(dataset, mask)
            sub_datasets.append(train_subset)

            concat_dataset = ConcatDataset(sub_datasets)
            # NOTE -> no shuffeling, because of gan noise cache
            datasets.append(concat_dataset)
            loaders.append(data.DataLoader(dataset=concat_dataset, batch_size=args.batch_size, shuffle=False, drop_last=drop_last)) # TODO -> fix last batch issue
        
        return loaders, datasets, n_tasks
