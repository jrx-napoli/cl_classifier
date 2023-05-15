import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, Dataset


def create_CI_eval_dataloaders(n_tasks, val_dataset_splits, args):
    """
    Evaluation CI dataset contains data from all previous tasks
    """
    eval_loaders = []
    for task_id in range(n_tasks):
        datasets = []

        for i in range(task_id + 1):
            datasets.append(val_dataset_splits[i])

        eval_data = data.ConcatDataset(datasets)
        eval_loader = data.DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True)
        eval_loaders.append(eval_loader)

    return eval_loaders


def split_data(args, dataset, drop_last):
    """
    Dataset-dependant data split
    """
    n_tasks = None
    mask = None
    loaders = []
    datasets = []

    if args.dataset.lower() == "cifar100":
        # 20 disjoint tasks, 5 classes each
        n_tasks = 20
        for task_id in range(n_tasks):

            idx = torch.zeros(len(dataset), dtype=torch.int)

            for class_id in range(5):
                class_idx = (torch.tensor(dataset.labels)).clone().detach() == ((task_id * 5) + class_id)
                idx = idx | class_idx
                mask = idx.nonzero().reshape(-1)

            train_subset = Subset(dataset, mask)
            # NOTE -> no shuffling, because of gan noise cache
            datasets.append(train_subset)
            loaders.append(data.DataLoader(dataset=train_subset, batch_size=args.batch_size, shuffle=False,
                                           drop_last=drop_last))  # TODO -> fix last batch issue

        return loaders, datasets, n_tasks

    elif args.dataset.lower() in ["mnist", "fashionmnist", "cifar10"]:
        # 5 disjoint tasks, 2 classes each
        n_tasks = 5
        for task_id in range(n_tasks):

            idx = torch.zeros(len(dataset), dtype=torch.int)

            for class_id in range(2):
                class_idx = (torch.tensor(dataset.labels)).clone().detach() == ((task_id * 2) + class_id)
                idx = idx | class_idx
                mask = idx.nonzero().reshape(-1)

            train_subset = Subset(dataset, mask)
            # NOTE -> no shuffling, because of gan noise cache
            datasets.append(train_subset)
            loaders.append(data.DataLoader(dataset=train_subset, batch_size=args.batch_size, shuffle=False,
                                           drop_last=False))
        return loaders, datasets, n_tasks

    else:
        raise NotImplementedError


def MNIST(dataroot, skip_normalization=False, train_aug=True):
    normalize = transforms.Normalize(mean=0.1307, std=0.3081)
    # normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))  # for GAN 28x28

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform

    if train_aug:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    train_dataset = DataWrapper(train_dataset)
    val_dataset = DataWrapper(val_dataset)

    return train_dataset, val_dataset


def FashionMNIST(dataroot, skip_normalization=False, train_aug=True):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform

    if train_aug:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    train_dataset = DataWrapper(train_dataset)
    val_dataset = DataWrapper(val_dataset)

    return train_dataset, val_dataset


def CIFAR10(dataroot, skip_normalization=False, train_aug=True):
    # normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # same normalization as for Multiband Gan training

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform

    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    train_dataset = DataWrapper(train_dataset)
    val_dataset = DataWrapper(val_dataset)

    return train_dataset, val_dataset


def CIFAR100(dataroot, skip_normalization=False, train_aug=True):
    # normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # same normalization as for Multiband Gan training

    if skip_normalization:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = val_transform

    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    train_dataset = DataWrapper(train_dataset)
    val_dataset = DataWrapper(val_dataset)

    return train_dataset, val_dataset


class DataWrapper(Dataset):
    """
    Dataset wrapper with access to class labels
    """

    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.labels = dataset.targets

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.labels)
