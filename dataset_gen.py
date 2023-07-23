import torch
import torch.utils.data as data
import torchvision
from torch.utils.data import Subset, ConcatDataset
from torchvision import transforms


def get_CI_eval_dataloaders(val_dataset_splits, n_tasks, batch_size):
    """
    Evaluation class incremental dataset contains data from all previous tasks
    """
    eval_loaders = []
    for task_id in range(n_tasks):
        datasets = []

        for i in range(task_id + 1):
            datasets.append(val_dataset_splits[i])

        eval_data = data.ConcatDataset(datasets)
        eval_loader = data.DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=True)
        eval_loaders.append(eval_loader)

    return eval_loaders


def get_CI_datasplit(dataset, n_tasks, n_classes_per_task, batch_size, drop_last):
    mask = None
    loaders = []
    datasets = []

    for task_id in range(n_tasks):
        idx = torch.zeros(len(dataset), dtype=torch.int)

        for class_id in range(n_classes_per_task):
            class_idx = (torch.tensor(dataset.targets)).clone().detach() == ((task_id * n_classes_per_task) + class_id)
            idx = idx | class_idx
            mask = idx.nonzero().reshape(-1)

        train_subset = Subset(dataset, mask)
        datasets.append(train_subset)
        loaders.append(data.DataLoader(dataset=train_subset,
                                       batch_size=batch_size,
                                       shuffle=False,  # NOTE -> no shuffling, because of gan noise cache
                                       drop_last=drop_last))  # TODO -> fix last batch issue

    return loaders, datasets


"""def split_data(args, dataset, drop_last):
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

    elif args.dataset.lower() in ["doublemnist"]:
        # 5 disjoint tasks, 2 classes each
        n_tasks = 10
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
        raise NotImplementedError"""


def MNIST(dataroot, skip_normalization=False, train_aug=True):
    # normalize = transforms.Normalize(mean=0.1307, std=0.3081)
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))  # for GAN 28x28

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
            # transforms.RandomCrop(32, padding=4),
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

    return train_dataset, val_dataset


def DoubleMNIST(dataroot, skip_normalization=False, train_aug=False):
    # normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
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
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset_fashion = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )

    train_dataset_mnist = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset_fashion = torchvision.datasets.FashionMNIST(
        dataroot,
        train=False,
        transform=val_transform
    )

    val_dataset_mnist = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    train_dataset_mnist.targets = train_dataset_mnist.targets + 10
    val_dataset_mnist.targets = val_dataset_mnist.targets + 10
    train_dataset = ConcatDataset([train_dataset_fashion, train_dataset_mnist])
    train_dataset.root = train_dataset_mnist.root
    val_dataset = ConcatDataset([val_dataset_fashion, val_dataset_mnist])
    val_dataset.root = val_dataset_mnist.root
    return train_dataset, val_dataset


def CIFAR10(dataroot, skip_normalization=False, train_aug=True):
    # normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

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

    return train_dataset, val_dataset


def CIFAR100(dataroot, skip_normalization=False, train_aug=True):
    # normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

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

    return train_dataset, val_dataset
