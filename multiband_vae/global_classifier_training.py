import copy
import numpy as np
import torch
import time
from torch import optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms
import wandb

def validate(model, data):
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        for i, batch in enumerate(data):
            images = batch[0]
            labels = batch[1]
            images = images.cuda()
            x = model(images)
            value, pred = torch.max(x,1)
            pred = pred.data.cpu()
            total += x.size(0)
            correct += torch.sum(pred == labels)
        return correct*100./total

def get_data_loader(dataset_splits, batch_size=32):
    datasets = []
    for i, split in enumerate(dataset_splits):
        datasets.append(dataset_splits[i])

    dataset = data.ConcatDataset(datasets)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def get_dataset(args):
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))  # for  28x28
    # normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root=args.dataroot,
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = torchvision.datasets.FashionMNIST(
        root=args.dataroot,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset):
    train_data_loader = data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_data_loader = data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    return train_data_loader, val_data_loader

def count_correct(pred, labels):
    value, prediction = torch.max(pred, 1)
    correct = torch.sum(prediction == labels)
    return correct.item()

def combine_models(feature_extractor, classifier):
    model = nn.Sequential(
        feature_extractor,
        classifier
    )
    return model

def test_architecture(args, feature_extractor, classifier, device, numb_epoch=100, lr=1e-3, scheduler_rate=0.99):

    model = combine_models(feature_extractor, classifier).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_rate)
    accuracies = []
    max_accuracy = 0
    wandb.watch(model)

    train_ds, val_ds = get_dataset(args)
    train_dl, val_dl = get_dataloaders(train_ds, val_ds)
    
    for epoch in range(numb_epoch):
        losses = []
        train_accuracy = 0
        total = 0
        start = time.time()

        for i, batch in enumerate(train_dl):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            pred = model(images)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_accuracy += count_correct(pred, labels)
            total += len(images)
            losses.append(loss.item())

        print(f'Epoch: {epoch}/{numb_epoch}, loss: {np.round(np.mean(losses), 3)}, '+
              f'train-accuracy: {np.round(train_accuracy*100/total, 3)} %, '+
              f'took: {np.round(time.time() - start, 3)}s.')
        
        if epoch % 1 == 0:
            # test split accuracy
            accuracy = float(validate(model, val_dl))
            accuracies.append(accuracy)
            print(f'test-accuracy: {np.round(accuracy, 3)} %')
            wandb.log({"Test-split accuracy: ": (np.round(accuracy, 3))})

            if accuracy > max_accuracy:
                best_model = copy.deepcopy(model)
                max_accuracy = accuracy
                print(f'New best test-split accuracy: {np.round(accuracy, 3)} %')
                wandb.log({"Best test-split accuracy: ": (np.round(accuracy, 3))})

        
        scheduler.step()
        
    plt.plot(accuracies)
    print(f'Best accuracy: {max_accuracy}')
    return best_model
