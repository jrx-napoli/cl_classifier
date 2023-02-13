import copy
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as data


def validate(model, data):
    with torch.no_grad():
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

def combine_models(feature_extractor, classifier):
    model = nn.Sequential(
        feature_extractor,
        classifier
    )
    return model

def test_architecture(feature_extractor, classifier, train_dataset_splits, val_dataset_splits, device, numb_epoch=3, lr=1e-3):

    model = combine_models(feature_extractor, classifier).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accuracies = []
    max_accuracy = 0

    train_dl = get_data_loader(train_dataset_splits)
    val_dl = get_data_loader(val_dataset_splits)
    
    for epoch in range(numb_epoch):
        for i, batch in enumerate(train_dl):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            optimizer.zero_grad()
            
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

        accuracy = float(validate(model, val_dl))
        accuracies.append(accuracy)
        
        print('Epoch:', epoch+1, "Accuracy :", accuracy, '%')
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(model)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        
    plt.plot(accuracies)
    return best_model
