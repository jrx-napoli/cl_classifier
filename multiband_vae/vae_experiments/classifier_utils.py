import pandas as pd

class ReplayImageDataset(Dataset):
    def __init__(self, dataset, encoder, translator, transform=None, target_transform=None):
        self.img_data = dataset[0]
        self.img_labels = encoder(translator(dataset[0]))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return (self.img_data[idx], self.img_labels[idx])

    def append_data(self, data):
        pass
