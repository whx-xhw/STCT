import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

def unpickle(file):
    # Tool for reading cifar data
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar10K(Dataset):
    def __init__(self, transform, dataset_path, path):
        self.transform = transform
        self.dataset_path = dataset_path
        train_data = []
        clean_label = []
        for n in range(1, 6):
            dpath = self.dataset_path + '/cifar-10-batches-py/data_batch_{}'.format(n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
            clean_label.append(data_dic['labels'])
        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))

        self.train_data = train_data

        self.targets = np.load(path)

    def __getitem__(self, index):
        img, noisy_target = self.train_data[index], self.targets[index]
        img = Image.fromarray(img)
        # return self.transform(img), noisy_target, clean_target
        return self.transform(img), noisy_target

    def __len__(self):
        return len(self.train_data)