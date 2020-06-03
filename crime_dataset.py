import numpy as np
from matplotlib import pyplot as plt
import os
import torch


class CrimeDataset():
    def __init__(self, device):
        reader = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/communities.data'))

        attributes = []
        while True:
            line = reader.readline().split(',')
            if len(line) < 128:
                break
            line = ['-1' if val == '?' else val for val in line]
            line = np.array(line[5:], dtype=np.float)
            attributes.append(line)
        reader.close()

        attributes = np.stack(attributes, axis=0)

        reader = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/names'))
        names = []
        for i in range(128):
            line = reader.readline().split()[1]
            if i >= 5:
                names.append(line)
        names = np.array(names)

        attributes = attributes[np.random.permutation(range(attributes.shape[0])), :]

        val_size = 500
        self.train_labels = attributes[val_size:, -1:]
        self.test_labels = attributes[:val_size:, -1:]

        attributes = attributes[:, :-1]
        selected = np.argwhere(np.array([np.min(attributes[:, i]) for i in range(attributes.shape[1])]) >= 0).flatten()
        self.train_features = attributes[val_size:, selected]
        self.test_features = attributes[:val_size:, selected]
        self.names = names[selected]

        self.train_ptr = 0
        self.test_ptr = 0
        self.x_dim = self.train_features.shape[1]

        self.train_size = self.train_features.shape[0]
        self.test_size = self.test_features.shape[0]
        self.device = device

    def train_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.train_features.shape[0]
            self.train_ptr = 0
        if self.train_ptr + batch_size > self.train_features.shape[0]:
            self.train_ptr = 0
        bx, by = self.train_features[self.train_ptr:self.train_ptr+batch_size], \
                 self.train_labels[self.train_ptr:self.train_ptr+batch_size]
        self.train_ptr += batch_size
        if self.train_ptr == self.train_features.shape[0]:
            self.train_ptr = 0
        return torch.from_numpy(bx).float().to(self.device), torch.from_numpy(by).float().to(self.device)

    def test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.test_features.shape[0]
            self.train_ptr = 0
        if self.test_ptr + batch_size > self.test_features.shape[0]:
            self.test_ptr = 0
        bx, by = self.test_features[self.test_ptr:self.test_ptr+batch_size], \
                 self.test_labels[self.test_ptr:self.test_ptr+batch_size]
        self.test_ptr += batch_size
        if self.test_ptr == self.test_features.shape[0]:
            self.test_ptr = 0
        return torch.from_numpy(bx).float().to(self.device), torch.from_numpy(by).float().to(self.device)


if __name__ == '__main__':
    dataset = CrimeDataset()
    print(dataset.names)
    print(dataset.train_features.shape, dataset.train_labels.shape)



