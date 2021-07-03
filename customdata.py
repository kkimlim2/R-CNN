from random import random
import numpy as np

import torch
from torch.utils.data import Sampler
from torch.utils.data import Dataset


## custom batch 만들기 32개의 positive와 96개의 negative
class CustomSampler(Sampler):

    def __init__(self, data, positive_num, negative_num):
        self.data = data
        self.total = len(self.data)
        self.positive_num = positive_num
        self.negative_num = negative_num

        self.batch_size = self.positive_num + self.negative_num
        self.iter = self.total // self.batch_size

        self.positive_list = len([i for i, predict in enumerate(self.data) if predict != 0])

        self.idx = list(range(self.total))

    def __iter__(self):
        batch = list()

        for i in range(self.iter):
            temp = np.concatenate((random.sample(self.idx[:self.positive_list], self.positive_num),
                                   random.sample(self.idx[self.positive_list:], self.negative_num)))

            random.shuffle(temp)
            batch.extend(temp)
        return iter(batch)

    def __len__(self) -> int:
        return self.iter * self.batch_size

    def get_num_batch(self) -> int:
        return self.iter


class CustomDataset(Dataset):

    def __init__(self, train_images, train_labels):
        self.x_data = train_images
        self.y_data = [[i] for i in train_labels]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.LongTensor(self.y_data[idx])

        return x, y
