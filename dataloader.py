import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flags import FLAGS


class DataLoader:
    '''
        Inherit this class, and implement your own dataloader, 
        you must implement next_batch function, which return a dict
        
    '''
    def __init__(self, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = 0
    def __iter__(self):
        return self
    def __next__(self):
        return self.next_batch(self.batch_size)
    def get_batch_data(self, start, end):
        raise NotImplementedError
    def next_batch(self, batch_size):
        if self.idx >= self.num_samples:
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.order)
            raise StopIteration
        start = self.idx
        end = min(self.idx + batch_size, self.num_samples)
        batch_data = self.get_batch_data(start, end)
        self.idx += batch_size
        return batch_data


class UserDataLoader(DataLoader):

    def __init__(self, file, batch_size, shuffle=True):
        super().__init__(batch_size, shuffle)
        self.data = self._preprocess(file)
        self.batch_num = int(np.ceil(self.num_samples / self.batch_size))
        self.order = list(np.arange(self.num_samples))
        if self.shuffle:
            np.random.shuffle(self.order)

    def _preprocess(self, file):
        def _normalize(data):
            return (data - np.mean(data, axis=0, keepdims=True)) / \
                    np.std(data, axis=0, keepdims=True)
        ori_data = pd.read_excel(file,header=None)
        labels = np.array(ori_data.iloc[1:,5].as_matrix(),dtype=np.int32)
        inputs = _normalize(ori_data.iloc[1:,:4].as_matrix())
        self.num_samples = inputs.shape[0]
        self.num_features = 4

        return {"labels":labels, "inputs": inputs}


    def get_batch_data(self, start, end):
        batch_data = {
            'inputs' : self.data['inputs'][start:end],
            'labels' : self.data['labels'][start:end]
        }
        return batch_data

    


if __name__ == '__main__':
    dataloader = UserDataLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','train.xlsx'), FLAGS.batch_size)
    for batch_data in dataloader:
        print(batch_data['inputs'].shape)
