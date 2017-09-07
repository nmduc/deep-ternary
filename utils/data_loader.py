import numpy as np 
import h5py
import random
import sys
import time 

class DataLoader():
    def __init__(self, db_fname, mean=0.0, scale=1.0, n_vals=1200):
        self.db_fname = db_fname
        self.n_vals = n_vals
        self.train_read_ind = 0
        self.val_read_ind = 0
        self.train_incides = None 
        self.val_incides = None
        self.x = None 
        self.x_train = None
        self.x_val = None 
        self.mean = mean
        self.scale = scale

    def prepare(self):
        self.load_data()
        self.split_train_val()

    def load_h5py(self, fname, n_samples=-1):
        with h5py.File(fname, 'r') as f:
            tic = time.time()
            data = f.get('patches')
            if n_samples > 0:
                data = data[:n_samples,:]
            np_data = np.array(data)
            toc = time.time()
            print 'Finish reading %d samples in %f seconds' %(data.shape[0], toc - tic) 
            return np_data

    def load_data(self):
        n_x = -1
        self.x = self.load_h5py(self.db_fname, n_x)

    def split_train_val(self):
        n_samples = self.x.shape[0]
        assert self.n_vals < n_samples, 'Validation set must be a subset of the whole dataset'
        train_cutoff = int(n_samples - self.n_vals)
        self.x_train = self.x[:train_cutoff,:]
        self.x_val = self.x[train_cutoff:,:]
        self.train_incides = np.arange(self.x_train.shape[0])
        self.val_incides = np.arange(self.x_val.shape[0])

    def shuffle_indices(self):
        print('Shuffle data indices')
        self.train_incides = np.random.permutation(self.x_train.shape[0])

    def get_num_samples(self, dataset):
        if dataset == 'train':
            return self.x_train.shape[0]
        if dataset == 'val':
            return self.x_val.shape[0]

    def get_data_dim(self):
        return self.x.shape[1]

    def next_batch(self, batch_size, dataset):
        ''' return next batch 
        Outputs:
            x: shape batch_size x dim_x
        '''
        x_set = None
        start_ind = None
        indices = None
        if dataset == 'train':
            x_set = self.x_train
            indices = self.train_incides
            start_ind = self.train_read_ind
        elif dataset == 'val':
            x_set = self.x_val
            indices = self.val_incides
            start_ind = self.val_read_ind
        end_ind = start_ind + batch_size
        n_samples = self.get_num_samples(dataset)
        flag = (end_ind >= n_samples)    # True if end of epoch
        if flag:
            end_ind = n_samples

        selected_indices = indices[start_ind:end_ind]

        x_indices = selected_indices
        x = x_set[x_indices]

        # update readers
        if flag:
            start_ind = 0
            if dataset == 'train':
                self.shuffle_indices()
        else:
            start_ind = end_ind
        if dataset == 'train':
            self.train_read_ind = start_ind
        elif dataset == 'val':
            self.val_read_ind = start_ind

        x = x.astype(np.float32)
        # normalize training data
        x -= self.mean
        x *= self.scale
        return (x, flag)
