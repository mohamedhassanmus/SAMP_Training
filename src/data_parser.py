import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from src import misc_utils


def load_norm_data(data_dir):
    # Always use norm of training data
    input_norm_data = np.float32(np.loadtxt(osp.join(data_dir, 'train', 'InputNorm.txt')))
    input_mean = input_norm_data[0]
    input_std = input_norm_data[1]
    for i in range(input_std.size):
        if input_std[i] == 0:
            input_std[i] = 1

    output_norm_data = np.float32(np.loadtxt(osp.join(data_dir, 'train', 'OutputNorm.txt')))
    output_mean = output_norm_data[0]
    output_std = output_norm_data[1]
    for i in range(output_std.size):
        if output_std[i] == 0:
            output_std[i] = 1
    return input_mean, input_std, output_mean, output_std


class GoalNetData(Dataset):
    def __init__(self, data_dir="", input_dim=None, output_dim=None, normalize=True, train_data=True, **kwargs):
        if train_data:
            folder_name = 'train'
        else:
            folder_name = 'test'
        self.input_data = np.loadtxt(osp.join(data_dir, folder_name, 'Input.txt')).astype(np.float32)
        self.output_data = np.loadtxt(osp.join(data_dir, folder_name, 'Output.txt')).astype(np.float32)
        self.input_mean, self.input_std, self.output_mean, self.output_std = load_norm_data(data_dir)

        if normalize:
            self.input_data = misc_utils.Normalize(self.input_data, self.input_mean, self.input_std)
            self.output_data = misc_utils.Normalize(self.output_data, self.output_mean, self.output_std)

        self.input_dim = input_dim
        self.output_dim = output_dim

        print("input data shape {}".format(self.input_data.shape))
        print("output data shape {}".format(self.output_data.shape))

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        y = self.output_data[idx, :]
        x = self.input_data[idx, :]
        return {'x': x, 'y': y}


class MotionNetData(Dataset):
    def __init__(self, dtype=None, data_dir="", state_dim=None, normalize=True, L=8, train_data=True, **kwargs):
        self.dtype = dtype if dtype is not None else torch.float32
        self.state_dim = state_dim
        if train_data:
            folder_name = 'train'
        else:
            folder_name = 'test'
        self.input_data = np.loadtxt(osp.join(data_dir, folder_name, 'Input.txt')).astype(np.float32)
        self.output_data = np.loadtxt(osp.join(data_dir, folder_name, 'Output.txt')).astype(np.float32)
        self.sequences = np.loadtxt(osp.join(data_dir, folder_name, 'Sequences.txt')).astype(np.float32)
        self.input_mean, self.input_std, self.output_mean, self.output_std = load_norm_data(data_dir)

        if normalize:
            self.input_data = misc_utils.Normalize(self.input_data, self.input_mean, self.input_std)
            self.output_data = misc_utils.Normalize(self.output_data, self.output_mean, self.output_std)

        N = self.input_data.shape[0]
        self.input_data = torch.tensor(self.input_data, dtype=torch.float32).split(L)
        self.output_data = torch.tensor(self.output_data, dtype=torch.float32).split(L)
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32).split(L)

        if N % L != 0:
            self.input_data = self.input_data[:-1]
            self.output_data = self.output_data[:-1]
            self.sequences = self.sequences[:-1]

        # Each rollout should contains frames from the same motion sequence only
        valid_ids = []
        for i, seq in enumerate(self.sequences):
            if seq[0] == seq[-1]:
                valid_ids.append(i)
        valid_ids = torch.tensor(valid_ids, dtype=torch.long)
        print("Total no of rollouts {}, valid {}, invalid {}".format(len(self.sequences), valid_ids.shape[0],
                                                                     len(self.sequences) - valid_ids.shape[0]))

        self.input_data = [self.input_data[id] for id in valid_ids]
        self.output_data = [self.output_data[id] for id in valid_ids]
        self.sequences = [self.sequences[id] for id in valid_ids]

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        y = self.output_data[idx]
        x1 = self.input_data[idx][:, :self.state_dim]
        x2 = self.input_data[idx][:, self.state_dim:]
        return {'x1': x1, 'x2': x2, 'y': y}
