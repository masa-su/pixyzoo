import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MazeDataset(Dataset):
    def __init__(self, length, partition='train', path="3dmaze_32.npy"):
        self.partition = partition
        dataset = np.load(path)
        num_seqs = int(dataset.shape[0]*0.8)
        if self.partition == 'train':
            # (B, T, C, H, W)
            self.state = dataset[:num_seqs].transpose(0, 1, 4, 2, 3) / 1.
        else:
            self.state = dataset[num_seqs:].transpose(0, 1, 4, 2, 3) / 1.
        # self._calculate_mean_std()
        self.mean = np.ones([self.state.shape[0], self.state.shape[1], 3, self.state.shape[3], self.state.shape[4]])
        self.std = np.ones([self.state.shape[0], self.state.shape[1], 3, self.state.shape[3], self.state.shape[4]])
        self.mean[:, :, 0, :, :] = 91.77488098
        self.mean[:, :, 1, :, :] = 101.29845624
        self.mean[:, :, 2, :, :] = 103.91033606

        self.std[:, :, 0, :, :] = 73.63031478
        self.std[:, :, 1, :, :] = 73.71162061
        self.std[:, :, 2, :, :] = 76.89208084

        self.state = (self.state - self.mean) / self.std

        self.length = length
        self.full_length = self.state.shape[1]
    
    def __getitem__(self, index):
        idx0 = np.random.randint(0, self.full_length - self.length)
        idx1 = idx0 + self.length

        state = self.state[index, idx0:idx1].astype(np.float32)
        return state
    
    def __len__(self):
        return self.state.shape[0]
    
    def _calculate_mean_std(self):
        # print(self.state.shape)
        std = np.std(self.state, axis=(0, 1, 3, 4))
        mean = np.mean(self.state, axis=(0, 1, 3, 4))
        print("mean: ", mean)
        print(mean.shape)
        print("std: ", std)
        print(std.shape)
        print(np.max(self.state))
        print(np.min(self.state))


class BounceDataset(Dataset):
    def __init__(self, length, partition='train', path="bouncing_ball_black.npy"):
        self.partition = partition
        dataset = np.load(path)
        num_seqs = int(dataset.shape[0]*0.8)
        if self.partition == 'train':
            # (B, T, C, H, W)
            self.state = dataset[:num_seqs].transpose(0, 1, 4, 2, 3) / 1.
        else:
            self.state = dataset[num_seqs:].transpose(0, 1, 4, 2, 3) / 1.
        # self._calculate_mean_std()
        self.mean = np.ones([self.state.shape[0], self.state.shape[1], 3, self.state.shape[3], self.state.shape[4]])
        self.std = np.ones([self.state.shape[0], self.state.shape[1], 3, self.state.shape[3], self.state.shape[4]])
        
        self.mean[:, :, 0, :, :] = 6.75852277
        self.mean[:, :, 1, :, :] = 6.67423723
        self.mean[:, :, 2, :, :] = 6.7084246

        self.std[:, :, 0, :, :] = 40.96029391
        self.std[:, :, 1, :, :] = 40.71099424
        self.std[:, :, 2, :, :] = 40.81231813

        self.state = (self.state - self.mean) / self.std

        self.length = length
        self.full_length = self.state.shape[1]
    
    def __getitem__(self, index):
        idx0 = np.random.randint(0, self.full_length - self.length)
        idx1 = idx0 + self.length

        state = self.state[index, idx0:idx1].astype(np.float32)
        return state
    
    def __len__(self):
        return self.state.shape[0]
    
    def _calculate_mean_std(self):
        # print(self.state.shape)
        std = np.std(self.state, axis=(0, 1, 3, 4))
        mean = np.mean(self.state, axis=(0, 1, 3, 4))
        print("mean: ", mean)
        print(mean.shape)
        print("std: ", std)
        print(std.shape)
        print(np.max(self.state))
        print(np.min(self.state))



def preprocess(image):
    return image


def postprocess_maze(image):
    mean = torch.ones_like(image)
    std = torch.ones_like(image)
    mean[:, :, 0, :, :] = 91.77488098
    mean[:, :, 1, :, :] = 101.29845624
    mean[:, :, 2, :, :] = 103.91033606

    std[:, :, 0, :, :] = 73.63031478
    std[:, :, 1, :, :] = 73.71162061
    std[:, :, 2, :, :] = 76.89208084

    image = image * std + mean
    image = torch.clamp(image, min=0.0, max=255.0) / 255.0
    return image


def postprocess_bounce(image):
    mean = torch.ones_like(image)
    std = torch.ones_like(image)
    mean[:, :, 0, :, :] = 6.75852277
    mean[:, :, 1, :, :] = 6.67423723
    mean[:, :, 2, :, :] = 6.7084246

    std[:, :, 0, :, :] = 40.96029391
    std[:, :, 1, :, :] = 40.71099424
    std[:, :, 2, :, :] = 40.81231813


    image = image * std + mean
    image = torch.clamp(image, min=0.0, max=255.0) / 255.0
    return image


def maze_dataloader(seq_size, init_size, batch_size, test_size=16, data_path="3dmaze_32.npy"):
    train_loader = MazeDataset(length=seq_size + init_size * 2, partition='train', path=data_path)
    test_loader = MazeDataset(length=seq_size + init_size * 2, partition='test', path=data_path)
    train_loader = DataLoader(dataset=train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_loader, batch_size=test_size, shuffle=False)
    return train_loader, test_loader


def bounce_dataloader(seq_size, init_size, batch_size, test_size=16, data_path="bouncing_ball_black.npy"):
    train_loader = BounceDataset(length=seq_size + init_size * 2, partition='train', path=data_path)
    test_loader = BounceDataset(length=seq_size + init_size * 2, partition='test', path=data_path)
    train_loader = DataLoader(dataset=train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_loader, batch_size=test_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    # MazeDataset()
    # BounceDataset(length=5)
    pass