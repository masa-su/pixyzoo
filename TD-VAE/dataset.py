import codecs
import errno
import numpy as np
import os
import random
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision import transforms


# https://github.com/proceduralia/tgan-pytorch/blob/master/dataset.py
class MovingMNIST(Dataset):
    def __init__(self, dataset_path, n_frames=16, norm_mean=0, norm_std=1, rescale=None):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.data = torch.from_numpy(np.float32(np.load(dataset_path)))
        #Dataset will be of the form (L, T, C, H, W)
        self.data = self.data.permute(1, 0, 2, 3).unsqueeze(2)
        # rescale if specified
        if rescale:
            self.data = interpolate(self.data, scale_factor=(1, rescale, rescale))
        self.n_frames = n_frames #This can't be greater than 20
        
        #self.normalize = transforms.Normalize(self.norm_mean, self.norm_std)
        self.normalize = lambda x: (x - 128)/128
        self.denormalize = lambda x: x*128 + 128

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, i):
        T = self.data.size(1)
        ot = np.random.randint(T - self.n_frames) if T > self.n_frames else 0
        x = self.data[i, ot:(ot + self.n_frames)]
        x = self.normalize(x)
        return x


class MovingMNISTLR(Dataset):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    def __init__(self, root, train=True, download=False):
        self.root = os.path.expanduser(root)
        self.train = train
        
        if download:
            self.download()
        
        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
    def __getitem__(self, index):
        
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = np.array(img) / 255.
        img = np.roll(img, random.randrange(28), 1)
        
        left = random.choice([0, 1])
        seq = [img]
        for i in range(19):
            img = np.roll(img, -1, 1) if left else np.roll(img, 1, 1)
            seq.append(img)
        
        seq = torch.from_numpy(np.array(seq)).view(-1, 1, 28, 28).float()

        return seq
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
        
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')
        
def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


if __name__ == "__main__":
    dset = MovingMNIST("data/mnist_test_seq.npy")
    zero = dset[0]
    print(type(zero))
    print(zero.size())
    