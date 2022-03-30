import os
import numpy as np
import torch
from tqdm import tqdm
import glob
import copy
import cv2

from utils import postprocess_observation, preprocess_observation_


def trans_data(data):
    keys = data[0].keys()
    n = len(data)
    output = dict()
    for key in tqdm(keys, desc="trans data"):
        output[key] = [data[i][key] for i in range(n)]

    return output


def clip_episode(data):
    data_length = []
    for k in data.keys():
        data_length.append(len(data[k]))
    episode_length = np.min(data_length)
    output = dict()
    for k in data.keys():
        output[k] = data[k][:episode_length]
    return output


def preprocess_image_data(auged_data, name, idx=0, size=(64, 64), h_split=2, w_split=2):
    # idx = int(idx % (w_split * h_split))
    # idx_w = int((idx % w_split))
    # idx_h = int((idx // w_split))

    # _, h, w, _ = auged_data[name].shape
    # dh = h - size[0]
    # dw = w - size[1]

    # ims = auged_data[name][:, dh * idx_h : size[0] + dh * idx_h, dw * idx_w : size[1] + dw * idx_w, :].transpose(0, 3, 1, 2)
    # auged_data[name] = np.array(ims)

    idx = int(idx % (w_split * h_split))
    w = int((idx % w_split) * (100 / (w_split - 1)))
    h = int((idx // h_split) * (100 / (h_split - 1)))

    ims = []
    episode_length = len(auged_data[name])
    for t in tqdm(range(episode_length), desc="clip image"):
        # im = auged_data["image"][t, :800, 560:-560]
        im = auged_data[name][t, 0:900, 510 : 510 + 900][h : h + 800, w : w + 800]

        im = cv2.resize(im, size).astype(np.int)[:, :, ::-1].transpose(2, 0, 1)
        ims.append(im)
    auged_data[name] = np.array(ims)
    return auged_data


def preprocess_data(data, idx=0, size=(64, 64), h_split=2, w_split=2, bit_depth=5):
    auged_data = copy.deepcopy(data)

    for name in auged_data.keys():
        if "image" in name:
            auged_data = preprocess_image_data(auged_data, name, idx=idx, size=size, h_split=h_split, w_split=w_split)

    auged_data = clip_episode(auged_data)

    output = dict()
    for k in auged_data.keys():
        output[k] = torch.tensor(auged_data[k], dtype=torch.float32)
        if "image" in k:
            preprocess_observation_(output[k], bit_depth)
    output["done"][-1] = 1.0
    output["nonterminals"] = 1.0 - output["done"].unsqueeze(-1)
    return output


def load_data_cobotta(dataset_dir, n_episode=1, n_augment=4):
    dataset = []
    file_names = glob.glob(os.path.join(dataset_dir, "*.npy"))
    if n_episode == None:
        n_episode = len(file_names)
    else:
        n_episode = np.min((n_episode, len(file_names)))
    print("find %d npy files!" % len(file_names))
    for file_name in file_names[:n_episode]:
        print(file_name)
        data = np.load(file_name, allow_pickle=True).item()
        for i in range(n_augment):
            auged_data = preprocess_data(data, i)
            dataset.append(auged_data)
    return dataset


class FixedExperienceReplay_Multimodal:
    def __init__(
        self,
        size,
        observation_names=["image", "sound"],
        observation_shapes=[[3, 64, 64], [20, 513]],
        action_name="action",
        action_size=2,
        bit_depth=5,
        device=torch.device("cpu"),
        dataset_type="mp4",
    ):
        self.dataset_type = dataset_type
        self.device = device
        self.size = size
        self.observation_names = observation_names
        self.action_name = action_name
        self.observations = dict()
        for name in observation_names:
            self.observations[name] = torch.empty((size, *observation_shapes[name]), dtype=torch.float32)
        self.actions = torch.empty((size, action_size), dtype=torch.float32)
        self.rewards = torch.empty((size,), dtype=torch.float32)
        self.nonterminals = torch.empty((size, 1), dtype=torch.float32)

        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        # Tracks how much experience has been used in total
        self.steps, self.episodes = 0, 0
        self.bit_depth = bit_depth
        self.episode_length = None

    def load_dataset_cobotta(self, dataset_dir, n_episode, n_augment=4):
        data = self.load_data_cobotta(dataset_dir, n_episode, n_augment)
        n_episode = len(data)
        for i in tqdm(range(n_episode), desc="set dataset"):
            episode_length = len(data[i]["done"])
            idx = np.arange(self.idx, self.idx + episode_length)

            for name in self.observation_names:
                self.observations[name][idx] = data[i][name]
            if self.action_name == "dummy":
                self.actions[idx] = torch.zeros((episode_length, self.actions.shape[-1]), dtype=torch.float32)
            else:
                self.actions[idx] = data[i][self.action_name]
            self.rewards[idx] = data[i]["reward"]
            self.nonterminals[idx] = data[i]["nonterminals"]

            self.idx = (self.idx + 1) % self.size
            self.full = self.full or self.idx == 0

            self.full = self.full or (self.idx + episode_length) / self.size >= 1
            self.idx = (self.idx + episode_length) % self.size
            self.steps = self.steps + episode_length
            self.episodes += 1

    def load_data(self, dataset_dir, episode_length=None, n_episode=None):
        dataset = []
        file_names = glob.glob(os.path.join(dataset_dir, "*.npy"))
        if n_episode == None:
            n_episode = len(file_names)
        else:
            n_episode = np.min((n_episode, len(file_names)))
        print("find %d npy files!" % len(file_names))
        for file_name in file_names[:n_episode]:
            print(file_name)
            data = np.load(file_name, allow_pickle=True).item()
            data["nonterminals"] = 1.0 - data["done"]  # .unsqueeze(-1)
            dataset.append(data)
        return dataset, n_episode

    def load_dataset_mujoco(self, dataset_dir, episode_length, n_episode, n_ep_per_data):
        data, n_episode = self.load_data(dataset_dir, episode_length, n_episode)
        for i in tqdm(range(n_episode), desc="set dataset"):
            idx = np.arange(i * episode_length, (i + 1) * episode_length)
            for name in self.observation_names:
                self.observations[name][idx] = torch.tensor(data[i][name], dtype=torch.float32)
            if self.action_name == "dummy":
                self.actions[idx] = torch.zeros((episode_length, self.actions.shape[-1]), dtype=torch.float32)
            else:
                self.actions[idx] = torch.tensor(data[i][self.action_name], dtype=torch.float32)
            self.rewards[idx] = torch.tensor(data[i]["reward"], dtype=torch.float32)
            self.nonterminals[idx] = torch.tensor(data[i]["nonterminals"], dtype=torch.float32).unsqueeze(-1)

        self.full = self.full or (self.idx + (n_episode * episode_length)) / self.size >= 1
        self.idx = (self.idx + (n_episode * episode_length)) % self.size
        self.steps = self.steps + (n_episode * episode_length)
        self.episodes = self.episodes + n_episode
        self.episode_length = episode_length

    def load_dataset(self, dataset_dir, episode_length=None, n_episode=None, n_ep_per_data=1000, n_augment=4):
        if self.dataset_type == "cobotta":
            self.load_dataset_cobotta(dataset_dir, n_episode=n_episode, n_augment=n_augment)
        elif self.dataset_type == "mujoco":
            self.load_dataset_mujoco(dataset_dir, episode_length=episode_length, n_episode=n_episode, n_ep_per_data=n_ep_per_data)

    # Returns an index for a valid single sequence chunk uniformly sampled from the memory
    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            # Make sure data does not cross the memory index
            valid_idx = not self.idx in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = dict()
        for name in self.observation_names:
            observations[name] = self.observations[name][vec_idxs].reshape(L, n, *self.observations[name].shape[1:]).to(self.device)

        actions = self.actions[vec_idxs].reshape(L, n, -1).to(self.device)
        rewards = self.rewards[vec_idxs].reshape(L, n).to(self.device)
        nonterminals = self.nonterminals[vec_idxs].reshape(L, n, 1).to(self.device)
        return observations, actions, rewards, nonterminals

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample(self, n, L):
        batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
        # print(np.asarray([self._sample_idx(L) for _ in range(n)]))
        # [1578 1579 1580 ... 1625 1626 1627]
        # [1049 1050 1051 ... 1096 1097 1098]
        # [1236 1237 1238 ... 1283 1284 1285]
        # ...
        # [2199 2200 2201 ... 2246 2247 2248]
        # [ 686  687  688 ...  733  734  735]
        # [1377 1378 1379 ... 1424 1425 1426]]
        return [torch.as_tensor(item).to(device=self.device) for item in batch]

    def _sample_fix_idx(self, L, batch_idx, episode_size=1000):
        valid_idx = False
        idx_max = self.size if self.full else self.idx - L
        while not valid_idx:
            idx = ((episode_size * batch_idx) % idx_max) + L * int(np.floor((episode_size * batch_idx) / idx_max)) % idx_max
            idxs = np.arange(idx, idx + L) % idx_max
            # Make sure data does not cross the memory index
            valid_idx = not self.idx in idxs[1:]
        return idxs

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def fix_sample(self, n, L, batch_idx, episode_size=None):
        if not (self.episode_length == None):
            episode_size = self.episode_length
        else:
            episode_size = n
        batch = self._retrieve_batch(np.asarray([self._sample_fix_idx(L, batch_idx * L + i, episode_size) for i in range(n)]), n, L)
        # print(np.asarray([self._sample_fix_idx(L, batch_idx*L+i, episode_size) for i in range(n)]))
        # [   0    1    2 ...   47   48   49]
        # [1000 1001 1002 ... 1047 1048 1049]
        # [ 100  101  102 ...  147  148  149]
        # ...
        # [1400 1401 1402 ... 1447 1448 1449]
        # [2400 2401 2402 ... 2447 2448 2449]
        # [1500 1501 1502 ... 1547 1548 1549]
        return [item for item in batch]
        # return [torch.as_tensor(item).to(device=self.device) for item in batch]


class EpisodeBuffer:
    def __init__(self, cfg, train_dataset_dir, test_dataset_dir, device=torch.device("cpu")):
        self.cfg = cfg
        self.train_dataset = []
        self.test_dataset = []
        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.device = device

    def load_dataset_cobbota(self, n_train_epi=6, n_test_epi=1, n_augment=4):
        self.n_argument = n_augment
        self.train_dataset = load_data_cobotta(self.train_dataset_dir, n_episode=6, n_augment=4)
        self.test_dataset = load_data_cobotta(self.test_dataset_dir, n_episode=1, n_augment=4)

        self.train_dataset_result_path = []
        train_file_names = glob.glob(os.path.join(self.train_dataset_dir, "*.npy"))

        for i in range(len(train_file_names)):
            for j in range(n_augment):
                self.train_dataset_result_path.append("{}_augment_{}".format(os.path.basename(train_file_names[i])[:-4], j))
        self.test_dataset_result_path = []
        test_file_names = glob.glob(os.path.join(self.test_dataset_dir, "*.npy"))
        for i in range(len(test_file_names)):
            for j in range(n_augment):
                self.test_dataset_result_path.append("{}_augment_{}".format(os.path.basename(test_file_names[i])[:-4], j))

    def result_path_train(self, epi_idx):
        return self.train_dataset_result_path[epi_idx]

    def result_path_test(self, epi_idx):
        return self.test_dataset_result_path[epi_idx]

    def train_epi(self, epi_idx):
        if self.train_dataset == []:
            print("error: dataset is empty")
        observations = dict()
        # add chunk size
        for key in self.train_dataset[epi_idx].keys():
            observations[key] = self.train_dataset[epi_idx][key].unsqueeze(1).to(self.device)
        actions = self.train_dataset[epi_idx]["action"].unsqueeze(1).to(self.device)
        rewards = self.train_dataset[epi_idx]["reward"].unsqueeze(1).to(self.device)
        nonterminals = self.train_dataset[epi_idx]["nonterminals"].unsqueeze(1).to(self.device)
        return observations, actions, rewards, nonterminals

    def test_epi(self, epi_idx):
        if self.test_dataset == []:
            print("error: dataset is empty")
        observations = dict()
        # add chunk size
        for key in self.test_dataset[epi_idx].keys():
            observations[key] = self.test_dataset[epi_idx][key].unsqueeze(1).to(self.device)
        actions = self.test_dataset[epi_idx]["action"].unsqueeze(1).to(self.device)
        rewards = self.test_dataset[epi_idx]["reward"].unsqueeze(1).to(self.device)
        nonterminals = self.test_dataset[epi_idx]["nonterminals"].unsqueeze(1).to(self.device)
        return observations, actions, rewards, nonterminals

    @property
    def n_total_epi(self):
        return len(self.train_dataset) + len(self.test_dataset)

    @property
    def n_train_epi(self):
        return len(self.train_dataset)

    @property
    def n_test_epi(self):
        return len(self.test_dataset)
