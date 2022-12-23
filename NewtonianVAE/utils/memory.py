import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from utils.env import postprocess_observation, _images_to_observation


class ExperienceReplay():
    def __init__(self, episode_size, sequence_size, action_size, bit_depth, device):
        self.device = device
        self.episode_size = episode_size
        self.sequence_size = sequence_size
        self.action_size = action_size
        self.bit_depth = bit_depth

        self.colors = np.empty(
            (episode_size, sequence_size, 3, 64, 64), dtype=np.float32)
        self.actions = np.empty(
            (episode_size, sequence_size, action_size), dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        colors = self.colors[index]
        actions = torch.from_numpy(self.actions[index])

        return _images_to_observation(colors, self.bit_depth), actions

    def append(self, color, action, batch):
        self.colors[batch] = postprocess_observation(color, self.bit_depth)
        self.actions[batch] = action

    def reset(self):
        self.colors = np.empty(
            (self.episode_size, 3, 64, 64), dtype=np.float32)
        self.actions = np.empty(
            (self.episode_size, self.sequence_size, self.action_size), dtype=np.float32)

    def save(self, path, filename):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        np.savez(f"{path}/{filename}", **
                {"colors": self.colors, "actions": self.actions})

    def load(self, path, filename):
        with np.load(f"{path}/{filename}", allow_pickle=True) as data:
            self.colors = data["colors"][0:self.episode_size]
            self.actions = data["actions"][0:self.episode_size]

def make_loader(cfg, mode):
    #==========================#
    # Define experiment replay #
    #==========================#
    replay = ExperienceReplay(**cfg["dataset"][mode]["memory"])
     
    #==============#
    # Load dataset #
    #==============#
    replay.load(**cfg["dataset"][mode]["data"])
     
    #====================#
    # Define data loader #
    #====================#
    loader = DataLoader(replay, **cfg["dataset"][mode]["loader"])

    return loader
