import cv2
import numpy as np
import torch


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
  # Quantise to given bit depth and centre
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 **
                                                       bit_depth).sub_(0.5)
  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
  # images = torch.tensor(cv2.resize(images, (64, 64)).transpose(
  #     2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  # Quantise, centre and dequantise inplace
  images = torch.from_numpy(images.copy()).to(torch.float32)
  preprocess_observation_(images, bit_depth)
  return images
