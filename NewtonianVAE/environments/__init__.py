# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""A collection of MuJoCo-based Reinforcement Learning environments."""

import collections
import inspect
import itertools
import cv2

from dm_control.rl import control

from dm_control.suite import acrobot
from dm_control.suite import ball_in_cup
from dm_control.suite import cartpole
from dm_control.suite import cheetah
from dm_control.suite import dog
from dm_control.suite import finger
from dm_control.suite import fish
from dm_control.suite import hopper
from dm_control.suite import humanoid
from dm_control.suite import humanoid_CMU
from dm_control.suite import lqr
from dm_control.suite import manipulator
from dm_control.suite import pendulum
from dm_control.suite import point_mass
from dm_control.suite import quadruped
from dm_control.suite import reacher
from dm_control.suite import stacker
from dm_control.suite import swimmer
from dm_control.suite import walker

from environments import reacher_nvae
from environments import point_mass_nvae
from environments import ycb_mass

from utils.env import preprocess_observation_, postprocess_observation, _images_to_observation

# Find all domains imported.
_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}


def _get_tasks(tag):
  """Returns a sequence of (domain name, task name) pairs for the given tag."""
  result = []

  for domain_name in sorted(_DOMAINS.keys()):

    domain = _DOMAINS[domain_name]

    if tag is None:
      tasks_in_domain = domain.SUITE
    else:
      tasks_in_domain = domain.SUITE.tagged(tag)

    for task_name in tasks_in_domain.keys():
      result.append((domain_name, task_name))

  return tuple(result)


def _get_tasks_by_domain(tasks):
  """Returns a dict mapping from task name to a tuple of domain names."""
  result = collections.defaultdict(list)

  for domain_name, task_name in tasks:
    result[domain_name].append(task_name)

  return {k: tuple(v) for k, v in result.items()}


# A sequence containing all (domain name, task name) pairs.
ALL_TASKS = _get_tasks(tag=None)

# Subsets of ALL_TASKS, generated via the tag mechanism.
BENCHMARKING = _get_tasks('benchmarking')
EASY = _get_tasks('easy')
HARD = _get_tasks('hard')
EXTRA = tuple(sorted(set(ALL_TASKS) - set(BENCHMARKING)))
NO_REWARD_VIZ = _get_tasks('no_reward_visualization')
REWARD_VIZ = tuple(sorted(set(ALL_TASKS) - set(NO_REWARD_VIZ)))

# A mapping from each domain name to a sequence of its task names.
TASKS_BY_DOMAIN = _get_tasks_by_domain(ALL_TASKS)


def load(domain_name, task_name, task_kwargs=None, environment_kwargs=None,
         visualize_reward=False):
  """Returns an environment from a domain name, task name and optional settings.
  ```python
  env = suite.load('cartpole', 'balance')
  ```
  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` of keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.
  Returns:
    The requested environment.
  """
  return build_environment(domain_name, task_name, task_kwargs,
                           environment_kwargs, visualize_reward)


def build_environment(domain_name, task_name, task_kwargs=None,
                      environment_kwargs=None, visualize_reward=False):
  """Returns an environment from the suite given a domain name and a task name.
  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` specifying keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.
  Raises:
    ValueError: If the domain or task doesn't exist.
  Returns:
    An instance of the requested environment.
  """
  if domain_name not in _DOMAINS:
    raise ValueError('Domain {!r} does not exist.'.format(domain_name))

  domain = _DOMAINS[domain_name]

  if task_name not in domain.SUITE:
    raise ValueError('Level {!r} does not exist in domain {!r}.'.format(
        task_name, domain_name))

  task_kwargs = task_kwargs or {}
  if environment_kwargs is not None:
    task_kwargs = dict(task_kwargs, environment_kwargs=environment_kwargs)
  env = domain.SUITE[task_name](**task_kwargs)
  env.task.visualize_reward = visualize_reward
  return env


class ControlSuiteEnv():
  def __init__(self, domain_name, task_name, max_episode_length, bit_depth):

    self._env = load(domain_name, task_name)
    self.max_episode_length = max_episode_length
    self.bit_depth = bit_depth

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    return _images_to_observation(self._env.physics.render(64, 64, camera_id=0), self.bit_depth)

  def step(self, action):
    action = action.detach().numpy()
    reward = 0

    state = self._env.step(action)
    if state.reward is not None:
        reward += state.reward
    self.t += 1  # Increment internal timer
    done = state.last() or self.t == self.max_episode_length

    observation = _images_to_observation(
        self._env.physics.render(64, 64, camera_id=0), self.bit_depth)

    return observation, reward, done

  def render(self):
    cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
    self._env.close()

  @property
  def observation_size(self):
    return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_spec().shape[0]

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    spec = self._env.action_spec()
    return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))
