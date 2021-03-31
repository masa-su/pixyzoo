from warnings import WarningMessage
from typing import Union
from abc import ABC
import re

def clamp(x: Union[float, int]):
    return max(min(1, x), 0)
class BaseScheduler(ABC):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, step: int) -> Union[float, int]:
        """Returns scheduled value accroding to the given step"""
        raise NotImplementedError


class ConstantScheduler(BaseScheduler):
    def __init__(self, value: Union[float, int]):
        self.value = value

    def __call__(self, step: int) -> Union[float, int]:
        return self.value


class LinearScheduler(BaseScheduler):
    def __init__(self, initial_value: Union[float, int], final_value: Union[float, int], num_step: int):
        self.initial = initial_value
        self.final = final_value
        self.num_step = num_step

    def __call__(self, step: int) -> Union[float, int]:
        ratio = clamp(step / self.num_step)
        return (1 - ratio) * self.initial + ratio * self.final


class WarmupScheduler(BaseScheduler):
    def __init__(self, warmup, value):
        self.warmup = warmup
        self.value = value

    def __call__(self, step: int) -> Union[float, int]:
        scale = clamp(step / self.warmup)
        return scale * self.value


class ExponentialScheduler(BaseScheduler):
    def __init__(self, initial_value: Union[int, float], final_value: Union[int, float], num_half_step: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_half_step = num_half_step

    def __call__(self, step: int) -> Union[float, int]:
        return (self.initial_value - self.final_value) * 0.5 ** (step / self.num_half_step) + self.final_value


class HorizonScheduler(BaseScheduler):
    def __init__(self, initial_value: Union[float, int], final_value: Union[float, int], num_step: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_step = num_step

    def __call__(self, step: int) -> Union[float, int]:
        ratio = clamp(step / self.num_step)
        horizon = (1 - ratio) * self.initial_value + ratio * self.final_value
        return 1 - 1 / horizon


def init_scheduler(config: str) -> BaseScheduler:
    try:
        value = float(config)
        return ConstantScheduler(value)
    except ValueError:

        match = re.match(r'linear\((.+),(.+),(.+)\)', config)
        if match:
            initial, final, num_step = [
                float(group) for group in match.groups()]
            return LinearScheduler(initial_value=initial, final_value=final, num_step=num_step)

        match = re.match(r'warmup\((.+),(.+)\)', config)
        if match:
          warmup, value = [float(group) for group in match.groups()]
          return WarmupScheduler(warmup=warmup, value=value)

        match = re.match(r'exp\((.+),(.+),(.+)\)', config)
        if match:
            initial, final, num_half_step = [
                float(group) for group in match.groups()]
            return ExponentialScheduler(initial_value=initial, final_value=final, num_half_step=num_half_step)

        match = re.match(r'horizon\((.+),(.+),(.+)\)', config)
        if match:
            initial, final, num_step = [
                float(group) for group in match.groups()]
            return HorizonScheduler(initial_value=initial, final_value=final, num_step=num_step)
        raise NotImplementedError(config)
