from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class ParameterSpec:
    name: str
    bounds: Tuple[float, float]
    default: Optional[float] = None
    active: bool = True
    description: str = ""


class BasePolicy:
    parameters: List[ParameterSpec] = []

    def active_parameters(self):
        return [p for p in self.parameters if p.active]

    def param_names(self):
        return [p.name for p in self.active_parameters()]

    def bounds(self):
        return [p.bounds for p in self.active_parameters()]

    def default_params(self):
        return {
            p.name: p.default for p in self.active_parameters() if p.default is not None
        }

    def set_params(self, params: dict):
        self.params = params
        self.reset()

    def reset(self):
        raise NotImplementedError

    def forget(self):
        raise NotImplementedError

    def logits(self):
        raise NotImplementedError

    def update(self, choice, reward):
        raise NotImplementedError
