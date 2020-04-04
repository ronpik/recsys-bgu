import abc
import pandas as pd


class AbstractSVDModelParams(abc.ABC):
    
    @abc.abstractmethod
    def initialize_parameters(self, data: pd.DataFrame, latent_dim: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, user: int, item: int, err: float, regularization: float, learning_rate: float):
        raise NotImplementedError()

    @abc.abstractmethod
    def estimate_rating(self, user: int, item: int) -> float:
        raise NotImplementedError()