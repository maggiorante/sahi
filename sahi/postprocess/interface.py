from abc import ABC, abstractmethod


class IPostProcessor(ABC):
    @abstractmethod
    def __call__():
        pass
