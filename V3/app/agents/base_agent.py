from abc import ABC, abstractmethod
from typing import Union

class BaseMatchingAgent(ABC):
    @abstractmethod
    def match(self, extracted: str, etrm_description: str) -> Union[bool, float]:
        """Base match method to be implemented by subclasses"""
        pass