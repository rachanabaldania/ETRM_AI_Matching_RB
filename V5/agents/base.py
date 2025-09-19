from abc import ABC, abstractmethod

class BaseMatchingAgent(ABC):
    @abstractmethod
    def match(self, extracted: str, etrm_df):
        """Return list of matches with scores"""
        pass
