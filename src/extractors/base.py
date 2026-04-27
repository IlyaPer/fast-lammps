from abc import ABC, abstractmethod

class Extractor(ABC):
    """
    Base class for any extractor
    """

    @abstractmethod
    def extract_interesting_regions(self):
        """Return list of masks with indexes to change"""
        pass

    # @abstractmethod
    # def check_condition_of_region(self, *_) -> bool:
    #     """Check condition of region"""
    #     pass

    # @abstractmethod
    # def visualize_interesting_regions(self, positions):
    #     """Visualizes regions extracted from extract_interesting_regions method"""
    #     pass