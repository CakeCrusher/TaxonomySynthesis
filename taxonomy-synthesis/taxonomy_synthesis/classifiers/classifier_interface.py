from abc import ABC, abstractmethod
from typing import List
from taxonomy_synthesis.models import Item, Category, ClassifiedItem


class IClassifier(ABC):
    @abstractmethod
    def classify_items(
        self, items: List[Item], categories: List[Category]
    ) -> List[ClassifiedItem]:
        """Classify items into categories."""
        pass
