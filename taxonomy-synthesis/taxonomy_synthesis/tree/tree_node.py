from typing import List, Optional
from taxonomy_synthesis.models import Item, Category


class TreeNode:
    def __init__(
        self, 
        value: Category, 
        parent: Optional["TreeNode"] = None
    ):
        self.value = value
        self.children: List["TreeNode"] = []
        self.parent = parent
        self.items: List[Item] = []

    def add_child(self, child: "TreeNode") -> None:
        """
        Add a child node to the current node.
        """
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "TreeNode") -> None:
        """
        Remove a child node from the current node.
        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def add_items(self, items: List[Item]) -> None:
        """
        Add items to the current node.
        """
        self.items.extend(items)

    def remove_item(self, item: Item) -> None:
        """
        Remove an item from the current node.
    """
        if item in self.items:
            self.items.remove(item)

    def get_all_items(self) -> List[Item]:
        """
        Recursively retrieve all items from the current node and its descendants.
        """  # noqa: E501
        all_items = list(self.items)
        for child in self.children:
            all_items.extend(child.get_all_items())
        return all_items
