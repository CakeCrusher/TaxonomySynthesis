from typing import List, Optional
from taxonomy_synthesis.models import Item, Category


class TreeNode:
    def __init__(self, value: Category, parent: Optional["TreeNode"] = None):
        self.value = value
        self.children: List["TreeNode"] = []
        self.parent = parent
        self.items: List[Item] = []

    def add_child(self, child: "TreeNode") -> None:
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "TreeNode") -> None:
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def add_items(self, items: List[Item]) -> None:
        self.items.extend(items)

    def remove_item(self, item: Item) -> None:
        if item in self.items:
            self.items.remove(item)
