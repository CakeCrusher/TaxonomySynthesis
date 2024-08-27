from typing import List, Optional
from taxonomy_synthesis.models import Item, Category, ClassifiedItem
from taxonomy_synthesis.tree.tree_node import TreeNode
from taxonomy_synthesis.classifiers.classifier_interface import IClassifier
from taxonomy_synthesis.generator.taxonomy_generator import TaxonomyGenerator


class NodeOperator:
    def __init__(self, classifier: IClassifier, generator: TaxonomyGenerator):
        self.classifier = classifier
        self.generator = generator

    def classify_items(self, node: TreeNode, items: List[Item]) -> List[ClassifiedItem]:
        """
        Classify the given items, remove any duplicates from the tree, and assign them to appropriate categories within the specified TreeNode.
        """  # noqa: E501
        categories = [child.value for child in node.children]
        all_items = node.get_all_items()  # Get all items recursively
        item_ids_to_classify = {item.id for item in items}

        # Remove items with the same ID from the tree
        for existing_item in all_items:
            if existing_item.id in item_ids_to_classify:
                self._remove_item_from_tree(node, existing_item)

        # Classify the new items
        classified_items = self.classifier.classify_items(items, categories)

        # Add classified items to appropriate nodes
        for classified_item in classified_items:
            item = classified_item.item
            category_name = classified_item.category.name

            # Find the appropriate category node
            category_node = next(
                (
                    child
                    for child in node.children
                    if child.value.name == category_name
                ),  # noqa: E501
                None,
            )

            if category_node is None:
                # Raise error if category node is not found
                raise ValueError(f"Category '{category_name}' not found in the tree")

            # Add the item to the category node
            category_node.add_items([item])

        return classified_items

    def _remove_item_from_tree(self, node: TreeNode, item: Item) -> None:
        """
        Helper method to remove an item from the tree starting from the given node.
        """  # noqa: E501
        # Check if the item is in the current node
        if item in node.items:
            node.remove_item(item)
        else:
            # Recursively check in the children
            for child in node.children:
                self._remove_item_from_tree(child, item)

    def generate_subcategories(
        self, node: TreeNode, max_categories: Optional[int] = None
    ) -> List[Category]:
        """
        Generate subcategories for the given TreeNode based on its items and a maximum number of categories.
        """  # noqa: E501
        parent_category = node.value
        items = node.items

        new_categories = self.generator.generate_categories(
            items, parent_category, max_categories
        )
        self.add_subcategories(node, new_categories)
        return new_categories

    def add_subcategories(self, node: TreeNode, categories: List[Category]) -> None:
        """
        Add new categories as children to the specified TreeNode.
        """
        for category in categories:
            new_node = TreeNode(value=category)
            node.add_child(new_node)
