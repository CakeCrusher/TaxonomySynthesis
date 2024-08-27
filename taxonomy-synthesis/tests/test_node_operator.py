import pytest
from unittest.mock import MagicMock
from taxonomy_synthesis.models import Item, Category, ClassifiedItem
from taxonomy_synthesis.tree.tree_node import TreeNode
from taxonomy_synthesis.tree.node_operator import NodeOperator
from taxonomy_synthesis.classifiers.classifier_interface import IClassifier
from taxonomy_synthesis.generator.taxonomy_generator import TaxonomyGenerator


@pytest.fixture
def mock_classifier():
    """Fixture to create a mock classifier."""
    mock = MagicMock(spec=IClassifier)
    return mock


@pytest.fixture
def mock_generator():
    """Fixture to create a mock taxonomy generator."""
    mock = MagicMock(spec=TaxonomyGenerator)
    return mock


def test_classify_items(mock_classifier, mock_generator):
    # Setup
    node_operator = NodeOperator(classifier=mock_classifier, generator=mock_generator)

    parent_category = Category(name="Parent", description="Parent category")
    child_category_1 = Category(name="Child1", description="Child category 1")

    parent_node = TreeNode(value=parent_category)
    child_node_1 = TreeNode(value=child_category_1)
    parent_node.add_child(child_node_1)

    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    item_dict2 = {"id": "2", "name": "Item 2", "value": 20}

    items = [Item(**item_dict1), Item(**item_dict2)]

    parent_node.add_items([items[0]])
    child_node_1.add_items([items[1]])

    # Mock classifier behavior
    classified_item1 = ClassifiedItem(item=items[0], category=child_category_1)
    classified_item2 = ClassifiedItem(item=items[1], category=child_category_1)
    mock_classifier.classify_items.return_value = [classified_item1, classified_item2]

    # Execute
    classified_items = node_operator.classify_items(parent_node, items)

    # Assert
    assert len(classified_items) == 2
    assert items[0] not in parent_node.items
    assert items[1] not in parent_node.items
    assert items[0] in child_node_1.items
    assert items[1] in child_node_1.items


def test_classify_items_with_missing_category(mock_classifier, mock_generator):
    # Setup
    node_operator = NodeOperator(classifier=mock_classifier, generator=mock_generator)

    parent_category = Category(name="Parent", description="Parent category")
    parent_node = TreeNode(value=parent_category)

    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}

    items = [Item(**item_dict1)]
    parent_node.add_items(items)

    # Mock classifier behavior with a non-existent category
    non_existent_category = Category(
        name="NonExistent", description="Non-existent category"
    )
    classified_item1 = ClassifiedItem(item=items[0], category=non_existent_category)
    mock_classifier.classify_items.return_value = [classified_item1]

    # Execute & Assert
    with pytest.raises(
        ValueError, match="Category 'NonExistent' not found in the tree"
    ):
        node_operator.classify_items(parent_node, items)
