from taxonomy_synthesis.tree.tree_node import TreeNode
from taxonomy_synthesis.models import Item, Category


def test_tree_node_initialization():
    category = Category(
        name="Root Category",
        description="Root Description"
    )
    node = TreeNode(value=category)

    assert node.value == category
    assert node.children == []
    assert node.parent is None
    assert node.items == []


def test_add_child():
    root_category = Category(
        name="Root Category",
        description="Root Description"
    )
    child_category = Category(
        name="Child Category",
        description="Child Description"
    )

    root_node = TreeNode(value=root_category)
    child_node = TreeNode(value=child_category)

    root_node.add_child(child_node)

    assert len(root_node.children) == 1
    assert root_node.children[0] == child_node
    assert child_node.parent == root_node


def test_remove_child():
    root_category = Category(
        name="Root Category",
        description="Root Description"
    )
    child_category = Category(
        name="Child Category",
        description="Child Description"
    )

    root_node = TreeNode(value=root_category)
    child_node = TreeNode(value=child_category)

    root_node.add_child(child_node)
    root_node.remove_child(child_node)

    assert len(root_node.children) == 0
    assert child_node.parent is None


def test_add_items():
    category = Category(
        name="Category",
        description="Description"
    )
    node = TreeNode(value=category)

    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    item_dict2 = {"id": "2", "name": "Item 2", "value": 20}

    items = [
        Item(**item_dict1),
        Item(**item_dict2),
    ]

    node.add_items(items)

    assert len(node.items) == 2
    assert node.items[0] == items[0]
    assert node.items[1] == items[1]


def test_remove_item():
    category = Category(
        name="Category",
        description="Description"
    )
    node = TreeNode(value=category)

    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    item_dict2 = {"id": "2", "name": "Item 2", "value": 20}

    item1 = Item(**item_dict1)
    item2 = Item(**item_dict2)

    node.add_items([item1, item2])
    node.remove_item(item1)

    assert len(node.items) == 1
    assert node.items[0] == item2
    assert item1 not in node.items
