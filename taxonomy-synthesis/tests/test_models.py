from taxonomy_synthesis.models import Item, Category, ClassifiedItem


def test_item_creation():
    item_dict = {"id": "1", "name": "Test Item", "value": 10}
    item = Item(**item_dict)

    # Checking the dynamically added attributes using getattr
    assert item.id == "1"
    assert item.dict() == item_dict

    assert isinstance(item, Item)


def test_category_creation():
    category = Category(
        name="Test Category",
        description="This is a test category"
    )
    assert category.name == "Test Category"
    assert category.description == "This is a test category"
    assert isinstance(category, Category)


def test_classified_item_creation():
    item_dict = {"id": "1", "name": "Test Item", "value": 10}
    item = Item(**item_dict)
    category = Category(
        name="Test Category",
        description="This is a test category"
    )
    classified_item = ClassifiedItem(item=item, category=category)
    assert classified_item.item == item
    assert classified_item.category == category
    assert isinstance(classified_item, ClassifiedItem)
