import pytest
from openai import OpenAI
from taxonomy_synthesis.classifiers.gpt_classifier import GPTClassifier
from taxonomy_synthesis.models import Item, Category, ClassifiedItem
from taxonomy_synthesis.generator.taxonomy_generator import TaxonomyGenerator
from dotenv import load_dotenv
import os

load_dotenv()


@pytest.fixture
def openai_client():
    """Fixture to create a mock OpenAI client."""
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return openai_client


def test_classify_items_success(openai_client):
    # Setup
    classifier = GPTClassifier(client=openai_client)

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    item_dict2 = {"id": "2", "name": "Item 2", "value": 20}

    items = [Item(**item_dict1), Item(**item_dict2)]

    categories = [
        Category(name="Category 1", description="Description 1"),
        Category(name="Category 2", description="Description 2"),
    ]

    # Execute
    print("pre")
    classified_items = classifier.classify_items(items, categories)
    print("post")
    # Assert
    assert all(isinstance(item, ClassifiedItem) for item in classified_items)
    assert len(classified_items) == 2
    item0 = None
    item1 = None
    for item in classified_items:
        if item.item.id == "1":
            item0 = item
        if item.item.id == "2":
            item1 = item
    if item0 is None or item1 is None:
        pytest.fail("Item not found in classified items.")
    assert item0.item.id == "1"
    assert item0.category.name == "Category 1"
    assert item1.item.id == "2"
    assert item1.category.name == "Category 2"


def test_generate_categories_success(openai_client):
    # Setup
    generator = TaxonomyGenerator(
        client=openai_client,
    )

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    item_dict2 = {"id": "2", "name": "Item 2", "value": 20}

    items = [Item(**item_dict1), Item(**item_dict2)]

    parent_category = Category(name="Parent Category", description="Parent Description")

    # Execute
    categories = generator.generate_categories(items, parent_category)

    # Assert
    assert len(categories) == 2
    assert categories[0].name is not None
    assert categories[0].description is not None
    assert categories[1].name is not None
    assert categories[1].description is not None


def test_generate_max_categories(openai_client):
    # Setup
    generator = TaxonomyGenerator(client=openai_client, max_categories=1)

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    item_dict2 = {"id": "2", "name": "Item 2", "value": 20}

    items = [Item(**item_dict1), Item(**item_dict2)]

    parent_category = Category(name="Parent Category", description="Parent Description")

    # Execute
    categories = generator.generate_categories(items, parent_category)

    # Assert
    assert len(categories) == 1
    assert categories[0].name is not None
    assert categories[0].description is not None


# below is dependent on the above test
@pytest.mark.dependency(depends=["test_generate_max_categories"])
def test_generate_specific_categories(openai_client):
    # Setup
    generator = TaxonomyGenerator(
        client=openai_client,
        generation_method="The category name must be 'all_numbers'.",
        max_categories=1,
    )

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    item_dict2 = {"id": "2", "name": "Item 2", "value": 20}

    items = [Item(**item_dict1), Item(**item_dict2)]

    parent_category = Category(name="Parent Category", description="Parent Description")

    # Execute
    categories = generator.generate_categories(items, parent_category)

    # Assert
    assert len(categories) == 1
    assert categories[0].name == "all_numbers"
    assert categories[0].description is not None
