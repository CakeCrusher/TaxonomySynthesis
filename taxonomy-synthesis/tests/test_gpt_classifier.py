import json
import pytest
from unittest.mock import MagicMock
from openai import OpenAI
from taxonomy_synthesis.classifiers.gpt_classifier import GPTClassifier
from taxonomy_synthesis.models import Item, Category, ClassifiedItem


@pytest.fixture
def mock_openai_client():
    """Fixture to create a mock OpenAI client."""
    mock_client = MagicMock(spec=OpenAI)
    # Adding the 'chat' attribute to the mock client
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = MagicMock()
    return mock_client


def test_classify_items_success(mock_openai_client):
    # Setup
    classifier = GPTClassifier(client=mock_openai_client)

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    item_dict2 = {"id": "2", "name": "Item 2", "value": 20}

    items = [
        Item(**item_dict1),
        Item(**item_dict2)
    ]
    
    categories = [
        Category(name="Category 1", description="Description 1"),
        Category(name="Category 2", description="Description 2")
    ]

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                tool_calls=[
                    MagicMock(
                        function=MagicMock(
                            arguments=json.dumps(
                                {
                                    "classified_items": [
                                        {"item_id": "1", "category_name": "Category 1"},  # noqa: E501
                                        {"item_id": "2", "category_name": "Category 2"}  # noqa: E501
                                    ]
                                }
                            )
                        )
                    )
                ]
            )
        )
    ]

    mock_openai_client.chat.completions.create.return_value = mock_response

    # Execute
    classified_items = classifier.classify_items(items, categories)
    
    # Assert
    assert all(isinstance(item, ClassifiedItem) for item in classified_items)
    assert len(classified_items) == 2
    assert classified_items[0].item.id == "1"
    assert classified_items[0].category.name == "Category 1"
    assert classified_items[1].item.id == "2"
    assert classified_items[1].category.name == "Category 2"


def test_classify_items_no_tool_calls(mock_openai_client):
    # Setup
    classifier = GPTClassifier(client=mock_openai_client)

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    items = [Item(**item_dict1)]

    categories = [Category(name="Category 1", description="Description 1")]

    mock_response = MagicMock()
    mock_response.choices = []

    mock_openai_client.chat.completions.create.return_value = mock_response

    # Execute & Assert
    with pytest.raises(
        ValueError,
        match="No tool calls found in the response from the model."
    ):
        classifier.classify_items(items, categories)


def test_classify_items_missing_arguments(mock_openai_client):
    # Setup
    classifier = GPTClassifier(client=mock_openai_client)

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    items = [Item(**item_dict1)]

    categories = [Category(name="Category 1", description="Description 1")]

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                tool_calls=[
                    MagicMock(
                        function=MagicMock(arguments=None)
                    )
                ]
            )
        )
    ]

    mock_openai_client.chat.completions.create.return_value = mock_response

    # Execute & Assert
    with pytest.raises(
        ValueError,
        match="Tool call arguments are missing in the model response."
    ):
        classifier.classify_items(items, categories)
