import json
import pytest
from unittest.mock import MagicMock
from openai import OpenAI
from taxonomy_synthesis.generator.taxonomy_generator import TaxonomyGenerator
from taxonomy_synthesis.models import Item, Category


@pytest.fixture
def mock_openai_client():
    """Fixture to create a mock OpenAI client."""
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = MagicMock()
    return mock_client


def test_generate_categories_success(mock_openai_client):
    # Setup
    generator = TaxonomyGenerator(
        client=mock_openai_client,
        max_categories=3,
        generation_method="Categorize based on type"
    )

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    item_dict2 = {"id": "2", "name": "Item 2", "value": 20}

    items = [
        Item(**item_dict1),
        Item(**item_dict2)
    ]
    
    parent_category = Category(
        name="Parent Category",
        description="Parent Description"
    )

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                tool_calls=[
                    MagicMock(
                        function=MagicMock(
                            arguments=json.dumps(
                                {
                                    "categories": [
                                        {"name": "Subcategory 1", "description": "Description 1"},  # noqa: E501
                                        {"name": "Subcategory 2", "description": "Description 2"}  # noqa: E501
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
    categories = generator.generate_categories(items, parent_category)
    
    # Assert
    assert len(categories) == 2
    assert categories[0].name == "Subcategory 1"
    assert categories[0].description == "Description 1"
    assert categories[1].name == "Subcategory 2"
    assert categories[1].description == "Description 2"


def test_generate_categories_no_tool_calls(mock_openai_client):
    # Setup
    generator = TaxonomyGenerator(
        client=mock_openai_client,
        max_categories=3,
        generation_method="Categorize based on type"
    )

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    items = [Item(**item_dict1)]

    parent_category = Category(
        name="Parent Category",
        description="Parent Description"
    )

    mock_response = MagicMock()
    mock_response.choices = []

    mock_openai_client.chat.completions.create.return_value = mock_response

    # Execute & Assert
    with pytest.raises(
        ValueError,
        match="No tool calls found in the response from the model."
    ):
        generator.generate_categories(items, parent_category)


def test_generate_categories_missing_arguments(mock_openai_client):
    # Setup
    generator = TaxonomyGenerator(
        client=mock_openai_client,
        max_categories=3,
        generation_method="Categorize based on type"
    )

    # Creating Item objects with additional attributes using **kwargs
    item_dict1 = {"id": "1", "name": "Item 1", "value": 10}
    items = [Item(**item_dict1)]

    parent_category = Category(
        name="Parent Category",
        description="Parent Description"
    )

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
        generator.generate_categories(items, parent_category)


def test_refine_categories_success(mock_openai_client):
    # Setup
    generator = TaxonomyGenerator(
        client=mock_openai_client,
        max_categories=3,
        generation_method="Categorize based on type"
    )

    feedback = "Please split the categories further."

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                tool_calls=[
                    MagicMock(
                        function=MagicMock(
                            arguments=json.dumps(
                                {
                                    "categories": [
                                        {"name": "Subcategory 1.1", "description": "Description 1.1"},  # noqa: E501
                                        {"name": "Subcategory 1.2", "description": "Description 1.2"}  # noqa: E501
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
    refined_categories = generator.refine_categories(feedback)
    
    # Assert
    assert len(refined_categories) == 2
    assert refined_categories[0].name == "Subcategory 1.1"
    assert refined_categories[0].description == "Description 1.1"
    assert refined_categories[1].name == "Subcategory 1.2"
    assert refined_categories[1].description == "Description 1.2"
