import json
from typing import List
from taxonomy_synthesis.models import (
    Item,
    Category,
    ClassifiedItem,
    ResponseItem,
)
from taxonomy_synthesis.classifiers.classifier_interface import IClassifier
from openai import OpenAI


class GPTClassifier(IClassifier):
    def __init__(self, client: OpenAI):
        self.client = client

    def classify_items(
        self, items: List[Item], categories: List[Category]
    ) -> List[ClassifiedItem]:
        prompt = f"""I will provide you with items and categories. You need to classify the items into the correct category.
ITEMS:
```
{[item.dict() for item in items]}
```
CATEGORIES:
```
{[category.dict() for category in categories]}
```"""  # noqa: E501

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "classifier",
                        "description": "Classify items into categories",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "classified_items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "item_id": {
                                                "type": "string",
                                                "description": "The id of the item",  # noqa: E501
                                            },
                                            "category_name": {
                                                "type": "string",
                                                "description": "The name of the category",  # noqa: E501
                                                "enum": [
                                                    category.name
                                                    for category in categories
                                                ],
                                            },
                                        },
                                        "required": [
                                            "item_id",
                                            "category_name",
                                        ],  # noqa: E501
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["classified_items"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "classifier"}},
        )

        # Check if response has the expected structure
        if not response.choices or not response.choices[0].message.tool_calls:
            raise ValueError("No tool calls found in the response from the model.")

        tool_call = response.choices[0].message.tool_calls[0]

        # Check if tool_call is not None before attempting to access it
        if tool_call is None or tool_call.function.arguments is None:
            raise ValueError("Tool call arguments are missing in the model response.")

        classified_items_data = json.loads(tool_call.function.arguments)

        response_items = [
            ResponseItem(**item) for item in classified_items_data["classified_items"]
        ]

        # Map response items to ClassifiedItem objects
        classified_items = [
            ClassifiedItem(
                item=next(item for item in items if item.id == response_item.item_id),
                category=next(
                    category
                    for category in categories
                    if category.name == response_item.category_name
                ),
            )
            for response_item in response_items
        ]

        return classified_items
