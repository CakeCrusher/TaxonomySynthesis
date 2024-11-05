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
        # if stringified items.model_dump() divided by 3 is longer than 60000 characters
        # then divide the items into the upper bound of divinding the stringified items.model_dump() by 60000 characters
        item_token_count = len(str([item.model_dump() for item in items])) / 3
        divisions = ((item_token_count + 30000) // 60000) + 1
        batches = [items]
        if divisions > 1:
            batches = [
                items[i : i + int(len(items) // divisions)]
                for i in range(0, len(items), int(len(items) // divisions))
            ]

        classified_items = []
        for batch in batches:
            prompt = f"""I will provide you with items and categories. You need to classify the items into the correct category.
    ITEMS:
    ```
    {[item.model_dump() for item in items]}
    ```
    CATEGORIES:
    ```
    {[category.model_dump() for category in categories]}
    ```"""  # noqa: E501
            item_ids = [item.id for item in batch]
            category_names = [category.name for category in categories]
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "classifier",
                            "strict": True,
                            "parameters": {
                                "$defs": {
                                    "classified_item": {
                                        "description": "Matches the item with its category.",
                                        "properties": {
                                            "item_id": {
                                                "description": "The id of the item",
                                                "enum": item_ids,
                                                "title": "Item Id",
                                                "type": "string",
                                            },
                                            "category_name": {
                                                "description": "The name of the category",
                                                "enum": category_names,
                                                "title": "Category Name",
                                                "type": "string",
                                            },
                                        },
                                        "required": ["item_id", "category_name"],
                                        "title": "ClassifiedItemModel",
                                        "type": "object",
                                        "additionalProperties": False,
                                    }
                                },
                                "description": "List of classified items.",
                                "properties": {
                                    "classified_items": {
                                        "description": "List of classified items",
                                        "items": {"$ref": "#/$defs/classified_item"},
                                        "title": "Classified Items",
                                        "type": "array",
                                    }
                                },
                                "required": ["classified_items"],
                                "title": "ClassifierModel",
                                "type": "object",
                                "additionalProperties": False,
                            },
                            "description": "List of classified items.",
                        },
                    }
                ],
            )

            # Check if response has the expected structure
            if (
                not response.choices
                or not response.choices[0]
                or not response.choices[0].message.tool_calls
            ):
                raise ValueError("Model response is missing the expected structure.")

            print("1")
            response_items = (
                response.choices[0].message.tool_calls[0].function.arguments
            )
            response_items = json.loads(response_items)
            response_items = response_items["classified_items"]
            print("2")

            # filter out redundant items
            unique_items = set()
            for item in response_items:
                unique_items.add(item["item_id"])

            print("2.5")
            response_items = [
                response_item
                for response_item in response_items
                if response_item["item_id"] in unique_items
            ]
            print("3")

            # missing items
            missing_items = set(item_ids) - unique_items
            if missing_items:
                missing_items_list = [
                    item for item in items if item.id in missing_items
                ]
                # recurse untill all items are classified
                classified_items = self.classify_items(missing_items_list, categories)
                response_items += [
                    {
                        "item_id": classified_item.item.id,
                        "category_name": classified_item.category.name,
                    }
                    for classified_item in classified_items
                ]
                response_items += self.classify_items(missing_items_list, categories)

            print("4")
            response_items = [
                ResponseItem(**response_item) for response_item in response_items
            ]

            print("5")
            # Map response items to ClassifiedItem objects
            response_items = [
                ClassifiedItem(
                    item=next(
                        item for item in items if item.id == response_item.item_id
                    ),
                    category=next(
                        category
                        for category in categories
                        if category.name == response_item.category_name
                    ),
                )
                for response_item in response_items
            ]

            classified_items += response_items

        return classified_items
