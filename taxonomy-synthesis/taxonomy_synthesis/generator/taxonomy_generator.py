import json
from typing import List, Optional
from taxonomy_synthesis.models import Item, Category
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)  # noqa
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)  # noqa


class TaxonomyGenerator:
    def __init__(
        self,
        client: OpenAI,
        generation_method: str = "",
        max_categories: Optional[int] = None,
    ):
        self.client = client
        self.max_categories = max_categories
        self.generation_method = generation_method
        self.chat_history: List[ChatCompletionMessageParam] = []

    def initialize_chat(self, items: List[Item], parent_category: Category):
        if self.max_categories:
            max_categories_prompt = (
                f"You can create at most {self.max_categories} subcategories."
            )
        else:
            max_categories_prompt = ""
        prompt = f"""I will provide you with items inside the parent category titled `{parent_category.name}` described as `{parent_category.description}`.
You need to create subcategories according to the following guideline:
{max_categories_prompt}
The created subcategories should not duplicate the parent category '{parent_category.name}'. {self.generation_method}
ITEMS:
```
{[item.model_dump() for item in items]}
```"""  # noqa
        self.chat_history = [{"role": "user", "content": prompt}]

    def generate_categories(
        self,
        items: List[Item],
        parent_category: Category,
        max_categories: Optional[int] = None,
    ) -> List[Category]:
        item_token_count = len(str([item.model_dump() for item in items])) / 3
        if item_token_count > 60000:
            print(
                "taxonomy-synthesis WARNING: Items will be truncated to just under 60000 tokens."
            )
            items = items.copy()
            while item_token_count > 60000:
                items.pop()
                item_token_count = len(str([item.model_dump() for item in items])) / 3

        self.initialize_chat(items, parent_category)

        if max_categories:
            self.max_categories = max_categories

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.chat_history,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "subcategories_list",
                        "strict": True,
                        "parameters": {
                            "$defs": {
                                "category": {
                                    "description": "Category for items.",
                                    "properties": {
                                        "name": {
                                            "description": "Name of the category.",
                                            "type": "string",
                                        },
                                        "description": {
                                            "description": "Description and instruction for how to use this category.",
                                            "type": "string",
                                        },
                                    },
                                    "required": ["name", "description"],
                                    "type": "object",
                                    "additionalProperties": False,
                                }
                            },
                            "description": "Matches the item with its category.",
                            "properties": {
                                "categories": {
                                    "description": "List of categories to match the item with.",
                                    "items": {"$ref": "#/$defs/category"},
                                    "type": "array",
                                }
                            },
                            "required": ["categories"],
                            "type": "object",
                            "additionalProperties": False,
                        },
                        "description": "Matches the item with its category.",
                    },
                },
            ],
        )

        # Check if response has the expected structure
        if (
            not response.choices
            or not response.choices[0]
            or not response.choices[0].message.tool_calls
        ):
            raise ValueError("Model response is missing the expected structure.")

        tool_call = response.choices[0].message.tool_calls[0]

        # Check if tool_call is not None before attempting to access it
        if tool_call is None or tool_call.function.arguments is None:
            raise ValueError(
                "Tool call arguments are missing in the model response."
            )  # noqa

        # Add assistant's response to chat history
        self.chat_history.append(
            ChatCompletionToolMessageParam(**response.choices[0].message.model_dump())
        )

        # Parse and return categories
        categories_data = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )
        categories_data = [Category(**cat) for cat in categories_data["categories"]]
        if self.max_categories:
            return categories_data[: self.max_categories]
        else:
            return categories_data

    def refine_categories(self, feedback: str) -> List[Category]:
        self.chat_history.append({"role": "user", "content": feedback})

        # Generate categories again based on feedback
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=self.chat_history
        )

        if not response.choices or not response.choices[0].message.tool_calls:
            raise ValueError(
                "No tool calls found in the response from the model."
            )  # noqa

        # Assuming response handling similar to `generate_categories`
        tool_call = response.choices[0].message.tool_calls[0]

        # Check if tool_call is not None before attempting to access it
        if tool_call is None or tool_call.function.arguments is None:
            raise ValueError(
                "Tool call arguments are missing in the model response."
            )  # noqa

        categories_data = json.loads(tool_call.function.arguments)
        return [Category(**cat) for cat in categories_data["categories"]]
