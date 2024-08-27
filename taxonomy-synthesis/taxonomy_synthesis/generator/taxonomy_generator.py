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
    def __init__(self, client: OpenAI, max_categories: int, generation_method: str):
        self.client = client
        self.max_categories = max_categories
        self.generation_method = generation_method
        self.chat_history: List[ChatCompletionMessageParam] = []

    def initialize_chat(self, items: List[Item], parent_category: Category):
        prompt = f"""I will provide you with items inside the parent category titled `{parent_category.name}` described as `{parent_category.description}`.
You need to create subcategories according to the following guideline:
There must be at most {self.max_categories} subcategories, and they should not duplicate the parent category '{parent_category.name}'. {self.generation_method}
ITEMS:
```
{[item.dict() for item in items]}
```"""  # noqa
        self.chat_history = [{"role": "user", "content": prompt}]

    def generate_categories(
        self,
        items: List[Item],
        parent_category: Category,
        max_categories: Optional[int] = None,
    ) -> List[Category]:
        self.initialize_chat(items, parent_category)

        if max_categories:
            self.max_categories = max_categories

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.chat_history,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "category_creator",
                        "description": "Create subcategories for items under the given parent category",  # noqa
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "categories": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "Snake case name of the subcategory",  # noqa
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "The description of the subcategory",  # noqa
                                            },
                                        },
                                        "required": ["name", "description"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["categories"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "category_creator"}},
        )

        # Check if response has the expected structure
        if not response.choices or not response.choices[0].message.tool_calls:
            raise ValueError(
                "No tool calls found in the response from the model."
            )  # noqa

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
        categories_data = json.loads(tool_call.function.arguments)
        return [Category(**cat) for cat in categories_data["categories"]]

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
