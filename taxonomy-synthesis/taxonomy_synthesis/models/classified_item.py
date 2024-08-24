from pydantic import BaseModel
from .item import Item
from .category import Category


class ClassifiedItem(BaseModel):
    item: Item
    category: Category
