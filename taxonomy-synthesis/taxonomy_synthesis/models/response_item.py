from pydantic import BaseModel


class ResponseItem(BaseModel):
    item_id: str
    category_name: str
