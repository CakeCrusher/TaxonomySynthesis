from pydantic import BaseModel


class Item(BaseModel):
    id: str

    class Config:
        extra = "allow"
