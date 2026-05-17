from typing import TypedDict
from torch import Tensor


class RawCategory(TypedDict):
    id: int
    name: str


class RawAnnotation(TypedDict):
    id: int
    image_id: int | str
    category_id: int
    area: float
    bbox: list[float]
    iscrowd: int


class ProcessedAnnotation(TypedDict):
    category_id: int
    bbox: list[float]


class ImageAnnotation(TypedDict):
    category_id: int
    bbox: list[float]
    iscrowd: int
    area: float
    image_id: int


class Label(TypedDict):
    boxes: Tensor
    labels: Tensor
    area: Tensor
    iscrowd: Tensor
    image_id: Tensor
