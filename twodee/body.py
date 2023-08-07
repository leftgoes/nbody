from dataclasses import dataclass
import numpy as np
from typing import Self


class Vector2(np.ndarray):
    def __new__(cls, x: float, y: float) -> Self:
        return np.asarray(np.array([x, y])).view(Vector2)
    
    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]


@dataclass
class Rect2:
    min: Vector2
    max: Vector2

    @property
    def center(self) -> Vector2:
        return (self.min + self.max) / 2

    def contains(self, position: Vector2) -> bool:
        return np.all(self.min <= position) and np.all(position < self.max)


@dataclass
class Body2:
    mass: float
    pos: Vector2
    vel: Vector2