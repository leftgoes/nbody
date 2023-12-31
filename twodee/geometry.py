from dataclasses import dataclass
import numpy as np
from typing import Self


class Vector2(np.ndarray):
    def __new__(cls, x: float, y: float) -> Self:
        return np.asarray(np.array([x, y], dtype=np.float_)).view(cls)
    
    def __repr__(self) -> str:
        if self.x == 0 == self.y:
            return '[0]'
        else:
            return f'[{self.x}, {self.y}]'

    @classmethod
    def from_polar(cls, r: float, theta: float) -> Self:
        return cls(r * np.cos(theta), r* np.sin(theta))
    
    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]
    
    @classmethod
    def zero(cls) -> Self:
        return cls(0, 0)


@dataclass(slots=True)
class Rect2:
    min: Vector2
    max: Vector2

    @property
    def center(self) -> Vector2:
        return (self.min + self.max) / 2

    def contains(self, position: Vector2) -> bool:
        return np.all(self.min <= position) and np.all(position < self.max)
