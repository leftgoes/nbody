from dataclasses import dataclass
import numpy as np
from typing import Self


class Vector2(np.ndarray):
    def __new__(cls, x: float, y: float) -> Self:
        return np.asarray(np.array([x, y], dtype=np.float_)).view(Vector2)
    
    def __repr__(self) -> str:
        if self.x == 0 == self.y:
            return '[0]'
        else:
            return f'[{self.x}, {self.y}]'

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]
    
    @classmethod
    def zero(cls) -> Self:
        return cls(0, 0)


@dataclass
class Rect2:
    min: Vector2
    max: Vector2

    @property
    def center(self) -> Vector2:
        return (self.min + self.max) / 2

    def contains(self, position: Vector2) -> bool:
        return np.all(self.min <= position) and np.all(position < self.max)


@dataclass(slots=True)
class Body2:
    mass: float
    pos: Vector2
    vel: Vector2

    @classmethod
    def zero(cls) -> Self:
        return cls(0, 0, 0)

    def __add__(self: Self, other: Self) -> Self:
        return Body2(self.mass + other.mass,
                     self.pos + other.pos,
                     self.vel + other.vel)