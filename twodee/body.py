from dataclasses import dataclass
import numpy as np
from typing import Self

from .geometry import Vector2

@dataclass(slots=True)
class Body2:
    mass: float
    pos: Vector2
    vel: Vector2

    @classmethod
    def from_posisitons(cls, mu: float, masses: np.ndarray, radii: np.ndarray, angles: np.ndarray) -> np.ndarray[Self]:
        length = radii.shape[0]
        velocities = np.sqrt(mu / radii)
        
        bodies = np.empty(length, dtype=cls)
        for i, (mass, radius, angle, velocity) in enumerate(zip(masses, radii, angles, velocities)):
            bodies[i] = cls(mass,
                            Vector2.from_polar(radius, angle),
                            Vector2.from_polar(velocity, angle + np.pi/2))
        return bodies

    @classmethod
    def zero(cls) -> Self:
        return cls(0, Vector2.zero(), Vector2.zero())

    def __add__(self: Self, other: Self) -> Self:
        return Body2(self.mass + other.mass,
                     self.pos + other.pos,
                     self.vel + other.vel)

