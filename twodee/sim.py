import matplotlib.pyplot as plt
import numpy as np

from .body import Body2
from . import random


G = 6.6743e-11


class Simulation:
    def __init__(self) -> None:
        self.bodies: np.ndarray[Body2] = None
    
    @property
    def count(self) -> int:
        return self.bodies.shape[0]

    def create_circular(self, count: int, mass_zero: float = 1, mass_half: float = 1.5,
                                      radius_zero: float = 1, radius_half: float = 1.5) -> None:
        masses = random.mass(count, mass_zero, mass_half)
        radii = random.radius(count, radius_zero, radius_half)
        angles = random.angle(count)
        
        self.bodies = Body2.from_posisitons(G * masses.sum(), masses, radii, angles)
