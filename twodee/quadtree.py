from dataclasses import dataclass
from typing import Iterable, Self

from .body import Body2, Vector2, Rect2


@dataclass
class Node4:
    boundary: Rect2

    northwest: Self = None
    northeast: Self = None
    southwest: Self = None
    southeast: Self = None
    body: Body2 = None

    @property
    def filled(self) -> bool:
        return self.body is not None

    @property
    def subdivided(self) -> bool:
        return self.northwest is not None

    def subdivide(self) -> None:
        boundary_center = self.boundary.center

        northwest = Rect2(Vector2(self.boundary.min.x, boundary_center.y),
                          Vector2(boundary_center.x, self.boundary.max.y))
        northeast = Rect2(boundary_center, self.boundary.max)
        southwest = Rect2(self.boundary.min, boundary_center)
        southeast = Rect2(Vector2(boundary_center.x, self.boundary.min.y),
                          Vector2(self.boundary.max.x, boundary_center.y))

        self.northwest = Node4(northwest)
        self.northeast = Node4(northeast)
        self.southwest = Node4(southwest)
        self.southeast = Node4(southeast)

    def insert(self, body: Body2):
        if not self.boundary.contains(body.pos):
            return

        if not self.filled:
            self.body = body
            return
        
        if not self.subdivided:
            self.subdivide()
        
        self.northwest.insert(body)
        self.northeast.insert(body)
        self.southwest.insert(body)
        self.southeast.insert(body)


class Quadtree:
    def __init__(self, boundary: Rect2) -> None:
        self.root = Node4(boundary)
    
    @classmethod
    def from_bodies(cls, bodies: Iterable[Body2], contraint: Rect2) -> Self:
        min_x = min(bodies, key=lambda i: i.pos.x).pos.x
        min_y = min(bodies, key=lambda i: i.pos.y).pos.y
        max_x = max(bodies, key=lambda i: i.pos.x).pos.x
        max_y = max(bodies, key=lambda i: i.pos.y).pos.y

        boundary = Rect2(Vector2(min(min_x, contraint.min.x), min(min_y, contraint.min.y)),
                         Vector2(max(max_x, contraint.max.x), max(max_y, contraint.max.y)))
        
        return cls(boundary)