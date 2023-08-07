from dataclasses import dataclass, field
import numpy as np
from typing import Iterable, Self, Generator, Literal

from .body import Body2
from .geometry import Vector2, Rect2

     
@dataclass(repr=False, slots=True)
class Node4:
    boundary: Rect2

    northwest: Self = None
    northeast: Self = None
    southwest: Self = None
    southeast: Self = None

    _barycenter: Body2 | None = field(default_factory=lambda: Body2.zero())
    _count: int = 0

    def __repr__(self) -> str:
        return f'Node4({self.count})' if self.count <= 1 else f'Node4({", ".join(str(child.count) for child in self.children)})'

    @property
    def barycenter(self) -> Body2:
        return self._barycenter

    @property
    def count(self):
        return self._count

    @property
    def children(self) -> tuple[Self]:
        return (self.northwest, self.northeast, self.southwest, self.southeast)

    @property
    def subdivided(self) -> bool:
        return self.northwest is not None

    def traverse(self, order: Literal['post', 'pre']) -> Generator[Self, None, None]:
        if self.count == 0:
            yield self
            return
        
        if order == 'pre':
            yield self

        if self.count != 1:
            for child in self.children:
                for node in child.traverse(order):
                    yield node
        
        if order == 'post':
            yield self

    def calculate_barycenter(self) -> None:
        self._barycenter = sum((child.barycenter for child in self.children), start=Body2.zero())
        self._barycenter.pos /= self.count
        self._barycenter.vel /= self.count

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

        if self.count == 0:
            self._barycenter = body
            self._count += 1
            return
        
        if not self.subdivided:
            self.subdivide()
            self.insert_children(self._barycenter)
            self._barycenter = None
        
        self.insert_children(body)
        self._count += 1
    
    def insert_children(self, body: Body2) -> None:
        for child in self.children:
            child.insert(body)


class Quadtree:
    def __init__(self, boundary: Rect2) -> None:
        self.root = Node4(boundary)
    
    def __repr__(self) -> str:
        return f'Quadtree({self.root.count})'

    @classmethod
    def from_bodies(cls, bodies: Iterable[Body2], contraint: Rect2) -> Self:
        min_x = min(bodies, key=lambda i: i.pos.x).pos.x
        min_y = min(bodies, key=lambda i: i.pos.y).pos.y
        max_x = max(bodies, key=lambda i: i.pos.x).pos.x
        max_y = max(bodies, key=lambda i: i.pos.y).pos.y

        boundary = Rect2(Vector2(min(min_x, contraint.min.x), min(min_y, contraint.min.y)),
                         Vector2(max(max_x, contraint.max.x), max(max_y, contraint.max.y)))
        
        tree = cls(boundary)
        for body in bodies:
            tree.insert(body)

        return tree

    def insert(self, body: Body2) -> None:
        self.root.insert(body)

    def calculate_barycenters(self) -> None:
        for node in self.root.traverse(order='post'):
            if node.subdivided:
                node.calculate_barycenter()
