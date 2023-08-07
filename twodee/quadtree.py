from dataclasses import dataclass
import numpy as np
from typing import Iterable, Self, Generator

from .body import Body2, Vector2, Rect2


@dataclass(repr=False)
class Node4:
    boundary: Rect2

    northwest: Self = None
    northeast: Self = None
    southwest: Self = None
    southeast: Self = None
    barycenter: Body2 = None
    _count: int = 0

    def __repr__(self) -> str:
        return f'Node4({self.count})' if self.count <= 1 else f'Node4({", ".join(str(child.count) for child in self.children)})'

    @property
    def count(self):
        return self._count

    @property
    def children(self) -> tuple[Self]:
        return (self.northwest, self.northeast, self.southwest, self.southeast)

    @property
    def subdivided(self) -> bool:
        return self.northwest is not None

    def traverse(self) -> Generator[Self, None, None]:  # post order traversal
        if self.count == 0:
            yield self
            return
        
        if self.count != 1:
            for child in self.children:
                for node in child.traverse():
                    yield node
        
        yield self

    def calculate_barycenter(self) -> None:
        self.barycenter = sum((child.barycenter for child in self.children), start=Body2.zero())
        self.barycenter.pos /= self.count
        self.barycenter.vel /= self.count

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
            self.barycenter = body
            self._count += 1
            return
        
        if not self.subdivided:
            self.subdivide()
            self.insert_children(self.barycenter)
            self.barycenter = None
        
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
        for node in self.root.traverse():
            if node.count == 0:
                node.barycenter = Body2(0, Vector2.zero(), Vector2.zero())
                
            if node.subdivided:
                node.calculate_barycenter()
            
            
            