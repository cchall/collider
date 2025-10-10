import numpy as np


class Element:
    __slots__ = ['center', 'width', 'interactions']

    def __init__(self, center: (float, float), width: (float, float)):
        self.center = center
        self.width = width
        self.interactions = 0

    @property
    def cx(self) -> float:
        return self.center[0]
    @property
    def cy(self) -> float:
        return self.center[1]
    @property
    def wx(self) -> float:
        return self.width[0]
    @property
    def wy(self) -> float:
        return self.width[1]
