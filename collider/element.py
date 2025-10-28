import numpy as np


class Element:
    __slots__ = ['center', 'width', 'interactions']

    def __init__(self, center: (float, float), width: (float, float)):
        self.center = center
        self.width = width
        self.interactions = 0

        # half widths are used internally to cut down on operations
        self._hwx = width[0]/2.
        self._hwy = width[1]/2.

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

    @property
    def vertices(self) -> np.ndarray:
        """Returns vertices in clockwise direction.
                v1 --------------- v2
                 |                  |
                 |        (0,0)     |
                 |         +        |
                 |                  |
                 |                  |
                v4 --------------- v3
        """
        v1 = [self.cx - self._hwx, self.cy + self._hwy]
        v2 = [self.cx + self._hwx, self.cy + self._hwy]
        v3 = [self.cx + self._hwx, self.cy - self._hwy]
        v4 = [self.cx - self._hwx, self.cy - self._hwy]

        return np.array([v1, v2, v3, v4])

