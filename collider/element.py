import numpy as np


class Element:
    __slots__ = ['center', 'width', 'interactions', '_hwx', '_hwy']

    def __init__(self, center: (float, float), width: (float, float), interactions: int = 0):
        self.center = center
        self.width = width
        self.interactions = interactions

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
                 |        (x,y)     |
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

    def __repr__(self) -> str:
        return f"Element(centroid=({self.center}), width={self.width})"


class GlobalElement(Element):
    """
    An Element view that stores global coordinates and proxies
    interactions to its original 'local_view' element.
    """

    def __init__(self, center: (float, float), width: (float, float), interactions: int, local_view: Element):
        # 2. Store the local_view. It is now a required argument.
        #    We no longer need the 'if/else' block.
        if local_view is None:
            raise ValueError("GlobalElement must be initialized with a valid local_view Element.")
        self._local_view = local_view

        # 1. Initialize the super() class with *global* coordinates
        super().__init__(center, width, interactions)


    @property
    def interactions(self) -> int:
        """Gets the interaction count from the underlying local element."""
        return self._local_view.interactions

    @interactions.setter
    def interactions(self, new_value: int):
        """Sets the interaction count on the underlying local element."""
        # 3. This is now a standard, predictable setter.
        self._local_view.interactions = new_value

    def __repr__(self) -> str:
        # A clearer repr for the global view
        return f"GlobalElement(global_centroid=({self.center}), width={self.width}, local_interactions={self.interactions})"
