import dataclasses
import json
import math
import pathlib
from collider.geometry import rotation_matrix
from typing import Iterator, List, Optional
from collider import element
import numpy as np
import matplotlib.pyplot as plt


@dataclasses.dataclass
class BeamInfo:
    Lx: float
    Ly: float
    dx: float
    dy: float
    Cx: float
    Cy: float
    vx: float
    vy: float

    @property
    def angle(self) -> float:
        return math.degrees(math.atan2(self.vy, self.vx))


@dataclasses.dataclass
class SetupInfo:
    Lx: float
    Ly: float
    vx: float
    vy: float

    @property
    def angle(self) -> float:
        return math.degrees(math.atan2(self.vy, self.vx))


class Beam:
    """
    Maintains a list of elements.
    Elements just know their center, in the local coordinates of the beam
    Beam maintains a global coordinate.
    Access to the global coordinate of an element should be through the Beam

    """

    def __init__(self, Lx: float, Ly: float, dx: float, dy: float, Cx: float, Cy: float, angle: float,
                 vx: float = 0.0, vy: float = 0.0, name: str = 'Beam') -> None:
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy
        self.name = name

        self._Cx = Cx
        self._Cy = Cy
        self.vx = vx
        self.vy = vy
        self._angle = angle

        self.Nx = int(Lx / dx)
        self.Ny = int(Ly / dy)

        self._elements: List[element.Element] = []

        # The cache. It is 'None' when dirty/invalid.
        self._cached_global_elements: Optional[List[element.GlobalElement]] = None

    def _invalidate_cache(self):
        """Marks the cache as dirty. This is the key to the solution."""
        self._cached_global_elements = None

    @property
    def Cx(self) -> float:
        """Get the Beam's global centroid."""
        return self._Cx

    @Cx.setter
    def Cx(self, new_Cx: float):
        """Set the Beam's global centroid and invalidate the cache."""
        if self._Cx != new_Cx:
            self._Cx = new_Cx
            self._invalidate_cache()

    @property
    def Cy(self) -> float:
        """Get the Beam's global centroid."""
        return self._Cy

    @Cy.setter
    def Cy(self, new_Cy: float):
        """Set the Beam's global centroid and invalidate the cache."""
        if self._Cy != new_Cy:
            self._Cy = new_Cy
            self._invalidate_cache()

    @property
    def angle(self) -> float:
        """Get the Beam's global angle in degrees."""
        return self._angle

    @angle.setter
    def angle(self, new_angle: float):
        """Set the Beam's global angle and invalidate the cache."""
        if self._angle != new_angle:
            self._angle = new_angle
            self._invalidate_cache()

    def update_position(self, dt):
        self.Cx += self.vx * dt
        self.Cy += self.vy * dt

    def create_elements(self):
        # Elements are in the local coordinate system. They don't know about the Beam center or angle.
        for cx in np.linspace(-(self.Lx - self.dx) / 2, (self.Lx - self.dx) / 2, self.Nx):
            for cy in np.linspace(-(self.Ly - self.dy) / 2, (self.Ly - self.dy) / 2, self.Ny):
                self._elements.append(element.Element(center=(cx, cy), width=(self.dx, self.dy)))

    def get_element_global(self, index: int):
        element = self._elements[index]
        v1 = element.cx

    def serialize(self, filename: str or pathlib.Path) -> None:
        state = {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'dx': self.dx,
            'dy': self.dy,
            'Cx': self.Cx,
            'Cy': self.Cy,
            'angle': self.angle,
            'interactions': [e.interactions for e in self._elements],
        }

        json.dump(state, open(f'{filename}.json', 'w'))

    def update_center_position(self, dx, dy):
        self.Cx += dx
        self.Cy += dy

    def _regenerate_cache(self):
        """
        Recalculates all global element views and fills the cache.
        This is the "expensive" operation that we now only run when needed.
        """
        # print("DEBUG: Regenerating cache...") # Uncomment for testing

        # Pre-calculate trig *once* for the entire batch
        matrix = rotation_matrix(self._angle)

        # angle_rad = math.radians(self._angle)
        # cos_a = math.cos(angle_rad)
        # sin_a = math.sin(angle_rad)
        #
        # beam_cx = self._centroid.x
        # beam_cy = self._centroid.y

        new_cache = []
        for ele in self._elements:
            # Perform rotation in local frame
            rotated_xy = np.dot(matrix, [ele.cx, ele.cy])

            # Perform translation
            rotated_xy += np.array([self.Cx, self.Cy])

            new_cache.append(
                element.GlobalElement(center=rotated_xy, width=ele.width,
                                      interactions=ele.interactions, local_view=ele)
            )

        self._cached_global_elements = new_cache

    def _get_valid_cache(self) -> List[element.GlobalElement]:
        """
        Internal helper that ensures the cache is valid,
        regenerating it *only* if it's currently invalid (None).
        """
        if self._cached_global_elements is None:
            # Cache is dirty, so we rebuild it
            self._regenerate_cache()

        # Now, the cache is guaranteed to be a valid list
        return self._cached_global_elements  # type: ignore

    # --- Container Protocol (Now reads from the cache) ---

    def __len__(self) -> int:
        """Returns the number of elements in the beam."""
        return len(self._elements)

    def __getitem__(self, index) -> element.GlobalElement:
        """
        Gets an element by index, reading from the cache.
        Regenerates cache first if it's dirty.
        """
        valid_cache = self._get_valid_cache()
        return valid_cache[index]

    def __iter__(self) -> Iterator[element.GlobalElement]:
        """
        Iterates over the elements, reading from the cache.
        Regenerates cache first if it's dirty.
        """
        valid_cache = self._get_valid_cache()
        yield from valid_cache

    def __repr__(self) -> str:
        return f"Beam(centroid=({self.Cx},{self.Cy}), angle={self.angle}, elements={len(self)})"


def plot_beam(beam, figure: List = None, bounds=None):
    if figure is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = figure

    outline_patch = plt.Rectangle((beam.Cx - beam.Lx / 2., beam.Cy - beam.Ly / 2.), beam.Lx, beam.Ly,
                                  facecolor='none', edgecolor='black', alpha=0.2,  # 'none' is standard for no fill
                                  rotation_point='center', angle=beam.angle)
    # ax.add_patch(outline_patch)

    # --- Start of Modifications ---

    # 1. Check if there are elements to plot
    if not beam._elements:
        # If no elements, just set limits, save, and return
        ax.set_xlim(-beam.Lx * 2, beam.Lx * 2)
        ax.set_ylim(-beam.Lx * 2, beam.Lx * 2)
        plt.savefig('beam.png')
        return fig, ax

    # 2. Get the data range for the color scale
    interactions = [ele.interactions for ele in beam._elements]
    min_val = min(interactions)
    max_val = max(interactions)

    # 3. Create a normalizer and a colormap
    # Use plt.Normalize to map the interaction values to the [0, 1] range
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    # Choose a colormap (e.g., 'viridis', 'plasma', 'inferno', 'jet', 'coolwarm')
    cmap = plt.cm.get_cmap('viridis')

    # 4. Create a ScalarMappable to handle color mapping and for the colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # --- End of Modifications ---

    for ele in beam:
        # x_center = ele.cx
        # y_center = ele.cy
        #
        # x_center_rot = x_center * np.cos(np.radians(beam.angle)) - y_center * np.sin(np.radians(beam.angle))
        # y_center_rot = x_center * np.sin(np.radians(beam.angle)) + y_center * np.cos(np.radians(beam.angle))
        # x_center_rot += beam.Cx
        # y_center_rot += beam.Cy

        # --- Modified facecolor ---
        # Get the RGBA color for this element's interaction value
        color = mappable.to_rgba(ele.interactions)

        patch = plt.Rectangle((ele.cx - beam.dx / 2., ele.cy - beam.dy / 2.), ele.wx, ele.wy,
                              rotation_point='center', angle=beam.angle,
                              facecolor=color, edgecolor='black',
                              alpha=0.42)  # Your original alpha is preserved
        ax.add_patch(patch)

    # 5. Add the colorbar to the figure
    cbar = fig.colorbar(mappable, ax=ax, orientation='vertical')
    cbar.set_label('Interactions')  # Set a label for the colorbar

    if bounds:
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
    else:
        ax.set_xlim(-(beam.Cx + beam.Lx) * 2, (beam.Cx + beam.Lx) * 2)
        ax.set_ylim(-(beam.Cy + beam.Ly) * 2, (beam.Cy + beam.Ly) * 2)
    ax.set_aspect('equal')
    plt.savefig('beam.png')

    return fig, ax