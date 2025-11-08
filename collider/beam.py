import dataclasses
import json
import math
import pathlib
from typing import Iterator, List, Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

from collider import element
from collider.geometry import rotation_matrix



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
    Elements store their centers fractionally (from -0.5 to 0.5) in the
    local coordinates of the beam.
    Resizing (Lx, Ly) is handled dynamically during cache generation.
    """

    def __init__(self, Lx: float, Ly: float, dx: float, dy: float, Cx: float, Cy: float, angle: float,
                 vx: float = 0.0, vy: float = 0.0, name: str = 'Beam') -> None:
        self._Lx = Lx
        self._Ly = Ly
        self.name = name

        self._Cx = Cx
        self._Cy = Cy
        self.vx = vx
        self.vy = vy
        self._angle = angle

        # Nx and Ny are fixed at init to define the grid topology
        self.Nx = int(Lx / dx)
        self.Ny = int(Ly / dy)

        self._elements: List[element.Element] = []

        # The cache. It is 'None' when dirty/invalid.
        self._cached_global_elements: Optional[List[element.GlobalElement]] = None

    def _invalidate_cache(self):
        """Marks the cache as dirty. This is the key to the solution."""
        self._cached_global_elements = None

    # --- Properties for all global transformations ---

    @property
    def Lx(self) -> float:
        return self._Lx

    @Lx.setter
    def Lx(self, new_Lx: float):
        """Set the Beam's physical length and invalidate the cache."""
        if self._Lx != new_Lx:
            self._Lx = new_Lx
            self._invalidate_cache()

    @property
    def Ly(self) -> float:
        return self._Ly

    @Ly.setter
    def Ly(self, new_Ly: float):
        """Set the Beam's physical width and invalidate the cache."""
        if self._Ly != new_Ly:
            self._Ly = new_Ly
            self._invalidate_cache()

    # --- Read-only properties for physical dx/dy ---

    @property
    def dx(self) -> float:
        """Get the current physical width of one element."""
        return self._Lx / self.Nx

    @property
    def dy(self) -> float:
        """Get the current physical height of one element."""
        return self._Ly / self.Ny

    @property
    def Cx(self) -> float:
        return self._Cx

    @Cx.setter
    def Cx(self, new_Cx: float):
        if self._Cx != new_Cx:
            self._Cx = new_Cx
            self._invalidate_cache()

    @property
    def Cy(self) -> float:
        return self._Cy

    @Cy.setter
    def Cy(self, new_Cy: float):
        if self._Cy != new_Cy:
            self._Cy = new_Cy
            self._invalidate_cache()

    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, new_angle: float):
        if self._angle != new_angle:
            self._angle = new_angle
            self._invalidate_cache()

    def update_position(self, dt):
        # Setters will handle cache invalidation
        self.Cx += self.vx * dt
        self.Cy += self.vy * dt

    def create_elements(self):
        """
        Populates the _elements list with a FRACTIONAL grid.
        Centers range from -0.5 to 0.5.
        This method is only called once.
        """

        # Calculate fractional dimensions
        frac_dx = 1.0 / self.Nx
        frac_dy = 1.0 / self.Ny
        frac_width = (frac_dx, frac_dy)

        # Create linspace for fractional centers
        # This is equivalent to np.linspace(-(Lx - dx) / 2, (Lx - dx) / 2, Nx) / Lx
        # Which simplifies to np.linspace(-(1 - frac_dx) / 2, (1 - frac_dx) / 2, Nx)
        cx_coords = np.linspace(-(1.0 - frac_dx) / 2.0, (1.0 - frac_dx) / 2.0, self.Nx)
        cy_coords = np.linspace(-(1.0 - frac_dy) / 2.0, (1.0 - frac_dy) / 2.0, self.Ny)

        self._elements = []
        for cx in cx_coords:
            for cy in cy_coords:
                self._elements.append(element.Element(center=(cx, cy), width=frac_width))

        self._invalidate_cache()  # Invalidate after creation

    def serialize(self, filename: Union[str, pathlib.Path]) -> None:
        """Saves the current beam state to a JSON file."""
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
        # Ensure filename has .json extension
        path = pathlib.Path(filename)
        if path.suffix != '.json':
            path = path.with_suffix('.json')

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def _regenerate_cache(self) -> None:
        """
        Recalculates all global element views.
        Performs vectorized scaling, rotation, and translation.
        """
        if not self._elements:
            self._cached_global_elements = []
            return

        # 1. Gather all fractional local coordinates into a (N, 2) array
        # Note: If performance is critical, consider maintaining _frac_coords
        # as a persistent numpy array alongside _elements.
        frac_coords = np.array([[e.cx, e.cy] for e in self._elements])

        # 2. Scale: fractional -> physical local dimensions
        local_physical_coords = frac_coords * np.array([self.Lx, self.Ly])

        # 3. Rotate
        # Transpose the rotation matrix to allow right-multiplication by row vectors: (M @ v).T == v @ M.T
        matrix = rotation_matrix(self._angle)
        rotated_coords = local_physical_coords @ matrix.T

        # 4. Translate: Add global beam centroid
        global_coords = rotated_coords + np.array([self.Cx, self.Cy])

        # 5. Rebuild the cache list
        current_physical_width = (self.dx, self.dy)
        self._cached_global_elements = [
            element.GlobalElement(
                center=tuple(gc),
                width=current_physical_width,
                interactions=ele.interactions,
                local_view=ele
            )
            for gc, ele in zip(global_coords, self._elements)
        ]

    def _get_valid_cache(self) -> List[element.GlobalElement]:
        """
        Internal helper that ensures the cache is valid,
        regenerating it *only* if it's currently invalid (None).
        """
        if self._cached_global_elements is None:
            self._regenerate_cache()

        return self._cached_global_elements  # type: ignore

    # --- Container Protocol (Now reads from the cache) ---

    def __len__(self) -> int:
        return len(self._elements)

    def __getitem__(self, index) -> element.GlobalElement:
        return self._get_valid_cache()[index]

    def __iter__(self) -> Iterator[element.GlobalElement]:
        yield from self._get_valid_cache()

    def __repr__(self) -> str:
        return (f"Beam(centroid=({self.Cx:.2f},{self.Cy:.2f}), angle={self.angle:.1f}, "
                f"Size=({self.Lx:.2f}x{self.Ly:.2f}), elements={len(self)})")


def plot_beam(beam, figure: List = None, bounds=None, filename: str = None):
    if figure is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = figure

    outline_patch = plt.Rectangle((beam.Cx - beam.Lx / 2., beam.Cy - beam.Ly / 2.), beam.Lx, beam.Ly,
                                  facecolor='none', edgecolor='black', alpha=0.2,  # 'none' is standard for no fill
                                  rotation_point='center', angle=beam.angle)
    # ax.add_patch(outline_patch)

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

    for ele in beam:
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

    if filename:
        plt.savefig(filename, dpi=600)

    return fig, ax