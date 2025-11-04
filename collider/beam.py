import json
from typing import List
from collider import element
import numpy as np
import matplotlib.pyplot as plt


class Beam:
    """
    Maintains a list of elements.
    Elements just know their center, in the local coordinates of the beam
    Beam maintains a global coordinate.
    Access to the global coordinate of an element should be through the Beam

    """

    def __init__(self, Lx: float, Ly: float, dx: float, dy: float, Cx: float, Cy: float, angle: float,
                 vx: float = 0.0, vy: float = 0.0):
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy

        self.Cx = Cx
        self.Cy = Cy
        self.vx = vx
        self.vy = vy
        self.angle = angle

        self.Nx = int(Lx / dx)
        self.Ny = int(Ly / dy)

        self.elements: List[element.Element] = []

    def update_position(self, dt):
        self.Cx += self.vx * dt
        self.Cy += self.vy * dt

    def create_elements(self):
        # Elements are in the local coordinate system. They don't know about the Beam center or angle.
        for cx in np.linspace(-(self.Lx - self.dx) / 2, (self.Lx - self.dx) / 2, self.Nx):
            for cy in np.linspace(-(self.Ly - self.dy) / 2, (self.Ly - self.dy) / 2, self.Ny):
                self.elements.append(element.Element(center=(cx, cy), width=(self.dx, self.dy)))

    def get_element_global(self, index: int):
        element = self.elements[index]
        v1 = element.cx

    def serialize(self, name):
        state = {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'dx': self.dx,
            'dy': self.dy,
            'Cx': self.Cx,
            'Cy': self.Cy,
            'angle': self.angle,
            'interactions': [e.interactions for e in self.elements],
        }

        json.dump(state, open(f'{name}.json', 'w'))

    def plot(self, figure: List = None):
        if figure is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig, ax = figure

        outline_patch = plt.Rectangle((self.Cx - self.Lx / 2., self.Cy - self.Ly / 2.), self.Lx, self.Ly,
                                      facecolor='none', edgecolor='black', alpha=0.2,  # 'none' is standard for no fill
                                      rotation_point='center', angle=self.angle)
        ax.add_patch(outline_patch)

        # --- Start of Modifications ---

        # 1. Check if there are elements to plot
        if not self.elements:
            # If no elements, just set limits, save, and return
            ax.set_xlim(-self.Lx * 2, self.Lx * 2)
            ax.set_ylim(-self.Lx * 2, self.Lx * 2)
            plt.savefig('beam.png')
            return fig, ax

        # 2. Get the data range for the color scale
        interactions = [ele.interactions for ele in self.elements]
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

        for ele in self.elements:
            x_center = ele.cx
            y_center = ele.cy

            x_center_rot = x_center * np.cos(np.radians(self.angle)) - y_center * np.sin(np.radians(self.angle))
            y_center_rot = x_center * np.sin(np.radians(self.angle)) + y_center * np.cos(np.radians(self.angle))
            x_center_rot += self.Cx
            y_center_rot += self.Cy

            # --- Modified facecolor ---
            # Get the RGBA color for this element's interaction value
            color = mappable.to_rgba(ele.interactions)

            patch = plt.Rectangle((x_center_rot - self.dx / 2., y_center_rot - self.dy / 2.), ele.wx, ele.wy,
                                  rotation_point='center', angle=self.angle,
                                  facecolor=color, edgecolor='black',
                                  alpha=0.2)  # Your original alpha is preserved
            ax.add_patch(patch)

        # 5. Add the colorbar to the figure
        cbar = fig.colorbar(mappable, ax=ax, orientation='vertical')
        cbar.set_label('Interactions')  # Set a label for the colorbar

        ax.set_xlim(-(self.Cx + self.Lx) * 2, (self.Cx + self.Lx)  * 2)
        ax.set_ylim(-(self.Cy + self.Ly)  * 2, (self.Cy + self.Ly)  * 2)
        ax.set_aspect('equal')
        plt.savefig('beam.png')

        return fig, ax

    def update_center_position(self, dx, dy):
        self.Cx += dx
        self.Cy += dy
