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

    def __init__(self, Lx: float, Ly: float, dx: float, dy: float):
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy
        self.Nx = int(Lx / dx)
        self.Ny = int(Ly / dy)

        self.elements: List[element.Element] = []

    def create_elements(self):
        for cx in np.linspace(-(self.Lx - self.dx) / 2, (self.Lx - self.dx) / 2, self.Nx):
            for cy in np.linspace(-(self.Ly - self.dy) / 2, (self.Ly - self.dy) / 2, self.Ny):
                self.elements.append(element.Element(center=(cx, cy), width=(self.dx, self.dy)))

    def plot(self):
        fig, ax = plt.subplots(1, 1)

        outline_patch = plt.Rectangle((-self.Lx/2., -self.Ly/2.), self.Lx, self.Ly,
                                      facecolor=None, edgecolor='black', alpha=0.2)
        ax.add_patch(outline_patch)
        for ele in self.elements:
            patch = plt.Rectangle((ele.cx - ele.wx / 2., ele.cy - ele.wy / 2.), ele.wx, ele.wy,
                                  facecolor='C4', edgecolor='black', alpha=0.2)
            ax.add_patch(patch)
        ax.set_xlim(-self.Lx, self.Lx)
        ax.set_ylim(-self.Ly, self.Ly)
        plt.savefig('beam.png')

        return fig, ax
