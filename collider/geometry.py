import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    angle_radians = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return rotation_matrix


def rotate_element_vertices(element, angle: float, centered: bool = True) -> np.ndarray:
    if centered:
        vertices = element.vertices.copy()
        vertices[:, 0] -= element.cx
        vertices[:, 1] -= element.cy
        vertices = vertices @ rotation_matrix(angle).T
        vertices[:, 0] += element.cx
        vertices[:, 1] += element.cy
    else:
        vertices = element.vertices @ rotation_matrix(angle).T

    return vertices