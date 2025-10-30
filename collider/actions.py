from collider import beam
from collider import element
import math
import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    angle_radians = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return rotation_matrix


def update_overlap(beam1: beam.Beam, beam2: beam.Beam):
    largest_edge1 = max(beam1.elements[0].wx, beam1.elements[0].wy) / 2.
    smallest_edge1 = min(beam1.elements[0].wx, beam1.elements[0].wy) / 2.
    largest_edge2 = max(beam2.elements[0].wx, beam2.elements[0].wy) / 2.
    smallest_edge2 = min(beam2.elements[0].wx, beam2.elements[0].wy) / 2.

    for e1 in beam1.elements:
        for e2 in beam2.elements:
            c2c_dist = math.sqrt((e1.cx - e2.cx)**2 + (e1.cy - e2.cy)**2)
            if c2c_dist > (largest_edge1 + largest_edge2):
                continue
            if c2c_dist < (smallest_edge1 + smallest_edge2):
                e1.interactions += 1
                e2.interactions += 1
            else:
                # TODO: sat_overlap_detection will run here
                pass


# TODO: Either implement angle and rotate the elements prior to check or pass in global coordinates so no rotation is needed
def elements_overlap(element1: element.Element, element2: element.Element, angle1: float, angle2: float) -> bool:
    """Returns True if the elements overlap, otherwise False"""
    vertices = []
    for ele, angle in zip((element1, element2), (angle1, angle2)):
        if angle != 0.:
            vertices.append(np.dot(ele.vertices, rotation_matrix(angle)))
        else:
            vertices.append(ele.vertices)
    vertices1, vertices2 = vertices[0], vertices[1]

    # For each rectangle (element) we find the two orthogonal edges (p)
    edges1 = [  # indexing: [edge][vertex][coordinate]
        [element1.vertices[0], element1.vertices[1]],
        [element1.vertices[1], element1.vertices[2]],
    ]
    edges2 = [ # indexing: [edge][vertex][coordinate]
        [element2.vertices[0], element2.vertices[1]],
        [element2.vertices[1], element2.vertices[2]],
    ]

    # For each edge we find the unit vector along that edge. This unit vector
    # defines the line that we will project a "shadow" of each rectangle onto
    projections1 = [  # indexing = [projection][coordinate]
        (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]) for edge in edges1
    ]
    projectsion2 = [  # indexing = [projection][coordinate]
        (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]) for edge in edges2
    ]  # TODO: Trying without making unit vector, need: / math.sqrt(edge11[1][0]**2 + edge11[1][1]**2)

    for proj in (*projections1, *projectsion2):  # 4 total projections to check
        ele1_proj = [np.dot(proj, v) for v in element1.vertices]
        ele2_proj = [np.dot(proj, v) for v in element2.vertices]

        # If cond1 is true ele1 is fully to the right of ele2 or if cond2 then it is fully to the left.
        # Otherwise, the two elements overlap on this projection.
        if min(ele1_proj) > max(ele2_proj) or max(ele1_proj) < min(ele2_proj):
            return False

    # The element shadows overlap on all projections so their areas must overlap
    return True





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    element1 = element.Element(center=(0.5, 0.24), width=(1.3, 0.89))
    element2 = element.Element(center=(-0.8, 0.24), width=(2.24, 0.1))

    overlap = elements_overlap(element1, element2, angle1=0, angle2=0)
    plt.figure()
    for v in element1.vertices:
        plt.scatter(*v, c='C1')
    for v in element2.vertices:
        plt.scatter(*v, c='C2')
    plt.savefig('overlap.png')
    print(f"{overlap=}")
