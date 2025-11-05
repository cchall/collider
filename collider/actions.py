from collider import beam
from collider import element
from collider.geometry import rotate_element_vertices
import math
import numpy as np


def update_overlap(beam1: beam.Beam, beam2: beam.Beam) -> None:
    """Count all overlapping elements between two beams. Updates interaction counters in place.
    (!) This function assumes all beam elements are the same width and height.
    """
    # Find extrema of each beam - all elements are the same size so we can check this for only one element
    largest_edge1 = max(beam1._elements[0].wx, beam1._elements[0].wy) / 2.
    smallest_edge1 = min(beam1._elements[0].wx, beam1._elements[0].wy) / 2.
    largest_edge2 = max(beam2._elements[0].wx, beam2._elements[0].wy) / 2.
    smallest_edge2 = min(beam2._elements[0].wx, beam2._elements[0].wy) / 2.

    # Perform a fast check if element centers are close enough to overlap
    # If they are not then continue
    # If they are under the length of the smallest edges there must be overlap
    # If we fall between these two extremes a full check of all projections for overlap is required
    for e1 in beam1:
        print(e1)
        for e2 in beam2:
            c2c_dist = math.sqrt((e1.cx - e2.cx)**2 + (e1.cy - e2.cy)**2)
            if c2c_dist > (largest_edge1 + largest_edge2):
                print(111)
                continue
            if c2c_dist < (smallest_edge1 + smallest_edge2):
                print(222)
                e1.interactions += 1
                e2.interactions += 1
            else:
                overlap = elements_overlap(e1, e2, beam1.angle, beam2.angle)
                print(333)
                if overlap:
                    e1.interactions += 1
                    e2.interactions += 1


def elements_overlap(element1: element.Element, element2: element.Element, angle1: float, angle2: float) -> bool:
    """Returns True if the elements overlap, otherwise False"""
    vertices1 = rotate_element_vertices(element1, angle1, centered=True)
    vertices2 = rotate_element_vertices(element2, angle2, centered=True)

    # For each rectangle (element) we find the two orthogonal edges (p)
    edges1 = [  # indexing: [edge][vertex][coordinate]
        [vertices1[0], vertices1[1]],
        [vertices1[1], vertices1[2]],
    ]
    edges2 = [ # indexing: [edge][vertex][coordinate]
        [vertices2[0], vertices2[1]],
        [vertices2[1], vertices2[2]],
    ]

    # For each edge we find the unit vector along that edge. This unit vector
    # defines the line that we will project a "shadow" of each rectangle onto
    projections1 = [  # indexing = [projection][coordinate]
        (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]) for edge in edges1
    ]
    projectsion2 = [  # indexing = [projection][coordinate]
        (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]) for edge in edges2
    ]

    for proj in (*projections1, *projectsion2):  # 4 total projections to check
        ele1_proj = [np.dot(proj, v) for v in vertices1]
        ele2_proj = [np.dot(proj, v) for v in vertices2]

        # If cond1 is true ele1 is fully to the right of ele2 or if cond2 then it is fully to the left.
        # If a vertex is shared this is not considered to be an overlap - so we use le/ge
        # Otherwise, the two elements overlap on this projection.
        if min(ele1_proj) >= max(ele2_proj) or max(ele1_proj) <= min(ele2_proj):
            return False

    # The element shadows overlap on all projections so their areas must overlap
    return True


if __name__ == "__main__":
    # Test
    import matplotlib.pyplot as plt
    element1 = element.Element(center=(0.5, 0.24), width=(1.3, 0.89))
    element2 = element.Element(center=(-0.8, 0.24), width=(2.24, 0.1))
    angle1 = 0
    angle2 = 30

    overlap = elements_overlap(element1, element2, angle1=angle1, angle2=angle2)
    plt.figure()
    for v in rotate_element_vertices(element1, angle1, centered=True):
        plt.scatter(*v, c='C1')
    for v in rotate_element_vertices(element2, angle2, centered=True):
        plt.scatter(*v, c='C2')
    plt.savefig('overlap.png')
    print(f"{overlap=}")
