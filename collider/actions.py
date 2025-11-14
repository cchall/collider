from collider import beam
from collider import element
from collider.geometry import rotate_element_vertices
import math
import numpy as np


def update_beam_overlap(beam1: beam.Beam, beam2: beam.Beam, fuzz: float = 1e-8) -> None:
    """Count all overlapping elements between two beams. Updates interaction counters in place.
    (!) This function assumes all beam elements are the same width and height.
    """
    # TODO: Need to account for rotation here
    # Find extrema of each beam - all elements are the same size so we can check this for only one element
    largest_edge1 = max(beam1[0].wx, beam1[0].wy) / 2.
    smallest_edge1 = min(beam1[0].wx, beam1[0].wy) / 2.
    largest_edge2 = max(beam2[0].wx, beam2[0].wy) / 2.
    smallest_edge2 = min(beam2[0].wx, beam2[0].wy) / 2.

    # Perform a fast check if element centers are close enough to overlap
    # If they are not then continue
    # If they are under the length of the smallest edges there must be overlap
    # If we fall between these two extremes a full check of all projections for overlap is required
    for e1 in beam1:
        for e2 in beam2:
            if e2._local_view in e1._local_view.interacted_with:
                continue

            c2c_dist = math.sqrt((e1.cx - e2.cx)**2 + (e1.cy - e2.cy)**2)
            if c2c_dist > (largest_edge1 + largest_edge2):
                continue
            # Floating point math can give false positives here. So add a small fuzz for cases where the edges overlap.
            if c2c_dist + fuzz < (smallest_edge1 + smallest_edge2):
                e1.interactions += e1.density * e2.density
                e2.interactions += e1.density * e2.density
                e1.flux += e1.density * e2.density
                e2.flux += e1.density * e2.density

                e1._local_view.interacted_with.add(e2._local_view)
                e2._local_view.interacted_with.add(e1._local_view)
            else:
                overlap = overlap_shadows(e1, e2, beam1.angle, beam2.angle, fuzz)
                if overlap:
                    e1.interactions += e1.density * e2.density
                    e2.interactions += e1.density * e2.density
                    e1.flux += e1.density * e2.density
                    e2.flux += e1.density * e2.density

                    e1._local_view.interacted_with.add(e2._local_view)
                    e2._local_view.interacted_with.add(e1._local_view)


def check_overlap(element1: element.Element, element2: element.Element, angle1: float, angle2: float) -> bool:
    """Fast test of where cases elements are guaranteed to be overlapping or not.
       Falls back to shadow test if fast test is inconclusive."""
    # Find extrema of each beam - all elements are the same size so we can check this for only one element
    largest_edge1 = max(element1.wx, element1.wy) / 2.
    smallest_edge1 = min(element1.wx, element1.wy) / 2.
    largest_edge2 = max(element2.wx, element2.wy) / 2.
    smallest_edge2 = min(element2.wx, element2.wy) / 2.

    # Perform a fast check if element centers are close enough to overlap
    # If they are not then continue
    # If they are under the length of the smallest edges there must be overlap
    # If we fall between these two extremes a full check of all projections for overlap is required
    c2c_dist = math.sqrt((element1.cx - element2.cx)**2 + (element1.cy - element2.cy)**2)
    if c2c_dist > (largest_edge1 + largest_edge2):
        return False
    if c2c_dist < (smallest_edge1 + smallest_edge2):
        return True
    else:
        overlap = overlap_shadows(element1, element2, angle1, angle2)
        return overlap


def check_beam_proximity(beam1: beam.Beam, beam2: beam.Beam) -> bool:
    """Returns False if no possibility of beams overlapping, otherwise True."""
    c2c_dist = math.sqrt((beam1.Cx - beam2.Cx) ** 2 + (beam1.Cy - beam2.Cy) ** 2)

    # Calculate the AABB of the rotated beam 1
    c1 = abs(math.cos(math.radians(beam1.angle)))
    s1 = abs(math.sin(math.radians(beam1.angle)))
    beam1_width_aabb = beam1.Lx * c1 + beam1.Ly * s1
    beam1_height_aabb = beam1.Lx * s1 + beam1.Ly * c1
    # Calculate the AABB of the rotated beam 2
    c2 = abs(math.cos(math.radians(beam2.angle)))
    s2 = abs(math.sin(math.radians(beam2.angle)))
    beam2_width_aabb = beam2.Lx * c2 + beam2.Ly * s2
    beam2_height_aabb = beam2.Lx * s2 + beam2.Ly * c2

    m1 = max(beam1_width_aabb, beam1_height_aabb) / 2.
    m2 = max(beam2_width_aabb, beam2_height_aabb) / 2.

    if c2c_dist > (m1 + m2):
        return False

    return True


def overlap_shadows(element1: element.Element, element2: element.Element, angle1: float, angle2: float,
                    fuzz: float = 1e-8) -> bool:
    """General overlap test, returns True if the elements overlap, otherwise False"""
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
    projections1 = np.array([  # indexing = [projection][coordinate]
        (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]) for edge in edges1
    ])
    projections1 = projections1 / np.linalg.norm(projections1, axis=0)
    projections2 = np.array([  # indexing = [projection][coordinate]
        (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]) for edge in edges2
    ])
    projections2 = projections2 / np.linalg.norm(projections2, axis=0)

    for proj in (*projections1, *projections2):  # 4 total projections to check
        ele1_proj = [np.dot(proj, v) for v in vertices1]
        ele2_proj = [np.dot(proj, v) for v in vertices2]

        # If cond1 is true ele1 is fully to the right of ele2 or if cond2 then it is fully to the left.
        # If a vertex is shared this is not considered to be an overlap - so we use le/ge
        # Otherwise, the two elements overlap on this projection.
        # Floating point math can give false positives here. So add a small fuzz for cases where the vertices overlap.
        if (min(ele1_proj) + fuzz) >= max(ele2_proj) or (max(ele1_proj) - fuzz) <= min(ele2_proj):
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

    overlap = overlap_shadows(element1, element2, angle1=angle1, angle2=angle2)
    plt.figure()
    for v in rotate_element_vertices(element1, angle1, centered=True):
        plt.scatter(*v, c='C1')
    for v in rotate_element_vertices(element2, angle2, centered=True):
        plt.scatter(*v, c='C2')
    plt.savefig('overlap.png')
    print(f"{overlap=}")
