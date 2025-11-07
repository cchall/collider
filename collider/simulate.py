import math
import pathlib

from collider.beam import Beam, SetupInfo
from collider.propagate import propagate
from collider import actions
from typing import List, Tuple
from itertools import combinations


def simulation_step(beams: List[Beam], time_step: float, save_directory: pathlib.Path = None) -> bool:

    for beam in beams:
        if save_directory is not None:
            beam.serialize(save_directory.joinpath(beam.name))

        _ = propagate(beam=beam, time_step=time_step)

    any_proximal = False
    for b1, b2 in combinations(beams, 2):
        proximal = actions.check_beam_proximity(b1, b2)
        any_proximal = any_proximal or proximal
        if proximal:
            actions.update_beam_overlap(b1, b2)

    return any_proximal


def set_initial_state(beam1: SetupInfo, beam2: SetupInfo) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Calculates initial center positions (Cx, Cy) for two beams and their
    collision time (t_c) in a *rotated* coordinate frame.

    The frame is rotated so that beam1's velocity vector (vx1, vy1)
    is aligned with the positive x-axis [1, 0].

    The logic finds the minimum collision time `t_c` such that the beams,
    when traced backward from a center-on-center collision at (0,0) at
    time `t_c`, are *not* overlapping at t=0 (plus a small padding).

    Args:
        beam1: SetupInfo for the first beam.
        beam2: SetupInfo for the second beam.

    Returns:
        A tuple containing:
        - (Cx1, Cy1): Initial center coordinates for beam 1 in the rotated frame.
        - (Cx2, Cy2): Initial center coordinates for beam 2 in the rotated frame.
        - t_c: The time of collision for the beam centers.
    """

    # --- 1. Calculate Rotated Velocities ---

    # Get the speed and angle of beam1
    v1_speed = math.sqrt(beam1.vx ** 2 + beam1.vy ** 2)

    # Define a small epsilon for float comparisons
    EPSILON = 1e-12

    if v1_speed < EPSILON:
        raise ValueError("Beam 1 velocity is zero, cannot align to x-axis.")

    # We need to rotate the system by -beam1.angle.
    # The rotation matrix for an angle `a` is:
    # [ cos(a) -sin(a) ]
    # [ sin(a)  cos(a) ]
    #
    # Our angle `a` is -beam1.angle.
    # cos(a) = cos(-beam1.angle) = cos(beam1.angle) = beam1.vx / v1_speed
    # sin(a) = sin(-beam1.angle) = -sin(beam1.angle) = -beam1.vy / v1_speed

    cos_a = beam1.vx / v1_speed
    sin_a = -beam1.vy / v1_speed

    # By definition, beam1's new velocity is (v1_speed, 0)
    v1_rot_x = v1_speed
    v1_rot_y = 0.0  # This should be zero within float precision

    # Apply the same rotation to beam2's velocity vector
    v2_rot_x = beam2.vx * cos_a - beam2.vy * sin_a
    v2_rot_y = beam2.vx * sin_a + beam2.vy * cos_a

    # --- 2. Calculate Collision Time (t_c) ---

    # We want the centers to collide at (0, 0) at time t_c.
    # P1(t) = C1 + v1_rot * t  => P1(t_c) = (Cx1 + v1_rot_x*t_c, Cy1 + v1_rot_y*t_c) = (0, 0)
    # P2(t) = C2 + v2_rot * t  => P2(t_c) = (Cx2 + v2_rot_x*t_c, Cy2 + v2_rot_y*t_c) = (0, 0)

    # This gives us the initial positions C1, C2 in terms of t_c:
    # Cx1 = -v1_rot_x * t_c
    # Cy1 = -v1_rot_y * t_c = 0
    # Cx2 = -v2_rot_x * t_c
    # Cy2 = -v2_rot_y * t_c

    # The initial separation (at t=0) is delta_C = C1 - C2
    # delta_Cx = Cx1 - Cx2 = (-v1_rot_x + v2_rot_x) * t_c = -(v1_rot_x - v2_rot_x) * t_c
    # delta_Cy = Cy1 - Cy2 = (-v1_rot_y + v2_rot_y) * t_c = -(0 - v2_rot_y) * t_c = v2_rot_y * t_c

    # Let v_rel = v1_rot - v2_rot
    rel_vx = v1_rot_x - v2_rot_x
    rel_vy = v1_rot_y - v2_rot_y  # This is just -v2_rot_y

    if abs(rel_vx) < EPSILON and abs(rel_vy) < EPSILON:
        raise ValueError("Beams have zero relative velocity. No collision will occur.")

    # The beams (as AABBs) do *not* overlap at t=0 if:
    # abs(delta_Cx) > (beam1.Lx + beam2.Lx) / 2  OR
    # abs(delta_Cy) > (beam1.Ly + beam2.Ly) / 2

    min_sep_x = (beam1.Lx + beam2.Lx) / 2.0
    min_sep_y = (beam1.Ly + beam2.Ly) / 2.0

    # Substitute delta_C equations (and t_c > 0):
    # t_c * abs(-rel_vx) > min_sep_x  OR  t_c * abs(-rel_vy) > min_sep_y
    # t_c * abs(rel_vx) > min_sep_x   OR  t_c * abs(rel_vy) > min_sep_y

    # This is equivalent to:
    # t_c > min_sep_x / abs(rel_vx)  (let's call this t_x) OR
    # t_c > min_sep_y / abs(rel_vy)  (let's call this t_y)

    # So, no overlap if t_c > min(t_x, t_y)

    # Calculate t_x and t_y, handling division by zero
    t_x = float('inf')
    if abs(rel_vx) > EPSILON:
        t_x = min_sep_x / abs(rel_vx)

    t_y = float('inf')
    if abs(rel_vy) > EPSILON:
        t_y = min_sep_y / abs(rel_vy)

    # t_sep is the critical time when they *just* stop overlapping
    t_sep = min(t_x, t_y)

    # Add a small padding to ensure they start separated
    # This padding value can be tuned.
    PADDING = 0.1
    t_c = t_sep + PADDING

    # --- 3. Calculate and Return Initial Positions ---

    # Use the t_c we found to set the initial positions
    Cx1 = -v1_rot_x * t_c
    Cy1 = 0.0  # Since v1_rot_y is 0

    Cx2 = -v2_rot_x * t_c
    Cy2 = -v2_rot_y * t_c

    return ((Cx1, Cy1), (Cx2, Cy2), t_c)



