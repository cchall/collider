import math
import pathlib

from collider.beam import Beam, SetupInfo
from collider.propagate import propagate
from collider import actions
from typing import List, Tuple
from itertools import combinations


def simulation_step(beams: List[Beam], time_step: float,
                    check_overlap: bool=True, save_directory: pathlib.Path = None) -> bool:

    for beam in beams:
        if save_directory is not None:
            beam.serialize(save_directory.joinpath(beam.name))

        _ = propagate(beam=beam, time_step=time_step)


    any_proximal = False
    if check_overlap:
        for b1, b2 in combinations(beams, 2):
            proximal = actions.check_beam_proximity(b1, b2)
            any_proximal = any_proximal or proximal
            if proximal:
                actions.update_beam_overlap(b1, b2)

    return any_proximal


def run_simulation(beam1, beam2, time_step: float, save_directory: pathlib.Path = None) -> None:
    _, _, t_c = set_initial_state(beam1, beam2)
    n_steps = int(2*t_c // time_step) + 1

    for _ in range(n_steps):
        proximal = simulation_step(beams=[beam1, beam2], time_step=time_step, save_directory=save_directory)

    return


def set_initial_state(beam1: SetupInfo, beam2: SetupInfo) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Calculates initial center positions (Cx, Cy) for two beams and their
    collision time (t_c) in a *rotated* coordinate frame.

    The frame is rotated so that beam1's velocity vector (vx1, vy1)
    is aligned with the positive x-axis.

    This function correctly calculates the required separation based on the
    axis-aligned bounding boxes (AABBs) of the beams in the new rotated frame.

    Assumption: beam.Lx is aligned with its velocity (beam.vx, beam.vy).
    """

    # --- 1. Calculate Rotated Velocities ---

    v1_speed = math.sqrt(beam1.vx ** 2 + beam1.vy ** 2)
    EPSILON = 1e-12

    if v1_speed < EPSILON:
        raise ValueError("Beam 1 velocity is zero, cannot align to x-axis.")

    # Rotation angle `a` is -beam1.angle
    # cos(a) = cos(-beam1.angle) = cos(beam1.angle) = beam1.vx / v1_speed
    # sin(a) = sin(-beam1.angle) = -sin(beam1.angle) = -beam1.vy / v1_speed

    cos_a = beam1.vx / v1_speed
    sin_a = -beam1.vy / v1_speed

    # Rotated velocity for beam 1 (by definition)
    v1_rot_x = v1_speed
    v1_rot_y = 0.0

    # Apply the same rotation to beam 2's velocity
    v2_rot_x = beam2.vx * cos_a - beam2.vy * sin_a
    v2_rot_y = beam2.vx * sin_a + beam2.vy * cos_a

    # --- 2. Calculate Required Separation (The Fix) ---

    # In the new frame, beam1's orientation is 0 degrees.
    # We need to find beam2's new orientation.
    # The rotation angle we applied to the system was -beam1.angle.
    # beam2's new angle = beam2.angle + (-beam1.angle)
    theta_2_rot = math.radians(beam2.angle - beam1.angle)

    # Calculate the AABB of the rotated beam 2
    c2 = abs(math.cos(theta_2_rot))
    s2 = abs(math.sin(theta_2_rot))

    beam2_width_aabb = beam2.Lx * c2 + beam2.Ly * s2
    beam2_height_aabb = beam2.Lx * s2 + beam2.Ly * c2

    # The minimum center-to-center separation to prevent overlap
    # is half the sum of their AABB dimensions in this frame.
    # beam1's AABB is (beam1.Lx, beam1.Ly) in this frame.
    min_sep_x = (beam1.Lx + beam2_width_aabb) / 2.0
    min_sep_y = (beam1.Ly + beam2_height_aabb) / 2.0

    # --- 3. Calculate Collision Time (t_c) ---

    # Relative velocity in the rotated frame
    rel_vx = v1_rot_x - v2_rot_x
    rel_vy = v1_rot_y - v2_rot_y  # This is just -v2_rot_y

    if abs(rel_vx) < EPSILON and abs(rel_vy) < EPSILON:
        raise ValueError("Beams have zero relative velocity. No collision will occur.")

    # Initial separation (delta_C) is -v_rel * t_c
    # delta_Cx = -rel_vx * t_c
    # delta_Cy = -rel_vy * t_c

    # We need the non-overlap condition at t=0:
    # abs(delta_Cx) > min_sep_x  OR  abs(delta_Cy) > min_sep_y

    # Substitute delta_C equations (and t_c > 0):
    # t_c * abs(rel_vx) > min_sep_x   OR  t_c * abs(rel_vy) > min_sep_y

    # This means:
    # t_c > (min_sep_x / abs(rel_vx))  (let's call this t_x)
    # OR
    # t_c > (min_sep_y / abs(rel_vy))  (let's call this t_y)

    # Calculate t_x and t_y, handling division by zero
    t_x = float('inf')
    if abs(rel_vx) > EPSILON:
        t_x = min_sep_x / abs(rel_vx)

    t_y = float('inf')
    if abs(rel_vy) > EPSILON:
        t_y = min_sep_y / abs(rel_vy)

    # t_sep is the critical time when they *just* stop overlapping
    t_sep = min(t_x, t_y)

    if t_sep == float('inf'):
        # This happens if e.g. rel_vx is 0 and rel_vy is 0
        # (already caught) or if min_sep_x/y is 0 and rel_v is 0.
        # Essentially, they are on a non-colliding path from the start.
        # Or, more likely, they are moving parallel and will never separate.
        # We can't satisfy the "will intersect" condition.
        raise ValueError("Beams move parallel and will never intersect (or are always intersecting).")

    # Add a small padding to ensure they start separated
    # This padding value can be tuned.
    PADDING = 0.1
    t_c = t_sep + PADDING

    # --- 4. Calculate and Return Initial Positions ---

    # Use the t_c we found to set the initial positions
    # C = -v_rot * t_c
    Cx1 = -v1_rot_x * t_c
    Cy1 = -v1_rot_y * t_c  # This is 0.0

    Cx2 = -v2_rot_x * t_c
    Cy2 = -v2_rot_y * t_c

    return ((Cx1, Cy1), (Cx2, Cy2), t_c)




