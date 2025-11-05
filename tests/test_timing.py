"""
Timing test for update_beam_overlap function.
Compares performance between:
1. Fast check cases (elements clearly overlapping or clearly not overlapping)
2. Cases requiring overlap_shadows (ambiguous zone requiring full shadow projection check)
"""

import time
import math
from collider.beam import Beam
from collider.actions import update_beam_overlap


def create_far_apart_beams(n_elements_per_side=10):
    """Create beams where elements are far apart - fast check will skip most pairs."""
    # Create two beams that are far apart
    dx, dy = 0.5, 0.5
    Lx, Ly = n_elements_per_side * dx, n_elements_per_side * dy

    # Beam 1: far left
    beam1 = Beam(Lx=Lx, Ly=Ly, dx=dx, dy=dy, Cx=-10.0, Cy=0.0, angle=0.0)
    beam1.create_elements()

    # Beam 2: far right
    beam2 = Beam(Lx=Lx, Ly=Ly, dx=dx, dy=dy, Cx=10.0, Cy=0.0, angle=0.0)
    beam2.create_elements()

    return beam1, beam2


def create_close_overlapping_beams(n_elements_per_side=10):
    """Create beams where elements are very close - fast check will detect overlap immediately."""
    dx, dy = 0.5, 0.5
    Lx, Ly = n_elements_per_side * dx, n_elements_per_side * dy

    # Beam 1: centered at origin
    beam1 = Beam(Lx=Lx, Ly=Ly, dx=dx, dy=dy, Cx=0.0, Cy=0.0, angle=0.0)
    beam1.create_elements()

    # Beam 2: slightly offset, overlapping
    beam2 = Beam(Lx=Lx, Ly=Ly, dx=dx, dy=dy, Cx=0.1, Cy=0.1, angle=0.0)
    beam2.create_elements()

    return beam1, beam2


def create_ambiguous_zone_beams(n_elements_per_side=10):
    """Create beams where elements are in the ambiguous zone - requires overlap_shadows check.

    For square elements (0.5x0.5), smallest_edge = largest_edge = 0.25.
    When rotated, the effective bounding box is larger, but the fast check still uses
    the original wx/wy. Positioning beams with moderate separation and rotation
    should create many cases in the ambiguous zone.
    """
    # Use rectangular elements to create a larger ambiguous zone
    # For rectangular elements, smallest_edge != largest_edge
    dx, dy = 1.0, 0.3  # Rectangular elements
    Lx, Ly = n_elements_per_side * dx, n_elements_per_side * dy

    # Beam 1: centered at origin, rotated
    beam1 = Beam(Lx=Lx, Ly=Ly, dx=dx, dy=dy, Cx=0.0, Cy=0.0, angle=30.0)
    beam1.create_elements()

    # Beam 2: positioned so many pairs fall in the ambiguous zone
    # Position so that center-to-center distances fall between thresholds
    beam2 = Beam(Lx=Lx, Ly=Ly, dx=dx, dy=dy, Cx=0.8, Cy=0.8, angle=-30.0)
    beam2.create_elements()

    return beam1, beam2


def time_function(func, *args, **kwargs):
    """Time a function call and return the elapsed time in seconds."""
    start = time.perf_counter()
    func(*args, **kwargs)
    end = time.perf_counter()
    return end - start


def count_interactions(beam1, beam2):
    """Count total interactions after update_beam_overlap."""
    total1 = sum(e.interactions for e in beam1._elements)
    total2 = sum(e.interactions for e in beam2._elements)
    return total1, total2


def reset_interactions(beam1, beam2):
    """Reset interaction counters."""
    for e in beam1._elements:
        e.interactions = 0
    for e in beam2._elements:
        e.interactions = 0


def count_check_types(beam1, beam2):
    """Count how many times each check type would be used (for analysis)."""
    largest_edge1 = max(beam1._elements[0].wx, beam1._elements[0].wy) / 2.
    smallest_edge1 = min(beam1._elements[0].wx, beam1._elements[0].wy) / 2.
    largest_edge2 = max(beam2._elements[0].wx, beam2._elements[0].wy) / 2.
    smallest_edge2 = min(beam2._elements[0].wx, beam2._elements[0].wy) / 2.

    skipped = 0
    fast_overlap = 0
    ambiguous = 0

    for e1 in beam1:
        for e2 in beam2:
            c2c_dist = math.sqrt((e1.cx - e2.cx) ** 2 + (e1.cy - e2.cy) ** 2)
            if c2c_dist > (largest_edge1 + largest_edge2):
                skipped += 1
            elif c2c_dist < (smallest_edge1 + smallest_edge2):
                fast_overlap += 1
            else:
                ambiguous += 1

    return skipped, fast_overlap, ambiguous


def run_timing_test():
    """Run comprehensive timing tests."""
    print("=" * 70)
    print("Timing Test for update_beam_overlap")
    print("=" * 70)

    # Test different sizes
    sizes = [10, 20, 30]

    results = []

    for n in sizes:
        print(f"\n{'=' * 70}")
        print(f"Testing with {n}x{n} elements per beam ({n * n} elements per beam)")
        print(f"{'=' * 70}")

        # Test 1: Far apart beams (fast check - mostly skips)
        print("\n1. Far Apart Beams (Fast check - mostly skips)")
        beam1, beam2 = create_far_apart_beams(n)
        reset_interactions(beam1, beam2)
        skipped, fast, ambig = count_check_types(beam1, beam2)
        print(f"   Check distribution: skipped={skipped}, fast_overlap={fast}, ambiguous={ambig}")
        time_far = time_function(update_beam_overlap, beam1, beam2)
        total1, total2 = count_interactions(beam1, beam2)
        print(f"   Time: {time_far * 1000:.3f} ms")
        print(f"   Total interactions: beam1={total1}, beam2={total2}")

        # Test 2: Close overlapping beams (fast check - immediate overlap detection)
        print("\n2. Close Overlapping Beams (Fast check - immediate overlap detection)")
        beam1, beam2 = create_close_overlapping_beams(n)
        reset_interactions(beam1, beam2)
        skipped, fast, ambig = count_check_types(beam1, beam2)
        print(f"   Check distribution: skipped={skipped}, fast_overlap={fast}, ambiguous={ambig}")
        time_close = time_function(update_beam_overlap, beam1, beam2)
        total1, total2 = count_interactions(beam1, beam2)
        print(f"   Time: {time_close * 1000:.3f} ms")
        print(f"   Total interactions: beam1={total1}, beam2={total2}")

        # Test 3: Ambiguous zone beams (requires overlap_shadows)
        print("\n3. Ambiguous Zone Beams (Requires overlap_shadows check)")
        beam1, beam2 = create_ambiguous_zone_beams(n)
        reset_interactions(beam1, beam2)
        skipped, fast, ambig = count_check_types(beam1, beam2)
        print(f"   Check distribution: skipped={skipped}, fast_overlap={fast}, ambiguous={ambig}")
        time_ambiguous = time_function(update_beam_overlap, beam1, beam2)
        total1, total2 = count_interactions(beam1, beam2)
        print(f"   Time: {time_ambiguous * 1000:.3f} ms")
        print(f"   Total interactions: beam1={total1}, beam2={total2}")

        # Comparison
        print("\n4. Comparison")
        if time_close > 0:
            print(f"   Far apart / Close overlapping ratio: {time_far / time_close:.2f}x")
        if time_far > 0:
            print(f"   Ambiguous / Far apart ratio: {time_ambiguous / time_far:.2f}x")
        if time_close > 0:
            print(f"   Ambiguous / Close overlapping ratio: {time_ambiguous / time_close:.2f}x")

        results.append({
            'size': n,
            'n_elements': n * n,
            'time_far': time_far,
            'time_close': time_close,
            'time_ambiguous': time_ambiguous,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"{'Size':<10} {'Elements':<10} {'Far (ms)':<12} {'Close (ms)':<12} {'Ambiguous (ms)':<15} {'Ratio':<10}")
    print("-" * 70)
    for r in results:
        ratio = r['time_ambiguous'] / r['time_close']
        print(f"{r['size']:<10} {r['n_elements']:<10} {r['time_far'] * 1000:<12.3f} "
              f"{r['time_close'] * 1000:<12.3f} {r['time_ambiguous'] * 1000:<15.3f} {ratio:<10.2f}x")

    print(f"\n{'=' * 70}")
    print("Key Observations:")
    print("- Fast checks (far apart or close overlapping) should be much faster")
    print("- Ambiguous zone checks (requiring overlap_shadows) should be slower")
    print("- The ratio shows the performance impact of shadow projection checks")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_timing_test()