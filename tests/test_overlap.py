import sys
sys.path.append("/Users/chall/research/collider/")

import time
import random
from typing import Tuple
from collider.actions import overlap_shadows
from collider.element import Element

VERBOSE = False

# If random floats in [-0.5, 0.5] match your Rust code:
def rand_float():
    return random.random() - 0.5


def overlap_timing_test(n_pairs: int):
    # ---- Generate the element sets ----
    element_set_1 = [
        Element(
            center=(rand_float(), rand_float()),
            width=(rand_float(), rand_float()),
            density=0.0,
            interactions=0,
            flux=0.0,
        )
        for _ in range(n_pairs)
    ]

    element_set_2 = [
        Element(
            center=(rand_float(), rand_float()),
            width=(rand_float(), rand_float()),
            density=0.0,
            interactions=0,
            flux=0.0,
        )
        for _ in range(n_pairs)
    ]

    # ---- Timing ----
    start = time.perf_counter()

    for i in range(n_pairs):
        if VERBOSE:
            print(f"Running pair {i}")
        overlap_result = overlap_shadows(element_set_1[i], element_set_2[i], 0.0, 0.0, 1e-8)
        if VERBOSE:
            print(f"\tOverlapping: {overlap_result}")

    elapsed = time.perf_counter() - start
    per_call = elapsed / n_pairs

    # ---- Report ----
    print("overlap_shadows benchmark")
    print(f"pairs:        {n_pairs}")
    print(f"total time:   {elapsed:.6f} s")
    print(f"per call:     {per_call:.3e} s ({per_call * 1e9:.1f} ns)")

if __name__ == "__main__":
    overlap_timing_test(n_pairs=4_000_000)
