from collider.beam import Beam

if __name__ == "__main__":

    Lx = 2.0
    Ly = 6.0
    dx = 0.25
    dy = 0.25
    Cx = -5.
    Cy = 5.
    angle = 0.

    beam = Beam(Lx, Ly, dx, dy, Cx, Cy, angle)
    beam.create_elements()

    print("--- First Access (Cache is built) ---")
    # This access will trigger _regenerate_cache()
    print(f"beam[0]: {beam[0]}")

    print("\n--- Second Access (Uses cache) ---")
    # This access is now a fast list lookup
    print(f"beam[1]: {beam[1]}")

    print("\n--- Iteration (Uses cache) ---")
    # This also uses the existing cache
    for el in beam:
        pass  # print(el.centroid)
    print("Iteration complete (no rebuild).")

    print("\n--- Changing Beam Angle ---")
    # This will call the @angle.setter, which calls _invalidate_cache()
    beam.angle = 90
    print(f"Beam angle set to {beam.angle}")
    print(f"Direct cache check: {beam._cached_global_elements} ")

    print("\n--- Access After Change (Cache is rebuilt) ---")
    # This access finds the cache is 'None' and triggers _regenerate_cache()
    print(f"beam[0]: {beam[0]}")
    # Local (10, 0) at angle 90, offset (100, 100) -> Global (100, 110)

    print("\n--- Subsequent Access (Uses new cache) ---")
    # This access is fast again
    print(f"beam[1]: {beam[1]}")