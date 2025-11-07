from collider.beam import Beam


def propagate(beam: Beam, time_step: float) -> Beam:
    beam.update_position(time_step)

    return beam
