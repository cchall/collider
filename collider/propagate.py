import numpy as np

from collider.beam import Beam


def propagate(beam: Beam, time_step: float) -> Beam:
    beam.update_position(time_step)

    return beam

def lorentz_transform(four_vector: np.ndarray, beta_vector: np.ndarray, gamma: float) -> np.ndarray:
    v0, v = four_vector[0], four_vector[1:]

    beta_dot_v = np.dot(beta_vector, v)
    v0_p = gamma * (v0 - beta_dot_v)

    v_p = v + beta_vector * (gamma ** 2 * beta_dot_v / (gamma + 1) - gamma * v0)

    return np.array([v0_p, *v_p])
