import abc
import numpy as np

from scipy import constants


def _sigma_t(tau_fwhm):
    return tau_fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))


class Profile(abc.ABC):
    @abc.abstractmethod
    def intensity(self, x, y, t):
        pass
    @abc.abstractmethod
    def sample_intensity(self, nx, ny, t):
        pass

class LaserPulse(Profile):
    # Always moves along x
    def __init__(self, w0: float, tau_fwhm: float, lambda0: float, I0: float, t0: float,
                 n_sigmax=2.25, n_sigmay=2.0
                 ):
        self.w0 = w0
        self.tau_fwhm = tau_fwhm
        self.lambda0 = lambda0
        self.I0 = I0
        self.t0 = t0

        self.xR = np.pi * self.w0**2 / self.lambda0
        self.sigma_t = _sigma_t(tau_fwhm)

        # Set the size of bounding box for sampling in terms of sigma_t and w0 in x and y respectively
        self.n_sigmax = n_sigmax
        self.n_sigmay = n_sigmay

    def wx(self, x):
        return  self.w0 * np.sqrt(1 + (x / self.xR) ** 2)

    def intensity(self, x, y, t):
        wx = self.w0 * np.sqrt(1 + (x / self.xR) ** 2)
        scale = (self.w0 / wx) ** 2
        Iy = np.exp(-2 * (y ** 2) / (wx ** 2))
        It = np.exp(-(((t - self.t0) - x / constants.c) ** 2) / (2 * self.sigma_t ** 2))
        I = self.I0 * Iy * It * scale

        return I

    def get_bounds(self, time, local=True):
        minx, maxx = -self.sigma_t * self.n_sigmax * constants.c, self.sigma_t * self.n_sigmax * constants.c
        if not local:
            minx += time * constants.c
            maxx += time * constants.c

        # Pulse width changes so set the bounding box size in x based on center of the pulse
        _wx = self.wx((maxx + minx) / 2.)
        miny, maxy = -_wx * self.n_sigmay, _wx * self.n_sigmay

        return minx, maxx, miny, maxy

    def sample_intensity(self, nx, ny, time):
        minx, maxx = -self.sigma_t * self.n_sigmax * constants.c, self.sigma_t * self.n_sigmax * constants.c
        minx += time * constants.c
        maxx += time * constants.c

        # Pulse width changes so set the bounding box size in x based on center of the pulse
        _wx = self.wx((maxx + minx) / 2.)
        miny, maxy = -_wx * self.n_sigmay, _wx * self.n_sigmay

        x = np.linspace(minx, maxx, nx)
        y = np.linspace(miny, maxy, ny)
        X, Y = np.meshgrid(x, y)

        intensity = self.intensity(X, Y, time)

        return intensity, (maxx - minx), (maxy - miny)

def plot_pulse(pulse, time, nx, ny, n_sigmax=2.25, n_sigmay=2.0, save_path=None, show=True, local=False):
    import matplotlib.pyplot as plt

    # local bases the grid around the center of pulse at t
    # if local is False the grid is based around the position of the waist
    if local:
        minx, maxx = -pulse.sigma_t * n_sigmax * constants.c, pulse.sigma_t * n_sigmax * constants.c
        minx += time * constants.c
        maxx += time * constants.c
    else:
        minx, maxx = -pulse.xR * n_sigmax, pulse.xR * n_sigmax
    miny, maxy = -pulse.w0 * n_sigmay, pulse.w0 * n_sigmay
    x = np.linspace(minx, maxx, nx)
    y = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(x, y)
    intensity = pulse.intensity(X, Y, time)

    plt.figure(figsize=(7.8,5.6))
    extent = [x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6]
    plt.imshow(intensity, origin='lower', aspect='auto', extent=extent)
    plt.colorbar(label='Normalized intensity (arb. units)')
    plt.xlabel('x (µm)'); plt.ylabel('y (µm)')
    plt.title('I(x, y) at fixed t = {:.2f} fs (zX = {:.3g} mm)'.format(time*1e15, pulse.xR*1e3))
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show: plt.show()
    plt.close()


if __name__ == '__main__':
    pulse = LaserPulse(w0=10e-6, tau_fwhm=250e-15, lambda0=1080e-9, I0=1.0, t0=0)

    plot_pulse(pulse=pulse, time=0*950e-15, nx=700, ny=700, local=True)