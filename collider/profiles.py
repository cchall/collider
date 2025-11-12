import abc
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


def _sigma_t(tau_fwhm):
    return tau_fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))


class Profile(abc.ABC):
    @abc.abstractmethod
    def intensity(self, x, y, t):
        """Return intensity array evaluated at (x,y,t)."""
        pass

    @abc.abstractmethod
    def get_bounds(self, time, local=True):
        """
        Return (minx, maxx, miny, maxy) for a reasonable plotting/sampling window
        around the profile at the given time.
        """
        pass

    def sample_intensity(self, nx, ny, time, local=True):
        """
        Generic sampler that relies only on get_bounds() and intensity().
        Returns (intensity, dx, dy) where dx, dy are domain extents.
        """
        minx, maxx, miny, maxy = self.get_bounds(time, local=local)
        x = np.linspace(minx, maxx, nx)
        y = np.linspace(miny, maxy, ny)
        X, Y = np.meshgrid(x, y)
        return self.intensity(X, Y, time), (maxx - minx), (maxy - miny)


class LaserPulse(Profile):
    # Always moves along x
    def __init__(self, w0: float, tau_fwhm: float, lambda0: float, I0: float, t0: float,
                 n_sigmax=2.25, n_sigmay=2.0):
        self.w0 = w0
        self.tau_fwhm = tau_fwhm
        self.lambda0 = lambda0
        self.I0 = I0
        self.t0 = t0

        self.xR = np.pi * self.w0**2 / self.lambda0
        self.sigma_t = _sigma_t(tau_fwhm)

        self.n_sigmax = n_sigmax
        self.n_sigmay = n_sigmay

    def wx(self, x):
        return self.w0 * np.sqrt(1 + (x / self.xR) ** 2)

    def intensity(self, x, y, t):
        wx = self.wx(x)
        scale = (self.w0 / wx) ** 2
        Iy = np.exp(-2 * (y ** 2) / (wx ** 2))
        It = np.exp(-(((t - self.t0) - x / constants.c) ** 2) / (2 * self.sigma_t ** 2))
        return self.I0 * Iy * It * scale

    def get_bounds(self, time, local=True):
        # Use longitudinal ± n_sigmax σ_t c; transverse from beam radius at the window center
        minx, maxx = -self.sigma_t * self.n_sigmax * constants.c, self.sigma_t * self.n_sigmax * constants.c
        if not local:
            # window centered at waist (x ≈ 0) rather than moving with the pulse
            pass
        else:
            minx += time * constants.c
            maxx += time * constants.c

        _wx = self.wx((maxx + minx) / 2.0)
        miny, maxy = -_wx * self.n_sigmay, _wx * self.n_sigmay
        return minx, maxx, miny, maxy


class ChargedBeam(Profile):
    def __init__(self,
                 sigma_x_rms: float,
                 sigma_y_rms: float,
                 sigma_z_rms: float,
                 emittance: float,
                 alpha0: float,
                 I0: float,
                 t0: float,
                 beta_rel: float = 1.0,
                 plane: str = 'x',
                 n_sigmax: float = 2.25,
                 n_sigmay: float = 2.0):
        self.sigma_x_rms = float(sigma_x_rms)
        self.sigma_y_rms = float(sigma_y_rms)
        self.sigma_z_rms = float(sigma_z_rms)
        self.emittance = float(emittance)
        self.alpha0 = float(alpha0)
        self.I0 = float(I0)
        self.t0 = float(t0)
        self.beta_rel = float(beta_rel)
        self.v = self.beta_rel * constants.c

        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'")
        self.plane = plane

        self.sigma_perp0 = self.sigma_x_rms if self.plane == 'x' else self.sigma_y_rms

        if self.emittance <= 0.0:
            raise ValueError("emittance must be positive")
        self.beta0 = (self.sigma_perp0 ** 2) / self.emittance
        self.gamma0 = (1.0 + self.alpha0 ** 2) / self.beta0

        if self.v <= 0.0:
            raise ValueError("beta_rel must be positive")
        self.sigma_t = self.sigma_z_rms / self.v

        self.n_sigmax = n_sigmax
        self.n_sigmay = n_sigmay

    def beta(self, x):
        return self.beta0 - 2.0 * self.alpha0 * x + self.gamma0 * x**2

    def sigma_perp(self, x):
        b = np.maximum(self.beta(x), 0.0)
        return np.sqrt(self.emittance * b)

    def intensity(self, x, y, t):
        sig_perp = self.sigma_perp(x)
        scale = (self.sigma_perp0 / np.where(sig_perp == 0, np.inf, sig_perp))
        Iy = np.exp(-0.5 * (y / np.where(sig_perp == 0, np.inf, sig_perp))**2)
        It = np.exp(-0.5 * (((t - self.t0) - x / self.v) / self.sigma_t) ** 2)
        return self.I0 * scale * Iy * It

    def get_bounds(self, time, local=True):
        half_span_x = self.sigma_t * self.n_sigmax * self.v  # = n_sigmax * σ_z
        if local:
            xc = time * self.v
            minx, maxx = xc - half_span_x, xc + half_span_x
        else:
            minx, maxx = -half_span_x, half_span_x

        x_center = 0.5 * (minx + maxx)
        sigma_here = self.sigma_perp(x_center)
        miny, maxy = -self.n_sigmay * sigma_here, self.n_sigmay * sigma_here
        return minx, maxx, miny, maxy


def plot_profile(profile: Profile, time, nx, ny, *, local=True, save_path=None, show=True):
    """
    Generic plotter for any Profile. Relies only on get_bounds() and intensity().
    """
    minx, maxx, miny, maxy = profile.get_bounds(time, local=local)
    x = np.linspace(minx, maxx, nx)
    y = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(x, y)
    intensity = profile.intensity(X, Y, time)

    plt.figure(figsize=(7.8, 5.6))
    extent = [x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6]
    plt.imshow(intensity, origin='lower', aspect='auto', extent=extent)
    plt.colorbar(label='Normalized intensity (arb. units)')
    plt.xlabel('x (µm)'); plt.ylabel('y (µm)')
    plt.title('I(x, y) at fixed t = {:.2f} fs'.format(time*1e15))
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show: plt.show()
    plt.close()




if __name__ == '__main__':
    pulse = LaserPulse(w0=10e-6, tau_fwhm=250e-15, lambda0=1080e-9, I0=1.0, t0=0)

    plot_profile(profile=pulse, time=0 * 950e-15, nx=700, ny=700, local=True)
    # Example: ultrarelativistic beam slice in the horizontal plane
    beam = ChargedBeam(
        sigma_x_rms=50e-6,   # 50 µm horizontal rms
        sigma_y_rms=30e-6,   # 30 µm vertical rms (unused here with plane='x')
        sigma_z_rms=100e-6,  # 100 µm bunch length
        emittance=1e-6,      # 1 mm·mrad geometric -> 1e-6 m·rad
        alpha0=0.0,          # at waist plane
        I0=1.0,
        t0=0.0,
        beta_rel=1.0,
        plane='x'
    )

    # Reuse your plotting util by passing the new object
    plot_profile(profile=beam, time=0, nx=700, ny=700, local=True)
