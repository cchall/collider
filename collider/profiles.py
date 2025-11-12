import abc
from typing import Tuple, Optional
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import constants
import matplotlib.pyplot as plt


def _sigma_t(tau_fwhm: float) -> float:
    """Convert FWHM pulse duration to the Gaussian rms width in time.

    Parameters
    ----------
    tau_fwhm : float
        Full width at half maximum (seconds).

    Returns
    -------
    float
        Temporal rms width σ_t (seconds).
    """
    return tau_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


class Profile(abc.ABC):
    """Abstract base class for 2D profiles propagating along x and sampled in (x, y, t)."""

    @abc.abstractmethod
    def intensity(self, x: ArrayLike, y: ArrayLike, t: ArrayLike) -> NDArray[np.float64]:
        """Return intensity evaluated at (x, y, t).

        Implementations should support broadcasting over `x`, `y`, and `t` as
        standard NumPy arrays.

        Parameters
        ----------
        x : ArrayLike
            Longitudinal coordinate(s) (meters).
        y : ArrayLike
            Transverse coordinate(s) (meters).
        t : ArrayLike
            Time(s) (seconds).

        Returns
        -------
        numpy.ndarray
            Intensity array with shape broadcast from (x, y, t).
        """
        pass

    @abc.abstractmethod
    def get_bounds(self, time: float, local: bool = True) -> Tuple[float, float, float, float]:
        """Return a reasonable sampling window around the profile at `time`.

        Parameters
        ----------
        time : float
            Time at which to define the sampling window (seconds).
        local : bool, optional
            If True, center the window on the profile's instantaneous position.
            If False, center on a fixed reference (e.g., waist at x=0).

        Returns
        -------
        (float, float, float, float)
            Tuple (minx, maxx, miny, maxy) in meters.
        """
        pass

    def sample_intensity(
        self, nx: int, ny: int, time: float, local: bool = True
    ) -> Tuple[NDArray[np.float64], float, float]:
        """Sample intensity on a regular (x, y) grid at a fixed `time`.

        Relies only on :meth:`get_bounds` and :meth:`intensity`.

        Parameters
        ----------
        nx : int
            Number of grid points along x.
        ny : int
            Number of grid points along y.
        time : float
            Fixed time at which to evaluate intensity (seconds).
        local : bool, optional
            Passed through to :meth:`get_bounds`.

        Returns
        -------
        (numpy.ndarray, float, float)
            Tuple (intensity, dx, dy), where `intensity` is ny×nx and
            `dx`, `dy` are the extents (max - min) along x and y in meters.
        """
        minx, maxx, miny, maxy = self.get_bounds(time, local=local)
        x = np.linspace(minx, maxx, nx)
        y = np.linspace(miny, maxy, ny)
        X, Y = np.meshgrid(x, y)
        return self.intensity(X, Y, time), (maxx - minx), (maxy - miny)


class LaserPulse(Profile):
    """Gaussian laser pulse propagating along +x with Gaussian time envelope and
    diffraction-limited transverse evolution (paraxial Gaussian beam).

    Notes
    -----
    - Beam radius evolves as w(x) = w0 * sqrt(1 + (x/xR)^2).
    - Intensity is separable in y and t, moving at speed c along x.
    """

    # Always moves along x
    def __init__(
        self,
        w0: float,
        tau_fwhm: float,
        lambda0: float,
        I0: float,
        t0: float,
        n_sigmax: float = 2.25,
        n_sigmay: float = 2.0,
    ) -> None:
        """
        Parameters
        ----------
        w0 : float
            Waist radius at focus (meters).
        tau_fwhm : float
            Temporal FWHM duration (seconds).
        lambda0 : float
            Central wavelength (meters).
        I0 : float
            Peak intensity scale (arbitrary units).
        t0 : float
            Reference time at which the pulse center is at x=0 (seconds).
        n_sigmax : float, optional
            Half-width of the sampling window along x in units of σ_t c.
        n_sigmay : float, optional
            Half-width of the sampling window along y in units of w(x).
        """
        self.w0 = w0
        self.tau_fwhm = tau_fwhm
        self.lambda0 = lambda0
        self.I0 = I0
        self.t0 = t0

        self.xR = np.pi * self.w0**2 / self.lambda0
        self.sigma_t = _sigma_t(tau_fwhm)

        self.n_sigmax = n_sigmax
        self.n_sigmay = n_sigmay

    def wx(self, x: ArrayLike) -> NDArray[np.float64]:
        """Rayleigh-evolving beam radius w(x) at longitudinal position(s) `x`.

        Parameters
        ----------
        x : ArrayLike
            Longitudinal coordinate(s) (meters).

        Returns
        -------
        numpy.ndarray
            Beam radius w(x) (meters), broadcast to the shape of `x`.
        """
        return self.w0 * np.sqrt(1 + (np.asarray(x) / self.xR) ** 2)

    def intensity(self, x: ArrayLike, y: ArrayLike, t: ArrayLike) -> NDArray[np.float64]:
        """Gaussian intensity I(x, y, t) with diffraction and temporal envelope.

        Returns
        -------
        numpy.ndarray
            Intensity array broadcast over `x`, `y`, and `t`.
        """
        wx = self.wx(x)
        scale = (self.w0 / wx) ** 2
        Iy = np.exp(-2.0 * (np.asarray(y) ** 2) / (wx ** 2))
        It = np.exp(-(((np.asarray(t) - self.t0) - np.asarray(x) / constants.c) ** 2) / (2.0 * self.sigma_t ** 2))
        return self.I0 * Iy * It * scale

    def get_bounds(self, time: float, local: bool = True) -> Tuple[float, float, float, float]:
        """Sampling bounds based on ± n_sigmax σ_t c along x and ± n_sigmay w(x_c) along y.

        Parameters
        ----------
        time : float
            Time (seconds) at which to center the local window if `local` is True.
        local : bool, optional
            If True, center at x = c * time; if False, center at x = 0.

        Returns
        -------
        (float, float, float, float)
            (minx, maxx, miny, maxy) in meters.
        """
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
        return float(minx), float(maxx), float(miny), float(maxy)


class ChargedBeam(Profile):
    """Single-plane Gaussian charged-particle beam slice propagating along +x.

    The transverse envelope evolves via Twiss parameters in a drift:
    β(x) = β0 − 2 α0 x + γ0 x², with γ0 = (1 + α0²)/β0 and σ⊥(x) = √(ε β(x)).
    The longitudinal envelope moves at v = β_rel c with σ_t = σ_z / v.
    """

    def __init__(
        self,
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
        n_sigmay: float = 2.0,
    ) -> None:
        """
        Parameters
        ----------
        sigma_x_rms : float
            Initial horizontal rms size (meters).
        sigma_y_rms : float
            Initial vertical rms size (meters).
        sigma_z_rms : float
            Initial rms bunch length (meters).
        emittance : float
            Geometric rms emittance (m·rad) for the chosen plane.
        alpha0 : float
            Initial Twiss α in the chosen plane.
        I0 : float
            Peak intensity scale (arbitrary units).
        t0 : float
            Reference time when the bunch center is at x=0 (seconds).
        beta_rel : float, optional
            v/c of the beam (dimensionless), default 1.0.
        plane : {'x', 'y'}, optional
            Which transverse plane to model as `y` in this 2D slice.
        n_sigmax : float, optional
            Half-width of the sampling window along x in units of σ_z.
        n_sigmay : float, optional
            Half-width of the sampling window along y in units of σ⊥.
        """
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

    def beta(self, x: ArrayLike) -> NDArray[np.float64]:
        """Twiss β(x) in a drift for position(s) `x` (meters)."""
        return self.beta0 - 2.0 * self.alpha0 * np.asarray(x) + self.gamma0 * np.asarray(x) ** 2

    def sigma_perp(self, x: ArrayLike) -> NDArray[np.float64]:
        """Transverse rms size σ⊥(x) = √(ε β(x)) at position(s) `x` (meters)."""
        b = np.maximum(self.beta(x), 0.0)
        return np.sqrt(self.emittance * b)

    def intensity(self, x: ArrayLike, y: ArrayLike, t: ArrayLike) -> NDArray[np.float64]:
        """Gaussian slice intensity with evolving transverse envelope and moving longitudinal center."""
        sig_perp = self.sigma_perp(x)
        scale = (self.sigma_perp0 / np.where(sig_perp == 0, np.inf, sig_perp))
        Iy = np.exp(-0.5 * (np.asarray(y) / np.where(sig_perp == 0, np.inf, sig_perp)) ** 2)
        It = np.exp(-0.5 * (((np.asarray(t) - self.t0) - np.asarray(x) / self.v) / self.sigma_t) ** 2)
        return self.I0 * scale * Iy * It

    def get_bounds(self, time: float, local: bool = True) -> Tuple[float, float, float, float]:
        """Sampling bounds based on ± n_sigmax σ_z along x and ± n_sigmay σ⊥(x_c) along y.

        Parameters
        ----------
        time : float
            Time (seconds) at which to center the local window if `local` is True.
        local : bool, optional
            If True, center at x = v * time; if False, center at x = 0.

        Returns
        -------
        (float, float, float, float)
            (minx, maxx, miny, maxy) in meters.
        """
        half_span_x = self.sigma_t * self.n_sigmax * self.v  # = n_sigmax * σ_z
        if local:
            xc = time * self.v
            minx, maxx = xc - half_span_x, xc + half_span_x
        else:
            minx, maxx = -half_span_x, half_span_x

        x_center = 0.5 * (minx + maxx)
        sigma_here = self.sigma_perp(x_center)
        miny, maxy = -self.n_sigmay * sigma_here, self.n_sigmay * sigma_here
        return float(minx), float(maxx), float(miny), float(maxy)


def plot_profile(
    profile: Profile,
    time: float,
    nx: int,
    ny: int,
    *,
    local: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Generic plotter for any :class:`Profile`.

    Relies only on :meth:`Profile.get_bounds` and :meth:`Profile.intensity`.

    Parameters
    ----------
    profile : Profile
        The profile object to visualize.
    time : float
        Fixed time at which to evaluate intensity (seconds).
    nx : int
        Number of grid points along x.
    ny : int
        Number of grid points along y.
    local : bool, optional
        Passed through to :meth:`Profile.get_bounds`.
    save_path : str, optional
        If provided, path where the figure will be saved.
    show : bool, optional
        If True, display the figure interactively.
    """
    minx, maxx, miny, maxy = profile.get_bounds(time, local=local)
    x = np.linspace(minx, maxx, nx)
    y = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(x, y)
    intensity = profile.intensity(X, Y, time)

    plt.figure(figsize=(7.8, 5.6))
    extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]
    plt.imshow(intensity, origin='lower', aspect='auto', extent=extent)
    plt.colorbar(label='Normalized intensity (arb. units)')
    plt.xlabel('x (µm)'); plt.ylabel('y (µm)')
    plt.title('I(x, y) at fixed t = {:.2f} fs'.format(time * 1e15))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
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
