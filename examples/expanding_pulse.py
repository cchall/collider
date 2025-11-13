import numpy as np
from scipy import constants

from collider.beam import SetupInfo, BeamInfo, Beam, plot_beam
from collider.profiles import LaserPulse


# Setup Laser Pulse
start_time = -500e-15
start_x = start_time * constants.c
pulse1 = LaserPulse(w0=10e-6, tau_fwhm=250e-15, lambda0=1080e-9, I0=1.0, t0=0)

minx, maxx, miny, maxy = pulse1.get_bounds(time=start_time)
print(pulse1.get_bounds(time=start_time))
Lx = (maxx - minx)
Ly = (maxy - miny)
angle1 = 45.0  # deg
vx1, vy1 = constants.c * np.cos(np.deg2rad(angle1)), constants.c * np.sin(np.deg2rad(angle1))

beam1 = Beam(Lx=Lx, Ly=Ly,
             dx=Lx/50, dy=Ly/50, Cx=(minx + maxx)/2,
             Cy=(miny+maxy)/2., angle=angle1,
             vx=vx1, vy=vy1, profile=pulse1, name='Pulse1')
beam1.create_elements()
beam1.update_from_profile(start_time*10)


fig, ax = plot_beam(beam1, bounds=((minx, maxx), (miny*5, 5*maxy)), color_by='density')


