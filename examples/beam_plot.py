from collider.beam import Beam, plot_beam

# Beam 1
Lx = 2.0
Ly = 6.0
dx = 0.25
dy = 0.25
Cx = -5.
Cy =  5.
angle = 45.

beam_obj = Beam(Lx, Ly, dx, dy, Cx, Cy, angle)

beam_obj.create_elements()
figure = plot_beam(beam_obj)

# Beam 2
Lx = 6.0
Ly = 3.0
dx = 0.25
dy = 0.25
Cx = 6.
Cy = 0.
angle = 0.

beam_obj = Beam(Lx, Ly, dx, dy, Cx, Cy, angle)

beam_obj.create_elements()
plot_beam(beam_obj, figure)
