from collider.beam import Beam

# Beam 1
Lx = 2.0
Ly = 6.0
dx = 0.25
dy = 0.25
Cx = -5.
Cy =  5.
angle = 79.

beam_obj = Beam(Lx, Ly, dx, dy, Cx, Cy, angle)

beam_obj.create_elements()
figure = beam_obj.plot()

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
beam_obj.plot(figure)
