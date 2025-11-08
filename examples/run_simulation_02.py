from collider.simulate import set_initial_state, simulation_step, run_simulation
from collider.beam import SetupInfo, BeamInfo, Beam, plot_beam

Lx1, Ly1 = 5, 1
Lx2, Ly2 = 5, 1
vx1, vy1 = 2.0, 0.0
vx2, vy2 = -5.0, -0.5
_beam1 = SetupInfo(Lx=Lx1, Ly=Ly1, vx=vx1, vy=vy1)
_beam2 = SetupInfo(Lx=Lx1, Ly=Ly1, vx=vx2, vy=vy2)

C1, C2, t_c = state = set_initial_state(_beam1, _beam2)

print(C1, C2, t_c)

beam1 = Beam(Lx=Lx1, Ly=Ly1, dx=0.5, dy=0.25, Cx=C1[0], Cy=C1[1], angle=_beam1.angle,
             vx=vx1, vy=vy1, name='Beam1')
beam2 = Beam(Lx=Lx1, Ly=Ly1, dx=0.5, dy=0.25, Cx=C2[0], Cy=C2[1], angle=_beam2.angle,
             vx=vx2, vy=vy2, name='Beam2')
beam1.create_elements()
beam2.create_elements()

# prox = simulation_step(beams=[beam1, beam2], time_step=6*t_c / 100.)
# print(prox)
run_simulation(beam1, beam2, time_step=t_c / 100., save_directory=None)

# proximal = simulation_step(beams=[beam1, beam2], time_step=0.5, save_directory=None)
# print(proximal)
# # beam1.update_position(dt=t_c)
# # beam2.update_position(dt=t_c)

figure = plot_beam(beam1)
plot_beam(beam2, figure, bounds=((-10, 10), (-10, 10)), filename='beam.png')
for ele in beam1:
    print(ele.interactions)
