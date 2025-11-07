from collider.simulate import set_initial_state, simulation_step, run_simulation
from collider.beam import SetupInfo, BeamInfo, Beam, plot_beam

# Mismatched. Parameters in setupinfo not copied into beams
_beam1 = SetupInfo(Lx=5, Ly=1, vx=2.0, vy=0.0)
_beam2 = SetupInfo(Lx=5, Ly=1, vx=-1.0, vy=-1.0)

C1, C2, t_c = state = set_initial_state(_beam1, _beam2)

print(C1, C2, t_c)

beam1 = Beam(Lx=5, Ly=1, dx=0.25, dy=0.25, Cx=C1[0], Cy=C1[1], angle=_beam1.angle,
             vx=2.0, vy=0.0, name='Beam1')
beam2 = Beam(Lx=5, Ly=1, dx=0.25, dy=0.25, Cx=C2[0], Cy=C2[1], angle=_beam2.angle,
             vx=-5.0, vy=-5.0, name='Beam2')
beam1.create_elements()
beam2.create_elements()


run_simulation(beam1, beam2, time_step=t_c / 200., save_directory=None)

# proximal = simulation_step(beams=[beam1, beam2], time_step=0.5, save_directory=None)
# print(proximal)
# # beam1.update_position(dt=t_c)
# # beam2.update_position(dt=t_c)

figure = plot_beam(beam1)
plot_beam(beam2, figure, bounds=((-10, 10), (-10, 10)))
