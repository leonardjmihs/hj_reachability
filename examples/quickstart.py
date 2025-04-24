import jax
import jax.numpy as jnp
import numpy as np

from IPython.display import HTML
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import plotly.graph_objects as go


import hj_reachability as hj


obs = jnp.array([
                 [2.0, 0.0, 1.0],
                 [1.5, 2.0, 1.25]
                 ])

dynamics = hj.systems.DubinsCarObs(obs,
                max_control = jnp.array([jnp.pi, 1.0]),
                 min_control = jnp.array([-jnp.pi, -1.0])
                 )
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo=np.array([-5., -5., -np.pi]),
                                                                           hi=np.array([5., 5.,  np.pi])),
                                                               (50, 50, 50),
                                                               periodic_dims=2)
values = jnp.linalg.norm(grid.states[..., :2], axis=-1) -0.25
# values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 2

solver_settings = hj.SolverSettings.with_accuracy("low")
time = 0.
target_time = -2.0
target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time, progress_bar=False)
breakpoint()

plt.jet()
plt.figure(figsize=(13, 8))
plt.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values[:, :, 25].T)
plt.colorbar()
plt.contour(grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            # target_values[:, :, 30].T,
            target_values[:, :, 25].T,
            levels=0,
            colors="black",
            linewidths=3)

for circ in obs:
    circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
    plt.gca().add_artist(circle)

plt.show()