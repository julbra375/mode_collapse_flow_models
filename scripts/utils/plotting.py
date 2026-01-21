from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
from scripts.utils.ode_solver import phi

def make_fig_ax(lim=None, grid=False, equal_aspect=False):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    if lim:
        ax.set_xlim((-lim, lim))
        ax.set_ylim((-lim, lim))
    if grid:
        ax.grid()
    if equal_aspect:
        ax.set_aspect('equal')
    return fig, ax

def plot_2d_generated_samples(model, num_samples, initial_samples=None):
    """
    Visualise generated samples, starting from standard 2d Gaussian samples
    """
    
    if initial_samples is None:
        key = jax.random.PRNGKey(0)
        initial_samples = jax.random.multivariate_normal(key, mean=jnp.zeros(2), cov=jnp.identity(2), shape=num_samples)

    ts = [0, 1, 0.05]
    
    generated_xs = jax.vmap(lambda x: phi(model, x, ts))(initial_samples)

    fig, ax = make_fig_ax(grid=True)
    ax.scatter(initial_samples[:, 0], initial_samples[:, 1], label="Initial samples", alpha=0.7)
    ax.scatter(generated_xs[:, 0], generated_xs[:, 1], label="Generated samples", alpha=0.7)
    ax.set_aspect('equal')
    ax.legend()
    return fig, ax