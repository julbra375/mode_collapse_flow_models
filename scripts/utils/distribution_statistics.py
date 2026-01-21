import jax
import jax.numpy as jnp
import equinox as eqx
from scripts.utils.ode_solver import phi, phi_with_logdet
from scripts.losses import CNF_single_example_loss
from matplotlib import pyplot as plt
from scripts.distributions import estimate_evolved_means

def kl_divergence_target_vs_learned(
    samples_from_target,
    target_distribution_pdf,
    init_distribution_pdf,
    model,
    ts,  # Time array [0, 1, dt],
    key,
    approx
):
    """
    Compute KL(p_target || q_learned).
    
    Args:
        samples_from_target: Samples from target distribution, shape (n, d)
        target_distribution_pdf: Function computing p_target(x)
        init_distribution_pdf: Function computing p_0(x)
        model: The flow model (neural network)
        ts: Time array for integration [t0, t1, dt0]
    
    Returns:
        float, estimated KL divergence
    """
    
    # Compute log probabilities
    p_target = jax.vmap(target_distribution_pdf)(samples_from_target)  # (n,)
    log_p_target = jnp.log(p_target)
    ode_keys = jax.random.split(key, samples_from_target.shape[0])
    log_q_learned = jax.vmap(CNF_single_example_loss, in_axes=(None, 0, None, None, 0, None))(model, samples_from_target, ts, init_distribution_pdf, ode_keys, approx)

    # KL divergence
    kl = jnp.mean(log_p_target - log_q_learned)
    
    return kl

def w_plus_minus(model, initial_distribution_sample_fn, key, mode1, mode2):
    """
    Calculates the proportion of sampled points in the positive half of R^d. 
    Separation is perpendicular to the line between the two modes.
    
    Args:
        mode1, mode2: Arrays of shape (d,), the two mode centers
    Returns:
        w_plus, w_minus
    """
    initial_samples = initial_distribution_sample_fn(key)
    
    ts = [0, 1, 0.01]
    generated_xs = jax.vmap(lambda x: eqx.filter_jit(phi)(model, x, ts))(initial_samples)
    
    # Direction from mode1 to mode2
    direction = mode2 - mode1
    direction_normalized = direction / jnp.linalg.norm(direction)
    
    # Project samples onto line
    centered_samples = generated_xs - mode1
    projections = jnp.dot(centered_samples, direction_normalized)
    
    # Midpoint projection
    midpoint = jnp.dot(direction, direction_normalized) / 2.0
    
    # Proportion on mode2 side
    proportion_positive = jnp.mean(projections > midpoint)
    
    return proportion_positive, 1-proportion_positive

def w_mu_plus_minus(model, initial_distribution_sample_fn, key, mode1, mode2, radius):
    """
    Calculates the proportion of sampled points in the positive half of R^d and average vector in each half. 
    Separation is perpendicular to the line between the two modes.
    
    Args:
        mode1, mode2: Arrays of shape (d,), the two mode centers
    Returns:
        w_plus, w_minus, mu_plus, mu_minus
    """
    w_plus, w_minus = w_plus_minus(model, initial_distribution_sample_fn, key, mode1, mode2)

    initial_samples = initial_distribution_sample_fn(key)
    
    ts = [0, 1, 0.01]
    generated_xs = jax.vmap(lambda x: eqx.filter_jit(phi)(model, x, ts))(initial_samples)
    
    # Direction from mode1 to mode2
    direction = mode2 - mode1
    direction_normalized = direction / jnp.linalg.norm(direction)
    
    # Project samples onto line
    centered_samples = generated_xs - mode1
    projections = jnp.dot(centered_samples, direction_normalized)
    
    # Midpoint projection
    midpoint = jnp.dot(direction, direction_normalized) / 2.0
    
    w_plus = jnp.mean(projections > midpoint)
    w_minus = 1 - w_plus

    x_plus = jnp.mean((generated_xs.T * (projections > midpoint)).T, axis=0)
    x_minus = jnp.mean((generated_xs.T * (projections <= midpoint)).T, axis=0)

    mu_plus = x_plus / w_plus
    mu_minus = x_minus / w_minus

    m_plus = mu_plus.T @ mode1 / radius**2
    m_minus = mu_minus.T @ mode1 / radius **2
    s = mu_plus.T @ mu_minus / radius**2

    return w_plus, w_minus, m_plus, m_minus, s
    

def plot_distribution_histogram_parallel(model, initial_distribution_sample_fn, key, mode1, mode2, num_bins=50, figsize=(10, 6), ax=None, midpoint_align=True):
    """
    Project generated samples onto the line between two modes and plot 1D histogram.
    Center of line is at 0.
    
    Args:
        model: The trained CNF model
        initial_distribution_sample_fn: Function to sample from initial distribution
        key: JAX random key
        mode1, mode2: Arrays of shape (d,), the two mode centers
        num_bins: Number of bins for histogram
        figsize: Figure size for plot
    
    Returns:
        projections: Array of projected samples
        fig, ax: Matplotlib figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    initial_samples = initial_distribution_sample_fn(key)

    ts = jnp.array([0, 1, 0.01])
    generated_xs = jax.vmap(lambda x: phi(model, x, ts))(initial_samples)
    
    # Direction from mode1 to mode2
    direction = mode2 - mode1
    direction_normalized = direction / jnp.linalg.norm(direction)
    
    # Midpoint between modes
    if midpoint_align:
        midpoint = (mode1 + mode2) / 2.0
    else:
        midpoint = mode2
    
    # Project samples onto line, centered at midpoint
    centered_samples = generated_xs - midpoint
    projections = jnp.dot(centered_samples, direction_normalized)
    
    # Plot on ax    
    ax.hist(projections, bins=num_bins, alpha=0.7, edgecolor='black', density=True)
    
    # Mark mode locations relative to center (0)
    half_distance = jnp.linalg.norm(direction) / 2.0
    if midpoint_align:
        ax.axvline(-half_distance, color='red', linestyle='--', linewidth=2, label=f'Mode 1')
        ax.axvline(half_distance, color='blue', linestyle='--', linewidth=2, label=f'Mode 2')

    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, label='Center')
    
    ax.set_xlabel('Projection onto mode-to-mode line')
    ax.set_ylabel('Density')
    # ax.set_title('Distribution of Generated Samples Projected onto Mode Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return projections, fig, ax



def plot_distribution_histogram_perpendicular(model, initial_distribution_sample_fn, key, mode1, mode2, num_bins=50, figsize=(10, 6), ax=None):
    """
    Plot histogram of signed perpendicular distances from samples to the line between modes.
    For N-dim, picks an arbitrary orthogonal direction to the mode line.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    initial_samples = initial_distribution_sample_fn(key)
    
    ts = jnp.array([0, 1, 0.01])
    generated_xs = jax.vmap(lambda x: eqx.filter_jit(phi)(model, x, ts))(initial_samples)
    
    # Direction from mode1 to mode2
    direction = mode2 - mode1
    direction_normalized = direction / jnp.linalg.norm(direction)
    
    # Create an orthogonal direction
    # Find a vector not parallel to direction, then orthogonalize
    dim = direction.shape[0]
    e = jnp.zeros(dim)
    e = e.at[0].set(1.0)  # Start with first standard basis vector
    
    # If direction is parallel to e, use second basis vector instead
    if jnp.abs(jnp.dot(direction_normalized, e)) > 0.9:
        e = e.at[0].set(0.0)
        e = e.at[1].set(1.0)
    
    # Gram-Schmidt orthogonalization
    perp_direction = e - jnp.dot(e, direction_normalized) * direction_normalized
    perp_direction = perp_direction / jnp.linalg.norm(perp_direction)
    
    # Vector from mode1 to each sample
    centered_samples = generated_xs - mode1
    
    # Signed perpendicular distance: dot product with perpendicular direction
    signed_perp_distances = jnp.dot(centered_samples, perp_direction)
    
    ax.hist(signed_perp_distances, bins=num_bins, alpha=0.7, edgecolor='black', density=True)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Mode line')
    ax.set_xlabel('Signed perpendicular distance from mode line')
    ax.set_ylabel('Density')
    # ax.set_title('Distribution of Signed Perpendicular Distances to Mode-Connecting Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return signed_perp_distances, fig, ax


def stable_rank_svd(A, eps=1e-10):
    """
    Compute stable rank using explicit SVD.
    
    stable_rank = (sum of all singular values)^2 / (largest singular value)^2
    
    This is mathematically equivalent but can be more interpretable.
    """
    # Compute SVD
    _, s, _ = jnp.linalg.svd(A, full_matrices=False)
    
    # Sum of squared singular values = Frobenius norm squared
    sum_sv_sq = jnp.sum(s ** 2)
    
    # Largest singular value squared
    max_sv_sq = s[0] ** 2
    
    # Stable rank
    stable_rank_val = sum_sv_sq / (max_sv_sq + eps)
    
    return stable_rank_val

def get_linear_summary_stats(W, initial_modes, target_modes, radius):
    
    E = jax.scipy.linalg.expm(W)
    
    mu_transformed = initial_modes @ E.T  # shape (k, D)
    
    M = (mu_transformed @ target_modes.T) / radius**2 # shape (k, k)
    
    S = (mu_transformed @ mu_transformed.T) / radius**2 # shape (k, k)

    eta = M[:, 0] - M[:, 1]

    v = E.T @ (target_modes[0].reshape((-1, 1)) - target_modes[1].reshape((-1, 1)))

    rho = jnp.linalg.norm(v)**2


    return M, S, eta, rho

def get_minimum_summary_stats(W, initial_modes, target_modes, time_derivs=False):

    M, D = initial_modes.shape
    M_star, _ = target_modes.shape

    # 1. Compute the Flow Operator (Matrix Exponential)
    E = jax.scipy.linalg.expm(W)
    
    # 2. Compute V (Transported Target Modes) - Shape (M_star, D)
    V = target_modes @ E
    
    # 3. Compute U (Transported Second Moment) - Shape (D, D)
    Sigma_init = jnp.eye(D) + (initial_modes.T @ initial_modes) / M
    U = E @ Sigma_init
    
    # 4. Compute Eta (Projected Means) - Shape (M, M_star)
    eta = initial_modes @ V.T
    
    # 5. Compute C (Gram Matrix of Logits) - Shape (M_star, M_star)
    C = V @ V.T
    
    if time_derivs:
        d_eta_dt = (initial_modes @ W.T) @ V.T
        return U, V, eta, C, d_eta_dt
    else:
        return U, V, eta, C