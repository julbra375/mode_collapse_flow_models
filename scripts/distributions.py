import jax
import jax.numpy as jnp
import equinox as eqx
from scripts.utils.ode_solver import phi

def define_distributions(dim, initial_radius, target_radius, num_initial_modes, num_target_modes, mode_arrangement, num_samples=10000, key=jax.random.PRNGKey(0), unimodal_init=False, eps_deg=0):

    key, train_key, mode_key1, mode_key2 = jax.random.split(key, 4)

    identity = jnp.identity(dim)

    # Define initial and target modes
    if mode_arrangement == 'orthogonal':
        initial_modes = identity[jnp.arange(num_initial_modes)] * initial_radius
        target_modes = identity[jnp.arange(num_initial_modes, num_initial_modes+num_target_modes)]*target_radius
    elif mode_arrangement == 'orthogonal_eps_overlap':
        initial_modes = identity[jnp.arange(num_initial_modes)] * initial_radius
        target_modes = identity[jnp.arange(num_initial_modes, num_initial_modes+num_target_modes)]*target_radius
        eps_rad = jnp.deg2rad(eps_deg)
        eps_rotation = jnp.array([[jnp.cos(eps_rad), 0, -jnp.sin(eps_rad), 0],
                                [0, jnp.cos(eps_rad), 0, -jnp.sin(eps_rad)],
                                [jnp.sin(eps_rad), 0 , jnp.cos(eps_rad), 0],
                                [0, jnp.sin(eps_rad), 0, jnp.cos(eps_rad)]])
        identity = jnp.identity(dim-4)
        zeros = jnp.zeros((dim-4, 4))
        rotation_block = jnp.block([[eps_rotation, zeros.T], [zeros, identity]])
        initial_modes = (rotation_block @ initial_modes.T).T
    elif mode_arrangement == 'symmetric':
        initial_modes = (jnp.repeat(identity, 2, axis=0) * (-1)**jnp.arange(2*dim)[:, jnp.newaxis])[jnp.arange(num_initial_modes)] * initial_radius
        target_modes = (jnp.repeat(identity, 2, axis=0) * (-1)**jnp.arange(2*dim)[:, jnp.newaxis])[jnp.arange(num_initial_modes, num_initial_modes+num_target_modes)] * target_radius
    elif mode_arrangement == 'random_hypersphere_symmetric':
        initial_mode1 = get_hypersphere_modes(dim, 1, initial_radius, jax.random.PRNGKey(0))[0]
        initial_modes = jnp.stack([initial_mode1, -initial_mode1])
        target_modes = find_orthogonal_points_symmetric(initial_mode1)
    elif mode_arrangement == 'random_hypersphere_orthogonal':
        initial_mode1 = get_hypersphere_modes(dim, 1, initial_radius, jax.random.PRNGKey(0))[0]
        new_initial_modes = find_3_orthogonal_points(initial_mode1.reshape((-1, 1)))
        initial_modes = jnp.stack([initial_mode1, new_initial_modes[0]])
        target_modes = new_initial_modes[1:]
  
    # Define initial distribution
    if unimodal_init:
        initial_sampler = lambda key: jax.random.multivariate_normal(key, mean=jnp.zeros(dim), cov=jnp.identity(dim), shape=num_samples)
        initial_pdf = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, mean=jnp.zeros(dim), cov=jnp.identity(dim))
    else:
        initial_covs = jnp.tile(jnp.identity(dim), (num_initial_modes, 1, 1))
        initial_weights = jnp.ones(num_initial_modes)
        initial_sampler = lambda key: sample_multimodal_gaussian(key, means=initial_modes, covs=initial_covs, weights=initial_weights, num_samples=num_samples)
        initial_pdf = lambda x: multimodal_gaussian_logpdf(x, means=initial_modes, covs=initial_covs, weights=initial_weights)

    # Define target distribution
    target_covs = jnp.tile(jnp.identity(dim), (num_target_modes, 1, 1))
    target_weights = jnp.ones(num_target_modes)
    target_sampler = lambda key: sample_multimodal_gaussian(key, target_modes, target_covs, target_weights, num_samples=num_samples)
    target_pdf = lambda x: multimodal_gaussian_logpdf(x, target_modes, target_covs, target_weights)

    return initial_modes, target_modes, initial_sampler, target_sampler, initial_pdf, target_pdf 


def sample_multimodal_gaussian(key, means, covs, weights, num_samples):
    """
    Sample from multi-modal Gaussian (full covariances)
    
    Args:
        key: JAX random key
        means: shape (k, d) - component means
        covs: shape (k, d, d) - full covariance matrices per component
        weights: shape (k,) - mixture weights (should sum to 1)
        num_samples: number of samples to draw
    
    Returns:
        samples: shape (num_samples, d)
    """
    key1, key2 = jax.random.split(key)
    
    k, d = means.shape
    weights_normed = weights / jnp.sum(weights)

    # Sample which component each sample comes from
    components = jax.random.choice(
        key1, 
        k,  # number of modes
        shape=(num_samples,),
        p=weights_normed
    )
    
    # Sample from standard normal
    z = jax.random.normal(key2, (num_samples, d))
    
    # Compute Cholesky factors for each component
    chol_factors = jax.vmap(jnp.linalg.cholesky)(covs)  # shape (k, d, d)
    
    # Transform: x = mean[component] + chol[component] @ z
    samples = means[components] + jax.vmap(
        lambda chol, z_sample: chol @ z_sample
    )(chol_factors[components], z)
    
    return samples

def multimodal_gaussian_pdf(x, means, covs, weights):
    """
    Multi-modal Gaussian (Mixture of Gaussians)
    
    Args:
        x: evaluation points, shape (..., d)
        means: component means, shape (k, d)
        covs: component covariances, shape (k, d, d)
        weights: mixture weights, shape (k,) - should sum to 1
    
    Returns:
        pdf values at x
    """
    
    # Compute pdf for each component
    log_probs = jax.vmap(lambda mean, cov: 
        jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)
    )(means, covs)  # shape (k, ...)

    # Log-sum-exp for numerical stability
    log_weights = jnp.log(weights)  # shape (k, 1)
    log_prob = jax.scipy.special.logsumexp(log_probs + log_weights, axis=0)
    
    return jnp.exp(log_prob)

def multimodal_gaussian_logpdf(x, means, covs, weights):
    """ Returns LOG pdf directly to prevent underflow. """
    
    # Compute log pdf for each component
    log_probs = jax.vmap(lambda mean, cov: 
        jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)
    )(means, covs)

    log_weights = jnp.log(weights)
    if x.ndim > 1:
        log_weights = log_weights[:, None]

    # Use logsumexp to combine them safely
    # Result is log( sum( exp(log_prob + log_weight) ) )
    final_log_prob = jax.scipy.special.logsumexp(log_probs + log_weights, axis=0)
    
    return final_log_prob  # <--- DO NOT EXP THIS


def sample_unit_sphere(dim, key):
    random_vector = jax.random.normal(key, shape=(dim,))
    unit_vector = random_vector / jnp.linalg.norm(random_vector) 
     
    return unit_vector

def get_hypersphere_modes(dim: int, num_modes: int, radius: float, key):
    """
    Generates 'num_modes' mean vectors spread on the surface of 
    a hypersphere of dimension 'dim' and radius 'radius'.
    
    Args:
        dim (int): The dimension of the space (e.g., 2 for a circle, 3 for a sphere).
        num_modes (int): The number of modes (mean vectors) to generate.
        radius (float): The radius of the hypersphere.
        key: A JAX PRNG key.

    Returns:
        jnp.array: An array of shape (num_modes, dim) containing the means.
    """

    keys = jax.random.split(key, num=num_modes)
    unit_means = jax.vmap(lambda k: sample_unit_sphere(dim, k))(keys)
    means = unit_means * radius
    
    return means

def estimate_evolved_means(key, component_idxs, means, covs, W, num_samples=1000):
    """
    Estimates the mean by sampling heavily from ONE component and flowing the samples.
    """
    # 1. Sample exclusively from the chosen component (no random choice needed)
    # Get mean and cov for the specific component index
    mus = means[component_idxs]
    covs = covs[component_idxs]
    d = mus.shape[1]
    
    # Standard sampling logic for a single Gaussian
    z = jax.random.normal(key, (num_samples, d))
    chols = jax.vmap(lambda cov: jnp.linalg.cholesky(cov))(covs)
    initial_samples = jax.vmap(lambda mu, chol: mu + z @ chol.T)(mus, chols) # shape (num_samples, d)
    
    model = lambda t, x: W @ x
    ts = [0, 1, 0.01]
    final_samples = jax.vmap(lambda zs: jax.vmap(lambda x: eqx.filter_jit(phi)(model, x, ts))(zs))(initial_samples)
    
    # 3. Compute empirical mean
    return jnp.mean(final_samples, axis=1)

def find_orthogonal_points_symmetric(mu1):
    # 1. Get the radius r
    r = jnp.linalg.norm(mu1)
    
    # 2. Create a basis where the first vector is mu1
    # We add a dummy dimension to use it as a column
    mu1_col = mu1[:, None]
    
    # 3. Use QR decomposition to find an orthonormal basis
    # We append a random identity-like matrix to ensure we have enough vectors to span D
    dummy = jnp.eye(mu1.shape[0])
    full_matrix = jnp.column_stack([mu1, dummy])
    q, r_tri = jnp.linalg.qr(full_matrix)
    
    # q[:, 0] is the normalized mu1. 
    # q[:, 1] is a unit vector guaranteed to be orthogonal to mu1.
    w = q[:, 1]
    
    # 4. Generate the two points
    p_plus = r * w
    p_minus = -r * w
    
    return jnp.array([p_plus, p_minus])

def find_3_orthogonal_points(mu1):
    r = jnp.linalg.norm(mu1)

    d = mu1.shape[0]
    if d < 4:
        raise ValueError(f"Input vector must be at least 4D to find 3 mutually orthogonal vectors. Found {d}D.")

    dummy = jnp.eye(d)
    full_matrix = jnp.column_stack([mu1, dummy])
    
    q, _ = jnp.linalg.qr(full_matrix)
    
    orthogonal_basis_vectors = q[:, 1:4] # Selects columns 1, 2, and 3
    
    points = r * orthogonal_basis_vectors.T
    
    return points # Returns an array of shape (3, d)