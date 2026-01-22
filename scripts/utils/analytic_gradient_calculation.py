import jax
from jax import jit
from jax.scipy.linalg import expm, expm_frechet
import jax.numpy as jnp
from scripts.distributions import sample_multimodal_gaussian
import flax.linen as nn
import optax

@jit
def integrand_at_t_symmetric(t, w, z, mu_star):
    """
    Computes the term inside the integral for a specific t.
    Term: e^{w.T(1-t)} * a_x(1) * z.T * e^{w.T * t}
    """
    # Matrix Exponentials
    term1 = expm(w.T * (1 - t))
    
    term2 = expm(w.T * t)
    

    # Reshaping z to be a row vector (1, d) for z^T
    z_T = z.reshape(1, -1) 
    
    a_val = (expm(w) @ z - mu_star * jnp.tanh(z_T @ expm(w.T) @ mu_star)).reshape(-1, 1)
    

    return term1 @ a_val @ z_T @ term2


def compute_integral(integrand_fn, w, z, mu_star, num_t_steps):
    """
    Approximates the integral from t=0 to 1 using a mean (Riemann sum).
    """
    # Create grid for t [0, 1]
    t_grid = jnp.linspace(0, 1, num_t_steps)
    
    batch_integrand = jax.vmap(integrand_fn, in_axes=(0, None, None, None))
    
    evaluations = batch_integrand(t_grid, w, z, mu_star)
    
    integral = jax.scipy.integrate.trapezoid(evaluations, t_grid, axis=0)

    return integral

def compute_gradient_symmetric(w, mu_star, radius, t_steps=1000, key=jax.random.PRNGKey(0)):
    dim = w.shape[0]
    zs = sample_multimodal_gaussian(key, means=jnp.array([jnp.identity(dim)[1], -jnp.identity(dim)[1]])*radius, 
                                    covs=jnp.tile(jnp.identity(dim), (2, 1, 1)), weights=jnp.array([0.5, 0.5]), num_samples=10000)
    integral = jax.vmap(compute_integral, in_axes=(None, None, 0, None, None))(integrand_at_t_symmetric, w, zs, mu_star, t_steps)

    return jnp.mean(integral, axis=0) - jnp.identity(dim)


def A_of_W(W, M):
    """
    Returns operator A(W)[M] using exact analytic algorithm.
    expm_frechet(A, E) returns a tuple: (expm(A), frechet_derivative)
    We only need the second element.
    """
    # Note: Your original code applied the exponential to W.T
    _, integral = expm_frechet(W.T, M)
    return integral

def compute_analytic_gradient_2modes(W, initial_modes, target_modes, key=jax.random.PRNGKey(0), num_samples=10000):
    dim = W.shape[0]
    zs = sample_multimodal_gaussian(key, means=initial_modes, 
                                    covs=jnp.tile(jnp.identity(dim), (2, 1, 1)), 
                                    weights=jnp.array([0.5, 0.5]), num_samples=num_samples)
    
    # Calculate first term
    # expectation_1 = jnp.mean(jax.vmap(lambda z: z.reshape((-1, 1)) @ z.reshape((1, -1)))(zs), axis=0)
    expectation_1 = jnp.identity(dim) + 0.5 * (initial_modes.T @ initial_modes)
    term_1 = A_of_W(W, expm(W) @ expectation_1)

    # Calculate second term
    mu1_star, mu2_star = target_modes
    v = expm(W.T) @ (mu1_star - mu2_star).reshape((-1, 1))
    expectation_2_term = lambda z: (mu1_star.reshape((-1, 1)) * jax.nn.sigmoid(z.reshape(1, -1) @ v) + mu2_star.reshape((-1, 1)) * jax.nn.sigmoid(-z.reshape((1, -1)) @ v)) @ z.reshape((1, -1))
    expectation_2 = jnp.mean(jax.vmap(expectation_2_term)(zs), axis=0)
    term_2 = A_of_W(W, expectation_2)

    # Combine terms
    gradient = -jnp.identity(dim) + term_1 - term_2

    return gradient

def compute_analytic_gradient_Mmodes(W, initial_modes, target_modes, weights, key=jax.random.PRNGKey(0), num_samples=10000):
    dim = W.shape[0]
    num_initial_modes = initial_modes.shape[0]
    zs = sample_multimodal_gaussian(key, means=initial_modes, 
                                    covs=jnp.tile(jnp.identity(dim), (num_initial_modes, 1, 1)), 
                                    weights=weights, num_samples=num_samples)
    
    # Calculate first term
    expectation_1 = jnp.identity(dim) + (initial_modes.T @ initial_modes) / num_initial_modes
    term_1 = A_of_W(W, expm(W) @ expectation_1)

    # Calculate second term
    def get_expectation_argument(z):
        softmax_z = jax.nn.softmax(z.reshape(1, -1) @ expm(W.T) @ target_modes.T)
        product_inside = jax.vmap(lambda t: t.reshape((-1, 1)) @ z.reshape(1, -1))(target_modes)
        expectation_argument = jnp.sum(softmax_z.reshape(-1, 1, 1) * product_inside, axis=0) # sum over M* targets
        return expectation_argument
    
    expectation_2 = jnp.mean(jax.vmap(get_expectation_argument)(zs), axis=0)
    term_2 = A_of_W(W, expectation_2)

    gradient = -jnp.identity(dim) + term_1 - term_2

    return gradient
    
def compute_analytic_gradient_batch(W, z_batch, latent_target_means, num_integral_iterations=1000):
    """
    Computes the analytic gradient of the KL divergence in latent space.
    Args:
        W: Flow matrix (d x d)
        z_batch: Encoded source data (batch_size x d)
        latent_target_means: Calculated means of encoded target data (2 x d)
    """
    dim = W.shape[0]
    
    # 1. Term 1: Source Covariance Alignment
    # E[zz^T] using the current batch
    expectation_1 = jnp.mean(jax.vmap(lambda z: z.reshape((-1, 1)) @ z.reshape((1, -1)))(z_batch), axis=0)
    term_1 = A_of_W(W, jax.scipy.linalg.expm(W) @ expectation_1, num_integral_iterations)

    # 2. Term 2: Target Mode Attraction
    mu1_star, mu2_star = latent_target_means[0], latent_target_means[1]
    
    # Projection vector v
    v = jax.scipy.linalg.expm(W.T) @ (mu1_star - mu2_star).reshape((-1, 1))
    
    def expectation_2_term(z):
        z_row = z.reshape((1, -1))
        # Sigmoid weighting
        s1 = jax.nn.sigmoid(z_row @ v)
        s2 = jax.nn.sigmoid(-z_row @ v)
        weighted_mu = mu1_star.reshape((-1, 1)) * s1 + mu2_star.reshape((-1, 1)) * s2
        return weighted_mu @ z_row

    expectation_2 = jnp.mean(jax.vmap(expectation_2_term)(z_batch), axis=0)
    term_2 = A_of_W(W, expectation_2, num_integral_iterations)

    # Gradient: -I + Term1 - Term2
    return -jnp.identity(dim) + term_1 - term_2



