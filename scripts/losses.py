import jax
import jax.numpy as jnp
import equinox as eqx
from scripts.utils.ode_solver import phi_with_logdet

"""
Using D_KL(q||p)

i.e. sampling from targets and pushing backwards
"""
def CNF_single_example_loss(f, x1, ts, init_distribution_pdf, key, approx):

    # Run backward and get log-det
    x0, log_det = eqx.filter_jit(phi_with_logdet)(f, x1, ts, key, approx=approx)

    # Log density under initial distribution
    log_q0 = jnp.log(init_distribution_pdf(x0))
    
    # Change of variables
    log_q_theta = log_q0 + log_det
    
    return log_q_theta

def CNF_batch_loss(model, x1s, ts, init_distribution_pdf, key, approx):
    """
    Calculate the KL-based loss for a batch of training example

    f: func, parameterised flow function. Should have arguments (t, f)
    x1: Array, batch of training examples from target distribution shape (batch, dim)
    ts: Array, list of time points at which to run the simulation. Should go from [1, 0] inclusive
    init_distribution_pdf: func, takes a point x and returns pdf of initial distribution. Default to multivariate Gaussian
    """
 
    # If not given, define initial distribution as multivariate normal, mean=0, covariance=identity
    if init_distribution_pdf is None:
        print("Need to define intial distribution!")

    # Split keys to be used in ODE solver
    ode_keys = jax.random.split(key, x1s.shape[0])

    # Calculate total loss for batch of examples
    total_loss = jax.vmap(CNF_single_example_loss, in_axes=(None, 0, None, None, 0, None))(model, x1s, ts, init_distribution_pdf, ode_keys, approx)
    
    return -jnp.mean(total_loss)

"""
Using D_KL(p||q)

i.e. sampling from initial and pushing forward
"""

def CNF_reverse_kl_single_loss(f, z, ts, target_log_prob_fn, key, approx):
    # 1. Run FORWARD (0 -> 1) to generate a sample x from latent z
    # Note: ts should be [0, 1, dt]
    x_model, log_det_J = eqx.filter_jit(phi_with_logdet)(f, z, ts, key, approx=approx)
    
    # 2. Evaluate how likely this sample is under the TARGET distribution
    log_p_target = jnp.log(target_log_prob_fn(x_model))
    
    # 3. Compute Reverse KL Loss
    loss =  - log_det_J - log_p_target
    
    return loss

def CNF_reverse_kl_single_loss_logprob(f, z, ts, target_log_prob_fn, key, approx):
    x_model, log_det_J = eqx.filter_jit(phi_with_logdet)(f, z, ts, key, approx=approx)
    
    # DIRECTLY use the log probability. Do not re-log it.
    log_p_target = target_log_prob_fn(x_model)
    
    loss = -log_det_J - log_p_target
    return loss

def CNF_reverse_kl_batch_loss(model, z_batch, ts, target_log_prob_fn, key, approx):
    """
    z_batch: Batch of samples from INITIAL distribution (e.g., Gaussian noise)
    target_log_prob_fn: Function that takes x and returns log p_target(x)
    """
    
    ode_keys = jax.random.split(key, z_batch.shape[0])
    
    total_loss = jax.vmap(CNF_reverse_kl_single_loss_logprob, in_axes=(None, 0, None, None, 0, None))(
        model, z_batch, ts, target_log_prob_fn, ode_keys, approx
    )
    
    return jnp.mean(total_loss)

def reconstruction_loss_koopman(model, x):
    # 1. Encode
    z = model.encoder(x)
    
    # 2. Flow (Transport)
    # We want the decoder to work on BOTH transported and non-transported coordinates
    # to ensure the coordinate system is valid everywhere.
    z_transported = model.flow(z, model.latent_targets) # Uses e^W
    
    # 3. Decode
    x_recon_source = model.decoder(z)
    x_recon_transported = model.decoder(z_transported)
    
    # 4. Reconstruction MSE
    # Reconstructing source ensures Z maps to X
    loss_source = jnp.mean((x - x_recon_source) ** 2)
    # Reconstructing transported ensures the target region maps to valid X
    # (Optional: Only if you have target data x_target to compare against. 
    #  If unsupervised, just reconstruct source).
    loss_total = loss_source 
    
    return loss_total, z # Return z to use in analytic step