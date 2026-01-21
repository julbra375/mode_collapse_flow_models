import diffrax
import jax
import jax.numpy as jnp
from functools import partial

def phi(f, y0, ts):
    """ Solves flow defined by f """
    def vector_field(t, y, args):
        fn = args[0]
        return fn(t, y)
    
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[1],
        dt0=ts[2],
        y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        args=(f,)
    )
    return solution.ys[0]

def approx_logp_wrapper(t, state, args):
    """ Uses Hutchinson's trace estimator for efficiency in high dimensions"""
    y, _ = state
    f, eps = args
    fn = lambda y: f(t, y)
    fp, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    logp = jnp.sum(eps_dfdy * eps)
    return fp, logp
    
def vector_field_with_trace(t, state, args):
    y, _ = state
    f, _ = args
    dy = f(t, y)
    
    jac = jax.jacobian(f, argnums=1)(t, y)
    dlogdet = jnp.trace(jac)
    
    return dy, dlogdet

def phi_with_logdet(f, y0, ts, key, approx=False):
    """ Solves flow defined by f and accumulates log|det J| 
    
        ts has form [t0, t1, dt0]
    """
    if approx:
        term = approx_logp_wrapper
    else:
        term = vector_field_with_trace

    y0_augmented = (y0, jnp.array(0.0))
    eps = jax.random.normal(key, y0.shape)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(term),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[1],
        dt0=ts[2],
        y0=y0_augmented,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        args=(f, eps)
    )
    final_y, final_logdet = solution.ys
    
    # Ensure correct shapes
    final_y = jnp.squeeze(final_y, axis=0) if final_y.ndim > 1 else final_y
    final_logdet = jnp.squeeze(final_logdet)
    
    return final_y, final_logdet # (trajectory, log_det_trajectory)