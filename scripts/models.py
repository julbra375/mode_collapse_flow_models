from typing import Callable, Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax
from jax import nn
from jax import custom_vjp
from scripts.utils.analytic_gradient_calculation import compute_analytic_gradient_batch

class MLP(eqx.Module):
    """
    A time-independent vector field for a Neural ODE,
    implemented using manually assigned linear layers and JAX LAX primitives
    for activation functions.
    This avoids issues with `jax.nn` functions that are themselves pre-JIT-compiled.
    """
    layers: list[eqx.nn.Linear]

    # Store activation functions as static fields.
    hidden_activation_fn: callable = eqx.field(static=True)
    final_activation_fn: callable = eqx.field(static=True) 

    def __init__(self, key, data_dim, width_size, depth, hidden_activation, final_activation=None, init_std=1.0, **kwargs):
        super().__init__(**kwargs)
        
        keys = jax.random.split(key, depth + 1) 

        layers = []
        if depth == 0:
            layers.append(eqx.nn.Linear(data_dim, data_dim, key=keys[0]))
        else:
            layers.append(eqx.nn.Linear(data_dim, width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i + 1]))
            layers.append(eqx.nn.Linear(width_size, data_dim, key=keys[-1]))

        self.layers = layers
        
        # Store the actual callable function directly as a static field
        # Ensure that `hidden_activation` passed here is a JAX LAX primitive or custom non-JITted function
        self.hidden_activation_fn = hidden_activation
        self.final_activation_fn = final_activation

        # Custom initialization
        self._reinitialize_weights(key, init_std)
        
    def _reinitialize_weights(self, key, init_std):
        """Reinitialize weights with custom scheme"""
        
        # Count total linear layers
        total_layers = len(self.layers)
        init_keys = jax.random.split(key, total_layers * 2)
        
        for layer_idx, layer in enumerate(self.layers):
            w_key = init_keys[layer_idx * 2]
            b_key = init_keys[layer_idx * 2 + 1]
            
            # Determine which initializer to use
            # First layer uses init_std, others use scaled version
            # if layer_idx == 0:
            #     w_stddev = init_std
            # else:
            #     w_stddev = jnp.sqrt(1 / jnp.sqrt(layer.in_features))
            w_stddev = init_std
            # Reinitialize weight
            new_weight = jax.nn.initializers.normal(stddev=w_stddev)(w_key, layer.weight.shape)
            new_layer = eqx.tree_at(lambda l: l.weight, layer, new_weight)
            
            # Reinitialize bias if it exists
            if layer.bias is not None:
                new_bias = jax.nn.initializers.normal(stddev=0)(b_key, layer.bias.shape)
                new_layer = eqx.tree_at(lambda l: l.bias, new_layer, new_bias)
            
            self.layers[layer_idx] = new_layer

    def __call__(self, t, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.hidden_activation_fn(x)
        
        # Apply the final activation function
        x = self.final_activation_fn(x)
            
        return x

class LinearFlow(eqx.Module):

    W: jax.Array
    
    def __init__(self, key, dim, init_var, initial_weight=None):
        # Initialize W randomly (arbitrary matrix)
        if initial_weight is not None:
            self.W = initial_weight * init_var
        else:
            self.W = jax.random.normal(key, (dim, dim)) * init_var
    
    def __call__(self, t, z):
        return self.W @ z
    
class LinearFlowWithBias(eqx.Module):

    W: jax.Array
    b: jax.Array

    def __init__(self, dim, key, init_var, initial_weight=None, initial_bias=None):
        # Initialize W randomly (arbitrary matrix)
        if initial_weight is not None:
            self.W = initial_weight * init_var
        else:
            self.W = jax.random.normal(key, (dim, dim)) * init_var
        if initial_bias is not None:
            self.b = initial_bias
        else:
            self.b = jnp.zeros(dim)
    
    def __call__(self, t, z):
        return self.W @ z + self.b
    

def linear_flow_op(W, z_in):
    U = jax.scipy.linalg.expm(W)
    return z_in @ U.T

class AnalyticLinearFlow(eqx.Module):
    W: jax.Array

    def __init__(self, dim, key):
        # Initialize W near identity (small random values)
        self.W = jax.random.normal(key, (dim, dim)) * 0.01

    def __call__(self, z):
        # Calls the custom VJP function
        return linear_flow_op(self.W, z)
    
class KoopmanAE(eqx.Module):
    encoder: eqx.nn.MLP
    flow: AnalyticLinearFlow
    decoder: eqx.nn.MLP
    latent_targets: jax.Array

    def __init__(self, input_dim, latent_dim, mlp_width, key, latent_target_radius=4.0):
        k1, k2, k3 = jax.random.split(key, 3)
        
        self.encoder = eqx.nn.MLP(input_dim, latent_dim, activation=jax.nn.relu, width_size=mlp_width, depth=2, key=k1)
        self.flow = AnalyticLinearFlow(latent_dim, key=k2)
        self.decoder = eqx.nn.MLP(latent_dim, input_dim, activation=jax.nn.relu, width_size=mlp_width, depth=2, key=k3)
        
        # Define Fixed Latent Targets (+/- 5.0)
        targets = jnp.zeros((2, latent_dim))
        targets = targets.at[0, 0].set(latent_target_radius)
        targets = targets.at[1, 0].set(-latent_target_radius)
        self.latent_targets = targets

    def __call__(self, x):
        z_in = self.encoder(x)
        z_in = z_in - jnp.mean(z_in, axis=0)
        z_out = self.flow(z_in)
        y_pred = self.decoder(z_out)
        return y_pred, z_in, z_out

class AntiSymmetricLinearFlow(eqx.Module):
    """
    Parameterize by unconstrained W, construct A = (W - W^T)/2
    This automatically stays anti-symmetric under optimization.
    """
    A: jax.Array  # Unconstrained parameter
    
    def __init__(self, dim, key, init_var):
        # Initialize W randomly (arbitrary matrix)
        self.A = jax.random.normal(key, (dim, dim)) * init_var
    
    def __call__(self, t, z, args=None):
        # Construct anti-symmetric A on-the-fly
        W = (self.A - self.A.T) / 2
        return W @ z

class ConcatMLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, data_size, width_size, depth, key, init_std=1.0, **kwargs):
        super().__init__(**kwargs)
        keys = jax.random.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(
                ConcatSquash(in_size=data_size, out_size=data_size, key=keys[0])
            )
        else:
            layers.append(
                ConcatSquash(in_size=data_size, out_size=width_size, key=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    ConcatSquash(
                        in_size=width_size, out_size=width_size, key=keys[i + 1]
                    )
                )
            layers.append(
                ConcatSquash(in_size=width_size, out_size=data_size, key=keys[-1])
            )
        self.layers = layers

        # Custom initialization
        self._reinitialize_weights(key, init_std)
    
    def _reinitialize_weights(self, key, init_std):
        """Reinitialize weights with custom scheme"""
        
        # Count total number of linear layers across all ConcatSquash modules
        total_linear_layers = sum(
            len([attr for attr in dir(layer) if isinstance(getattr(layer, attr), eqx.nn.Linear)])
            for layer in self.layers
        )
        
        init_keys = jax.random.split(key, total_linear_layers * 2)
        key_counter = 0
        
        for layer_idx, concat_squash in enumerate(self.layers):
            # Iterate through all attributes of the ConcatSquash
            for attr_name in dir(concat_squash):
                attr = getattr(concat_squash, attr_name)
                
                # Check if it's a Linear layer
                if isinstance(attr, eqx.nn.Linear):
                    # Determine which initializer to use
                    # First layer (layer_idx=0) uses init_std, others use scaled version
                    # if layer_idx == 0:
                    #     w_stddev = init_std
                    # else:
                    #     w_stddev = jnp.sqrt(1 / jnp.sqrt(attr.in_features))
                    w_stddev = init_std

                    w_key = init_keys[key_counter * 2]
                    b_key = init_keys[key_counter * 2 + 1]
                    key_counter += 1
                    
                    # Reinitialize weight
                    new_weight = jax.nn.initializers.normal(stddev=w_stddev)(w_key, attr.weight.shape)
                    new_linear = eqx.tree_at(lambda l: l.weight, attr, new_weight)
                    
                    # Reinitialize bias if it exists
                    if attr.bias is not None:
                        new_bias = jax.nn.initializers.normal(stddev=0)(b_key, attr.bias.shape)
                        new_linear = eqx.tree_at(lambda l: l.bias, new_linear, new_bias)
                    
                    # Update the attribute in the ConcatSquash
                    concat_squash = eqx.tree_at(lambda l: getattr(l, attr_name), concat_squash, new_linear)
            
            self.layers[layer_idx] = concat_squash

    def __call__(self, t, y):
        t = jnp.asarray(t)[None]
        for layer in self.layers[:-1]:
            y = layer(t, y)
            y = jax.nn.tanh(y)
        y = self.layers[-1](t, y)
        return y


# Credit: this layer, and some of the default hyperparameters below, are taken from the
# FFJORD repo.
class ConcatSquash(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    lin3: eqx.nn.Linear

    def __init__(self, *, in_size, out_size, key, **kwargs):
        super().__init__(**kwargs)
        key1, key2, key3 = jax.random.split(key, 3)
        self.lin1 = eqx.nn.Linear(in_size, out_size, key=key1)
        self.lin2 = eqx.nn.Linear(1, out_size, key=key2)
        self.lin3 = eqx.nn.Linear(1, out_size, use_bias=False, key=key3)

    def __call__(self, t, y):
        return self.lin1(y) * jax.nn.sigmoid(self.lin2(t)) + self.lin3(t)
    
class Autoencoder(eqx.Module):
    """
    Autoencoder with shared weight vector w across encoder/decoder.
    
    f(x) = bx + (W2/sqrt(h)) * sigma((W1^T x)/sqrt(d))
    
    where:
    - b: learnable bias, shape (d,)
    - W1: learnable input-hidden weight matrix, shape (h, d), encoder
    - W2: learnable hidden-output weight matrix, shape (o, h), decoder
    - d: input dimension
    - h: hidden dimension
    - sigma: activation function
    """
    
    W1: jax.Array  # Encoder weight matrix, shape (h, d)
    W2: jax.Array # Decoder weight matrix, shape (o, h)
    b: jax.Array  # Bias, shape (d,)
    d: int        # Data dimension
    h: int        # Hidden dimension
    activation: callable
    
    def __init__(self, data_dim, hidden_dim, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)):
        """
        Initialize the autoencoder.
        
        Args:
            data_dim: Dimension of input/output (d)
            hidden_dim: Hidden dimension (h)
            activation: Activation function (default: tanh)
            key: JAX random key
        """
        self.d = data_dim
        self.h = hidden_dim
        self.activation = activation
        
        key_W1, key_W2, key_b = jax.random.split(key, 3)
        
        # Initialize W: shape (d, h)
        # Scale by 1/sqrt(d) for stable initialization
        self.W1 = jax.random.normal(key_W1, (hidden_dim, data_dim)) / jnp.sqrt(data_dim)
        self.W2 = jax.random.normal(key_W2, (data_dim, hidden_dim)) / jnp.sqrt(data_dim)
        
        # Initialize b to identity (ones)
        self.b = jnp.ones(data_dim)
    
    def __call__(self, t, x):
        """
        Forward pass: f(x) = b*x + (W/sqrt(h)) * sigma((W^T x)/sqrt(d))
        
        Args:
            x: Input array, shape (d,)
        
        Returns:
            Output array, shape (d,)
        """
        # Encoder: W^T x, shape (h,)
        encoded = jnp.dot(self.W1, x)  # (h, d) @ (d,) = (h,)
        
        # Normalize by sqrt(d)
        encoded_normalized = encoded / jnp.sqrt(self.d)
        
        # Apply activation: shape (h,)
        activated = self.activation(encoded_normalized)
        
        # Decoder: W @ activated, shape (d,)
        # Scale by 1/sqrt(h)
        scaled_W2 = self.W2 / jnp.sqrt(self.h)
        decoded = scaled_W2 @ activated  # (d, h) @ (h,) = (d,)
        
        # Linear term: b * x (element-wise)
        linear_term = self.b * x
        
        # Combine
        output = linear_term + decoded
        
        return output


class SharedWeightAutoencoder(eqx.Module):
    """
    Autoencoder with shared weight vector w across encoder/decoder.
    
    f(x) = bx + (w/sqrt(h)) * sigma((W^T x)/sqrt(d))
    
    where:
    - b: learnable bias, shape (d,)
    - W: learnable weight matrix, shape (d, h) — shared encoder/decoder
    - d: input dimension
    - h: hidden dimension
    - sigma: activation function
    """
    
    W: jax.Array  # Shared weight matrix, shape (d, h)
    b: jax.Array  # Bias, shape (d,)
    d: int        # Input dimension
    h: int        # Hidden dimension
    # activation: callable
    
    def __init__(self, input_dim, hidden_dim, activation, key):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Dimension of input/output (d)
            hidden_dim: Hidden dimension (h)
            activation: Activation function (default: tanh)
            key: JAX random key
        """
        self.d = input_dim
        self.h = hidden_dim
        # self.activation = activation
        
        key_W, key_b = jax.random.split(key)
        
        # Initialize W: shape (d, h)
        # Scale by 1/sqrt(d) for stable initialization
        self.W = jax.random.normal(key_W, (input_dim, hidden_dim)) / jnp.sqrt(input_dim)
        
        # Initialize b to identity (ones)
        self.b = jnp.ones(input_dim)
    
    def __call__(self, t, x):
        """
        Forward pass: f(x) = b*x + (W/sqrt(h)) * sigma((W^T x)/sqrt(d))
        
        Args:
            x: Input array, shape (d,)
        
        Returns:
            Output array, shape (d,)
        """
        # Encoder: W^T x, shape (h,)
        encoded = jnp.dot(self.W.T, x)  # (d, h)^T @ (d,) = (h,)
        
        # Normalize by sqrt(d)
        encoded_normalized = encoded / jnp.sqrt(self.d)
        
        # Apply activation: shape (h,)
        # activated = self.activation(encoded_normalized)
        activated = jax.nn.softplus(encoded_normalized)
        
        # Decoder: W @ activated, shape (d,)
        # Scale by 1/sqrt(h)
        scaled_W = self.W / jnp.sqrt(self.h)
        decoded = scaled_W @ activated  # (d, h) @ (h,) = (d,)
        
        # Linear term: b * x (element-wise)
        linear_term = self.b * x
        
        # Combine
        output = linear_term + decoded
        
        return output