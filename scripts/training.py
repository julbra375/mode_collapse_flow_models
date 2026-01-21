import equinox as eqx
from tqdm import tqdm
import jax
import jax.numpy as jnp
from scripts.utils.analytic_gradient_calculation import compute_analytic_gradient_2modes, compute_analytic_gradient_Mmodes, compute_analytic_gradient_batch
from scripts.losses import reconstruction_loss_koopman

def train_CNF(model, sample_dataset_fn, loss_fn, optimizer, key, training_iterations=1000, calc_rank_fn=None, save_weights_and_grads=False, save_biases=False):
    """
    Train a continuous normalizing flow model 

    Args:
        model, model to train args=x
        sample_dataset_fn, returns batch of samples from target distribution args=(key)
        loss_fn, computes loss for batch of samples args=(model, samples, key)
        optimizer, optax optimizer
        key, PRNG jax key
        training_iterations: integer
    """

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    iterator = tqdm(range(training_iterations))

    losses = []
    weights_rank = []
    weights = []
    biases = []
    gradients = []
    
    for i, iteration in enumerate(iterator):
        key, data_key, loss_key = jax.random.split(key, 3)

        target_xs = sample_dataset_fn(data_key)

        # Compute gradients
        current_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, target_xs, loss_key)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        
        # Save weights and gradients
        current_weights = jax.tree.leaves(eqx.filter(model, eqx.is_array))

        if save_weights_and_grads:
            if save_biases:
                weights.append(current_weights[0])
                grads_flat = jax.tree_util.tree_leaves(grads)[0]
                gradients.append(grads_flat)
            else:
                weights.append(current_weights[0])
                grads_flat = jax.tree_util.tree_leaves(grads)[0]
                gradients.append(grads_flat)
    
        # Save biases
        if save_biases:
            biases.append(current_weights[1])

        # Save weight ranks
        weights_rank.append(calc_rank_fn(current_weights))

        # Save loss
        losses.append(current_loss)

        iterator.set_description(f'Loss: {current_loss:.5f}')


    losses = jnp.array(losses)
    weights_rank = jnp.array(weights_rank)
    weights = jnp.array(weights)
    biases = jnp.array(biases)
    gradients = jnp.array(gradients)

    return model, losses, weights, biases, weights_rank, gradients
   

def train_CNF_parallel(models, sample_dataset_fn, loss_fn, optimizer, key, 
                       calc_rank_parallel, initial_modes_batch, target_modes_batch, training_iterations=1000, 
                       save_weights_and_grads=True, verbose=True):
    """
    Train multiple CNF models in parallel using vmap over keys.
    
    Args:
        models: Array of models, shape (num_models,) — each is a pytree
        sample_dataset_fn: Function taking key, returns batch of samples
        loss_fn: Function (model, target_xs, loss_key) -> loss
        optimizer: optax optimizer
        key: PRNG jax key
        training_iterations: Number of training steps
        track_weights_interval: Interval for tracking weights
        save_model_interval: Interval for saving models
        verbose: Whether to print progress
    
    Returns:
        models: Trained models, shape (num_models,)
        losses: Loss history, shape (training_iterations, num_models)
        weights: Tracked weights history
    """
    num_models = jax.tree.leaves(models)[0].shape[0]
    
    # Initialize optimizer states for all models
    opt_states = jax.vmap(lambda m: optimizer.init(eqx.filter(m, eqx.is_array)))(models)
    
    losses_history = []
    rank_history = []
    weights_history = []
    grads_history = []
    
    iterator = tqdm(range(training_iterations)) if verbose else range(training_iterations)
    
    for i in iterator:
        # Split keys for each model
        key, *subkeys = jax.random.split(key, num_models + 1)
        data_keys = jnp.array(subkeys[:num_models])
        loss_keys = jax.random.split(key, num_models)
        
        # Sample data for each model (can be same or different)
        initial_xs_all = jax.vmap(sample_dataset_fn)(data_keys, initial_modes_batch)  # (num_models, batch_size, d)
        
        # Define single training step
        def train_step(model, opt_state, initial_xs, loss_key, target_modes):
            """Train step for a single model"""
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model, initial_xs, loss_key, target_modes)
            updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, grads
        
        # vmap over all models
        models, opt_states, current_losses, grads = jax.vmap(train_step)(
            models, opt_states, initial_xs_all, loss_keys, target_modes_batch
        )
        
        losses_history.append(current_losses)
        
        # Track weights
        if save_weights_and_grads:
            current_weights = jax.vmap(lambda model: jax.tree.leaves(eqx.filter(model, eqx.is_array))[0])(models)
            weights_history.append(current_weights)
            ranks_parallel = jax.vmap(calc_rank_parallel)(current_weights)
            rank_history.append(ranks_parallel)
            grads_flat = jax.vmap(lambda grad: jax.tree_util.tree_leaves(grad)[0])(grads)
            grads_history.append(grads_flat)
        
        if verbose:
            mean_loss = jnp.mean(current_losses)
            std_loss = jnp.std(current_losses)
            iterator.set_description(f'Mean Loss: {mean_loss:.5f} ± {std_loss:.5f}')
    

    losses_history = jnp.array(losses_history)  # (training_iterations, num_models)
    ranks_history = jnp.array(rank_history)
    weights_history = jnp.array(weights_history)
    grads_history = jnp.array(grads_history)

    return models, losses_history, weights_history, ranks_history, grads_history


def compute_analytic_grads_tree(model, initial_modes, target_modes, key, num_samples=10000):
    """
    Computes the analytic gradient for the weight matrix and returns a PyTree 
    of gradients matching the model structure.
    """
    # 1. Extract the weight matrix W. 
    W = model.W
    
    # 2. Compute the analytic gradient matrix
    raw_grad_W = compute_analytic_gradient_Mmodes(
        W, initial_modes, target_modes, key, num_samples
    )

    # 3. Create a PyTree of zeros matching the model structure
    grads = jax.tree_util.tree_map(jnp.zeros_like, model)
    
    # 4. Inject the computed gradient into the correct leaf
    grads = eqx.tree_at(lambda m: m.W, grads, raw_grad_W)
    
    return grads

def train_CNF_analytic(model, initial_modes, target_modes, optimizer, key, 
                       training_iterations=1000, num_samples=10000, calc_rank_fn=None, 
                       save_weights_and_grads=False, save_biases=False):
    """
    Train a CNF model using analytic gradients (no ODE solver backprop).
    """

    # Initialize optimizer
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    iterator = tqdm(range(training_iterations))

    losses = []
    weights_rank = []
    weights_store = []
    biases_store = []
    gradients_store = []

    # --- Define the Step Function (JIT Compiled) ---
    @eqx.filter_jit
    def step(model, opt_state, key):
        # 1. Compute Gradients Analytically
        grads = compute_analytic_grads_tree(
            model, initial_modes, target_modes, key, num_samples=num_samples
        )
        
        # 2. Apply Updates using Optax
        updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        
        # Return grads so we can log them if needed
        return new_model, new_opt_state, grads

    # --- Training Loop ---
    for i in iterator:
        key, step_key = jax.random.split(key)

        # Perform one update step
        model, opt_state, grads = step(model, opt_state, step_key)
        
        # --- Logging and Storage ---
        
        # Since we aren't calculating loss, we log 0.0 or Gradient Norm as a proxy
        grad_norm = jnp.linalg.norm(jax.tree_util.tree_leaves(grads)[0])
        losses.append(grad_norm)
        
        if save_weights_and_grads:
            # Extract leaves (Assumes index 0 is Weight, index 1 is Bias if present)
            params = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
            grad_leaves = jax.tree_util.tree_leaves(grads)
            
            weights_store.append(params[0]) 
            gradients_store.append(grad_leaves[0])
            
            if save_biases and len(params) > 1:
                biases_store.append(params[1])

        if calc_rank_fn:
            params = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
            weights_rank.append(calc_rank_fn(params))

        # Update progress bar with Grad Norm instead of Loss
        iterator.set_description(f'Grad Norm: {grad_norm:.5f}')

    # Convert lists to JAX arrays
    return (model, jnp.array(losses), jnp.array(weights_store), 
        jnp.array(biases_store), jnp.array(weights_rank), jnp.array(gradients_store))

def train_CNF_analytic_parallel(
    models, 
    initial_modes_batch, 
    target_modes_batch, 
    optimizer, 
    key, 
    calc_rank_parallel=None, 
    training_iterations=1000, 
    num_samples=10000, 
    save_weights_and_grads=True, 
    save_biases=False,
    verbose=True
):
    """
    Train multiple CNF models in parallel using analytic gradients (vmapped).
    
    Args:
        models: PyTree of models with leading batch dimension (num_models, ...)
        initial_modes_batch: Batch of initial modes (num_models, ...)
        target_modes_batch: Batch of target modes (num_models, ...)
        optimizer: optax optimizer
        key: JAX PRNG key
        calc_rank_parallel: Function to calculate rank on a batch of weights
    """
    
    # 1. Determine number of models from the batch dimension of the first leaf
    num_models = jax.tree_util.tree_leaves(models)[0].shape[0]

    # 2. Initialize optimizer states for all models (vmapped)
    # We map the init function over the batch of models
    opt_states = jax.vmap(lambda m: optimizer.init(eqx.filter(m, eqx.is_array)))(models)

    # Storage lists
    grad_norms_history = []
    weights_history = []
    biases_history = []
    grads_history = []
    rank_history = []

    # --- Define Single Analytic Step ---
    def train_step_analytic(model, opt_state, step_key, init_modes, targ_modes):
        """
        Performs one analytic update step for a single model.
        This function will be vmapped below.
        """
        # 1. Compute Gradients Analytically using your provided function
        # Note: We rely on compute_analytic_grads_tree being available in scope
        grads = compute_analytic_grads_tree(
            model, init_modes, targ_modes, step_key, num_samples=num_samples
        )
        
        # 2. Apply Updates using Optax
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        
        return new_model, new_opt_state, grads

    # --- Define Parallel Step (JIT Compiled) ---
    @eqx.filter_jit
    def step_parallel(models, opt_states, step_keys, init_modes_batch, targ_modes_batch):
        # We vmap over: models, opt_states, keys, and the mode batches
        return jax.vmap(train_step_analytic)(
            models, opt_states, step_keys, init_modes_batch, targ_modes_batch
        )

    # --- Training Loop ---
    iterator = tqdm(range(training_iterations)) if verbose else range(training_iterations)

    for i in iterator:
        # Split keys: one main key for next iter, and `num_models` keys for the batch
        key, step_key_base = jax.random.split(key)
        step_keys = jax.random.split(step_key_base, num_models)

        # Perform parallel update
        models, opt_states, grads = step_parallel(
            models, opt_states, step_keys, initial_modes_batch, target_modes_batch
        )

        # --- Logging ---
        # Calculate Grad Norms (proxy for loss) for all models
        # We assume the first leaf is the Weight matrix (similar to your single loop)
        current_grad_norms = jax.vmap(lambda g: jnp.linalg.norm(jax.tree_util.tree_leaves(g)[0]))(grads)
        grad_norms_history.append(current_grad_norms)

        if save_weights_and_grads:
            # Extract weights (index 0) and biases (index 1 if exists)
            # This creates a batch of leaves: (num_models, weight_shape)
            params_batch = jax.vmap(lambda m: jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array)))(models)
            grad_leaves_batch = jax.vmap(jax.tree_util.tree_leaves)(grads)

            # Store Weights (index 0)
            weights_history.append(params_batch[0]) 
            
            # Store Gradients (index 0 - Weights)
            grads_history.append(grad_leaves_batch[0])

            # Store Biases (index 1) if requested and present
            if save_biases and len(params_batch) > 1:
                biases_history.append(params_batch[1])

            # Calculate Ranks if function provided
            if calc_rank_parallel:
                # Assumes calc_rank_parallel takes a batch of weights and returns batch of ranks
                rank_history.append(calc_rank_parallel(params_batch[0]))

        # Update Progress Bar
        if verbose:
            mean_norm = jnp.mean(current_grad_norms)
            iterator.set_description(f'Mean Grad Norm: {mean_norm:.5f}')

    # --- Formatting Returns ---
    grad_norms_history = jnp.array(grad_norms_history) # (iterations, num_models)
    weights_history = jnp.array(weights_history)       # (iterations, num_models, ...)
    grads_history = jnp.array(grads_history)           # (iterations, num_models, ...)
    
    # Return tuple matching your preferred style
    return (
        models, 
        grad_norms_history, 
        weights_history, 
        jnp.array(biases_history) if save_biases else None,
        jnp.array(rank_history) if calc_rank_parallel else None,
        grads_history
    )

def train_KoopmanAE(model, source_data, target_data, optimizer, key,
                    batch_size=256, training_iterations=5000, 
                    lambda_recon=1.0, lambda_anchor=2.0, lambda_analytic=1.0,
                    lambda_analytic_switch=2000):
    """
    Train a Koopman Autoencoder with Hybrid Gradients.
    Args:
        source_data: Array of shape (N_source, input_dim)
        target_data: Array of shape (N_target, input_dim)
    """
    
    # Initialize Optimizer
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Storage
    losses = []
    grads = []
    
    # --- 1. Define Loss Function (Standard Backprop Component) ---
    def reconstruction_loss_fn(model, x_s, x_t):
        # 1. Encode
        z_s = jax.vmap(model.encoder)(x_s)
        z_t = jax.vmap(model.encoder)(x_t)
        
        # 2. Decode (Standard Reconstruction)
        x_s_recon = jax.vmap(model.decoder)(z_s)
        x_t_recon = jax.vmap(model.decoder)(z_t)
        
        loss_recon = jnp.mean((x_s - x_s_recon)**2) + jnp.mean((x_t - x_t_recon)**2)
        
        target_pos = model.latent_targets[0] # Shape (D,)
        target_neg = model.latent_targets[1] # Shape (D,)
        
        # Calculate distance to BOTH anchors for every point in batch
        # z_t shape: (Batch, D)
        dist_pos = jnp.sum((z_t - target_pos)**2, axis=1) # (Batch,)
        dist_neg = jnp.sum((z_t - target_neg)**2, axis=1) # (Batch,)
        
        # Take the minimum (Greedy assignment: map to the closest anchor)
        loss_anchor = jnp.mean(jnp.minimum(dist_pos, dist_neg))
        
        # Total Loss
        total_loss = lambda_recon * loss_recon + lambda_anchor * loss_anchor
        
        return total_loss, (z_s, z_t)

    # --- 2. Define The Hybrid Step ---
    @eqx.filter_jit
    def step(model, opt_state, x_s_batch, x_t_batch, lambda_analytic):
        # -- Phase 1: Standard Gradients (Reconstruction) --
        (loss, (z_s, z_t)), grads = eqx.filter_value_and_grad(
            reconstruction_loss_fn, has_aux=True
        )(model, x_s_batch, x_t_batch)
        
        # -- Phase 2: Analytic Gradient (Latent Dynamics) --
        # We stop gradients here because we only want to update W based on the *current* geometry
        z_s_detached = jax.lax.stop_gradient(z_s)
        z_t_detached = jax.lax.stop_gradient(z_t)
        
        latent_targets = model.latent_targets 
        
        # Compute the specific gradient for W
        w_analytic_grad = compute_analytic_gradient_batch(
            model.flow.W, 
            z_s_detached, 
            latent_targets
        )
        
        # -- Phase 3: Hybrid Injection --
        # Add the analytic gradient to the existing gradient for W
        # We perform the manual addition: grad_W_total = grad_W_recon + lambda * grad_W_analytic
        
        grads = eqx.tree_at(
            lambda g: g.flow.W,  # Target the W matrix in the gradient tree
            grads,
            grads.flow.W + lambda_analytic * w_analytic_grad # Injection
        )

        # -- Phase 4: Update --
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        
        return new_model, new_opt_state, loss, grads

    # --- 3. Training Loop ---
    iterator = tqdm(range(training_iterations))
    
    n_source = source_data.shape[0]
    n_target = target_data.shape[0]
    
    for i in iterator:
        if i < lambda_analytic_switch:
            lambda_analytic_current = 0
        else:
            lambda_analytic_current = lambda_analytic

        key, k_s, k_t = jax.random.split(key, 3)
        
        # Sample Batches
        idx_s = jax.random.choice(k_s, n_source, shape=(batch_size,))
        idx_t = jax.random.choice(k_t, n_target, shape=(batch_size,))
        
        x_s_batch = source_data[idx_s]
        x_t_batch = target_data[idx_t]
        
        # Perform Step
        model, opt_state, loss_val, grads_current = step(model, opt_state, x_s_batch, x_t_batch, lambda_analytic_current)
        
        losses.append(loss_val)
        grads.append(grads_current)
        
        if i % 10 == 0:
            iterator.set_description(f'Loss: {loss_val:.5f}')

    return model, jnp.array(losses), grads

