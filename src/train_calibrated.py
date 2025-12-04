import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import numpy as np
import os
from tqdm import tqdm

# Reuse your existing model and config structure
from src.model import NREClassifier
from src.train_config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    VAL_SPLIT, SEED, CKPT_DIR
)
# We need to access min/max for proper normalization
from src.sim_config import ETA_MIN, ETA_MAX, B_MIN, B_MAX, NU_MIN, NU_MAX

def normalize_theta(theta):
    """
    Min-Max Normalization to [-1, 1] range.
    """
    # theta shape: (..., 3) -> [eta, B, nu]
    eta = theta[..., 0]
    B   = theta[..., 1]
    nu  = theta[..., 2]
    
    eta_norm = 2 * (eta - ETA_MIN) / (ETA_MAX - ETA_MIN) - 1
    B_norm   = 2 * (B - B_MIN) / (B_MAX - B_MIN) - 1
    nu_norm  = 2 * (nu - NU_MIN) / (NU_MAX - NU_MIN) - 1
    
    return jnp.stack([eta_norm, B_norm, nu_norm], axis=-1)

@jax.jit
def train_step_clean(state, batch_x, batch_theta, rng):
    """
    Strict NRE training step WITHOUT Label Smoothing.
    """
    batch_size = batch_x.shape[0]
    
    # 1. Positive Pairs (Joint Distribution)
    pos_x = batch_x
    pos_theta = batch_theta
    pos_labels = jnp.ones((batch_size, 1)) # Strict 1.0
    
    # 2. Negative Pairs (Product of Marginals)
    rng, subkey = jax.random.split(rng)
    perm = jax.random.permutation(subkey, batch_size)
    neg_theta = batch_theta[perm]
    neg_x = batch_x
    neg_labels = jnp.zeros((batch_size, 1)) # Strict 0.0
    
    # Concatenate
    train_x = jnp.concatenate([pos_x, neg_x], axis=0)
    train_theta = jnp.concatenate([pos_theta, neg_theta], axis=0)
    train_labels = jnp.concatenate([pos_labels, neg_labels], axis=0)
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, train_x, train_theta)
        loss = optax.sigmoid_binary_cross_entropy(logits, train_labels).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, rng

def create_train_state(rng, learning_rate, input_shape):
    model = NREClassifier()
    dummy_x = jnp.ones((1, *input_shape)) 
    dummy_theta = jnp.ones((1, 3))
    variables = model.init(rng, dummy_x, dummy_theta)
    tx = optax.adamw(learning_rate, weight_decay=1e-4)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=tx,
    )

def main():
    print("--- Starting Calibrated Training (No Label Smoothing) ---")
    
    # [FIX] Convert checkpoint directory to absolute path
    abs_ckpt_dir = os.path.abspath(CKPT_DIR)
    if not os.path.exists(abs_ckpt_dir):
        os.makedirs(abs_ckpt_dir)
    print(f"Checkpoints will be saved to: {abs_ckpt_dir}")
    
    # Check data paths
    possible_paths = [f"data/dataset_{64}.npz", f"data/dataset_{32}.npz"]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if data_path is None:
        print("Data not found. Please run generate_data.py first.")
        return

    print(f"Loading data from {data_path}...")
    # Use mmap_mode to avoid loading everything into RAM instantly
    try:
        data = np.load(data_path, mmap_mode='r')
        theta = data['theta']
        x = data['x']
        print(f"Data loaded. Shape: {x.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Train/Val Split
    n_total = len(theta)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val
    
    # Init Model
    rng = jax.random.PRNGKey(SEED)
    state = create_train_state(rng, LEARNING_RATE, x.shape[1:])
    
    steps_per_epoch = n_train // BATCH_SIZE
    best_val_loss = float('inf')
    
    print(f"Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        # Epoch setup
        rng, shuffle_key = jax.random.split(rng)
        
        # Shuffle indices
        perm = np.random.permutation(n_train)
        
        epoch_losses = []
        
        # Training Loop
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for i in pbar:
            # Batch indices
            batch_idx = perm[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            
            # Load batch into RAM now
            batch_x = jnp.array(x[batch_idx])
            batch_theta = jnp.array(theta[batch_idx])
            
            state, loss, rng = train_step_clean(state, batch_x, batch_theta, rng)
            epoch_losses.append(loss)
            pbar.set_postfix(loss=f"{loss:.4f}")
            
        # Validation Loop
        val_loss_acc = 0
        val_steps = min(len(range(n_val // BATCH_SIZE)), 50) 
        
        for i in range(val_steps):
            idx_start = n_train + i*BATCH_SIZE
            idx_end = n_train + (i+1)*BATCH_SIZE
            
            bx = jnp.array(x[idx_start:idx_end])
            bt = jnp.array(theta[idx_start:idx_end])
            
            pos_logits = state.apply_fn({'params': state.params}, bx, bt)
            l1 = optax.sigmoid_binary_cross_entropy(pos_logits, jnp.ones_like(pos_logits)).mean()
            
            bt_neg = jnp.roll(bt, 1, axis=0)
            neg_logits = state.apply_fn({'params': state.params}, bx, bt_neg)
            l2 = optax.sigmoid_binary_cross_entropy(neg_logits, jnp.zeros_like(neg_logits)).mean()
            
            val_loss_acc += (l1 + l2) / 2
            
        val_loss = val_loss_acc / val_steps if val_steps > 0 else 0
        train_loss = np.mean(epoch_losses)
        
        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save Checkpoint with absolute path
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoints.save_checkpoint(
                ckpt_dir=abs_ckpt_dir, 
                target=state, 
                step=epoch, 
                prefix="calibrated_", 
                keep=1, 
                overwrite=True
            )
            
    print("Training Done. Best model saved.")

if __name__ == "__main__":
    main()