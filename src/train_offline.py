# src/train_offline.py
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import numpy as np
import os
from dataclasses import dataclass
from tqdm import tqdm

# Import the established model
from src.model import NREClassifier

from src.train_config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    VAL_SPLIT, SEED, CKPT_DIR
)
from src.sim_config import GRID_SIZE, CHANNELS

@dataclass
class OfflineConfig:
    data_path: str = f"data/dataset_{GRID_SIZE}.npz" 
    batch_size: int = BATCH_SIZE
    n_epochs: int = EPOCHS        
    learning_rate: float = LEARNING_RATE
    seed: int = SEED
    val_split: float = VAL_SPLIT   # portion of verification dataset
    ckpt_dir: str = CKPT_DIR


def create_train_state(rng, learning_rate, input_shape):
    """Initializes model and optimizer."""
    model = NREClassifier()
    
    # Dummy inputs for shape inference
    dummy_x = jnp.ones((1, *input_shape)) 
    # 3 parameters [eta, B, nu]
    dummy_theta = jnp.ones((1, 3))
    
    variables = model.init(rng, dummy_x, dummy_theta)
    tx = optax.adamw(learning_rate, weight_decay=1e-4)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )

@jax.jit
def train_step(state, batch_x, batch_theta):
    """
    Contrastive training step (Same logic as online, but JIT-ed for speed).
    """
    batch_size = batch_x.shape[0]
    
    # === Label smoothing ===
    smoothing = 0.1  
    
    # Positive pairs
    pos_x = batch_x
    pos_theta = batch_theta
    pos_labels = jnp.ones((batch_size, 1)) * (1 - smoothing) + smoothing * 0.5  # 0.95 instead of 1.0
    
    # Negative pairs (Shuffle theta)
    neg_x = batch_x
    neg_theta = jnp.roll(batch_theta, shift=1, axis=0)
    neg_labels = jnp.zeros((batch_size, 1)) + smoothing * 0.5  # 0.05 instead of 0.0
    
    # Combine
    train_x = jnp.concatenate([pos_x, neg_x], axis=0)
    train_theta = jnp.concatenate([pos_theta, neg_theta], axis=0)
    train_labels = jnp.concatenate([pos_labels, neg_labels], axis=0)
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, train_x, train_theta)
        loss = optax.sigmoid_binary_cross_entropy(logits, train_labels).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@jax.jit
def eval_step(state, batch_x, batch_theta):
    """
    Evaluation step (Evaluates both Positive and Negative pairs).
    """
    batch_size = batch_x.shape[0]
    
    
    pos_x = batch_x
    pos_theta = batch_theta
    pos_labels = jnp.ones((batch_size, 1))
    
    
    neg_x = batch_x
    neg_theta = jnp.roll(batch_theta, shift=1, axis=0)
    neg_labels = jnp.zeros((batch_size, 1))
    
   
    eval_x = jnp.concatenate([pos_x, neg_x], axis=0)
    eval_theta = jnp.concatenate([pos_theta, neg_theta], axis=0)
    eval_labels = jnp.concatenate([pos_labels, neg_labels], axis=0)
    
    
    logits = state.apply_fn({'params': state.params}, eval_x, eval_theta)
    loss = optax.sigmoid_binary_cross_entropy(logits, eval_labels).mean()
    
    
    preds = jax.nn.sigmoid(logits) > 0.5
    accuracy = jnp.mean(preds == eval_labels)
    
    return loss, accuracy

def monitor_B_sensitivity(state, val_ds, batch_size=32):
    """
    Test the sensitivity of the model in B
    How: Pick a set of samples with very close eta values
    The model should distinguish the data based on B
    """
    # 1. Pick the sample with eta around 0.8 
    etas = val_ds['theta'][:, 0]
    mask = (etas > 0.75) & (etas < 0.85)
    
    if np.sum(mask) < batch_size:
        return 0.5 
        
    
    sub_x = val_ds['x'][mask][:batch_size]
    sub_theta = val_ds['theta'][mask][:batch_size]
    
    # 2. eval_step
    loss, acc = eval_step(state, jnp.array(sub_x), jnp.array(sub_theta))
    return acc

def load_data(path, val_split=0.1, seed=42):
    """Loads and splits the .npz dataset."""
    print(f"📂 Loading data from {path}...")
    data = np.load(path)
    theta = data['theta']
    x = data['x']
    
    n_total = theta.shape[0]
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    # Shuffle
    np.random.seed(seed)
    perm = np.random.permutation(n_total)
    theta = theta[perm]
    x = x[perm]
    
    # Split
    train_ds = {'theta': theta[:n_train], 'x': x[:n_train]}
    val_ds = {'theta': theta[n_train:], 'x': x[n_train:]}
    
    print(f" Loaded {n_total} samples. Train: {n_train}, Val: {n_val}")
    return train_ds, val_ds

def get_batch(ds, batch_size, idx):
    """Simple batch getter."""
    batch_theta = ds['theta'][idx : idx + batch_size]
    batch_x = ds['x'][idx : idx + batch_size]
    return jnp.array(batch_x), jnp.array(batch_theta)

if __name__ == "__main__":
    cfg = OfflineConfig()
    
    cfg.ckpt_dir = os.path.abspath(cfg.ckpt_dir)
    
    # 1. Load Data
    if not os.path.exists(cfg.data_path):
        print(f"Data file not found: {cfg.data_path}")
        print("Please run 'python generate_data.py' first!")
        exit()
        
    train_ds, val_ds = load_data(cfg.data_path, cfg.val_split, cfg.seed)
    
    # Check if information from B-field is too weak
    x_sample = train_ds['x'][:100] # averaged over 100 data points
    print("\n Data Stats Check:")
    print(f"   Density Mean: {np.mean(x_sample[..., :2]):.4f}")
    print(f"   Curl Mean:    {np.mean(x_sample[..., 2]):.4f}")
    print(f"   Curl Max:     {np.max(x_sample[..., 2]):.4f}")
    print(f"   Curl Min:     {np.min(x_sample[..., 2]):.4f}")
    print("--------------------------------------------------\n")
    
    # 2. Setup Model
    rng = jax.random.PRNGKey(cfg.seed)
    input_shape = train_ds['x'].shape[1:] # (128, 128, 3)
    state = create_train_state(rng, cfg.learning_rate, input_shape)
    
    print(f"Starting Offline Training | Epochs: {cfg.n_epochs} | Batch: {cfg.batch_size}")
    
    # 3. Training Loop
    steps_per_epoch = len(train_ds['theta']) // cfg.batch_size
    
    for epoch in range(cfg.n_epochs):
        # Shuffle training data each epoch
        perm = np.random.permutation(len(train_ds['theta']))
        train_ds['theta'] = train_ds['theta'][perm]
        train_ds['x'] = train_ds['x'][perm]
        
        epoch_loss = []
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{cfg.n_epochs}")
        for i in pbar:
            idx = i * cfg.batch_size
            batch_x, batch_theta = get_batch(train_ds, cfg.batch_size, idx)
            
            state, loss = train_step(state, batch_x, batch_theta)
            epoch_loss.append(loss)
            
            pbar.set_postfix(loss=f"{loss:.4f}")
        
        for i in range(steps_per_epoch):
            idx = i * cfg.batch_size
            batch_x, batch_theta = get_batch(train_ds, cfg.batch_size, idx)
            
            state, loss = train_step(state, batch_x, batch_theta)
            epoch_loss.append(loss)
            
        # Validation
        print("  Running Validation...", flush=True)
        val_loss = []
        val_acc = []
        val_steps = len(val_ds['theta']) // cfg.batch_size
        for i in range(val_steps):
            idx = i * cfg.batch_size
            b_x, b_t = get_batch(val_ds, cfg.batch_size, idx)
            l, acc = eval_step(state, b_x, b_t)
            val_loss.append(l)
            val_acc.append(acc)
        
        val_acc_b = monitor_B_sensitivity(state, val_ds, cfg.batch_size)    
        print(f"Epoch {epoch+1:02d} | Train Loss: {np.mean(epoch_loss):.4f} | Val Loss: {np.mean(val_loss):.4f} | Val Acc: {np.mean(val_acc):.2f} | B-Sens Acc: {val_acc_b:.2f}")
        
        # Save Checkpoint (Best model logic can be added here)
        # if (epoch + 1) % 10 == 0:
        checkpoints.save_checkpoint(ckpt_dir=cfg.ckpt_dir, target=state, step=epoch, overwrite=True)

    print("🎉 Training Complete. Model saved.")