import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import numpy as np
import os
from dataclasses import dataclass

# Import the established model
from src.model import NREClassifier

@dataclass
class OfflineConfig:
    data_path: str = "data/dataset_128.npz" # Aim to data file
    batch_size: int = 64      # Offline training permits larger Batch
    n_epochs: int = 50        
    learning_rate: float = 1e-3
    seed: int = 42
    val_split: float = 0.1    # portion of verification dataset
    ckpt_dir: str = "checkpoints" 

def create_train_state(rng, learning_rate, input_shape):
    """Initializes model and optimizer."""
    model = NREClassifier()
    
    # Dummy inputs for shape inference
    dummy_x = jnp.ones((1, *input_shape)) # (1, 128, 128, 2)
    dummy_theta = jnp.ones((1, 2))
    
    variables = model.init(rng, dummy_x, dummy_theta)
    tx = optax.adam(learning_rate)
    
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
    
    # Positive pairs
    pos_x = batch_x
    pos_theta = batch_theta
    pos_labels = jnp.ones((batch_size, 1))
    
    # Negative pairs (Shuffle theta)
    neg_x = batch_x
    neg_theta = jnp.roll(batch_theta, shift=1, axis=0)
    neg_labels = jnp.zeros((batch_size, 1))
    
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
    """Evaluation step (No gradients)."""
    batch_size = batch_x.shape[0]
    pos_labels = jnp.ones((batch_size, 1))
    
    # Evaluation only on positive pairs for simplicity (or standard constrastive accuracy)
    logits = state.apply_fn({'params': state.params}, batch_x, batch_theta)
    loss = optax.sigmoid_binary_cross_entropy(logits, pos_labels).mean()
    
    # Accuracy (Prediction > 0.5 is True)
    preds = jax.nn.sigmoid(logits) > 0.5
    accuracy = jnp.mean(preds == pos_labels)
    return loss, accuracy

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
    
    print(f"✅ Loaded {n_total} samples. Train: {n_train}, Val: {n_val}")
    return train_ds, val_ds

def get_batch(ds, batch_size, idx):
    """Simple batch getter."""
    batch_theta = ds['theta'][idx : idx + batch_size]
    batch_x = ds['x'][idx : idx + batch_size]
    return jnp.array(batch_x), jnp.array(batch_theta)

if __name__ == "__main__":
    cfg = OfflineConfig()
    
    # 1. Load Data
    if not os.path.exists(cfg.data_path):
        print(f"Data file not found: {cfg.data_path}")
        print("Please run 'python generate_data.py' first!")
        exit()
        
    train_ds, val_ds = load_data(cfg.data_path, cfg.val_split, cfg.seed)
    
    # 2. Setup Model
    rng = jax.random.PRNGKey(cfg.seed)
    input_shape = train_ds['x'].shape[1:] # (128, 128, 2)
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
        
        for i in range(steps_per_epoch):
            idx = i * cfg.batch_size
            batch_x, batch_theta = get_batch(train_ds, cfg.batch_size, idx)
            
            state, loss = train_step(state, batch_x, batch_theta)
            epoch_loss.append(loss)
            
        # Validation
        val_loss = []
        val_acc = []
        val_steps = len(val_ds['theta']) // cfg.batch_size
        for i in range(val_steps):
            idx = i * cfg.batch_size
            b_x, b_t = get_batch(val_ds, cfg.batch_size, idx)
            l, acc = eval_step(state, b_x, b_t)
            val_loss.append(l)
            val_acc.append(acc)
            
        print(f"Epoch {epoch+1:02d} | Train Loss: {np.mean(epoch_loss):.4f} | Val Loss: {np.mean(val_loss):.4f} | Val Acc: {np.mean(val_acc):.2f}")
        
        # Save Checkpoint (Best model logic can be added here)
        if (epoch + 1) % 10 == 0:
            checkpoints.save_checkpoint(ckpt_dir=cfg.ckpt_dir, target=state, step=epoch, overwrite=True)

    print("🎉 Training Complete. Model saved.")