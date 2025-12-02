# src/train_drag.py
import jax
import jax.numpy as jnp
import optax
import numpy as np
import os
from flax.training import train_state, checkpoints
from src.model import NREClassifier
from src.train_offline import create_train_state, train_step


DATA_PATH = "data/dataset_drag_t1.npz" 
CKPT_DIR = "checkpoints_drag"          
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4

def main():
    if not os.path.exists(DATA_PATH):
        print("Data missing! Run generate_drag_data.py first.")
        return
        
    print(f"--- 🧠 Training Drag Detector (on T=1 data) ---")
    data = np.load(DATA_PATH)
    theta = jnp.array(data['theta']) # (N, 3)
    x = jnp.array(data['x'])         # (N, 64, 64, 3)
    
    # Init Model
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, LEARNING_RATE, x.shape[1:])
    
    # Training Loop
    steps_per_epoch = len(theta) // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        # Simple shuffling
        perm = jax.random.permutation(jax.random.fold_in(rng, epoch), len(theta))
        theta = theta[perm]
        x = x[perm]
        
        epoch_loss = []
        for i in range(steps_per_epoch):
            batch_x = x[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            batch_theta = theta[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            state, loss = train_step(state, batch_x, batch_theta)
            epoch_loss.append(loss)
            
        print(f"Epoch {epoch+1:02d} | Loss: {np.mean(epoch_loss):.4f}")
        
    # Save
    if not os.path.exists(CKPT_DIR): os.makedirs(CKPT_DIR)
    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=EPOCHS, overwrite=True)
    print("✅ Drag Detector Trained!")

if __name__ == "__main__":
    main()