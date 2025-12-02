# src/generate_data.py
import jax
import numpy as np
import time
import os
from src.simulator import DataGenerator 
from src.sim_config import GRID_SIZE, EVOLVE_STEPS, N_SAMPLES, BATCH_SIZE

# --- Config ---
SAVE_DIR = "data"
FILENAME = f"dataset_{GRID_SIZE}.npz"


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print(f"Offline Data Factory | Target: {N_SAMPLES} | Grid: {GRID_SIZE}")
    
    # Instantiate using config constants
    data_gen = DataGenerator(grid_size=GRID_SIZE, evolve_steps=EVOLVE_STEPS)
    batched_sim = jax.jit(jax.vmap(data_gen.sample_batch))
    
    master_key = jax.random.PRNGKey(42)
    current_key = master_key
    
    all_thetas = []
    all_labels = []
    all_images = []
    
    n_batches = N_SAMPLES // BATCH_SIZE
    start_time = time.time()
    
    for i in range(n_batches):
        current_key, subkey = jax.random.split(current_key)
        batch_keys = jax.random.split(subkey, BATCH_SIZE)
        
        # Unpack the nested structure from simulator.py: ((params, label), x)
        (batch_theta, batch_label), batch_img = batched_sim(batch_keys)

        # Block until computation is done
        batch_img.block_until_ready()
        
        # Save for NRE
        all_thetas.append(np.array(batch_theta))
        all_labels.append(np.array(batch_label))
        all_images.append(np.array(batch_img))
        
        # Simple progress tracking
        done = (i + 1) * BATCH_SIZE
        elapsed = time.time() - start_time
        speed = done / elapsed
        print(f"Batch {i+1}/{n_batches} | Speed: {speed:.2f} img/s")

    print("Saving...")
    full_thetas = np.concatenate(all_thetas, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    full_images = np.concatenate(all_images, axis=0)
    
    save_path = os.path.join(SAVE_DIR, FILENAME)
    # create theta key to match training script expectation
    np.savez_compressed(save_path, theta=full_thetas, label=full_labels, x=full_images)
    print(f"Data Saved: {save_path}")
    print(f"   Shapes -> Theta: {full_thetas.shape}, Label: {full_labels.shape}, X: {full_images.shape}")