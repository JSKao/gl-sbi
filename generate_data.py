import jax
import numpy as np
import time
import os
from src.simulator import DataGenerator # Reuse logic

# --- Config ---
SAVE_DIR = "data"
FILENAME = "dataset_128.npz"
N_SAMPLES = 5000
BATCH_SIZE = 10
# Production quality settings
GRID_SIZE = 128 
STEPS = 1000

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print(f"🏭 Offline Data Factory | Target: {N_SAMPLES} | Grid: {GRID_SIZE}")
    
    # Instantiate shared logic
    data_gen = DataGenerator(grid_size=GRID_SIZE, evolve_steps=STEPS)
    batched_sim = jax.jit(jax.vmap(data_gen.sample_batch))
    
    master_key = jax.random.PRNGKey(42)
    current_key = master_key
    
    all_thetas = []
    all_images = []
    
    n_batches = N_SAMPLES // BATCH_SIZE
    start_time = time.time()
    
    for i in range(n_batches):
        current_key, subkey = jax.random.split(current_key)
        batch_keys = jax.random.split(subkey, BATCH_SIZE)
        
        batch_theta, batch_img = batched_sim(batch_keys)
        batch_img.block_until_ready()
        
        all_thetas.append(np.array(batch_theta))
        all_images.append(np.array(batch_img))
        
        # Simple progress tracking
        done = (i + 1) * BATCH_SIZE
        elapsed = time.time() - start_time
        speed = done / elapsed
        print(f"Batch {i+1}/{n_batches} | Speed: {speed:.2f} img/s")

    print("💾 Saving...")
    full_thetas = np.concatenate(all_thetas, axis=0)
    full_images = np.concatenate(all_images, axis=0)
    
    save_path = os.path.join(SAVE_DIR, FILENAME)
    np.savez_compressed(save_path, theta=full_thetas, x=full_images)
    print(f"✅ Saved to {save_path}")