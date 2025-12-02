# /generate_drag_data.py
import jax
import numpy as np
import os
from src.simulator import DataGenerator
from src.sim_config import GRID_SIZE, ETA_MAX, B_MAX, NU_MAX

#--- Settings only for this script ---
SAVE_DIR = "data"
FILENAME = f"dataset_drag_t1.npz" 
N_SAMPLES = 2000 
BATCH_SIZE = 10
EARLY_STEPS = 1000  #  t=1.0 

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    print(f"--- 🌪️ Generating Transient Dynamics Data (T={EARLY_STEPS*0.001:.1f}) ---")
    
    # Overwrite the default parameters in DataGenerator
    data_gen = DataGenerator(grid_size=GRID_SIZE, evolve_steps=EARLY_STEPS)
    batched_sim = jax.jit(jax.vmap(data_gen.sample_batch))
    
    master_key = jax.random.PRNGKey(999)
    current_key = master_key
    
    all_thetas = []
    all_images = []
    
    for i in range(N_SAMPLES // BATCH_SIZE):
        current_key, subkey = jax.random.split(current_key)
        batch_keys = jax.random.split(subkey, BATCH_SIZE)
        
        (batch_theta, _), batch_img = batched_sim(batch_keys)
        batch_img.block_until_ready()
        
        all_thetas.append(np.array(batch_theta))
        all_images.append(np.array(batch_img))
        print(f"Batch {i+1}/{N_SAMPLES//BATCH_SIZE}", end='\r')

    full_thetas = np.concatenate(all_thetas, axis=0)
    full_images = np.concatenate(all_images, axis=0)
    
    
    save_path = os.path.join(SAVE_DIR, FILENAME)
    np.savez_compressed(save_path, theta=full_thetas, x=full_images)
    print(f"\n Saved kinetic dataset to {save_path}")