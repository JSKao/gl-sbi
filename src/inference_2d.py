# src/inference_2d.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from flax.training import train_state, checkpoints
import optax

# Import physics
from src.gl_jax import GLSolverJAX, SimConfig
# Import models
from src.model import NREClassifier

from src.sim_config import (
    GRID_SIZE, EVOLVE_STEPS, ETA_MIN, ETA_MAX, 
    B_MIN, B_MAX , TEST_CASE_ETA, TEST_CASE_B,
    TEST_CASE_NU
)

CKPT_DIR = os.path.abspath("checkpoints") 

def create_train_state(rng, input_shape):
    """Rebuild model with trained weights"""
    model = NREClassifier()
    dummy_x = jnp.ones((1, *input_shape))
    dummy_theta = jnp.ones((1, 3))
    variables = model.init(rng, dummy_x, dummy_theta)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(1e-3),
    )

def main():
    # Make sure you've executed train_offline
    input_shape = (GRID_SIZE, GRID_SIZE, 3) 
    
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, input_shape)
    
   
    if os.path.exists(CKPT_DIR):
        state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)
        print(f"Loaded checkpoint from {CKPT_DIR}")
    else:
        print("No checkpoint found! Please run train_offline.py first.")
        return

    print(f"Generating observation for eta={TEST_CASE_ETA}, B={TEST_CASE_B}, nu={TEST_CASE_NU}...")
    
    key_sim = jax.random.PRNGKey(999)
    config = SimConfig(eta=TEST_CASE_ETA, B=TEST_CASE_B, nu=TEST_CASE_NU, N=GRID_SIZE)
    solver = GLSolverJAX(config)
    
    # Physics evolution
    psi1, psi2 = GLSolverJAX.initialize_state(config, key_sim)
    p1_f, p2_f = solver.evolve(psi1, psi2, EVOLVE_STEPS)
    
    # --- Sync with the character enforcement in simulator.py ---
    rho1 = jnp.abs(p1_f)**2
    rho2 = jnp.abs(p2_f)**2
    
    Jx, Jy = GLSolverJAX.compute_current(p1_f, config)
    curl_J = GLSolverJAX.compute_curl_J(Jx, Jy, config)
    
    # Stack 3 layers (Density1, Density2, Curl)
    obs_img = jnp.stack([rho1, rho2, curl_J], axis=-1)
    obs_img_batch = jnp.expand_dims(obs_img, axis=0) 

    # Scanning 2D parameters (batch splitting to avoid OOM)
    print("Scanning 2D parameter space...")
    res = 50 
    eta_range = jnp.linspace(ETA_MIN, ETA_MAX, res)
    B_range = jnp.linspace(B_MIN, B_MAX, res)
    
    Eta_grid, B_grid = jnp.meshgrid(eta_range, B_range)
    # Nu is fixed at default number
    Nu_flat = jnp.full_like(Eta_grid.ravel(), TEST_CASE_NU)
    grid_flat = jnp.stack([Eta_grid.ravel(), B_grid.ravel(), Nu_flat], axis=-1) 
    
    print(f"Predicting {len(grid_flat)} points in chunks...")
    INFER_BATCH = 100  
    logits_list = []
    num_chunks = int(np.ceil(len(grid_flat) / INFER_BATCH))
    
    for i in range(num_chunks):
        start = i * INFER_BATCH
        end = min((i + 1) * INFER_BATCH, len(grid_flat))
        batch_theta = grid_flat[start:end]
        
       
        batch_img = jnp.repeat(obs_img_batch, batch_theta.shape[0], axis=0)
        
        # inference
        batch_logits = state.apply_fn({'params': state.params}, batch_img, batch_theta)
        logits_list.append(batch_logits)
        print(f"Processed chunk {i+1}/{num_chunks}", end='\r')
        
    logits = jnp.concatenate(logits_list, axis=0)
    probs = jax.nn.sigmoid(logits).reshape(res, res)
    print("\nInference done.")

    print("Plotting...")
    plt.figure(figsize=(8, 6))
    plt.imshow(probs, origin='lower', extent=[ETA_MIN, ETA_MAX, B_MIN, B_MAX], aspect='auto', cmap='viridis')
    plt.colorbar(label='Posterior Probability (Unnormalized)')
    plt.scatter([TEST_CASE_ETA], [TEST_CASE_B], color='red', marker='*', s=200, label='Ground Truth')
    plt.xlabel(r'Coupling $\eta$')
    plt.ylabel(r'Magnetic Field $B$')
    plt.title(r'Joint Posterior $p(\eta, B | x)$')
    plt.legend()
    plt.tight_layout()
    

    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    plt.savefig("assets/joint_posterior_2d.png")
    print("Saved 2D heatmap to assets/joint_posterior_2d.png")

if __name__ == "__main__":
    main()