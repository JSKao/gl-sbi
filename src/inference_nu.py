# src/inference_nu.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from flax.training import checkpoints
from src.inference_1d import create_train_state, CKPT_DIR
from src.gl_jax import GLSolverJAX, SimConfig
from src.sim_config import GRID_SIZE, EVOLVE_STEPS, NU_MIN, NU_MAX


TEST_ETA = 0.8
TEST_B   = 0.04
TEST_NU  = 0.4  # <--- test this

CKPT_DIR = os.path.abspath("checkpoints_drag")

def main():
    print(f"--- Probing Drag: Scanning Nu (True Nu={TEST_NU}) ---")
    
    # 1. Load Model
    rng = jax.random.PRNGKey(0)
    input_shape = (GRID_SIZE, GRID_SIZE, 3)
    state = create_train_state(rng, input_shape)
    
    if not os.path.exists(CKPT_DIR):
        print("No checkpoint found.")
        return
    state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)

    # 2. Generate ONE synthetic image with Drag
    print("Generating synthetic observation...")
   
    
    # overwrite evolve_steps
    EARLY_STEPS = 1000 
    
    config = SimConfig(eta=TEST_ETA, B=TEST_B, nu=TEST_NU, N=GRID_SIZE)
    key_sim = jax.random.PRNGKey(42)
    psi1, psi2 = GLSolverJAX.initialize_state(config, key_sim)
    solver = GLSolverJAX(config)
    p1_f, p2_f = solver.evolve(psi1, psi2, EARLY_STEPS) # force to read early dynamics
    
    rho1, rho2 = jnp.abs(p1_f)**2, jnp.abs(p2_f)**2
    Jx, Jy = GLSolverJAX.compute_current(p1_f, config)
    curl_J = GLSolverJAX.compute_curl_J(Jx, Jy, config)
    obs_img = jnp.stack([rho1, rho2, curl_J], axis=-1)
    obs_img_batch = jnp.expand_dims(obs_img, axis=0)

    # 3. Scan Nu
    print("Scanning nu range...")
    test_nus = jnp.linspace(0.0, 0.6, 100) # include true nu
    scores = []
    
    for nu_val in test_nus:
        # Query: [Fixed Eta, Fixed B, Scanning Nu]
        theta_test = jnp.array([[TEST_ETA, TEST_B, nu_val]])
        logit = state.apply_fn({'params': state.params}, obs_img_batch, theta_test)
        scores.append(jax.nn.sigmoid(logit)[0, 0])
        
    # 4. Plot
    plt.figure(figsize=(8, 5))
    plt.plot(test_nus, scores, linewidth=3, color='darkorange', label='NRE Posterior Proxy')
    plt.axvline(x=TEST_NU, color='black', linestyle='--', label=f'Ground Truth ($\\nu={TEST_NU}$)')
    plt.title(r"Inference of Drag Coefficient $\nu$ (from early-time snapshot)")
    plt.xlabel(r"Drag Coefficient $\nu$")
    plt.ylabel("Ratio Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("assets/inference_nu.png")
    print("Saved to assets/inference_nu.png")

if __name__ == "__main__":
    main()