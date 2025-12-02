# src/inference_1d.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from flax.training import train_state, checkpoints
import optax

from src.gl_jax import GLSolverJAX, SimConfig
from src.model import NREClassifier

from src.sim_config import (
    GRID_SIZE, EVOLVE_STEPS,
    ETA_MIN, ETA_MAX, B_MIN, B_MAX,
    TEST_CASE_ETA, TEST_CASE_B, TEST_CASE_NU
)

CKPT_DIR = os.path.abspath("checkpoints")

def create_train_state(rng, input_shape):
  
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
    input_shape = (GRID_SIZE, GRID_SIZE, 3)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, input_shape)
    
    if os.path.exists(CKPT_DIR):
        state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)
        print(f" Loaded checkpoint from {CKPT_DIR}")
    else:
        print(" No checkpoint found! Please run train_offline.py first.")
        return


    print(f"Generating observation for eta={TEST_CASE_ETA}, B={TEST_CASE_B}...")
    
    key_sim = jax.random.PRNGKey(999)
    config = SimConfig(eta=TEST_CASE_ETA, B=TEST_CASE_B, nu=TEST_CASE_NU, N=GRID_SIZE)
    solver = GLSolverJAX(config)
    
   
    psi1, psi2 = GLSolverJAX.initialize_state(config, key_sim)
    p1_f, p2_f = solver.evolve(psi1, psi2, EVOLVE_STEPS)
    
    
    rho1 = jnp.abs(p1_f)**2
    rho2 = jnp.abs(p2_f)**2
    Jx, Jy = GLSolverJAX.compute_current(p1_f, config)
    curl_J = GLSolverJAX.compute_curl_J(Jx, Jy, config)
    
    # 3-Channel Input
    obs_img = jnp.stack([rho1, rho2, curl_J], axis=-1)
    obs_img_batch = jnp.expand_dims(obs_img, axis=0)

    # 3. 1D parameter scanning (only scan eta, fix B)
    print(f" Scanning eta range [{ETA_MIN}, {ETA_MAX}]...")
    test_etas = jnp.linspace(ETA_MIN, ETA_MAX, 100)
    scores = []
    
    for test_eta in test_etas:
        # Construct query: [eta, fixed_B, fixed_nu]
        theta_test = jnp.array([[test_eta, TEST_CASE_B, TEST_CASE_NU]])
        
        # AI score
        logit = state.apply_fn({'params': state.params}, obs_img_batch, theta_test)
        prob = jax.nn.sigmoid(logit)
        scores.append(prob[0, 0])
    
    
    scores = np.array(scores)
    
    plt.figure(figsize=(8, 5))
    plt.plot(test_etas, scores, label='Posterior Proxy (AI)', color='blue', linewidth=2.5)
    plt.axvline(x=TEST_CASE_ETA, color='red', linestyle='--', linewidth=2, label=f'Ground Truth ($\eta={TEST_CASE_ETA}$)')
    
    plt.title(f"1D Posterior Inference (Grid {GRID_SIZE}x{GRID_SIZE})", fontsize=14)
    plt.xlabel(r"Coupling Strength $\eta$", fontsize=12)
    plt.ylabel("Ratio Score $r(x, \\theta)$", fontsize=12)
    plt.xlim(ETA_MIN, ETA_MAX)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if not os.path.exists("assets"):
        os.makedirs("assets")
    
    save_path = "assets/inference_result_1d.png"
    plt.savefig(save_path, dpi=150)
    print(f" Saved 1D plot to {save_path}")

if __name__ == "__main__":
    main()