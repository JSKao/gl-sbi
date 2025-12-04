import jax
import jax.numpy as jnp
from flax.training import checkpoints
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model import NREClassifier
from src.sim_config import ETA_MIN, ETA_MAX, B_MIN, B_MAX, NU_MIN, NU_MAX

# --- Config ---
CKPT_DIR = "checkpoints"
DATA_PATH = "data/dataset_64.npz" # check your grid size
GRID_SIZE = 1000  # Resolution for 1D integration
TEST_SAMPLES = 200 # Number of SBC samples

def load_data():
    if not os.path.exists(DATA_PATH):
        # Fallback to 32 if 64 not found
        alt_path = "data/dataset_32.npz"
        if os.path.exists(alt_path):
            return np.load(alt_path)
        raise FileNotFoundError("Run generate_data.py first.")
    return np.load(DATA_PATH)

def run_sbc_diagnostic():
    print("--- Running Simulation-Based Calibration (SBC) ---")
    
    # 1. Load Model
    model = NREClassifier()
    # Dummy init
    dummy_x = jnp.ones((1, 64, 64, 3)) 
    dummy_theta = jnp.ones((1, 3))
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, dummy_x, dummy_theta)
    
    # 2. Restore 'calibrated_' checkpoint
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=None, prefix="calibrated_")
    if state_dict is None:
        print("Warning: 'calibrated_' checkpoint not found. Loading default.")
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=None)
    
    # 3. Load Test Data
    data = load_data()
    x_test = data['x'][-TEST_SAMPLES:]
    theta_test = data['theta'][-TEST_SAMPLES:]
    
    # JIT inference
    @jax.jit
    def get_logits(x_b, theta_b):
        return model.apply(state_dict, x_b, theta_b)
    
    # Perform SBC for Parameter 0 (Eta) as an example
    # We compute the rank of the true parameter within the 1D conditional posterior
    ranks = []
    param_idx = 0  # Eta
    p_min, p_max = ETA_MIN, ETA_MAX
    
    print(f"Computing ranks for {TEST_SAMPLES} samples...")
    for i in range(TEST_SAMPLES):
        x_obs = x_test[i]      # (64, 64, 3)
        true_theta = theta_test[i] # (3,)
        true_val = true_theta[param_idx]
        
        # Create a grid of hypothesis for Eta
        grid_vals = jnp.linspace(p_min, p_max, GRID_SIZE)
        
        # Batch: (GRID, 64, 64, 3)
        batch_x = jnp.tile(x_obs[None, ...], (GRID_SIZE, 1, 1, 1))
        
        # Batch: (GRID, 3) - Fix other params to truth (Conditional Slice)
        # For full marginal SBC, we would need to integrate others, but this is a standard proxy.
        batch_theta = jnp.tile(true_theta[None, :], (GRID_SIZE, 1))
        batch_theta = batch_theta.at[:, param_idx].set(grid_vals)
        
        # Inference
        logits = get_logits(batch_x, batch_theta).flatten()
        
        # Unnormalized Posterior ~ Likelihood Ratio ~ exp(logit)
        # (Assuming uniform prior)
        probs = jnp.exp(logits)
        probs = probs / jnp.sum(probs) # Normalize
        
        # Calculate Rank (CDF)
        cdf = jnp.cumsum(probs)
        rank = jnp.interp(true_val, grid_vals, cdf)
        ranks.append(rank)
        
    ranks = np.array(ranks)
    
    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    
    # 1. Rank Histogram
    plt.subplot(1, 2, 1)
    plt.hist(ranks, bins=20, density=True, alpha=0.7, color='teal', edgecolor='black')
    plt.axhline(1.0, color='k', linestyle='--', linewidth=2, label="Ideal Uniform")
    plt.title("SBC Rank Histogram")
    plt.xlabel("Rank Statistic")
    plt.legend()
    
    # 2. ECDF (PP Plot)
    plt.subplot(1, 2, 2)
    # Sort ranks
    sorted_ranks = np.sort(ranks)
    ideal = np.linspace(0, 1, len(ranks))
    plt.plot(ideal, sorted_ranks, lw=2, label="Empirical")
    plt.plot([0, 1], [0, 1], 'k--', label="Ideal")
    
    # Calculate MAD (Area between curves)
    mad = np.mean(np.abs(sorted_ranks - ideal))
    
    plt.title(f"Calibration Curve (PP-Plot)\nMAD Score = {mad:.4f}")
    plt.xlabel("Expected Confidence")
    plt.ylabel("Observed Confidence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "results/sbc_diagnostic.png"
    plt.savefig(save_path)
    print(f"Diagnostics saved to {save_path}")
    print(f"Final MAD Score: {mad:.4f} (Lower is better, <0.05 is excellent)")

if __name__ == "__main__":
    run_sbc_diagnostic()