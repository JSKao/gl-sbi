import jax
import jax.numpy as jnp
from flax.training import checkpoints
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model import NREClassifier
from src.sim_config import ETA_MIN, ETA_MAX, B_MIN, B_MAX, NU_MIN, NU_MAX
from src.train_config import CKPT_DIR

# --- Config ---
# NOTE: CKPT_DIR is imported from train_config
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

def load_model_and_data():
    # --- 1. Use Absolute Path for Checkpoint ---
    abs_ckpt_dir = os.path.abspath(CKPT_DIR)

    # 2. Load Dummy for Init
    model = NREClassifier()
    # Assuming standard grid size, adjust if necessary
    dummy_x = jnp.ones((1, 64, 64, 3)) 
    dummy_theta = jnp.ones((1, 3))
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, dummy_x, dummy_theta)
    
    # 3. Restore Checkpoint using the absolute path
    print(f"Attempting to restore from absolute path: {abs_ckpt_dir}")
    
    # Try to restore the 'calibrated_' checkpoint
    state_dict = checkpoints.restore_checkpoint(
        ckpt_dir=abs_ckpt_dir, 
        target=None, 
        prefix="calibrated_",
        # Find the latest step number automatically
        step=checkpoints.latest_step(abs_ckpt_dir, prefix="calibrated_") 
    )
    
    if state_dict is None:
        raise FileNotFoundError(f"No checkpoint found in {abs_ckpt_dir} with prefix 'calibrated_'.")
        
    # 4. Load Test Data
    data = load_data()
    # Take the validation set (tail end)
    x_test = data['x'][-TEST_SAMPLES:]
    theta_test = data['theta'][-TEST_SAMPLES:]
    
    return model, state_dict, theta_test, x_test

def run_sbc_diagnostic():
    print("--- Running Simulation-Based Calibration (SBC) ---")
    
    try:
        model, state_dict, theta, x = load_model_and_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Fatal Error during model loading: {e}")
        return

    # JIT the inference call
    @jax.jit
    def predict_logits(x_batch, theta_batch):
        # We need to explicitly handle the nested structure if state_dict is params only
        # Assuming state_dict is the nested dictionary returned by restore_checkpoint
        return model.apply(state_dict, x_batch, theta_batch)

    # We analyze Parameter 0 (Eta) as primary example
    target_idx = 0  # Eta
    p_min, p_max = ETA_MIN, ETA_MAX
    ranks = []
    
    print(f"Computing ranks for {TEST_SAMPLES} samples...")
    for i in range(len(theta)):
        true_theta = theta[i] 
        x_obs = x[i]
        true_val = true_theta[target_idx]
        
        grid_vals = jnp.linspace(p_min, p_max, GRID_SIZE)
        
        # Batch: (GRID, 64, 64, 3)
        x_batch = jnp.tile(x_obs[None, ...], (GRID_SIZE, 1, 1, 1))
        
        # Batch: (GRID, 3) - Fix other params to truth (Conditional Slice)
        batch_theta = jnp.tile(true_theta[None, :], (GRID_SIZE, 1))
        batch_theta = batch_theta.at[:, target_idx].set(grid_vals)
        
        # Inference
        logits = predict_logits(x_batch, batch_theta).flatten()
        
        # Unnormalized Posterior ~ Likelihood Ratio ~ exp(logit)
        probs = jnp.exp(logits)
        probs = probs / jnp.sum(probs) # Normalize
        
        # Calculate Rank (CDF)
        cdf = jnp.cumsum(probs)
        # Find index of true value in grid
        rank = jnp.interp(true_val, grid_vals, cdf)
        ranks.append(rank)
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(theta)} samples")

    ranks = np.array(ranks)
    
    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    
    # 1. Rank Histogram
    plt.subplot(1, 2, 1)
    plt.hist(ranks, bins=20, density=True, alpha=0.7, color='teal', edgecolor='black')
    plt.axhline(1.0, color='k', linestyle='--', linewidth=2, label="Ideal Uniform")
    plt.title("SBC Rank Histogram (Eta)")
    plt.xlabel("Rank Statistic")
    
    # 2. ECDF (PP Plot)
    plt.subplot(1, 2, 2)
    sorted_ranks = np.sort(ranks)
    ideal = np.linspace(0, 1, len(ranks))
    plt.plot(ideal, sorted_ranks, lw=2, label="Empirical CDF")
    plt.plot([0, 1], [0, 1], 'k--', label="Ideal Uniform")
    
    mad = np.mean(np.abs(sorted_ranks - ideal))
    
    plt.title(f"Calibration Curve (PP-Plot)\nMAD Score = {mad:.4f}")
    plt.xlabel("Expected Confidence")
    plt.ylabel("Observed Coverage")
    
    save_path = "results/sbc_diagnostic.png"
    plt.savefig(save_path)
    print(f"\nDiagnostics saved to {save_path}")
    print(f"Final MAD Score: {mad:.4f} (Lower is better, <0.05 is excellent)")

if __name__ == "__main__":
    run_sbc_diagnostic()