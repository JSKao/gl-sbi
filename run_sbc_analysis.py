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
DATA_PATH = "data/dataset_64.npz"
GRID_SIZE = 1000  # Resolution for 1D integration
TEST_SAMPLES = 200  # Number of SBC samples

def load_data():
    if not os.path.exists(DATA_PATH):
        alt_path = "data/dataset_32.npz"
        if os.path.exists(alt_path):
            return np.load(alt_path)
        raise FileNotFoundError("Run generate_data.py first.")
    return np.load(DATA_PATH)

def load_model_and_data():
    abs_ckpt_dir = os.path.abspath(CKPT_DIR)

    # 1. Initialize model
    model = NREClassifier()
    dummy_x = jnp.ones((1, 64, 64, 3)) 
    dummy_theta = jnp.ones((1, 3))
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, dummy_x, dummy_theta)
    
    # 2. Restore checkpoint
    print(f"Attempting to restore from: {abs_ckpt_dir}")
    
    try:
        all_steps = checkpoints.list_steps(abs_ckpt_dir, prefix="calibrated_")
        if not all_steps:
            raise FileNotFoundError
        latest_step = max(all_steps)
        print(f"Found checkpoint at step {latest_step}")
    except Exception:
        latest_step = None
        
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=abs_ckpt_dir, 
        target=None, 
        prefix="calibrated_",
        step=latest_step
    )
    
    if restored_state is None:
        raise FileNotFoundError(f"No checkpoint found in {abs_ckpt_dir}")
    
    # 3. Extract params properly
    # The checkpoint might contain a TrainState or just params
    if hasattr(restored_state, 'params'):
        # It's a TrainState object
        params = restored_state.params
    elif 'params' in restored_state:
        # It's a dict with 'params' key
        params = restored_state['params']
    else:
        # It might be the params directly
        params = restored_state
    
    # Create the proper variables dict for apply
    variables_dict = {'params': params}
    
    # 4. Load test data
    data = load_data()
    x_test = data['x'][-TEST_SAMPLES:]
    theta_test = data['theta'][-TEST_SAMPLES:]
    
    return model, variables_dict, theta_test, x_test

def run_sbc_diagnostic():
    print("--- Running Simulation-Based Calibration (SBC) ---")
    
    try:
        model, variables, theta, x = load_model_and_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Fatal Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return

    # JIT the inference call
    @jax.jit
    def predict_logits(x_batch, theta_batch):
        return model.apply(variables, x_batch, theta_batch)

    # Analyze Parameter 0 (Eta)
    target_idx = 0  # Eta
    p_min, p_max = ETA_MIN, ETA_MAX
    ranks = []
    
    print(f"Computing ranks for {TEST_SAMPLES} samples...")
    for i in range(len(theta)):
        true_theta = theta[i] 
        x_obs = x[i]
        true_val = true_theta[target_idx]
        
        grid_vals = jnp.linspace(p_min, p_max, GRID_SIZE)
        
        # Create batched inputs
        x_batch = jnp.tile(x_obs[None, ...], (GRID_SIZE, 1, 1, 1))
        batch_theta = jnp.tile(true_theta[None, :], (GRID_SIZE, 1))
        batch_theta = batch_theta.at[:, target_idx].set(grid_vals)
        
        # Inference
        logits = predict_logits(x_batch, batch_theta).flatten()
        
        # Compute posterior (unnormalized)
        probs = jnp.exp(logits)
        probs = probs / jnp.sum(probs)
        
        # Calculate rank (CDF at true value)
        cdf = jnp.cumsum(probs)
        rank = jnp.interp(true_val, grid_vals, cdf)
        ranks.append(rank)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(theta)} samples")

    ranks = np.array(ranks)
    
    # --- Visualization ---
    os.makedirs("results", exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Rank Histogram
    plt.subplot(1, 2, 1)
    plt.hist(ranks, bins=20, density=True, alpha=0.7, color='teal', edgecolor='black')
    plt.axhline(1.0, color='k', linestyle='--', linewidth=2, label="Ideal Uniform")
    plt.title("SBC Rank Histogram (Eta)", fontsize=14, fontweight='bold')
    plt.xlabel("Rank Statistic", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. PP Plot
    plt.subplot(1, 2, 2)
    sorted_ranks = np.sort(ranks)
    ideal = np.linspace(0, 1, len(ranks))
    plt.plot(ideal, sorted_ranks, lw=2, label="Empirical CDF", color='teal')
    plt.plot([0, 1], [0, 1], 'k--', label="Ideal Uniform", linewidth=2)
    
    mad = np.mean(np.abs(sorted_ranks - ideal))
    
    plt.title(f"Calibration Curve (PP-Plot)\nMAD = {mad:.4f}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Expected Confidence", fontsize=12)
    plt.ylabel("Observed Coverage", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = "results/sbc_diagnostic.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Diagnostics saved to {save_path}")
    print(f"✓ Final MAD Score: {mad:.4f}")
    
    if mad < 0.05:
        print("  → Excellent calibration! Model is well-calibrated.")
    elif mad < 0.10:
        print("  → Good calibration. Minor deviations detected.")
    else:
        print("  → Poor calibration. Consider retraining with adjustments.")

if __name__ == "__main__":
    run_sbc_diagnostic()