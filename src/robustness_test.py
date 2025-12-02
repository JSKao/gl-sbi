# robustness_test.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from flax.training import train_state, checkpoints
import optax

# Import project modules
from src.gl_jax import GLSolverJAX, SimConfig
from src.model import NREClassifier
from src.sim_config import (
    GRID_SIZE, EVOLVE_STEPS,
    ETA_MIN, ETA_MAX, 
    # Fallback defaults in case they are missing in sim_config
    DEFAULT_B, DEFAULT_NU
)

# Configuration for the Experiment
CKPT_DIR = os.path.abspath("checkpoints")
NOISE_LEVEL = 0.15      # 15% Noise intensity (Adjust this if needed)
TEST_ETA = 0.8          # Ground Truth Eta
TEST_B = DEFAULT_B      # Ground Truth B
TEST_NU = DEFAULT_NU    # Ground Truth Nu (usually 0.0 for baseline)

def create_train_state(rng, input_shape):
    """Reconstruct the model state structure"""
    model = NREClassifier()
    dummy_x = jnp.ones((1, *input_shape))
    dummy_theta = jnp.ones((1, 3))
    variables = model.init(rng, dummy_x, dummy_theta)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(1e-3),
    )

def add_physics_aware_noise(key, image, noise_scale):
    """
    Adds Gaussian noise but respects physical constraints:
    - Density (channels 0, 1) cannot be negative.
    - Curl (channel 2) can be negative.
    """
    noise = jax.random.normal(key, image.shape) * noise_scale
    noisy_image = image + noise
    
    # Split channels
    rho1 = noisy_image[..., 0]
    rho2 = noisy_image[..., 1]
    curl = noisy_image[..., 2]
    
    # Clip densities to be non-negative
    rho1 = jnp.clip(rho1, a_min=0.0)
    rho2 = jnp.clip(rho2, a_min=0.0)
    
    # Re-stack
    return jnp.stack([rho1, rho2, curl], axis=-1)

def main():
    print(" Starting Robustness Check ...")
    
    # 1. Setup & Load Model
    input_shape = (GRID_SIZE, GRID_SIZE, 3)
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, input_shape)
    
    if os.path.exists(CKPT_DIR):
        state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)
        print(f" Loaded checkpoint from {CKPT_DIR}")
    else:
        print(" No checkpoint found! Please run train_offline.py first.")
        return

    # 2. Generate Ground Truth Physics Data (Clean)
    print(f"⚗️  Simulating Clean Physics (eta={TEST_ETA})...")
    key_sim = jax.random.PRNGKey(101) # Specific seed for reproducibility
    config = SimConfig(eta=TEST_ETA, B=TEST_B, nu=TEST_NU, N=GRID_SIZE)
    solver = GLSolverJAX(config)
    
    psi1, psi2 = GLSolverJAX.initialize_state(config, key_sim)
    p1_f, p2_f = solver.evolve(psi1, psi2, EVOLVE_STEPS)
    
    # Construct Observation
    rho1 = jnp.abs(p1_f)**2
    rho2 = jnp.abs(p2_f)**2
    Jx, Jy = GLSolverJAX.compute_current(p1_f, config)
    curl_J = GLSolverJAX.compute_curl_J(Jx, Jy, config)
    
    clean_img = jnp.stack([rho1, rho2, curl_J], axis=-1) # (H, W, 3)
    
    # 3. Generate Noisy Observation
    print(f"⚡ Injecting {NOISE_LEVEL*100}% Gaussian Noise...")
    key_noise = jax.random.PRNGKey(777)
    noisy_img = add_physics_aware_noise(key_noise, clean_img, NOISE_LEVEL)

    # Prepare batches for Inference
    # We want to scan Eta, keeping B and Nu fixed
    scan_resolution = 100
    eta_grid = jnp.linspace(ETA_MIN, ETA_MAX, scan_resolution)
    
    # Create the parameter batch: [eta_i, B_fixed, nu_fixed]
    B_col = jnp.full((scan_resolution,), TEST_B)
    Nu_col = jnp.full((scan_resolution,), TEST_NU)
    theta_batch = jnp.stack([eta_grid, B_col, Nu_col], axis=-1)
    
    # Create image batches (repeat single image to match theta batch size)
    clean_batch = jnp.repeat(clean_img[None, ...], scan_resolution, axis=0)
    noisy_batch = jnp.repeat(noisy_img[None, ...], scan_resolution, axis=0)

    # 4. Run Inference (Vectorized)
    print(" Running NRE Inference on both Clean and Noisy data...")
    
    # Clean Inference
    logits_clean = state.apply_fn({'params': state.params}, clean_batch, theta_batch)
    probs_clean = jax.nn.sigmoid(logits_clean).flatten()
    
    # Noisy Inference
    logits_noisy = state.apply_fn({'params': state.params}, noisy_batch, theta_batch)
    probs_noisy = jax.nn.sigmoid(logits_noisy).flatten()

    # 5. Visualization (The "Money Plot")
    print("📊 Plotting results...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Visual Comparison
    ax_img = axes[0]
    # Visualize Channel 0 (Density Band 1) + Channel 2 (Vortex Cores)
    # Just showing Density 1 for clarity
    ax_img.set_title(f"Input Data: Clean vs Noisy ($\sigma={NOISE_LEVEL}$)")
    vis_concat = jnp.concatenate([clean_img[..., 0], noisy_img[..., 0]], axis=1)
    ax_img.imshow(vis_concat, cmap='inferno', origin='lower')
    ax_img.axis('off')
    ax_img.text(10, 10, "Clean", color='white', fontsize=12, fontweight='bold')
    ax_img.text(GRID_SIZE + 10, 10, "Noisy", color='white', fontsize=12, fontweight='bold')

    # Subplot 2: Posterior Comparison
    ax_post = axes[1]
    ax_post.plot(eta_grid, probs_clean, label='Original (Clean)', color='#1f77b4', linewidth=2.5)
    ax_post.plot(eta_grid, probs_noisy, label=f'Perturbed ($\sigma={NOISE_LEVEL}$)', color='#ff7f0e', linestyle='--', linewidth=2.5)
    
    # Ground Truth Line
    ax_post.axvline(TEST_ETA, color='red', linestyle=':', linewidth=2, label=f'Ground Truth $\eta={TEST_ETA}$')
    
    ax_post.set_title("Robustness of Posterior Estimation", fontsize=14)
    ax_post.set_xlabel(r"Coupling Strength $\eta$", fontsize=12)
    ax_post.set_ylabel("Posterior Score (Unnormalized)", fontsize=12)
    ax_post.legend(fontsize=11)
    ax_post.grid(True, alpha=0.3)
    
    # Save
    if not os.path.exists("assets"):
        os.makedirs("assets")
    save_path = "assets/robustness_check.png"
    plt.savefig(save_path, dpi=150)
    print(f" Saved robustness plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()