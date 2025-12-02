# src/analysis_drag_evolution.py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gl_jax import GLSolverJAX, SimConfig
from src.sim_config import GRID_SIZE, ALPHA1, BETA1, D1, ALPHA2, BETA2, D2

# --- Experiment Settings ---
N_SAMPLES_PER_MODEL = 100  # 100 for A, 100 for B (Lightweight)
TOTAL_TIME = 5.0           # Total dimensionless time
STEPS_PER_UNIT_TIME = 1000 # Assuming dt=0.001, so 1000 steps = T=1.0
CHECKPOINTS = 5            # Measure at T=1, 2, 3, 4, 5

# Physics Params
TEST_ETA = 0.8
TEST_B   = 0.04
TEST_NU  = 0.4  # Drag strength for Model B

def compute_sk_k1(images):
    """
    Computes the Structure Factor intensity specifically at k=1 (Low-k limit).
    Returns scalar mean for the batch.
    """
    rho = images[..., 0] # Use channel 0 density
    N = rho.shape[0]
    L = rho.shape[1]
    
    # FFT
    rho_mean = jnp.mean(rho, axis=(1, 2), keepdims=True)
    rho_fluc = rho - rho_mean
    rho_k = jnp.fft.fft2(rho_fluc)
    rho_k_shifted = jnp.fft.fftshift(rho_k, axes=(1, 2))
    S_k_2d = jnp.abs(rho_k_shifted)**2 / (L * L)
    
    # Radial Average logic simplified for k=1
    # Center is at (L/2, L/2). k=1 corresponds to pixels at distance 1 from center.
    # In a 64x64 grid, center is roughly (32, 32).
    # We define a mask for k=1 ring.
    y, x = jnp.indices((L, L))
    center = (L-1)/2
    r = jnp.sqrt((x - center)**2 + (y - center)**2)
    
    # Mask for k=1 bin (e.g., 0.5 < r < 1.5)
    mask = (r >= 0.5) & (r < 1.5)
    
    # Extract S(k) values in this ring
    Sk_val = jnp.sum(S_k_2d * mask, axis=(1, 2)) / jnp.sum(mask)
    
    return Sk_val # Returns shape (N,)

def run_evolution_experiment():
    print(f"--- 🧪 Starting Drag Evolution Experiment ---")
    print(f"    Samples: {N_SAMPLES_PER_MODEL} per model")
    print(f"    Checkpoints: T=1 to T={CHECKPOINTS}")
    
    # 1. Initialize State
    print("    Initializing states...")
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    # Configs
    config_A = SimConfig(eta=TEST_ETA, B=TEST_B, nu=0.0, N=GRID_SIZE)     # No Drag
    config_B = SimConfig(eta=TEST_ETA, B=TEST_B, nu=TEST_NU, N=GRID_SIZE) # With Drag
    
    # Solver instances
    solver_A = GLSolverJAX(config_A)
    solver_B = GLSolverJAX(config_B)
    
    # Initial batch of wavefunctions
    # We use vmap to initialize batch
    init_batch = jax.vmap(GLSolverJAX.initialize_state, in_axes=(None, 0))
    
    keys_A = jax.random.split(k1, N_SAMPLES_PER_MODEL)
    psi1_A, psi2_A = init_batch(config_A, keys_A)
    
    keys_B = jax.random.split(k2, N_SAMPLES_PER_MODEL)
    psi1_B, psi2_B = init_batch(config_B, keys_B)
    
    # Evolution Loop
    history = {
        'time': [],
        'Sk_A_mean': [], 'Sk_A_err': [],
        'Sk_B_mean': [], 'Sk_B_err': []
    }
    
    # Compiled step function (evolve for 1 unit time)
    # We compile the loop of 1000 steps
    @jax.jit
    def evolve_segment_A(p1, p2):
        return solver_A.evolve(p1, p2, STEPS_PER_UNIT_TIME)
    
    @jax.jit
    def evolve_segment_B(p1, p2):
        return solver_B.evolve(p1, p2, STEPS_PER_UNIT_TIME)
    
    # We vmap the evolution over the batch
    batch_evolve_A = jax.vmap(evolve_segment_A)
    batch_evolve_B = jax.vmap(evolve_segment_B)
    
    # Data containers for plotting
    t_axis = []
    
    # --- Main Loop ---
    for t in range(1, CHECKPOINTS + 1):
        print(f"    Processing Time T={t}.0 ...", end='\r')
        
        # Evolve one segment
        psi1_A, psi2_A = batch_evolve_A(psi1_A, psi2_A)
        psi1_B, psi2_B = batch_evolve_B(psi1_B, psi2_B)
        
        # Compute Observables (Density)
        rho1_A = jnp.abs(psi1_A)**2
        # Need to stack for our compute_sk function expectation
        # Make dummy channels to match shape (N, H, W, 1)
        imgs_A = jnp.expand_dims(rho1_A, -1) 
        
        rho1_B = jnp.abs(psi1_B)**2
        imgs_B = jnp.expand_dims(rho1_B, -1)
        
        # Calculate S(k=1)
        sk_A = compute_sk_k1(imgs_A)
        sk_B = compute_sk_k1(imgs_B)
        
        # Stats
        history['time'].append(t)
        history['Sk_A_mean'].append(float(jnp.mean(sk_A)))
        history['Sk_A_err'].append(float(jnp.std(sk_A) / np.sqrt(N_SAMPLES_PER_MODEL)))
        
        history['Sk_B_mean'].append(float(jnp.mean(sk_B)))
        history['Sk_B_err'].append(float(jnp.std(sk_B) / np.sqrt(N_SAMPLES_PER_MODEL)))
        
    print("\n    Simulation complete.")
    return history

def plot_results(history):
    times = history['time']
    
    plt.figure(figsize=(8, 6))
    
    # Plot Model A (Blue)
    plt.errorbar(times, history['Sk_A_mean'], yerr=history['Sk_A_err'], 
                 fmt='o-', linewidth=2, capsize=5, label='Model A (No Drag)', color='blue')
    
    # Plot Model B (Red)
    plt.errorbar(times, history['Sk_B_mean'], yerr=history['Sk_B_err'], 
                 fmt='s-', linewidth=2, capsize=5, label='Model B (With Drag)', color='red')
    
    plt.xlabel(r"Evolution Time $t$ (Simulation Units)")
    plt.ylabel(r"Structure Factor Intensity $S(k \approx 1)$")
    plt.title(r"Dynamical Fingerprint of Drag")
    
    # --- 關鍵修正：使用對數坐標 ---
    plt.yscale('log') 
    # ---------------------------

    # Add annotations (調整位置以適應 Log 圖)
    plt.axvspan(0.5, 1.5, color='yellow', alpha=0.1, label='Relaxation Phase')
    
    plt.legend()
    plt.grid(True, alpha=0.3, which="both") # which='both' 顯示次格線
    plt.tight_layout()
    
    save_path = "assets/drag_time_evolution_log.png"
    if not os.path.exists("assets"): os.makedirs("assets")
    plt.savefig(save_path, dpi=150)
    print(f"✅ Log-scale Plot saved to {save_path}")
    
    # --- 加碼：直接印出數值來確認 T=1 ---
    print("\n--- Numerical Check at T=1 ---")
    print(f"Model A (No Drag): {history['Sk_A_mean'][0]:.6f}")
    print(f"Model B (With Drag): {history['Sk_B_mean'][0]:.6f}")
    
    if history['Sk_B_mean'][0] < history['Sk_A_mean'][0]:
        print("✅ SUCCESS: Drag suppression observed at T=1!")
    else:
        print("⚠️ WARNING: No suppression observed at T=1.")

if __name__ == "__main__":
    hist = run_evolution_experiment()
    plot_results(hist)