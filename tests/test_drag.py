import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import time
from src.gl_jax import GLSolverJAX, SimConfig

def run_simulation(nu_value, label):
    print(f"Testing Drag Effect: nu = {nu_value} ({label})...")
    
    # 1. Use small N to test
    config = SimConfig(
        N=64, 
        L=32.0, 
        dt=0.001, 
        B=0.02, 
        nu=nu_value 
    )
    
    # 2. Initialize
    sim = GLSolverJAX(config)
    key = jax.random.PRNGKey(42)
    psi1, psi2 = GLSolverJAX.initialize_state(config, key)
    
    # 3. Evolve
    steps = 2000 
    psi1_final, psi2_final = sim.evolve(psi1, psi2, steps)
    
    return psi1_final, psi2_final

def plot_overlay(psi1, psi2, title, filename):
    rho1 = jnp.abs(psi1)**2
    rho2 = jnp.abs(psi2)**2
    
    plt.figure(figsize=(6, 6))
    
    # 
    # Band 1: Reds
    plt.imshow(rho1, cmap='Reds', origin='lower', alpha=0.5, label='Band 1')
    # Band 2: Blues
    plt.imshow(rho2, cmap='Blues', origin='lower', alpha=0.5, label='Band 2')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    # --- A: No drag (Control Group) ---
    p1_no, p2_no = run_simulation(0.0, "Control")
    plot_overlay(p1_no, p2_no, 
                 "No Drag (nu=0): Independent Vortices?", 
                 "results/drag_test_nu_0.png")
    
    # --- B: Strong Drag ---
    p1_drag, p2_drag = run_simulation(10.0, "Strong Drag")
    plot_overlay(p1_drag, p2_drag, 
                 "Strong Drag (nu=10): Locked Vortices?", 
                 "results/drag_test_nu_10.png")
                 
    print("Test Complete. Check 'results/' for comparison images.")

if __name__ == "__main__":
    main()