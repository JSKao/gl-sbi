import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import time

from src.gl_jax import GLSolverJAX, SimConfig
from src.hessian_analysis import compute_hessian_eig

def main():
    # --- Configuration ---
    print("Starting Hessian Analysis Task...")
    
    # Try N=40, small size
    # Hessian matrix with size (4*N^2)^2。
    # N=40 -> matrix 6400 -> number of elements 40 millions (about 300MB RAM)
    # N=64 matrix becomes 6.5 larger 
    config = SimConfig(
        N=40, 
        L=20.0,      
        dt=0.001,    
        eta=0.8,     # Type-1.5 
        B=0.02       
    )
    
    if not os.path.exists("results"):
        os.makedirs("results")

    # --- Simulation ---
    print(f"Initializing Simulation (Grid: {config.N}x{config.N})...")
    key = jax.random.PRNGKey(42) 
    sim = GLSolverJAX(config)
    
    psi1, psi2 = GLSolverJAX.initialize_state(config, key)
    
    # Evolve to Metastable State
    steps = 4000
    print(f"Evolving for {steps} steps to reach a glassy state...")
    start_time = time.time()
    psi1_final, psi2_final = sim.evolve(psi1, psi2, steps)
    print(f" Simulation done in {time.time() - start_time:.2f}s")

    plt.figure(figsize=(6, 6))
    plt.imshow(jnp.abs(psi1_final)**2, origin='lower', cmap='inferno')
    plt.title("Vortex Configuration (State for Hessian)")
    plt.savefig("results/vortex_state.png")
    plt.close()

    # --- Hessian analysis (The Glassy Proof) ---
    print("Computing Hessian Eigenvalues (This is the heavy lifting)...")
    
    eigvals = compute_hessian_eig(psi1_final, psi2_final, config)
    
    # Drop large eigenvalues
    eigvals = np.array(eigvals) # transform to numpy for matplotlib
    
    print(f"  Done! Found {len(eigvals)} eigenvalues.")
    print(f"  Min eigenvalue: {np.min(eigvals):.5f}")
    
    # --- Visualization ---
    print("Plotting Hessian Spectrum...")
    
    plt.figure(figsize=(10, 6))
    # Histogram - DOS
    plt.hist(eigvals, bins=100, range=(0, 20), color='teal', alpha=0.7, density=True)
    
    plt.title(f"Hessian Spectrum (N={config.N}, $\eta$={config.eta})")
    plt.xlabel(r"Eigenvalue $\lambda$ (Curvature)")
    plt.ylabel(r"Density of States $\rho(\lambda)$")
    plt.grid(True, alpha=0.3)
    
    # Remark: For glassy state, we expect many modes nearby lambda=0
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Mode')
    plt.legend()
    
    save_path = "results/hessian_spectrum.png"
    plt.savefig(save_path)
    print(f"Analysis Complete! Plot saved to {save_path}")

if __name__ == "__main__":
    import numpy as np 
    main()
