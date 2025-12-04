import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

from src.gl_jax import GLSolverJAX, SimConfig

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

class StructureFactorAnalyzer:
    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.dx = L / N
        kx = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self.K_mag = np.sqrt(KX**2 + KY**2)
        self.k_bins = np.linspace(0, np.max(self.K_mag)/2, N//2)
        self.k_centers = 0.5 * (self.k_bins[1:] + self.k_bins[:-1])

    def compute_sk_radial(self, density_field):
        delta_rho = density_field - jnp.mean(density_field)
        rho_k = jnp.fft.fft2(delta_rho)
        S_k_2d = (jnp.abs(rho_k)**2) / (self.N**2)
        S_k_2d_np = np.array(S_k_2d)
        digitized = np.digitize(self.K_mag.ravel(), self.k_bins)
        S_k_1d = [S_k_2d_np.ravel()[digitized == i].mean() for i in range(1, len(self.k_bins))]
        return self.k_centers, np.array(S_k_1d)

def run_noise_demo():
    print("--- Visualizing Intrinsic Quench Noise ---")
    
    # Single Parameter Set (The Clean System)
    cfg = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=0.15)
    
    # Run 10 realizations with DIFFERENT seeds
    n_runs = 10
    keys = jax.random.split(jax.random.PRNGKey(42), n_runs)
    
    analyzer = StructureFactorAnalyzer(cfg.N, cfg.L)
    solver = GLSolverJAX(cfg)
    
    plt.figure(figsize=(8, 6))
    
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...")
        p1, p2 = GLSolverJAX.initialize_state(cfg, keys[i])
        # Evolve to the "formation epoch"
        p1, p2 = solver.evolve(p1, p2, steps=1500) 
        
        rho = jnp.abs(p1)**2 + jnp.abs(p2)**2
        k, sk = analyzer.compute_sk_radial(rho)
        
        plt.loglog(k, sk, 'k-', alpha=0.3, linewidth=1)
        
    plt.xlabel(r"$k$")
    plt.ylabel(r"$S(k)$")
    plt.title(f"Intrinsic Variance (Same Parameters, Diff. Seeds)")
    
    # Add a text box explaining
    plt.text(0.1, 0.1, "The spread here is PURELY\ndue to quench stochasticity\n(No thermal noise)", 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/quench_noise_demo.png")
    print("Saved to results/quench_noise_demo.png")

if __name__ == "__main__":
    run_noise_demo()