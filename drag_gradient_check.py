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

    def compute_low_k_mean(self, density_field, k_limit_idx=5):
        delta_rho = density_field - jnp.mean(density_field)
        rho_k = jnp.fft.fft2(delta_rho)
        S_k_2d = (jnp.abs(rho_k)**2) / (self.N**2)
        S_k_2d_np = np.array(S_k_2d)
        digitized = np.digitize(self.K_mag.ravel(), self.k_bins)
        
        # Fast compute only low k bins
        vals = []
        for i in range(1, k_limit_idx + 1):
            v = S_k_2d_np.ravel()[digitized == i].mean()
            vals.append(v)
        return np.mean(vals)

def run_gradient_study():
    print("--- Starting Drag Gradient Study (Dose-Response) ---")
    
    # Define a gradient of nu values
    # Let's push nu to 0.3 to see if the effect is drastic
    nu_values = [0.0, 0.15, 0.30] 
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    labels = [r'Baseline ($\nu=0$)', r'Weak Drag ($\nu=0.15$)', r'Strong Drag ($\nu=0.30$)']
    
    steps = 2500
    snapshot_interval = 25
    n_chunks = steps // snapshot_interval
    
    # Common Parameters
    base_config = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04)
    tau_GL = 1.0 / abs(base_config.alpha1)
    
    # Use a larger ensemble to smooth out curves, but sum them up
    n_ensemble_per_nu = 10 
    keys = jax.random.split(jax.random.PRNGKey(999), n_ensemble_per_nu)
    
    analyzer = StructureFactorAnalyzer(base_config.N, base_config.L)
    
    plt.figure(figsize=(8, 6))
    
    for idx, nu in enumerate(nu_values):
        print(f"  Simulating nu = {nu}...")
        
        # Storage for ensemble average
        avg_trace = np.zeros(n_chunks)
        time_axis = np.arange(n_chunks) * snapshot_interval * base_config.dt / tau_GL
        
        for k in range(n_ensemble_per_nu):
            # Init solver with specific nu
            cfg = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=nu)
            solver = GLSolverJAX(cfg)
            
            # Same seeds for each nu to reduce variance between lines
            p1, p2 = GLSolverJAX.initialize_state(cfg, keys[k])
            
            trace = []
            for _ in range(n_chunks):
                p1, p2 = solver.evolve(p1, p2, snapshot_interval)
                rho = jnp.abs(p1)**2 + jnp.abs(p2)**2
                val = analyzer.compute_low_k_mean(rho)
                trace.append(val)
            
            avg_trace += np.array(trace)
            
        avg_trace /= n_ensemble_per_nu
        
        # Plot
        plt.plot(time_axis, avg_trace, label=labels[idx], color=colors[idx], linewidth=2)

    plt.xlabel(r"Time ($t/\tau_{GL}$)")
    plt.ylabel(r"Low-$k$ Structure Factor $S(k \to 0)$")
    plt.title("Dose-Response: Effect of Drag Strength on Fluctuations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in on the peak area
    # plt.xlim(1.5, 4.0) 
    
    os.makedirs("results", exist_ok=True)
    save_path = "results/drag_gradient_proof.png"
    plt.savefig(save_path)
    print(f"Gradient plot saved to {save_path}")

if __name__ == "__main__":
    run_gradient_study()