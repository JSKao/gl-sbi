import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import os

# Import modules
from src.gl_jax import GLSolverJAX, SimConfig
from src.sim_config import GRID_SIZE

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
        
        # Bins for radial averaging
        self.k_bins = np.linspace(0, np.max(self.K_mag)/2, N//2)
        self.k_centers = 0.5 * (self.k_bins[1:] + self.k_bins[:-1])

    def compute_sk_radial(self, density_field):
        delta_rho = density_field - jnp.mean(density_field)
        rho_k = jnp.fft.fft2(delta_rho)
        S_k_2d = (jnp.abs(rho_k)**2) / (self.N**2)
        
        # Vectorized binning is tricky in JAX, so we use numpy for analysis step
        # Since this is analysis (not training), it's fine.
        S_k_2d_np = np.array(S_k_2d)
        digitized = np.digitize(self.K_mag.ravel(), self.k_bins)
        
        S_k_1d = [S_k_2d_np.ravel()[digitized == i].mean() for i in range(1, len(self.k_bins))]
        return self.k_centers, np.array(S_k_1d)

def run_ensemble_simulation(n_ensembles=20, steps=2500, nu_drag=0.15):
    """
    Runs an ENSEMBLE of simulations to get error bars.
    """
    print(f"--- Starting Ensemble Simulation (N={n_ensembles}) ---")
    
    # 1. Configs
    cfg_base = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=0.0)
    cfg_drag = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=nu_drag)
    
    # 2. Prepare Batched Initial Conditions
    # We use vmap to run multiple seeds at once if GPU memory allows, 
    # or just loop if memory is tight. For N=64, looping is safer for stability.
    
    keys = jax.random.split(jax.random.PRNGKey(42), n_ensembles)
    
    # Storage for all runs
    # Shape: (n_ensembles, n_time_points)
    results_base = []
    results_drag = []
    times = []
    
    snapshot_interval = 25
    n_chunks = steps // snapshot_interval
    
    # Calculate characteristic time tau_GL
    tau_GL = 1.0 / abs(cfg_base.alpha1)
    
    analyzer = StructureFactorAnalyzer(cfg_base.N, cfg_base.L)
    
    for i in range(n_ensembles):
        print(f"  Running Realization {i+1}/{n_ensembles}...")
        
        # Init Solvers
        solver_b = GLSolverJAX(cfg_base)
        solver_d = GLSolverJAX(cfg_drag)
        
        # Common Initial Condition for this realization
        p1_b, p2_b = GLSolverJAX.initialize_state(cfg_base, keys[i])
        p1_d, p2_d = p1_b, p2_b # Clone for fair comparison
        
        sk_time_trace_b = []
        sk_time_trace_d = []
        
        # Time Loop
        for chunk in range(n_chunks):
            # Evolve
            p1_b, p2_b = solver_b.evolve(p1_b, p2_b, snapshot_interval)
            p1_d, p2_d = solver_d.evolve(p1_d, p2_d, snapshot_interval)
            
            # Analyze Low-k S(k) (k index 0-4 average)
            rho_b = jnp.abs(p1_b)**2 + jnp.abs(p2_b)**2
            _, sk_b_full = analyzer.compute_sk_radial(rho_b)
            sk_val_b = np.mean(sk_b_full[0:4]) # Average low k
            
            rho_d = jnp.abs(p1_d)**2 + jnp.abs(p2_d)**2
            _, sk_d_full = analyzer.compute_sk_radial(rho_d)
            sk_val_d = np.mean(sk_d_full[0:4])
            
            sk_time_trace_b.append(sk_val_b)
            sk_time_trace_d.append(sk_val_d)
            
            if i == 0: # Record time only once
                times.append(chunk * snapshot_interval * cfg_base.dt / tau_GL)
                
        results_base.append(sk_time_trace_b)
        results_drag.append(sk_time_trace_d)

    return np.array(times), np.array(results_base), np.array(results_drag)

def plot_with_error_bars(times, data_base, data_drag):
    """
    Plots mean curves with standard deviation shading.
    data shape: (n_ensembles, n_time_points)
    """
    # Statistics
    mean_base = np.mean(data_base, axis=0)
    std_base = np.std(data_base, axis=0) / np.sqrt(data_base.shape[0]) # Standard Error
    
    mean_drag = np.mean(data_drag, axis=0)
    std_drag = np.std(data_drag, axis=0) / np.sqrt(data_drag.shape[0])
    
    # Calculate Statistical Significance (Z-score approximate)
    # Where do the error bars NOT overlap?
    diff = np.abs(mean_base - mean_drag)
    combined_err = std_base + std_drag
    significant_mask = diff > combined_err
    
    # --- Plot 1: S(k) Evolution ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Plot Baseline
    plt.plot(times, mean_base, label=r'Baseline ($\nu=0$)', color='tab:blue')
    plt.fill_between(times, mean_base - std_base, mean_base + std_base, color='tab:blue', alpha=0.2)
    
    # Plot Drag
    plt.plot(times, mean_drag, label=r'Drag ($\nu=0.15$)', color='tab:orange')
    plt.fill_between(times, mean_drag - std_drag, mean_drag + std_drag, color='tab:orange', alpha=0.2)
    
    plt.xlabel(r"Time ($t/\tau_{GL}$)")
    plt.ylabel(r"Structure Factor $S(k \to 0)$")
    plt.title("Ensemble Averaged Dynamics ($N_{runs}=20$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- Plot 2: Difference with Significance ---
    plt.subplot(1, 2, 2)
    
    delta_S = mean_drag - mean_base # Should be negative usually (suppression)
    # Actually, let's plot absolute difference to match previous logic, 
    # OR plot (Base - Drag) to show "Suppression Magnitude" (Positive)
    suppression = mean_base - mean_drag 
    
    plt.plot(times, suppression, color='firebrick', label='Suppression Magnitude')
    plt.fill_between(times, suppression - combined_err, suppression + combined_err, color='firebrick', alpha=0.2)
    
    # Highlight significant region
    # plt.fill_between(times, 0, suppression, where=significant_mask, color='green', alpha=0.1, label='Statistically Significant')
    
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel(r"Time ($t/\tau_{GL}$)")
    plt.ylabel(r"Suppression $\Delta S = S_{base} - S_{drag}$")
    plt.title("Net Suppression Effect")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/ensemble_statistics.png")
    print("Plot saved to results/ensemble_statistics.png")

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
        
    times, res_b, res_d = run_ensemble_simulation(n_ensembles=20, steps=2000)
    plot_with_error_bars(times, res_b, res_d)