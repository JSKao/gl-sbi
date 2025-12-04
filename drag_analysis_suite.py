import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import scipy.stats as stats

# Import your existing modules
from src.gl_jax import GLSolverJAX, SimConfig
from src.sim_config import GRID_SIZE, L_SIZE

# Ensure 64-bit for precision analysis
jax.config.update("jax_enable_x64", True)

class StructureFactorAnalyzer:
    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.dx = L / N
        # Pre-compute k-grid for radial averaging
        kx = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self.K_mag = np.sqrt(KX**2 + KY**2)
        
        # Define bins for radial averaging
        self.k_bins = np.linspace(0, np.max(self.K_mag)/2, N//2)
        self.k_centers = 0.5 * (self.k_bins[1:] + self.k_bins[:-1])

    def compute_sk_radial(self, density_field):
        """
        Computes the radially averaged static structure factor S(k).
        Input: density_field (N, N)
        Output: k_centers, S_k_1d
        """
        # 1. 2D FFT
        # Subtract mean density to focus on fluctuations
        delta_rho = density_field - jnp.mean(density_field)
        rho_k = jnp.fft.fft2(delta_rho)
        S_k_2d = (jnp.abs(rho_k)**2) / (self.N**2)
        
        # 2. Radial Averaging
        S_k_2d_np = np.array(S_k_2d) # Convert to numpy for binning
        digitized = np.digitize(self.K_mag.ravel(), self.k_bins)
        
        S_k_1d = [S_k_2d_np.ravel()[digitized == i].mean() for i in range(1, len(self.k_bins))]
        return self.k_centers, np.array(S_k_1d)

def run_comparison_simulation(steps=2000, nu_drag=0.15):
    """
    Runs two simulations with IDENTICAL initial conditions:
    1. Baseline (nu=0)
    2. Drag (nu=nu_drag)
    Returns time series of S(k) for both.
    """
    print("--- Starting Comparative Simulation ---")
    
    # Common Config
    cfg_base = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=0.0)
    cfg_drag = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=nu_drag)
    
    # Initialize Solver
    solver_base = GLSolverJAX(cfg_base)
    solver_drag = GLSolverJAX(cfg_drag)
    
    # Same Initial Condition (Crucial for direct comparison)
    key = jax.random.PRNGKey(42)
    psi1_0, psi2_0 = GLSolverJAX.initialize_state(cfg_base, key)
    
    # Initialize Analysis
    analyzer = StructureFactorAnalyzer(cfg_base.N, cfg_base.L)
    
    # Storage
    times = []
    sk_time_series_base = []
    sk_time_series_drag = []
    
    # Manual Evolution Loop to track Time
    # Using smaller chunks to record data
    snapshot_interval = 20
    n_chunks = steps // snapshot_interval
    
    p1_b, p2_b = psi1_0, psi2_0
    p1_d, p2_d = psi1_0, psi2_0
    
    for i in range(n_chunks):
        t_current = i * snapshot_interval * cfg_base.dt
        times.append(t_current)
        
        # --- Evolve Baseline ---
        p1_b, p2_b = solver_base.evolve(p1_b, p2_b, snapshot_interval)
        rho_b = jnp.abs(p1_b)**2 + jnp.abs(p2_b)**2 # Total density
        _, sk_b = analyzer.compute_sk_radial(rho_b)
        sk_time_series_base.append(sk_b)
        
        # --- Evolve Drag ---
        p1_d, p2_d = solver_drag.evolve(p1_d, p2_d, snapshot_interval)
        rho_d = jnp.abs(p1_d)**2 + jnp.abs(p2_d)**2
        k_axis, sk_d = analyzer.compute_sk_radial(rho_d)
        sk_time_series_drag.append(sk_d)
        
        if i % 10 == 0:
            print(f"  Step {i*snapshot_interval}/{steps} complete.")
            
    return np.array(times), np.array(sk_time_series_base), np.array(sk_time_series_drag), k_axis, cfg_base

def analyze_results(times, sk_base, sk_drag, k_axis, config):
    """
    Performs the 3 specific analyses requested by the user.
    """
    
    # ==========================================
    # Task 2: Timescale Normalization (Do this first to plot x-axis correctly)
    # ==========================================
    # Theory: tau_GL = 1 / |alpha| for dimensionless GL equations.
    # In user's code: alpha1 = -1.0. 
    # Therefore, tau_GL = 1.0 / 1.0 = 1.0 (Simulation Time Unit)
    # But for generality, we calculate it:
    tau_GL = 1.0 / abs(config.alpha1) 
    times_norm = times / tau_GL
    
    print(f"\n[Task 2] Timescale Normalization:")
    print(f"  alpha = {config.alpha1}")
    print(f"  tau_GL (Characteristic Relaxation Time) = {tau_GL:.2f} simulation units")
    print("  The time axis in plots will be normalized by this value.")

    # ==========================================
    # Task 1: Integrated Difference Metric
    # ==========================================
    # Focus on low-k region (e.g., first 5 bins) where clustering happens
    k_idx_low = slice(0, 5) 
    S_low_k_base = np.mean(sk_base[:, k_idx_low], axis=1)
    S_low_k_drag = np.mean(sk_drag[:, k_idx_low], axis=1)
    
    # Calculate difference
    delta_S = np.abs(S_low_k_drag - S_low_k_base)
    
    # Integrate over time
    integrated_diff = simpson(delta_S, x=times_norm)
    
    print(f"\n[Task 1] Integrated Difference Metric:")
    print(f"  Integrated Difference I = {integrated_diff:.4f}")
    
    # Plotting Task 1
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(times_norm, S_low_k_base, label='Baseline ($\\nu=0$)', alpha=0.7)
    plt.plot(times_norm, S_low_k_drag, label='Drag ($\\nu=0.15$)', alpha=0.7)
    plt.xlabel(r"Time ($t/\tau_{GL}$)")
    plt.ylabel(r"$S(k \to 0)$")
    plt.title("Evolution of Low-k Structure Factor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(times_norm, delta_S, color='red')
    plt.fill_between(times_norm, delta_S, color='red', alpha=0.2)
    plt.xlabel(r"Time ($t/\tau_{GL}$)")
    plt.ylabel(r"Difference $\Delta S(t)$")
    plt.title(f"Dynamical Signature\nIntegrated Diff = {integrated_diff:.2f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/task1_integrated_difference.png")
    plt.close()

    # ==========================================
    # Task 3: Hyperuniformity Exponent Fitting
    # ==========================================
    # Select a snapshot near t/tau = 1.0 (Peak difference)
    target_time_idx = np.argmin(np.abs(times_norm - 1.5)) # Look at t=1.5 tau
    print(f"\n[Task 3] Hyperuniformity Exponent (at t = {times_norm[target_time_idx]:.2f} tau):")
    
    S_snap_base = sk_base[target_time_idx]
    S_snap_drag = sk_drag[target_time_idx]
    
    # Filter for low k (k < 1.5, before the structure peak)
    mask = (k_axis > 0.3) & (k_axis < 1.5)
    k_fit = k_axis[mask]
    
    # Linear Regression in Log-Log
    def fit_exponent(S_data, label):
        S_fit = S_data[mask]
        log_k = np.log(k_fit)
        log_S = np.log(S_fit)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_S)
        print(f"  {label}: zeta = {slope:.4f} +/- {std_err:.4f} (R2={r_value**2:.3f})")
        return slope, intercept, log_k, log_S
        
    zeta_b, int_b, lk_b, ls_b = fit_exponent(S_snap_base, "Baseline")
    zeta_d, int_d, lk_d, ls_d = fit_exponent(S_snap_drag, "Drag")
    
    # Plotting Task 3
    plt.figure(figsize=(6, 6))
    plt.loglog(k_axis, S_snap_base, 'o-', label=f'Baseline $\\zeta={zeta_b:.2f}$', alpha=0.5, markersize=4)
    plt.loglog(k_axis, S_snap_drag, 's-', label=f'Drag $\\zeta={zeta_d:.2f}$', alpha=0.5, markersize=4)
    
    # Plot fit lines
    plt.loglog(k_fit, np.exp(int_b + zeta_b * np.log(k_fit)), 'k--', linewidth=1)
    plt.loglog(k_fit, np.exp(int_d + zeta_d * np.log(k_fit)), 'k--', linewidth=1)

    plt.xlabel(r"Wavevector $k$")
    plt.ylabel(r"Structure Factor $S(k)$")
    plt.title(f"Hyperuniformity Fit ($t={times_norm[target_time_idx]:.1f}\\tau_{{GL}}$)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.savefig("results/task3_hyperuniformity.png")
    plt.close()

if __name__ == "__main__":
    # Ensure results dir exists
    import os
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # Run Pipeline
    times, sk_base, sk_drag, k_axis, config = run_comparison_simulation(steps=1500)
    analyze_results(times, sk_base, sk_drag, k_axis, config)
    print("\nAnalysis Complete. Check 'results/' folder for plots.")