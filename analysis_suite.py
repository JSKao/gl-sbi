import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.integrate import simpson

# Import your simulation modules
from src.gl_jax import GLSolverJAX, SimConfig

# Enable 64-bit precision for stability
jax.config.update("jax_enable_x64", True)

class StructureFactorAnalyzer:
    """
    Helper class to compute radially averaged Structure Factor S(k).
    """
    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.dx = L / N
        kx = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self.K_mag = np.sqrt(KX**2 + KY**2)
        
        # Define bins for radial averaging
        # Binning up to Nyquist frequency
        self.k_bins = np.linspace(0, np.max(self.K_mag)/2, N//2)
        self.k_centers = 0.5 * (self.k_bins[1:] + self.k_bins[:-1])

    def compute_sk_radial(self, density_field):
        """
        Computes S(k) for a single snapshot.
        """
        delta_rho = density_field - jnp.mean(density_field)
        rho_k = jnp.fft.fft2(delta_rho)
        S_k_2d = (jnp.abs(rho_k)**2) / (self.N**2)
        
        # Radial averaging using numpy
        S_k_2d_np = np.array(S_k_2d)
        digitized = np.digitize(self.K_mag.ravel(), self.k_bins)
        
        S_k_1d = [S_k_2d_np.ravel()[digitized == i].mean() for i in range(1, len(self.k_bins))]
        return self.k_centers, np.array(S_k_1d)

    def compute_low_k_mean(self, density_field, k_limit_idx=4):
        """
        Fast computation of mean S(k) in the low-k limit.
        """
        k, sk = self.compute_sk_radial(density_field)
        if len(sk) < k_limit_idx:
            return np.mean(sk)
        return np.mean(sk[:k_limit_idx])

# --- Task 1: Dose-Response Gradient Study ---
def run_gradient_study():
    print("--- Task 1: Dose-Response Gradient Study ---")
    nu_values = [0.0, 0.15, 0.30]
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    labels = [r'Baseline ($\nu=0$)', r'Weak Drag ($\nu=0.15$)', r'Strong Drag ($\nu=0.30$)']
    
    # Common Parameters
    steps = 2500
    snapshot_interval = 25
    base_config = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04)
    tau_GL = 1.0 / abs(base_config.alpha1)
    
    n_ensemble = 10 # Average over 10 runs to smooth curves
    keys = jax.random.split(jax.random.PRNGKey(100), n_ensemble)
    analyzer = StructureFactorAnalyzer(base_config.N, base_config.L)
    
    n_chunks = steps // snapshot_interval
    time_axis = np.arange(n_chunks) * snapshot_interval * base_config.dt / tau_GL
    
    plt.figure(figsize=(8, 6))
    
    for idx, nu in enumerate(nu_values):
        print(f"  Simulating nu = {nu}...")
        avg_trace = np.zeros(n_chunks)
        
        for k in range(n_ensemble):
            cfg = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=nu)
            solver = GLSolverJAX(cfg)
            p1, p2 = GLSolverJAX.initialize_state(cfg, keys[k]) # Same seeds for fair comparison
            
            trace = []
            for _ in range(n_chunks):
                p1, p2 = solver.evolve(p1, p2, snapshot_interval)
                rho = jnp.abs(p1)**2 + jnp.abs(p2)**2
                trace.append(analyzer.compute_low_k_mean(rho))
            avg_trace += np.array(trace)
            
        avg_trace /= n_ensemble
        plt.plot(time_axis, avg_trace, label=labels[idx], color=colors[idx], linewidth=2)

    plt.xlabel(r"Time ($t/\tau_{GL}$)")
    plt.ylabel(r"Low-$k$ Structure Factor $S(k \to 0)$")
    plt.title("Dose-Response: Kinetic Locking Enhances Clustering")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/drag_gradient_proof.png")
    print("  Saved: results/drag_gradient_proof.png")

# --- Task 2: Ensemble Statistics ---
def run_ensemble_statistics():
    print("--- Task 2: Ensemble Statistics ---")
    n_runs = 20
    steps = 2500
    snapshot_interval = 25
    
    cfg_base = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=0.0)
    cfg_drag = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=0.15)
    tau_GL = 1.0 / abs(cfg_base.alpha1)
    
    keys = jax.random.split(jax.random.PRNGKey(200), n_runs)
    analyzer = StructureFactorAnalyzer(cfg_base.N, cfg_base.L)
    
    res_base = []
    res_drag = []
    times = []
    
    for i in range(n_runs):
        if i % 5 == 0: print(f"  Run {i}/{n_runs}...")
        solver_b = GLSolverJAX(cfg_base)
        solver_d = GLSolverJAX(cfg_drag)
        p1_b, p2_b = GLSolverJAX.initialize_state(cfg_base, keys[i])
        p1_d, p2_d = p1_b, p2_b
        
        trace_b, trace_d = [], []
        
        for chunk in range(steps // snapshot_interval):
            p1_b, p2_b = solver_b.evolve(p1_b, p2_b, snapshot_interval)
            p1_d, p2_d = solver_d.evolve(p1_d, p2_d, snapshot_interval)
            
            rho_b = jnp.abs(p1_b)**2 + jnp.abs(p2_b)**2
            rho_d = jnp.abs(p1_d)**2 + jnp.abs(p2_d)**2
            
            trace_b.append(analyzer.compute_low_k_mean(rho_b))
            trace_d.append(analyzer.compute_low_k_mean(rho_d))
            
            if i == 0:
                times.append(chunk * snapshot_interval * cfg_base.dt / tau_GL)
                
        res_base.append(trace_b)
        res_drag.append(trace_d)
        
    # Plotting
    res_base = np.array(res_base)
    res_drag = np.array(res_drag)
    mean_base = np.mean(res_base, axis=0)
    std_base = np.std(res_base, axis=0) / np.sqrt(n_runs)
    mean_drag = np.mean(res_drag, axis=0)
    std_drag = np.std(res_drag, axis=0) / np.sqrt(n_runs)
    
    plt.figure(figsize=(10, 5))
    plt.plot(times, mean_base, label=r'Baseline ($\nu=0$)', color='tab:blue')
    plt.fill_between(times, mean_base - std_base, mean_base + std_base, color='tab:blue', alpha=0.2)
    plt.plot(times, mean_drag, label=r'Drag ($\nu=0.15$)', color='tab:orange')
    plt.fill_between(times, mean_drag - std_drag, mean_drag + std_drag, color='tab:orange', alpha=0.2)
    
    plt.xlabel(r"Time ($t/\tau_{GL}$)")
    plt.ylabel(r"$S(k \to 0)$")
    plt.title(f"Ensemble Statistics ($N={n_runs}$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/ensemble_statistics.png")
    print("  Saved: results/ensemble_statistics.png")

# --- Task 3: Intrinsic Noise Demo ---
def run_noise_demo():
    print("--- Task 3: Intrinsic Quench Noise ---")
    cfg = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=0.15)
    n_runs = 8
    keys = jax.random.split(jax.random.PRNGKey(300), n_runs)
    analyzer = StructureFactorAnalyzer(cfg.N, cfg.L)
    solver = GLSolverJAX(cfg)
    
    plt.figure(figsize=(7, 6))
    for i in range(n_runs):
        p1, p2 = GLSolverJAX.initialize_state(cfg, keys[i])
        p1, p2 = solver.evolve(p1, p2, steps=2000) # Go to formation epoch
        rho = jnp.abs(p1)**2 + jnp.abs(p2)**2
        k, sk = analyzer.compute_sk_radial(rho)
        plt.loglog(k, sk, 'k-', alpha=0.3)
        
    plt.xlabel(r"$k$")
    plt.ylabel(r"$S(k)$")
    plt.title("Intrinsic Config. Variance (Same Params)")
    plt.savefig("results/quench_noise_demo.png")
    print("  Saved: results/quench_noise_demo.png")

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Run all analyses
    # Comment out if you only want to run specific ones
    run_gradient_study()
    run_ensemble_statistics()
    run_noise_demo()
    print("\nAll analysis tasks complete.")