"""
Unified Analysis Suite for GL-SBI Project
Combines and optimizes functionalities from both analysis scripts.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import simpson
from scipy import stats

from src.gl_jax import GLSolverJAX, SimConfig

jax.config.update("jax_enable_x64", True)

# ========================================
# SHARED UTILITY: Structure Factor Analyzer
# ========================================

class StructureFactorAnalyzer:
    """Centralized tool for computing S(k) - SINGLE SOURCE OF TRUTH"""
    
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
        S_k_1d = [S_k_2d_np.ravel()[digitized == i].mean() 
                  for i in range(1, len(self.k_bins))]
        return self.k_centers, np.array(S_k_1d)

    def compute_low_k_mean(self, density_field, k_limit_idx=4):
        k, sk = self.compute_sk_radial(density_field)
        return np.mean(sk[:min(k_limit_idx, len(sk))])

# ========================================
# ANALYSIS 1: Dose-Response Gradient Study
# ========================================

def run_gradient_study(nu_values=[0.0, 0.15, 0.30], n_ensemble=10, steps=2500):
    """
    Multi-level drag comparison with ensemble averaging.
    Best for: Publication-quality dose-response curves.
    """
    print("=== Analysis 1: Dose-Response Gradient ===")
    
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    labels = [f'$\\nu={nu}$' for nu in nu_values]
    base_config = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04)
    tau_GL = 1.0 / abs(base_config.alpha1)
    
    snapshot_interval = 25
    n_chunks = steps // snapshot_interval
    time_axis = np.arange(n_chunks) * snapshot_interval * base_config.dt / tau_GL
    
    keys = jax.random.split(jax.random.PRNGKey(100), n_ensemble)
    analyzer = StructureFactorAnalyzer(base_config.N, base_config.L)
    
    plt.figure(figsize=(10, 6))
    
    for idx, nu in enumerate(nu_values):
        print(f"  Simulating ν={nu} ({n_ensemble} runs)...")
        avg_trace = np.zeros(n_chunks)
        
        for k in range(n_ensemble):
            cfg = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=nu)
            solver = GLSolverJAX(cfg)
            p1, p2 = GLSolverJAX.initialize_state(cfg, keys[k])
            
            for t in range(n_chunks):
                p1, p2 = solver.evolve(p1, p2, snapshot_interval)
                rho = jnp.abs(p1)**2 + jnp.abs(p2)**2
                avg_trace[t] += analyzer.compute_low_k_mean(rho)
        
        avg_trace /= n_ensemble
        plt.plot(time_axis, avg_trace, label=labels[idx], 
                color=colors[idx], linewidth=2.5, alpha=0.8)

    plt.xlabel(r"Time ($t/\tau_{GL}$)", fontsize=13)
    plt.ylabel(r"Low-$k$ Structure Factor", fontsize=13)
    plt.title("Dose-Response: Kinetic Drag Enhances Clustering", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/analysis1_gradient.png", dpi=150)
    print("  ✓ Saved: results/analysis1_gradient.png\n")

# ========================================
# ANALYSIS 2: Ensemble Statistics + Integrated Difference
# ========================================

def run_ensemble_with_metrics(n_runs=20, steps=2000, nu_drag=0.15):
    """
    Combines ensemble averaging with integrated difference metric.
    Best for: Statistical significance + quantitative comparison.
    """
    print("=== Analysis 2: Ensemble Statistics + Metrics ===")
    
    cfg_base = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=0.0)
    cfg_drag = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=nu_drag)
    tau_GL = 1.0 / abs(cfg_base.alpha1)
    
    snapshot_interval = 20
    n_chunks = steps // snapshot_interval
    keys = jax.random.split(jax.random.PRNGKey(200), n_runs)
    analyzer = StructureFactorAnalyzer(cfg_base.N, cfg_base.L)
    
    res_base, res_drag = [], []
    
    for i in range(n_runs):
        if i % 5 == 0: 
            print(f"  Run {i+1}/{n_runs}...")
        
        solver_b = GLSolverJAX(cfg_base)
        solver_d = GLSolverJAX(cfg_drag)
        p1, p2 = GLSolverJAX.initialize_state(cfg_base, keys[i])
        p1_b, p2_b = p1, p2
        p1_d, p2_d = p1, p2
        
        trace_b, trace_d = [], []
        
        for _ in range(n_chunks):
            p1_b, p2_b = solver_b.evolve(p1_b, p2_b, snapshot_interval)
            p1_d, p2_d = solver_d.evolve(p1_d, p2_d, snapshot_interval)
            
            rho_b = jnp.abs(p1_b)**2 + jnp.abs(p2_b)**2
            rho_d = jnp.abs(p1_d)**2 + jnp.abs(p2_d)**2
            
            trace_b.append(analyzer.compute_low_k_mean(rho_b))
            trace_d.append(analyzer.compute_low_k_mean(rho_d))
        
        res_base.append(trace_b)
        res_drag.append(trace_d)
    
    res_base = np.array(res_base)
    res_drag = np.array(res_drag)
    times = np.arange(n_chunks) * snapshot_interval * cfg_base.dt / tau_GL
    
    mean_base = np.mean(res_base, axis=0)
    std_base = np.std(res_base, axis=0) / np.sqrt(n_runs)
    mean_drag = np.mean(res_drag, axis=0)
    std_drag = np.std(res_drag, axis=0) / np.sqrt(n_runs)
    
    # Calculate integrated difference
    delta_S = np.abs(mean_drag - mean_base)
    integrated_diff = simpson(delta_S, x=times)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Time series with error bands
    ax = axes[0]
    ax.plot(times, mean_base, label=r'Baseline ($\nu=0$)', 
            color='tab:blue', linewidth=2)
    ax.fill_between(times, mean_base - std_base, mean_base + std_base, 
                     color='tab:blue', alpha=0.2)
    ax.plot(times, mean_drag, label=f'Drag ($\\nu={nu_drag}$)', 
            color='tab:orange', linewidth=2)
    ax.fill_between(times, mean_drag - std_drag, mean_drag + std_drag, 
                     color='tab:orange', alpha=0.2)
    ax.set_xlabel(r"Time ($t/\tau_{GL}$)", fontsize=12)
    ax.set_ylabel(r"$S(k \to 0)$", fontsize=12)
    ax.set_title(f"Ensemble Average (N={n_runs})", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Right: Integrated difference
    ax = axes[1]
    ax.plot(times, delta_S, color='red', linewidth=2)
    ax.fill_between(times, delta_S, color='red', alpha=0.2)
    ax.set_xlabel(r"Time ($t/\tau_{GL}$)", fontsize=12)
    ax.set_ylabel(r"Difference $|\Delta S(t)|$", fontsize=12)
    ax.set_title(f"Integrated Diff = {integrated_diff:.3f}", 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/analysis2_ensemble.png", dpi=150)
    print(f"  ✓ Integrated Difference: {integrated_diff:.4f}")
    print("  ✓ Saved: results/analysis2_ensemble.png\n")

# ========================================
# ANALYSIS 3: Hyperuniformity Exponent Fitting
# ========================================

def run_hyperuniformity_analysis(steps=1500, nu_drag=0.15):
    """
    Analyzes power-law scaling S(k) ~ k^ζ in the low-k regime.
    Best for: Quantifying ordering quality.
    """
    print("=== Analysis 3: Hyperuniformity Exponent ===")
    
    cfg_base = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=0.0)
    cfg_drag = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=nu_drag)
    tau_GL = 1.0 / abs(cfg_base.alpha1)
    
    solver_b = GLSolverJAX(cfg_base)
    solver_d = GLSolverJAX(cfg_drag)
    
    key = jax.random.PRNGKey(300)
    p1, p2 = GLSolverJAX.initialize_state(cfg_base, key)
    
    p1_b, p2_b = solver_b.evolve(p1, p2, steps)
    p1_d, p2_d = solver_d.evolve(p1, p2, steps)
    
    analyzer = StructureFactorAnalyzer(cfg_base.N, cfg_base.L)
    
    rho_b = jnp.abs(p1_b)**2 + jnp.abs(p2_b)**2
    rho_d = jnp.abs(p1_d)**2 + jnp.abs(p2_d)**2
    
    k_axis, S_b = analyzer.compute_sk_radial(rho_b)
    _, S_d = analyzer.compute_sk_radial(rho_d)
    
    # Fit power law in range [0.3, 1.5]
    mask = (k_axis > 0.3) & (k_axis < 1.5)
    k_fit = k_axis[mask]
    
    def fit_exponent(S_data, label):
        S_fit = S_data[mask]
        slope, intercept, r_val, _, std_err = stats.linregress(
            np.log(k_fit), np.log(S_fit))
        print(f"  {label}: ζ = {slope:.3f} ± {std_err:.3f} (R²={r_val**2:.3f})")
        return slope, intercept
    
    zeta_b, int_b = fit_exponent(S_b, "Baseline")
    zeta_d, int_d = fit_exponent(S_d, "Drag")
    
    # Plotting
    plt.figure(figsize=(7, 7))
    plt.loglog(k_axis, S_b, 'o-', label=f'Baseline $\\zeta={zeta_b:.2f}$', 
               alpha=0.6, markersize=5)
    plt.loglog(k_axis, S_d, 's-', label=f'Drag $\\zeta={zeta_d:.2f}$', 
               alpha=0.6, markersize=5)
    
    plt.loglog(k_fit, np.exp(int_b + zeta_b * np.log(k_fit)), 
               'k--', linewidth=2, label='Power-law fits')
    plt.loglog(k_fit, np.exp(int_d + zeta_d * np.log(k_fit)), 'k--', linewidth=2)
    
    plt.xlabel(r"Wavevector $k$", fontsize=13)
    plt.ylabel(r"Structure Factor $S(k)$", fontsize=13)
    plt.title(f"Hyperuniformity Analysis ($t={steps*cfg_base.dt/tau_GL:.1f}\\tau_{{GL}}$)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig("results/analysis3_hyperuniformity.png", dpi=150)
    print("  ✓ Saved: results/analysis3_hyperuniformity.png\n")

# ========================================
# ANALYSIS 4: Intrinsic Noise Demonstration
# ========================================

def run_noise_demo(n_runs=8, steps=2000, nu=0.15):
    """
    Demonstrates sensitivity to initial conditions.
    Best for: Showing why ensemble averaging is necessary.
    """
    print("=== Analysis 4: Intrinsic Quench Noise ===")
    
    cfg = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=nu)
    keys = jax.random.split(jax.random.PRNGKey(400), n_runs)
    analyzer = StructureFactorAnalyzer(cfg.N, cfg.L)
    solver = GLSolverJAX(cfg)
    
    plt.figure(figsize=(7, 7))
    for i in range(n_runs):
        p1, p2 = GLSolverJAX.initialize_state(cfg, keys[i])
        p1, p2 = solver.evolve(p1, p2, steps=steps)
        rho = jnp.abs(p1)**2 + jnp.abs(p2)**2
        k, sk = analyzer.compute_sk_radial(rho)
        plt.loglog(k, sk, 'k-', alpha=0.3, linewidth=1.5)
    
    plt.xlabel(r"Wavevector $k$", fontsize=13)
    plt.ylabel(r"Structure Factor $S(k)$", fontsize=13)
    plt.title("Configurational Variance (Same Parameters)", 
              fontsize=14, fontweight='bold')
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig("results/analysis4_noise.png", dpi=150)
    print("  ✓ Saved: results/analysis4_noise.png\n")

# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    print("\n" + "="*60)
    print("  UNIFIED ANALYSIS PIPELINE FOR GL-SBI")
    print("="*60 + "\n")
    
    # Run all analyses (comment out if you want only specific ones)
    run_gradient_study(nu_values=[0.0, 0.15, 0.30], n_ensemble=10)
    run_ensemble_with_metrics(n_runs=20, nu_drag=0.15)
    run_hyperuniformity_analysis(steps=1500, nu_drag=0.15)
    run_noise_demo(n_runs=8, nu=0.15)
    
    print("="*60)
    print("  ALL ANALYSES COMPLETE")
    print("  Results saved in 'results/' directory")
    print("="*60)