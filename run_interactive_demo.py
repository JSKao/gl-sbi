import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

# Ensure the 'src' directory is in the Python path to import modules
# (Uncomment the line below if your gl_jax.py is inside a 'src' folder)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.gl_jax import GLSolverJAX, SimConfig

# --- 1. Experimental Setup ---
config = SimConfig(
    N = 128,            
    L = 64.0,           
    dt = 0.01,
    eta = 0.8,          # Initial coupling
    B = 0.005,          # Initial magnetic field
    
    # Physics: Two-component competition
    D1 = 4.0,           # Component 1: Stiff / Large coherence length (Type-I like)
    D2 = 1.0,           # Component 2: Soft / Small coherence length (Type-II like)
    steps_per_frame = 5
)

print(f"Initializing Virtual Laboratory...")
print(f"System Size: {config.N}x{config.N}, Box Length: {config.L}")

# Initialize Solver and State
# Note: GLSolverJAX is stateless. We must manage 'psi' explicitly.
sim = GLSolverJAX(config)
key = jax.random.PRNGKey(42)
psi1, psi2 = GLSolverJAX.initialize_state(config, key)

# --- 2. Visualization Layout (2x2 Grid) ---
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
plt.subplots_adjust(bottom=0.2, hspace=0.3) # Reserve space for sliders
fig.suptitle(fr"Multi-Component Ginzburg-Landau Laboratory", fontsize=16)

# Assign subplots
ax_d1, ax_d2 = axs[0] # Top row: Density
ax_p1, ax_p2 = axs[1] # Bottom row: Phase

# Convert JAX arrays to NumPy for plotting
d1 = np.array(jnp.abs(psi1)**2)
d2 = np.array(jnp.abs(psi2)**2)
p1 = np.array(jnp.angle(psi1))
p2 = np.array(jnp.angle(psi2))

# --- Top Row: Density Maps ---
# Component 1 (Top-Left)
im_d1 = ax_d1.imshow(d1, cmap='inferno', origin='lower', vmin=0, vmax=2.0)
ax_d1.set_title(r"Density $|\psi_1|^2$ (Type-I behavior)")
plt.colorbar(im_d1, ax=ax_d1, fraction=0.046, pad=0.04)

# Component 2 (Top-Right)
im_d2 = ax_d2.imshow(d2, cmap='inferno', origin='lower', vmin=0, vmax=2.0)
ax_d2.set_title(r"Density $|\psi_2|^2$ (Type-II behavior)")
plt.colorbar(im_d2, ax=ax_d2, fraction=0.046, pad=0.04)

# --- Bottom Row: Phase Maps ---
# Component 1 (Bottom-Left)
im_p1 = ax_p1.imshow(p1, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
ax_p1.set_title(r"Phase $\theta_1$")
plt.colorbar(im_p1, ax=ax_p1, fraction=0.046, pad=0.04)

# Component 2 (Bottom-Right)
im_p2 = ax_p2.imshow(p2, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
ax_p2.set_title(r"Phase $\theta_2$")
plt.colorbar(im_p2, ax=ax_p2, fraction=0.046, pad=0.04)

# Quiver Plot for Supercurrent (Overlay on Top-Left)
skip = 8
Y_grid, X_grid = np.mgrid[0:config.N:skip, 0:config.N:skip]
Q = ax_d1.quiver(X_grid, Y_grid, np.zeros_like(X_grid), np.zeros_like(Y_grid), 
                 color='white', scale=20, width=0.005, alpha=0.7)

# Text Annotations
time_text = ax_d1.text(5, 5, '', color='white', fontsize=10, fontweight='bold')
param_text = ax_d2.text(5, 5, '', color='white', fontsize=10, fontweight='bold')

# --- Interactive Controls ---
ax_slider_B = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_eta = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgoldenrodyellow')

s_B = Slider(ax_slider_B, 'Mag Field B', 0.0, 0.02, valinit=config.B, valstep=0.001)
s_eta = Slider(ax_slider_eta, 'Coupling $\eta$', 0.0, 1.5, valinit=config.eta, valstep=0.1)

def update_params(val):
    # Update the config object in real-time
    sim.cfg.B = s_B.val
    sim.cfg.eta = s_eta.val

s_B.on_changed(update_params)
s_eta.on_changed(update_params)

# Keyboard Interaction (Gauge Transformation Demo)
view_state = {'chi': jnp.zeros((config.N, config.N))}

def on_key(event):
    if event.key == 'g':
        # Apply random gauge transformation
        print("Applying Gauge Transformation...")
        view_state['chi'] = jnp.array(np.random.rand(config.N, config.N) * 2 * np.pi)
    elif event.key == 'r':
        # Reset gauge
        view_state['chi'] = jnp.zeros((config.N, config.N))
        s_B.reset()
        s_eta.reset()

fig.canvas.mpl_connect('key_press_event', on_key)


# --- 3. Animation Loop ---
def update(frame):
    global psi1, psi2 # We need to update the global state variables
    
    # Evolve the system (Stateless update)
    psi1, psi2 = sim.evolve(psi1, psi2, config.steps_per_frame)
    
    # Apply visualization gauge (if any)
    chi = view_state['chi']
    p1_disp = psi1 * jnp.exp(1j * chi)
    p2_disp = psi2 * jnp.exp(1j * chi)
    
    # Update Density Plots
    im_d1.set_data(np.array(jnp.abs(p1_disp)**2))
    im_d2.set_data(np.array(jnp.abs(p2_disp)**2))
    
    # Update Phase Plots
    im_p1.set_data(np.array(jnp.angle(p1_disp)))
    im_p2.set_data(np.array(jnp.angle(p2_disp)))
    
    # Update Supercurrent (Compute for Component 1)
    Jx, Jy = sim.compute_current(psi1, sim.cfg.dx, sim.cfg.B)
    Jx_plot = np.array(Jx)[::skip, ::skip]
    Jy_plot = np.array(Jy)[::skip, ::skip]
    Q.set_UVC(Jx_plot, Jy_plot)
    
    # Update Info Text
    time_text.set_text(f"Step: {(frame+1)*config.steps_per_frame}")
    param_text.set_text(fr"$B={sim.cfg.B:.3f}, \eta={sim.cfg.eta:.1f}$")
    
    return im_d1, im_d2, im_p1, im_p2, time_text, param_text, Q

print("Simulation started. Use sliders to adjust B and eta.")
ani = FuncAnimation(fig, update, frames=500, interval=50, blit=False)
plt.show()