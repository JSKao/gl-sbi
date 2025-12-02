# src/analysis_drag.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sim_config import GRID_SIZE

def compute_structure_factor(images):
    """
    Computes the radially averaged Structure Factor S(k).
    S(k) = < |FFT(rho)|^2 >
    
    This helps us detect 'Hyperuniformity' (suppression of density fluctuations at low k).
    """
    # 1. Take density channel (channel 0: |psi1|^2)
    # images shape: (N_samples, H, W, 3)
    rho = images[..., 0] 
    N_batch = rho.shape[0]
    L_pixels = rho.shape[1] # grid size (e.g., 64)
    
    print(f"   Computing FFT for {N_batch} images (Grid: {L_pixels}x{L_pixels})...")

    # 2. Subtract mean density (we care about fluctuations)
    # rho_fluc: deviation from the average density of that specific image
    rho_mean = np.mean(rho, axis=(1, 2), keepdims=True)
    rho_fluc = rho - rho_mean
    
    # 3. FFT (2D)
    rho_k = np.fft.fft2(rho_fluc)
    rho_k_shifted = np.fft.fftshift(rho_k, axes=(1, 2)) # Shift zero freq to center
    
    # Power Spectrum P(k) ~ |rho_k|^2
    S_k_2d = np.mean(np.abs(rho_k_shifted)**2, axis=0) # Average over batch
    
    # Normalization (convention varies, but we care about relative shape)
    S_k_2d /= (L_pixels * L_pixels)
    
    # 4. Radial Averaging
    # Create coordinate grid relative to center
    y, x = np.indices((L_pixels, L_pixels))
    center = np.array([(L_pixels-1)/2, (L_pixels-1)/2])
    
    # Calculate radius r for every pixel
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Binning by integer radius
    r_int = r.astype(int)
    
    # Sum S(k) values in each bin
    tbin = np.bincount(r_int.ravel(), S_k_2d.ravel())
    # Count number of pixels in each bin
    nr = np.bincount(r_int.ravel())
    
    # Avoid division by zero
    valid_mask = nr > 0
    radial_profile = np.zeros_like(tbin, dtype=float)
    radial_profile[valid_mask] = tbin[valid_mask] / nr[valid_mask]
    
    # We typically ignore the DC component (r=0) and go up to Nyquist (L/2)
    return radial_profile[1 : L_pixels//2]

def main():
    # 1. Load Data
    data_path = f"data/dataset_{GRID_SIZE}.npz"
    print(f"🔍 Loading dataset: {data_path}")
    
    if not os.path.exists(data_path):
        print(" Data not found! Please run generate_data.py first.")
        return

    data = np.load(data_path)
    
    # Critical Check: Do we have labels?
    if 'label' not in data:
        print(" Error: 'label' key missing in dataset!")
        print("   Please make sure your generate_data.py saves 'label' (Model A vs Model B).")
        return
        
    x = data['x']
    labels = data['label'].ravel() # (N,)
    theta = data['theta']
    
    print(f"   Total Samples: {len(labels)}")
    print(f"   Model A (No Drag, nu=0): {np.sum(labels < 0.5)}")
    print(f"   Model B (Drag, nu>0):    {np.sum(labels > 0.5)}")
    
    # --- 2. Physical Layer: Hyperuniformity Analysis ---
    print("\n🔬 Performing Structure Factor Analysis...")
    
    # Split data based on label
    mask_model_A = (labels < 0.5) # No Drag
    mask_model_B = (labels > 0.5) # With Drag
    
    if np.sum(mask_model_A) == 0 or np.sum(mask_model_B) == 0:
        print(" Warning: One of the classes is empty. Cannot compare.")
        return
    
    img_A = x[mask_model_A]
    img_B = x[mask_model_B]
    
    Sk_A = compute_structure_factor(img_A)
    Sk_B = compute_structure_factor(img_B)
    
    k_axis = np.arange(1, len(Sk_A) + 1)
    
    # --- 3. Visualization ---
    print("\n🎨 Plotting Physics Proof...")
    
    plt.figure(figsize=(12, 5))
    
    # Left Panel: Full S(k)
    plt.subplot(1, 2, 1)
    plt.plot(k_axis, Sk_A, label='Model A (Standard)', color='blue', alpha=0.7, linewidth=2)
    plt.plot(k_axis, Sk_B, label='Model B (Drag/Andreev-Bashkin)', color='red', alpha=0.7, linewidth=2)
    plt.title(f"Structure Factor $S(k)$ (Grid {GRID_SIZE}x{GRID_SIZE})")
    plt.xlabel("Wavevector $k$ (Radial)")
    plt.ylabel("Intensity $S(k)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right Panel: Zoom at Low-k (The Physics Signature)
    plt.subplot(1, 2, 2)
    limit_k = 10 # Zoom in first 10 modes
    plt.plot(k_axis[:limit_k], Sk_A[:limit_k], 'o-', label='Standard', color='blue')
    plt.plot(k_axis[:limit_k], Sk_B[:limit_k], 's-', label='Drag Enhanced', color='red')
    
    # Annotation
    plt.title(r"Low-$k$ Region (Large Scale Structure)")
    plt.xlabel("$k$")
    plt.ylabel("$S(k)$")
    plt.grid(True, alpha=0.3)
    
    # Physics Interpretation Text
    diff = Sk_A[0] - Sk_B[0]
    note = "Drag suppresses fluctuations!" if diff > 0 else "Drag enhances fluctuations!"
    plt.figtext(0.5, 0.01, f"Observation: {note}", ha="center", fontsize=12, color='darkgreen')

    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    save_path = "assets/drag_physics_proof.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f" Analysis saved to {save_path}")
    print(f"   Interpretation: Check if the Red curve is lower/higher than Blue at k=1.")

if __name__ == "__main__":
    main()