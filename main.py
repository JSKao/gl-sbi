import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np
import jax.numpy as jnp
from gl_jax import GLSolverJAX, SimConfig 

# --- 1. 實驗參數設定 ---
config = SimConfig(
    N = 128,            
    L = 64.0,           
    dt = 0.01,
    eta = 0.8,          
    B = 0.005,          # 小磁場
    
    # [物理關鍵] 讓兩個分量性質不同，才能看出差異
    D1 = 4.0,           # Component 1: 硬/擴散快 (Type-I 傾向)
    D2 = 1.0,           # Component 2: 軟/擴散慢 (Type-II 傾向)
    steps_per_frame = 5
)

print(f"正在啟動四分割虛擬實驗室...")
sim = GLSolverJAX(config)

# --- 2. 準備畫布 (2x2 Layout) ---
# figsize=(10, 8) 讓視窗大一點
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
plt.subplots_adjust(bottom=0.2, hspace=0.3) # 留白給 Slider
fig.suptitle(fr"Multi-Component Ginzburg-Landau Lab", fontsize=16)

# 分配子圖
ax_d1, ax_d2 = axs[0] # 上排: 密度
ax_p1, ax_p2 = axs[1] # 下排: 相位

# 初始資料
d1 = np.array(jnp.abs(sim.psi1)**2)
d2 = np.array(jnp.abs(sim.psi2)**2)
p1 = np.array(jnp.angle(sim.psi1))
p2 = np.array(jnp.angle(sim.psi2))

# --- 上排: 密度圖 (Density) ---
# Component 1 (左上)
im_d1 = ax_d1.imshow(d1, cmap='inferno', origin='lower', vmin=0, vmax=2.0)
ax_d1.set_title(r"Density $|\psi_1|^2$ (Type-I like)")
plt.colorbar(im_d1, ax=ax_d1, fraction=0.046, pad=0.04)

# Component 2 (右上)
im_d2 = ax_d2.imshow(d2, cmap='inferno', origin='lower', vmin=0, vmax=2.0)
ax_d2.set_title(r"Density $|\psi_2|^2$ (Type-II like)")
plt.colorbar(im_d2, ax=ax_d2, fraction=0.046, pad=0.04)

# --- 下排: 相位圖 (Phase) ---
# Component 1 (左下)
im_p1 = ax_p1.imshow(p1, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
ax_p1.set_title(r"Phase $\theta_1$")
plt.colorbar(im_p1, ax=ax_p1, fraction=0.046, pad=0.04)

# Component 2 (右下)
im_p2 = ax_p2.imshow(p2, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
ax_p2.set_title(r"Phase $\theta_2$")
plt.colorbar(im_p2, ax=ax_p2, fraction=0.046, pad=0.04)


# 初始化 Quiver (只畫在左上角 Component 1，避免太亂)
skip = 8
Y_grid, X_grid = np.mgrid[0:config.N:skip, 0:config.N:skip]
Q = ax_d1.quiver(X_grid, Y_grid, np.zeros_like(X_grid), np.zeros_like(Y_grid), 
                 color='white', scale=20, width=0.005, alpha=0.7)

# 文字顯示
time_text = ax_d1.text(5, 5, '', color='white', fontsize=10, fontweight='bold')
param_text = ax_d2.text(5, 5, '', color='white', fontsize=10, fontweight='bold')

# --- [互動控制區] ---
ax_slider_B = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_eta = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgoldenrodyellow')

s_B = Slider(ax_slider_B, 'Mag Field B', 0.0, 0.02, valinit=config.B, valstep=0.001)
s_eta = Slider(ax_slider_eta, 'Coupling $\eta$', 0.0, 1.5, valinit=config.eta, valstep=0.1)

def update_params(val):
    sim.cfg.B = s_B.val
    sim.cfg.eta = s_eta.val

s_B.on_changed(update_params)
s_eta.on_changed(update_params)

# 鍵盤互動 (保留 Gauge 功能)
view_state = {'chi': jnp.zeros((config.N, config.N))}
def on_key(event):
    if event.key == 'g':
        view_state['chi'] = jnp.array(np.random.rand(config.N, config.N) * 2 * np.pi)
    elif event.key == 'r':
        view_state['chi'] = jnp.zeros((config.N, config.N))
        s_B.reset()
        s_eta.reset()

fig.canvas.mpl_connect('key_press_event', on_key)


# --- 3. 動畫更新 ---
def update(frame):
    sim.evolve(config.steps_per_frame)
    
    # 取得兩組波函數
    psi1 = sim.psi1
    psi2 = sim.psi2
    
    # 應用展示用的規範變換
    chi = view_state['chi']
    p1_disp = psi1 * jnp.exp(1j * chi)
    p2_disp = psi2 * jnp.exp(1j * chi)
    
    # 更新圖像
    im_d1.set_data(np.array(jnp.abs(p1_disp)**2))
    im_d2.set_data(np.array(jnp.abs(p2_disp)**2))
    
    im_p1.set_data(np.array(jnp.angle(p1_disp)))
    im_p2.set_data(np.array(jnp.angle(p2_disp)))
    
    # 更新電流 (只畫 C1)
    Jx, Jy = sim.compute_current(psi1, sim.cfg.dx, sim.cfg.B)
    Jx_plot = np.array(Jx)[::skip, ::skip]
    Jy_plot = np.array(Jy)[::skip, ::skip]
    Q.set_UVC(Jx_plot, Jy_plot)
    
    # 更新文字
    time_text.set_text(f"Step: {(frame+1)*config.steps_per_frame}")
    param_text.set_text(fr"$B={sim.cfg.B:.3f}, \eta={sim.cfg.eta:.1f}$")
    
    return im_d1, im_d2, im_p1, im_p2, time_text, param_text, Q

print("視窗已開啟... 左右滑動 eta 來觀察 Vortex 鎖定與解鎖！")
ani = FuncAnimation(fig, update, frames=500, interval=50, blit=False)
plt.show()