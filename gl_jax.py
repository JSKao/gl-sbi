import jax
import jax.numpy as jnp
import numpy as np # 還是需要 numpy 來做一些初始化，因為 jax 的隨機數生成比較複雜
import time
from dataclasses import dataclass

# 啟用 64 位元精度 (物理模擬通常需要高精度)
jax.config.update("jax_enable_x64", True)

# --- 定義參數配置類別 ---
@dataclass
class SimConfig:
    """
    集中管理所有模擬參數。
    這樣做的好處是：以後要加新參數，只要改這裡，不用改一堆函數簽名。
    """
    # 網格與時間參數
    N: int = 128           # 格點數
    L: float = 64.0        # 系統物理尺寸
    dt: float = 0.01       # 時間步長
    steps_per_frame: int = 20

    # 全域物理參數
    eta: float = 0.8       # Josephson Coupling 強度
    B: float = 0.005       # 外加磁場強度 (Flux per plaquette ~ B*dx^2)

    # Component 1 (傾向 Type-I / 吸引)
    alpha1: float = -1.0
    beta1: float = 1.0
    D1: float = 4.0        # 大擴散係數 -> 大相干長度 xi

    # Component 2 (傾向 Type-II / 排斥)
    alpha2: float = -1.0
    beta2: float = 1.0
    D2: float = 1.0

    @property
    def dx(self):
        """自動計算 dx，避免手動算錯"""
        return self.L / self.N


class GLSolverJAX:
    
    @staticmethod
    def initialize_state(config, key):
        # 我們需要生成 4 個隨機矩陣: psi1_real, psi1_imag, psi2_real, psi2_imag
        # 分裂鑰匙 (產生 4 把子鑰匙)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # 2. 生成隨機數 
        # 記得傳入 key 和 shape=(config.N, config.N)
        # 實部
        r1 = jax.random.uniform(k1, shape=(config.N, config.N), minval=-0.5, maxval=0.5)
        # 虛部
        i1 = jax.random.uniform(k2, shape=(config.N, config.N), minval=-0.5, maxval=0.5)
        
        # 組合 psi1
        psi1 = r1 + 1j * i1
        
        # (psi2 的部分同理...)
        # 實部
        r2 = jax.random.uniform(k3, shape=(config.N, config.N), minval=-0.5, maxval=0.5)
        # 虛部
        i2 = jax.random.uniform(k4, shape=(config.N, config.N), minval=-0.5, maxval=0.5)
        
        # 組合 psi2
        psi2 = r2 + 1j * i2
        
        # 3. 組合複數並回傳
        return psi1, psi2

    
    def __init__(self, config: SimConfig):
        """
        初始化 Solver。
        我們只需要傳入一個 config 物件。
        """
        self.cfg = config  # 儲存配置，隨時可用

    # --- 靜態核心運算 (Physics Kernel) ---
    @staticmethod # JAX 喜歡靜態函數，不依賴 self 的狀態
    @jax.jit      # 自動編譯加速
    def laplacian(field, dx, B):
        """
        拉普拉斯算子。
        注意：這裡的 field 是一個不可變的 JAX array。
        包含磁場效應的拉普拉斯算子 (Peierls Substitution)
        Gauge: A = (0, B*x, 0) -> A_y 隨 x (col index) 變化
        """
        
        # 1. 取得矩陣形狀
        # jax.numpy 的 shape 是 (rows, cols)
        rows, cols = field.shape
        
        # 2. 建立座標網格 (只跟 column index j 有關)
        # 我們需要一個 row vector: [0, 1, 2, ..., cols-1]
        j_indices = jnp.arange(cols)
        
        # 3. 計算相位因子 U_y (你的公式!)
        # 這裡 q 我們設為 1，Flux = B * dx^2
        flux_per_cell = B * (dx**2)
        
        # U_y = exp(-i * flux * j)
        # 這是 shape 為 (cols,) 的向量
        uy = jnp.exp(-1j * flux_per_cell * j_indices)
        
        # 因為 field 是 2D 的，我們要把 uy 廣播 (broadcast) 成 (rows, cols)
        # 這樣每一列 (row) 都會乘上相同的相位因子
        U = jnp.tile(uy, (rows, 1))
        
        # 任務 2: JAX 也有 jnp.roll，用法跟 numpy 一模一樣
        # axis=0 是行 (垂直移動), axis=1 是列 (水平移動)
        # shift=1 (向下/向右), shift=-1 (向上/向左)
        
        # ip1 (向下跳 i -> i+1): 順著 A_y 方向，乘上 U
        # im1 (向上跳 i -> i-1): 逆著 A_y 方向，乘上 conjugate(U)
        ip1 = jnp.roll(field, shift=-1, axis=0) * U # i+1 (向下)
        im1 = jnp.roll(field, shift=1, axis=0) * jnp.conj(U) # i-1 (向上)
        
        # x 方向沒有磁場分量 (A_x = 0)，所以 U_x = 1，不用變
        jp1 = jnp.roll(field, shift=-1, axis=1) # j+1 (向右)
        jm1 = jnp.roll(field, shift=1, axis=1) # j-1 (向左)
        
        center = field
        
        # 任務 3: 組合公式 (Discrete Laplacian)
        # (up + down + left + right - 4*center) / dx^2
        lap = (ip1 + im1 + jp1 + jm1 - 4 * center) / (dx**2)
        
        return lap
    
    @staticmethod
    @jax.jit
    def potential_force(psi, alpha, beta):
        """
        定義讓超導體「凝聚」的力量。
        對應公式: - dV/d(psi*) = - (alpha * psi + beta/2 * |psi|^4) 的導數
        結果應該是: - alpha * psi - beta * |psi|^2 * psi (注意正負號定義)
        
        但在 TDGL 標準式中，我們通常寫成:
        d_psi/dt = ... - (alpha*psi + beta*|psi|^2*psi)
        如果是 alpha < 0 (超導態), 我們希望這一項把 psi 推離 0。
        標準形式通常寫為: (1 - |psi|^2)psi (當 alpha=-1, beta=1)
        
        請填空完成通式:
        force = - (alpha * psi + beta * |psi|^2 * psi)
        注意：這裡的 alpha 通常是負的 (-1)，所以 -alpha*psi 是正的（推力）。
        """
        # 提示: jnp.abs(psi)**2 取得模平方
        density = jnp.abs(psi)**2
        force = - (alpha * psi + beta * density * psi) 
        return force
    
    @staticmethod
    @jax.jit
    def interaction_force(psi_self, psi_other, eta):
        """
        耦合力。
        僅包含標準的 Josephson 耦合項: - eta * (psi1 * psi2* + c.c.)
        Force ~ eta * psi_other
        """
        return eta * psi_other
    
    @staticmethod
    @jax.jit
    def compute_current(psi, dx, B):
        """計算超導電流 (用於視覺化)"""
        rows, cols = psi.shape
        j_indices = jnp.arange(cols)
        flux = B * (dx**2)
        uy = jnp.exp(-1j * flux * j_indices)
        U = jnp.tile(uy, (rows, 1))
        
        ip1 = jnp.roll(psi, shift=-1, axis=0)
        # J_y (向下流)
        J_y = jnp.imag(jnp.conj(psi) * U * ip1) / dx
        
        jp1 = jnp.roll(psi, shift=-1, axis=1)
        # J_x (向右流)
        J_x = jnp.imag(jnp.conj(psi) * jp1) / dx
        
        return J_x, J_y
    
    @staticmethod
    @jax.jit
    def compute_curl_J(Jx, Jy, dx):
        """
        計算超導電流的旋度 (Curl)
        Curl_z = d(Jy)/dx - d(Jx)/dy
        物理意義：這張圖會顯示 '磁通量' 或 '曲率' 最集中的地方 (Vortex Cores)。
        """
        # 計算 d(Jy)/dx
        # 向右推一格 (shift=-1 on axis 1) 減去 自己
        jp1_y = jnp.roll(Jy, shift=-1, axis=1)
        dJy_dx = (jp1_y - Jy) / dx
        
        # 計算 d(Jx)/dy
        # 向下推一格 (shift=-1 on axis 0) 減去 自己
        ip1_x = jnp.roll(Jx, shift=-1, axis=0)
        dJx_dy = (ip1_x - Jx) / dx
        
        # Curl = dJy/dx - dJx/dy
        curl_z = dJy_dx - dJx_dy
        
        return curl_z
    
    @staticmethod
    @jax.jit
    def update_step(psi1, psi2, D1, D2, alpha1, beta1, alpha2, beta2, eta, dx, dt, B):
        """
        將所有力加總，進行一步時間演化 (Euler Method)。
        """
        # 1. 動能 (擴散)
        kin1 = D1 * GLSolverJAX.laplacian(psi1, dx, B)
        kin2 = D2 * GLSolverJAX.laplacian(psi2, dx, B)

        # 2. 本徵位能 (凝聚)
        pot1 = GLSolverJAX.potential_force(psi1, alpha1, beta1)
        pot2 = GLSolverJAX.potential_force(psi2, alpha2, beta2)

        # 3. 交互作用 (耦合)
        coup1 = GLSolverJAX.interaction_force(psi1, psi2, eta)
        coup2 = GLSolverJAX.interaction_force(psi2, psi1, eta)

        # 4. 更新 (歐拉法: New = Old + dt * Total_Force)
        # 任務 3: 把上面算出來的三股力量加起來
        new_psi1 = psi1 + dt * (kin1 + pot1 + coup1)
        new_psi2 = psi2 + dt * (kin2 + pot2 + coup2)

        return new_psi1, new_psi2

    
    def evolve(self, psi1, psi2, steps):
        """
        演化函數從 self.cfg 中讀取參數，
        再傳給靜態的 update_step。
        """
        c = self.cfg  # 縮寫方便讀取
        
        # 1. 準備參數 (這是常數，scan 裡面直接讀取即可)
        params = (c.D1, c.D2, c.alpha1, c.beta1, 
                  c.alpha2, c.beta2, c.eta, c.dx, c.dt, c.B)
        
        # 2. 定義迴圈本體 (Closure)
        # scan 要求函數簽名必須是 (carry, x) -> (new_carry, output)
        def body_fun(carry, _):
            # 解包目前的狀態
            p1, p2 = carry
            
            # 呼叫物理核心
            # 在這裡呼叫 self.update_step
            new_p1, new_p2 = self.update_step(p1, p2, *params)
            
            new_carry = (new_p1, new_p2)
            return new_carry, None # None 代表我們只在乎最後狀態，不需要浪費記憶體去存中間過程
        
        # 3. 啟動 JAX 迴圈引擎
        init_carry = (psi1, psi2)
        # 呼叫 jax.lax.scan
        final_state, _ = jax.lax.scan(body_fun, init_carry, jnp.arange(steps))
    
        # 4. 回傳
        return final_state    
        

# --- 測試區 ---
if __name__ == "__main__":
    # 測試時，我們可以隨意覆寫預設值
    test_config = SimConfig(N=64, B=0.05, D1=2.0)
    
    # 先實例化 Solver，因為 evolve 需要讀取 self.cfg
    sim = GLSolverJAX(test_config)
    
    # 準備 JAX 的隨機鑰匙
    key = jax.random.PRNGKey(42)
    
    # 明確傳入 config 和 key，並記得加括號呼叫
    psi1, psi2 = GLSolverJAX.initialize_state(test_config, key)
    
    print(f"Running simulation with B={sim.cfg.B}, D1={sim.cfg.D1}")
    
    # [修正 2] 必須使用實例 'sim' 來呼叫 evolve，而不是類別 'GLSolverJAX'
    final_psi1, final_psi2 = sim.evolve(psi1, psi2, steps=100)
    
    print(f"Done. Final shape: {final_psi1.shape}")