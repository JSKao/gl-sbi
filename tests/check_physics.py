import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from src.gl_jax import GLSolverJAX, SimConfig

def main():
    print("🧪 正在進行物理差異性測試...")
    
    # 固定亂數種子
    key = jax.random.PRNGKey(555)
    key_sim = jax.random.split(key)[0]
    
    # 設定 1: 低磁場 (B=0)
    cfg_low = SimConfig(eta=0.8, B=0.000, N=128)
    # 設定 2: 高磁場 (B=0.02)
    cfg_high = SimConfig(eta=0.8, B=0.20, N=128)
    
    print("1. 執行 B=0.000 模擬...")
    solver_low = GLSolverJAX(cfg_low)
    # [修正] 解包：分別接住 psi1 和 psi2
    psi1_init, psi2_init = GLSolverJAX.initialize_state(cfg_low, key_sim)
    
    # [修正] 傳遞：將兩個參數分開傳入 evolve
    p1_low, p2_low = solver_low.evolve(psi1_init, psi2_init, 1000)
    
    print("2. 執行 B=0.20 模擬...")
    solver_high = GLSolverJAX(cfg_high)
    # 注意：使用完全相同的初始態 (控制變因)
    p1_high, p2_high = solver_high.evolve(psi1_init, psi2_init, 1000)
    
    # 定義特徵提取函數 (針對單一分量計算)
    def get_features(solver, psi, cfg):
        rho = jnp.abs(psi)**2
        Jx, Jy = solver.compute_current(psi, cfg.dx, cfg.B)
        curl = solver.compute_curl_J(Jx, Jy, cfg.dx)
        return rho, curl

    # 我們比較 Component 1 即可
    print("   計算特徵中...")
    rho_low, curl_low = get_features(solver_low, p1_low, cfg_low)
    rho_high, curl_high = get_features(solver_high, p1_high, cfg_high)
    
    # 計算差異
    diff_rho = jnp.abs(rho_high - rho_low)
    diff_curl = jnp.abs(curl_high - curl_low)
    
    print(f"\n📊 差異統計 (Mean Absolute Difference):")
    print(f"密度圖差異: {jnp.mean(diff_rho):.6f}")
    print(f"旋度圖差異: {jnp.mean(diff_curl):.6f}")
    print(f"旋度圖最大差異: {jnp.max(diff_curl):.6f}")

    # 判斷標準
    if jnp.mean(diff_curl) < 1e-6:
        print("\n🔴 嚴重警告：物理模擬對 B 不敏感！改變 B 沒有造成顯著差異！")
        print("   可能原因：步數太少、系統尚未演化出渦旋、或 B 參數未正確傳入。")
    else:
        print("\n🟢 物理層通過：改變 B 確實會改變輸出圖像。")
        print("   證明：物理引擎是正常的，問題可能出在 AI 的學習過程。")

if __name__ == "__main__":
    main()