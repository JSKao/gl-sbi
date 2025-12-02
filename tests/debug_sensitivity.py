import jax
import jax.numpy as jnp
import numpy as np
import os
from src.gl_jax import GLSolverJAX, SimConfig
from src.model import NREClassifier

# 設定
CKPT_DIR = os.path.abspath("checkpoints")
GRID_SIZE = 128
STEPS = 1000  # 確保這裡是 1000

def main():
    print("🩺 啟動 AI 敏感度診斷...")
    
    # 1. 載入模型
    model = NREClassifier()
    dummy_x = jnp.ones((1, GRID_SIZE, GRID_SIZE, 3))
    dummy_theta = jnp.ones((1, 2))
    key = jax.random.PRNGKey(0)
    variables = model.init(key, dummy_x, dummy_theta)
    
    # 嘗試載入訓練好的權重
    from flax.training import checkpoints
    if os.path.exists(CKPT_DIR):
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=None)
        if state_dict:
            variables = {'params': state_dict['params']}
            print("✅ 已載入訓練權重")
        else:
            print("⚠️ 未找到權重，使用隨機初始化")
    
    # 2. 生成兩組對照數據 (Control vs Experiment)
    print("⚗️  生成測試數據 (Steps=1000)...")
    key_sim = jax.random.PRNGKey(42)
    
    # Case A: B = 0.000
    cfg_a = SimConfig(eta=0.8, B=0.000, N=GRID_SIZE)
    solver_a = GLSolverJAX(cfg_a)
    psi1_init_a, psi2_init_a = GLSolverJAX.initialize_state(cfg_a, key_sim)
    # [修正] 解包回傳值
    p1_a, p2_a = solver_a.evolve(psi1_init_a, psi2_init_a, STEPS)
    
    # 製作 3-channel
    rho1_a = jnp.abs(p1_a)**2
    rho2_a = jnp.abs(p2_a)**2 # 這裡雖然沒用到 rho2_a 但保持對稱
    # 計算電流特徵
    Jx_a, Jy_a = GLSolverJAX.compute_current(p1_a, cfg_a.dx, cfg_a.B)
    cur_a = GLSolverJAX.compute_curl_J(Jx_a, Jy_a, cfg_a.dx)
    
    # 疊加: 使用 rho1_a, rho2_a (模擬 simulator 的行為)
    # 這裡假設 simulator 是 stack([rho1, rho2, curl])
    # 但為了診斷 B，其實只要 curl 對了就好
    # 修正: simulator 是用 rho1 和 rho2
    rho2_a = jnp.abs(p2_a)**2
    img_a = jnp.stack([rho1_a, rho2_a, cur_a], axis=-1)[None, ...] 

    # Case B: B = 0.020
    cfg_b = SimConfig(eta=0.8, B=0.020, N=GRID_SIZE)
    solver_b = GLSolverJAX(cfg_b)
    psi1_init_b, psi2_init_b = GLSolverJAX.initialize_state(cfg_b, key_sim)
    # [修正] 解包回傳值
    p1_b, p2_b = solver_b.evolve(psi1_init_b, psi2_init_b, STEPS)
    
    rho1_b = jnp.abs(p1_b)**2
    rho2_b = jnp.abs(p2_b)**2
    Jx_b, Jy_b = GLSolverJAX.compute_current(p1_b, cfg_b.dx, cfg_b.B)
    cur_b = GLSolverJAX.compute_curl_J(Jx_b, Jy_b, cfg_b.dx)
    
    img_b = jnp.stack([rho1_b, rho2_b, cur_b], axis=-1)[None, ...]

    # 3. 讓 AI 看圖
    # 固定 theta = [0.8, 0.01]
    theta_fixed = jnp.array([[0.8, 0.01]])
    
    logit_a = model.apply(variables, img_a, theta_fixed)
    logit_b = model.apply(variables, img_b, theta_fixed)
    
    print(f"\n📊 診斷結果:")
    print(f"Image A (B=0.00) -> Logit: {logit_a[0,0]:.4f}")
    print(f"Image B (B=0.02) -> Logit: {logit_b[0,0]:.4f}")
    diff = jnp.abs(logit_a - logit_b)[0,0]
    print(f"差異 (Delta): {diff:.4f}")
    
    if diff < 0.01:
        print("\n🔴 結論：模型對 B '無感'。")
        print("   原因可能是：1. 訓練不足  2. 特徵被 GAP 吃掉  3. 權重壞掉")
    else:
        print("\n🟢 結論：模型能區分不同的 B！")
        print("   如果熱力圖還是錯的，那可能是 inference_2d.py 的掃描邏輯或參數範圍寫錯了。")

if __name__ == "__main__":
    main()