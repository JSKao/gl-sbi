import jax
import jax.numpy as jnp
from flax.training import train_state
import optax # 需要 pip install optax

from gl_jax import GLSolverJAX, SimConfig
from model import NREClassifier # 假設你的檔案叫 model.py
from dataclasses import dataclass

# --- 1. 設定區 ---
@dataclass
class TrainConfig:
    # 數據生成
    batch_size: int = 8
    n_samples: int = 1000 # 總共訓練幾個 Step (為了演示先設小一點)
    
    # 訓練參數
    learning_rate: float = 1e-3
    seed: int = 42

# --- 2. 模擬器 (Data Factory) ---
# 這是我們之前寫好的，保持不變
def simulator(key):
    # 1. Split key
    key_eta, key_B, key_sim = jax.random.split(key, 3)
    
    # 2. Prior Sampling
    eta = jax.random.uniform(key_eta, minval=0.0, maxval=1.5)
    B = jax.random.uniform(key_B, minval=0.0, maxval=0.02)
    
    # 3. Config & Solver
    config = SimConfig(eta=eta, B=B, N=32)
    solver = GLSolverJAX(config)
    
    # 4. Evolution
    psi1_init, psi2_init = GLSolverJAX.initialize_state(config, key_sim)
    psi1_final, psi2_final = solver.evolve(psi1_init, psi2_init, 100)
    
    # 5. Feature Extraction (Density)
    rho1 = jnp.abs(psi1_final) ** 2
    rho2 = jnp.abs(psi2_final) ** 2
    density = jnp.stack([rho1, rho2], axis=-1) # 改成 channel last: (128, 128, 2) 符合 CNN 習慣
    
    # 6. Return
    params = jnp.array([eta, B])
    return params, density

# --- 3. 訓練狀態管理 (Deep Skill: Flax Pattern) ---
def create_train_state(rng, learning_rate):
    """初始化模型參數與優化器"""
    model = NREClassifier()
    
    # 假數據用於初始化形狀
    dummy_x = jnp.ones((1, 128, 128, 2))
    dummy_theta = jnp.ones((1, 2))
    
    # 初始化參數
    variables = model.init(rng, dummy_x, dummy_theta)
    
    # 設定優化器 (Adam)
    tx = optax.adam(learning_rate)
    
    # 建立 TrainState (它會幫我們保管 params 和 opt_state)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )

# --- 4. 損失函數與更新步 (The Brain) ---
@jax.jit
def train_step(state, batch_x, batch_theta, key_shuffle):
    """
    執行一步訓練：
    1. 準備正樣本 (Joint)
    2. 製造負樣本 (Marginal, by shuffling)
    3. 計算 Loss
    4. 反向傳播 (Backprop)
    """
    
    # A. 準備數據 ------------------------------------------------
    # 正樣本: (x, theta) -> Label 1
    pos_x = batch_x
    pos_theta = batch_theta
    pos_labels = jnp.ones((batch_x.shape[0], 1))
    
    # 負樣本: (x, theta_shuffled) -> Label 0
    # Deep Skill: 透過 "大風吹" (Roll) 快速製造不匹配的參數
    # 這樣 x 還是合法的 x，theta 還是合法的 theta，但配在一起就是錯的
    neg_x = batch_x
    neg_theta = jnp.roll(batch_theta, shift=1, axis=0) 
    neg_labels = jnp.zeros((batch_x.shape[0], 1))
    
    # 合併數據 (Concatenate)
    train_x = jnp.concatenate([pos_x, neg_x], axis=0)
    train_theta = jnp.concatenate([pos_theta, neg_theta], axis=0)
    train_labels = jnp.concatenate([pos_labels, neg_labels], axis=0)
    
    # B. 定義 Loss Function (Closure) ---------------------------
    def loss_fn(params):
        # Forward pass (計算 Logits)
        logits = state.apply_fn({'params': params}, train_x, train_theta)
        
        # 計算 Binary Cross Entropy
        # 使用 optax 的穩定版本 (內含 Sigmoid)
        loss = optax.sigmoid_binary_cross_entropy(logits, train_labels).mean()
        return loss

    # C. 計算梯度與更新 -----------------------------------------
    # jax.value_and_grad 會同時算出 loss 值和 gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # 更新模型參數
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss

# --- 5. 主程式 ---
if __name__ == "__main__":
    cfg = TrainConfig()
    print(f"🚀 開始訓練 NRE 模型 | Batch: {cfg.batch_size} | Steps: {cfg.n_samples}")
    
    # 1. 準備隨機鑰匙
    master_key = jax.random.PRNGKey(cfg.seed)
    key_init, key_train = jax.random.split(master_key)
    
    # 2. 初始化 AI (TrainState)
    state = create_train_state(key_init, cfg.learning_rate)
    print("✅ 模型與優化器初始化完成")
    
    # 3. 準備批次模擬器 (vmap)
    batched_simulator = jax.vmap(simulator)
    
    # 4. 訓練迴圈 (Training Loop)
    # 在這裡我們「邊跑模擬、邊訓練」(Online Training)
    # 這比先存硬碟再讀取更省空間，且利用 GPU 高併發優勢
    
    key_current = key_train
    
    for step in range(cfg.n_samples):
        # A. 更新隨機鑰匙
        key_current, key_sim_batch, key_shuffle = jax.random.split(key_current, 3)
        sim_keys = jax.random.split(key_sim_batch, cfg.batch_size)
        
        # B. 產生數據 (Physics Engine)
        # 這裡會呼叫 GPU 進行 32 個宇宙的平行演化
        batch_theta, batch_x = batched_simulator(sim_keys)
        
        # C. 訓練一步 (AI Brain)
        state, loss = train_step(state, batch_x, batch_theta, key_shuffle)
        
        # D. 監控
        if step % 10 == 0:
            # 簡單的進度條
            print(f"Step {step:04d} | Loss: {loss:.4f} | (Physics + AI running...)")

    print("\n🎉 訓練完成！")
    
    # --- 6. 驗收時刻：AI 真的懂物理嗎？ ---
    print("\n🔍 開始進行推論測試 (Inference Demo)...")
    import matplotlib.pyplot as plt
    import numpy as np

    # A. 產生一個 "真實觀測" (Ground Truth)
    # 我們設定一個已知的物理情況
    true_eta = 0.8
    true_B = 0.01
    
    # 為了公平，我們用一把全新的鑰匙來模擬
    key_infer, key_sim = jax.random.split(jax.random.PRNGKey(999))
    
    # 跑一次模擬拿到 "觀測圖片" (Observation)
    print(f"1. 正在生成真實觀測數據 (True eta={true_eta}, True B={true_B})...")
    config = SimConfig(eta=true_eta, B=true_B, N=32) # 保持 N=32 與訓練一致
    solver = GLSolverJAX(config)
    psi1, psi2 = GLSolverJAX.initialize_state(config, key_sim)
    p1_f, p2_f = solver.evolve(psi1, psi2, 100) # 保持 steps=100
    
    # 轉成密度圖 (128, 128, 2)
    obs_img = jnp.stack([jnp.abs(p1_f)**2, jnp.abs(p2_f)**2], axis=-1)
    # 增加 Batch 維度 -> (1, 128, 128, 2)
    obs_img_batch = jnp.expand_dims(obs_img, axis=0)

    # B. 讓 AI 進行 "掃描式推論"
    # 我們固定 B，掃描 eta 從 0.0 到 1.5，看 AI 覺得哪個最像
    print("2. AI 正在掃描參數空間，尋找最可能的 eta...")
    
    test_etas = jnp.linspace(0.0, 1.5, 100) # 測試 100 個不同的 eta
    scores = []
    
    for test_eta in test_etas:
        # 建立測試參數對 (test_eta, true_B)
        # 注意：我們作弊告訴它 B 是對的，只考它 eta (為了 demo 簡單)
        theta_test = jnp.array([[test_eta, true_B]])
        
        # 讓 AI 打分數 (Forward Pass)
        # 不需要算梯度，直接 apply
        logit = state.apply_fn({'params': state.params}, obs_img_batch, theta_test)
        
        # 轉成機率 (Sigmoid)
        prob = jax.nn.sigmoid(logit)
        scores.append(prob[0, 0])
    
    scores = np.array(scores)

    # C. 畫圖驗證
    print("3. 繪製結果...")
    plt.figure(figsize=(10, 6))
    plt.plot(test_etas, scores, label='AI Confidence', color='blue', linewidth=2)
    plt.axvline(x=true_eta, color='red', linestyle='--', label=f'True Eta ({true_eta})')
    plt.title("Neural Ratio Estimation: Inference Result", fontsize=14)
    plt.xlabel("Eta (Coupling Strength)", fontsize=12)
    plt.ylabel("AI Probability Score (Posterior Proxy)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 儲存圖片
    plt.savefig("inference_result.png")
    print(f"✅ 推論完成！結果已存為 'inference_result.png'")
    print("請打開圖片，看看紅線(真實值)是否落在藍線(AI預測)的高峰附近？")
    
