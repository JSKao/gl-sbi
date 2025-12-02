import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.model import NREClassifier # 測試模型架構
import os

def diagnose():
    print("🔍 全面診斷開始...")
    
    # 1. 檢查數據檔案
    data_path = "data/dataset_128.npz"
    if not os.path.exists(data_path):
        print("❌ 數據檔不存在！")
        return
    
    try:
        data = np.load(data_path)
        theta = data['theta']
        x = data['x']
        print(f"✅ 數據載入成功: X={x.shape}, Theta={theta.shape}")
    except:
        print("❌ 數據檔損毀！")
        return

    # 2. 檢查 B 的分布 (是否都一樣？)
    Bs = theta[:, 1]
    print(f"\n📊 B 參數統計:")
    print(f"   Min: {Bs.min():.5f}")
    print(f"   Max: {Bs.max():.5f}")
    print(f"   Mean: {Bs.mean():.5f}")
    print(f"   Std:  {Bs.std():.5f}")
    
    if Bs.std() < 1e-6:
        print("🔴 致命錯誤：所有樣本的 B 幾乎都一樣！AI 當然學不到！")
        print("   -> 請檢查 generate_data.py 的隨機生成部分。")
        return
    else:
        print("🟢 B 參數有變化，正常。")

    # 3. 檢查 Curl 通道 (是否太暗？)
    curl = x[..., 2]
    print(f"\n📊 Curl (磁場) 通道統計:")
    print(f"   Max: {curl.max():.5f}")
    print(f"   Mean: {curl.mean():.5f}")
    
    if curl.max() < 0.1:
        print("🔴 致命錯誤：磁場訊號太弱！Max < 0.1。")
        print("   -> 請檢查 simulator.py 是否正確計算 curl。")
    else:
        print("🟢 磁場訊號強度足夠。")

    # 4. 視覺化檢查 (物理是否可辨識？)
    print("\n👁️ 視覺化檢查：挑選 B 最小與最大的樣本...")
    idx_min = np.argmin(Bs)
    idx_max = np.argmax(Bs)
    
    print(f"   樣本 A (Index {idx_min}): B = {Bs[idx_min]:.5f}")
    print(f"   樣本 B (Index {idx_max}): B = {Bs[idx_max]:.5f}")
    
    # 畫圖
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # 畫 Curl Channel
    im1 = axs[0].imshow(x[idx_min, ..., 2], cmap='seismic', origin='lower')
    axs[0].set_title(f"Low B ({Bs[idx_min]:.4f})\nCurl Sum: {np.sum(x[idx_min, ..., 2]):.2f}")
    plt.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(x[idx_max, ..., 2], cmap='seismic', origin='lower')
    axs[1].set_title(f"High B ({Bs[idx_max]:.4f})\nCurl Sum: {np.sum(x[idx_max, ..., 2]):.2f}")
    plt.colorbar(im2, ax=axs[1])
    
    plt.savefig("diagnosis_plot.png")
    print("✅ 已儲存比較圖 'diagnosis_plot.png'")
    
    # 關鍵判斷：總和是否不同？
    sum_diff = np.abs(np.sum(x[idx_max, ..., 2]) - np.sum(x[idx_min, ..., 2]))
    print(f"   兩圖 Curl 總和差異: {sum_diff:.2f}")
    
    if sum_diff < 1.0:
        print("🔴 物理警告：高低磁場的圖片總特徵量幾乎一樣！")
        print("   -> 這代表物理模擬對 B 不敏感，或者 B 範圍太小。")
    else:
        print("🟢 物理差異顯著，AI 理論上應該看得到。")

    # 5. 模型架構檢查 (Pooling 測試)
    print("\n🧠 模型架構檢查 (Pooling Test)...")
    model = NREClassifier()
    # 模擬兩個 Batch 的輸入
    mock_x = jnp.array([x[idx_min], x[idx_max]]) # (2, 128, 128, 3)
    mock_theta = jnp.array([theta[idx_min], theta[idx_max]])
    
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, mock_x, mock_theta)
    
    # 提取 CNN Encoder 的輸出 (手動模擬，因為我們無法直接 access 中間層，這裡用推論代替)
    # 我們檢查 logit 是否不同
    logits = model.apply(variables, mock_x, mock_theta)
    print(f"   Logit A (Low B): {logits[0,0]:.4f}")
    print(f"   Logit B (High B): {logits[1,0]:.4f}")
    
    if jnp.abs(logits[0] - logits[1]) < 1e-4:
        print("🔴 模型警告：未經訓練的模型對兩張圖的反應完全一樣！")
        print("   -> 這可能代表初始化權重下，GAP/GMP 把差異抹平了。")
    else:
        print("🟢 模型初始化正常，對不同輸入有不同反應。")

if __name__ == "__main__":
    diagnose()