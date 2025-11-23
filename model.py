import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

# --- 1. 通用組件 (Reusable Components) ---

class CNNEncoder(nn.Module):
    """
    通用的圖像特徵提取器。
    無論輸入是什麼物理系統的圖片，都將其壓縮成一個特徵向量。
    
    架構：ConvBlock -> ConvBlock -> ... -> Global Average Pooling(GAP)
    特點：具備平移不變性 (Translation Invariance)，適合物理晶格系統。
    """
    # 每一層卷積的濾波器數量，例如 [32, 64, 128]
    features: Sequence[int] = (32, 64, 64)
    # 輸出的特徵向量維度
    output_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, height, width, channels)
        
        # 1. 卷積層堆疊 (Feature Extraction)
        for feat in self.features:
            x = nn.Conv(features=feat, kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            
        # 2. Global Average Pooling (GAP)
        # 這是你的關鍵決策！
        # 不管圖片剩多大，直接對長寬 (axis 1, 2) 取平均
        # shape 變換: (batch, H', W', C) -> (batch, C)
        x = jnp.mean(x, axis=(1, 2))
        
        # 3. 投影到統一的輸出維度
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.relu(x)
        return x

class DataEmbedding(nn.Module):
    """
    處理物理參數 (eta, B) 的嵌入層。
    將只有 2 個數字的參數擴展成高維向量，以便跟圖片特徵融合。
    """
    output_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, 2)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.relu(x)
        return x


# --- 2. 特定任務組裝 (Task-Specific Model) ---

class NREClassifier(nn.Module):
    """
    Project 1 專用的 NRE 分類器 (雙塔架構)。
    
    Left Tower: CNNEncoder 看圖片
    Right Tower: DataEmbedding 看參數
    Head: 融合後判斷這組 (圖片, 參數) 是不是原配
    """
    
    @nn.compact
    def __call__(self, x, theta):
        """
        Args:
            x: 圖片數據 (Batch, 128, 128, 2)
            theta: 物理參數 (Batch, 2)
        Returns:
            logit: 分數 (Scalar)
        """
        
        # 1. 左塔：提取圖片特徵
        # 我們直接實例化上面的通用組件
        h_img = CNNEncoder(output_dim=64)(x)
        
        # 2. 右塔：提取參數特徵
        h_param = DataEmbedding(output_dim=64)(theta)
        
        # 3. 融合 (Fusion)
        # 將兩個向量串接起來: [h_img, h_param]
        h_joint = jnp.concatenate([h_img, h_param], axis=-1)
        
        # 4. 決策頭 (Decision Head / MLP)
        h = nn.Dense(features=64)(h_joint)
        h = nn.relu(h)
        h = nn.Dense(features=64)(h)
        h = nn.relu(h)
        
        # 最後輸出一顆神經元 (Logit)，代表 "是真配對的信心分數"
        # 注意：這裡不加 Sigmoid，因為我們之後會配合 BCEWithLogitsLoss 使用 (數值較穩定)
        logit = nn.Dense(features=1)(h)
        
        return logit
    
    
if __name__ == "__main__":
    # 測試區：驗證模型架構與維度
    
    # 1. 假造一些數據 (Mock Data)
    # 模擬 batch_size = 5 的輸入
    dummy_image = jnp.ones((5, 128, 128, 2)) # (Batch, H, W, C)
    dummy_theta = jnp.ones((5, 2))           # (Batch, Params)
    
    # 2. 初始化模型
    # 實例化模型結構
    model = NREClassifier()
    
    # 準備一把隨機鑰匙來初始化參數
    key = jax.random.PRNGKey(0)
    
    # 呼叫 model.init 來生成參數字典 'params'
    # 這一步只是為了把參數形狀定下來
    variables = model.init(key, dummy_image, dummy_theta)
    
    print("模型初始化成功！參數結構樹如下：")
    # 可以用 jax.tree_util.tree_map 來看結構，這裡簡單印出
    print(variables.keys()) # 應該會看到 'params'
    
    # 3. 試跑一次前向傳播 (Forward Pass)
    # 注意：在 Flax 裡，執行模型需要把 'variables' (參數) 顯式地傳進去
    output_logit = model.apply(variables, dummy_image, dummy_theta)
    
    print(f"\n輸入圖片: {dummy_image.shape}")
    print(f"輸入參數: {dummy_theta.shape}")
    print(f"輸出 Logit Shape: {output_logit.shape}")
    
    # 驗證輸出是否為 (Batch, 1)
    assert output_logit.shape == (5, 1)
    print("\n✅ 測試通過：模型架構與 GAP 運作正常！")    