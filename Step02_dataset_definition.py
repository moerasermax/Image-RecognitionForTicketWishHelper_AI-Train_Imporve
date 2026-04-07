import os
import torch
import string
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# =================================================================
# ⚙️ 配置區 (Configuration)
# =================================================================
# 指向你剛抓完數據的路徑
DATA_PATH = Path("raw_data/Tixcraft")
# 拓元驗證碼特徵：純小寫英文 26 碼
CHARACTERS = string.ascii_lowercase 
# 圖片標準化尺寸 (拓元通常為 160x60)
IMG_W, IMG_H = 160, 60

# =================================================================
# 🛠️ 核心轉譯器 (Label Converter)
# =================================================================
class LabelConverter:
    """
    負責將「文字(String)」與「數字(Tensor)」進行雙向轉換。
    就像是 C# 中的 Enum 或 Dictionary 映射。
    """
    def __init__(self, alphabet):
        # ⚠️ 重要：索引 0 必須保留給 CTC 的 "Blank" (空白標籤)
        # 這是處理「黏連字體」的關鍵。
        self.alphabet = '-' + alphabet  # 結果為: -abcdefghijklmnopqrstuvwxyz
        self.dict = {char: i for i, char in enumerate(self.alphabet)}

    def encode(self, text):
        """文字 -> 數字 (例如: 'abc' -> [1, 2, 3])"""
        return [self.dict[char] for char in text.lower()]

    def decode(self, res):
        """數字 -> 文字 (處理模型輸出的序列，折疊重複字元)"""
        char_list = []
        for i in range(len(res)):
            # CTC 解碼邏輯：跳過 0 (blank) 且跳過連續重複的字元
            if res[i] != 0 and (not (i > 0 and res[i] == res[i-1])):
                char_list.append(self.alphabet[res[i]])
        return "".join(char_list)

# =================================================================
# 📦 數據提供者 (Dataset Class)
# =================================================================
class TixcraftDataset(Dataset):
    """
    實作 PyTorch 的數據協議。
    這就像是 C# 的 IEnumerable<T>，負責產出模型訓練用的每一筆樣本。
    """
    def __init__(self, root_dir, converter):
        self.root_dir = Path(root_dir)
        self.converter = converter
        # 取得所有圖片路徑
        self.samples = list(self.root_dir.glob("*.png"))
        
        # 數據轉換 Pipeline：縮放 -> 轉為張量 -> 歸一化
        self.transform = T.Compose([
            T.Resize((IMG_H, IMG_W)),      # 確保所有輸入圖片尺寸一致
            T.ToTensor(),                 # 將像素 [0, 255] 轉為 [0.0, 1.0] 的 Tensor
            T.Normalize([0.5], [0.5])      # 歸一化至 [-1, 1]，助於神經網路穩定收斂
        ])

    def __len__(self):
        """回傳總樣本數"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根據索引抓取一張圖及其標籤。
        這是訓練時會被萬次呼叫的核心方法。
        """
        img_path = self.samples[idx]
        
        # 1. 讀取圖片並轉為 RGB
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 2. 從檔名提取標籤 (格式: label_uuid.png)
        # 範例: 'cfra_a1b2c3.png' -> 'cfra'
        label_text = img_path.stem.split('_')[0]
        
        # 3. 將標籤轉為數字 Tensor
        label = torch.LongTensor(self.converter.encode(label_text))
        
        return image, label

# =================================================================
# 🧪 測試運行 (測試腳本是否能正確解析你的 raw_data)
# =================================================================
if __name__ == "__main__":
    conv = LabelConverter(CHARACTERS)
    dataset = TixcraftDataset(DATA_PATH, conv)
    
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"✅ 數據集讀取成功！共 {len(dataset)} 張圖片。")
        print(f"🖼️ 圖片 Tensor 形狀: {img.shape} (C, H, W)")
        print(f"🏷️ 樣本標籤索引: {label.tolist()}")
    else:
        print(f"❌ 找不到圖片，請檢查路徑: {DATA_PATH.absolute()}")