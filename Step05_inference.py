import torch
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

# 確保匯入路徑正確
from Step02_dataset_definition import LabelConverter, CHARACTERS
from Step03_model_architecture import TixcraftCRNN

# =================================================================
# ⚙️ 配置區
# =================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "tixcraft_ocr_v1.pth"
DATA_DIR = Path("test_data/Tixcraft") 
TARGET_KEYWORD = "test"              

# =================================================================
# 🛠️ 預測器類別
# =================================================================
class TixcraftPredictor:
    def __init__(self, model_path):
        self.converter = LabelConverter(CHARACTERS)
        self.num_class = len(CHARACTERS) + 1
        
        self.model = TixcraftCRNN(self.num_class).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval() 
        
        self.transform = T.Compose([
            T.Resize((60, 160)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            
        probs = logits.argmax(2).squeeze(0).cpu().numpy()
        prediction = self.converter.decode(probs)
        return prediction

# =================================================================
# 🧪 驗收執行邏輯 (針對 test_{label}_{id} 格式優化)
# =================================================================
if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        print(f"❌ 找不到模型權重檔: {MODEL_PATH}")
        exit()

    predictor = TixcraftPredictor(MODEL_PATH)
    
    # 取得所有包含 'test' 的檔案
    test_targets = [f for f in DATA_DIR.glob("*.png") if TARGET_KEYWORD.lower() in f.name.lower()]

    if not test_targets:
        print(f"⚠️ 在 {DATA_DIR} 找不到符合格式的驗收圖片。")
    else:
        print(f"✅ 找到 {len(test_targets)} 個驗收目標 (格式: test_label_id.png)")
        print("-" * 60)
        
        correct_count = 0
        for img_path in test_targets:
            # 1. 模型預測
            pred = predictor.predict(img_path)
            
            # 2. 解析真實標籤 
            # 檔名: test_abcd_001.png -> stem: test_abcd_001 -> split: ['test', 'abcd', '001']
            # 取索引 [1] 得到 'abcd'
            try:
                parts = img_path.stem.split('_')
                actual = parts[1] if len(parts) > 1 else "Unknown"
            except Exception:
                actual = "Error"
            
            status = "✅" if pred == actual else "❌"
            if pred == actual: correct_count += 1
            
            print(f"{status} 檔案: {img_path.name:30} | 預測: {pred:6} | 真實: {actual:6}")

        print("-" * 60)
        acc = (correct_count / len(test_targets)) * 100
        print(f"🏁 驗收完成！ 成功率: {correct_count}/{len(test_targets)} ({acc:.2f}%)")