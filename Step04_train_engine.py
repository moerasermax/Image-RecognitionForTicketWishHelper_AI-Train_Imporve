import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging

# 匯入你定義好的元件
from Step02_dataset_definition import TixcraftDataset, LabelConverter, CHARACTERS
from Step03_model_architecture import TixcraftCRNN

# =================================================================
# ⚙️ 配置與路徑
# =================================================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_PATH = BASE_DIR / "raw_data" / "Tixcraft"
MODEL_PREFIX = "tixcraft_ocr_v" # 統一前綴

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | INFO | train:main:60 - [%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 訓練超參數
BATCH_SIZE = 64
# --- v7 建議設定為 0.00001 ---
LR = 0.000005  
#====學習率調整終極指南====
#初次訓練 (From Scratch) 0.001 ~ 0.0005  快速建立基本認知，讓模型知道什麼是字、什麼是背景。
#再次訓練 (Incremental) 0.0001 ~ 0.00005 加入新資料，修正環境偏差（如貼邊、藍底），但不破壞舊大腦。
#極致微調 (Fine-tuning) 0.00001 ~ 0.000005 針對最後那 4 張魔鬼題進行「手術級」修正，追求 100% 準確率。
#====學習階段=====
#v1 ~ v3：教模型認字（幼兒園）。
#v4 ~ v6：教模型適應實戰環境（小學）。
#v7 ~ v10：教模型處理極端魔鬼題（大學專家）。
#=====EPOCHS=====
#因狀況而定 50(適量)~150(深度) 都可以

EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================================================================
# 📏 準確率計算器
# =================================================================
def calculate_accuracy(logits, targets, converter):
    preds = logits.argmax(2).permute(1, 0) 
    correct_count = 0
    batch_size = preds.size(0)
    for i in range(batch_size):
        if converter.decode(preds[i].tolist()) == converter.decode(targets[i].tolist()):
            correct_count += 1
    return correct_count / batch_size

# =================================================================
# 📂 智慧版本管理
# =================================================================
def get_version_info():
    max_v = 0
    latest_file = None
    pattern = re.compile(rf"{MODEL_PREFIX}(\d+)\.pth")

    for file in os.listdir(BASE_DIR):
        match = pattern.match(file)
        if match:
            v_num = int(match.group(1))
            if v_num > max_v:
                max_v = v_num
                latest_file = BASE_DIR / file
    return latest_file, max_v + 1

# =================================================================
# 🔥 訓練主邏輯
# =================================================================
def train():
    converter = LabelConverter(CHARACTERS)
    dataset = TixcraftDataset(DATA_PATH, converter)
    
    if len(dataset) == 0:
        logger.error(f"找不到圖片於: {DATA_PATH}")
        return

    latest_path, next_v = get_version_info()
    target_filename = f"{MODEL_PREFIX}{next_v}.pth"
    TARGET_SAVE_PATH = BASE_DIR / target_filename

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_class = len(CHARACTERS) + 1
    model = TixcraftCRNN(num_class).to(DEVICE)

    if latest_path:
        logger.info(f"♻️ 載入目前最高版本: {latest_path.name}")
        model.load_state_dict(torch.load(latest_path, map_location=DEVICE))
    else:
        logger.info("🆕 未發現現有權重，將從 v1 開始訓練")
        target_filename = f"{MODEL_PREFIX}1.pth"
        TARGET_SAVE_PATH = BASE_DIR / target_filename

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    logger.info(f"🚀 啟動微調！目標存檔: {target_filename} | LR: {LR} | 樣本數: {len(dataset)}")
    
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for i, (imgs, labels) in enumerate(train_loader):
            global_step += 1
            imgs = imgs.to(DEVICE)
            logits = model(imgs) 
            logits_for_loss = logits.permute(1, 0, 2) 
            
            loss = criterion(logits_for_loss, labels, 
                             torch.full((imgs.size(0),), 39, dtype=torch.long), 
                             torch.full((imgs.size(0),), 4, dtype=torch.long))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if global_step % 10 == 0:
                acc = calculate_accuracy(logits_for_loss, labels, converter)
                # --- 獲取即時學習率 ---
                current_lr = optimizer.param_groups[0]['lr']
                
                # 更新後的日誌格式，包含 Lr 顯示
                logger.info(f"Epoch: {epoch:02d} Step: {global_step:04d} "
                            f"AvgLoss: {epoch_loss/(i+1):.6f} "
                            f"Lr: {current_lr:.6f} " # 👈 這裡會顯示即時 LR
                            f"Acc: {acc:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), TARGET_SAVE_PATH)
            logger.info(f"💾 暫存更新至: {target_filename}")

    torch.save(model.state_dict(), TARGET_SAVE_PATH)
    logger.info("=" * 50)
    logger.info(f"🎉 訓練完成！")
    logger.info(f"✅ 版本進化: {latest_path.name if latest_path else 'None'} -> {target_filename}")
    logger.info("=" * 50)

if __name__ == "__main__":
    train()