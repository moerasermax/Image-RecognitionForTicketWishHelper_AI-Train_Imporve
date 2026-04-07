import os
import pathlib
import secrets
import string
from typing import List

def batch_rename_with_short_uuid(target_dir: str = ".", id_length: int = 7):
    """
    執行 PNG 檔案重命名任務。
    格式：{原檔名前四位}_{7位隨機碼}.png
    """
    # 1. 初始化路徑物件 (pathlib 提供了類比 C# System.IO.Path 的物件導向操作)
    base_path = pathlib.Path(target_dir).resolve()
    
    # 2. 獲取所有 PNG 檔案
    # 使用 list 固化結果，避免在遍歷時因為檔案名稱變更導致迭代器行為異常
    png_files: List[pathlib.Path] = [f for f in base_path.glob("*.png") if f.is_file()]
    
    if not png_files:
        print("[Log] 此目錄下無 PNG 檔案，任務終止。")
        return

    print(f"[Log] 偵測到 {len(png_files)} 個檔案，開始執行重構...")

    # 定義隨機字元集 (使用 Base62: 大小寫字母+數字，提供更高的熵)
    # 若您嚴格要求 UUID 風格的 16 進位，可改為 string.hexdigits
    alphabet = string.ascii_letters + string.digits

    for file_path in png_files:
        # 3. 提取前綴與生成隨機碼
        # stem 取得不含副檔名的部分 (類比 C# Path.GetFileNameWithoutExtension)
        original_stem = file_path.stem
        prefix = original_stem[:4]
        
        # 使用 secrets 模組產生具備密碼學強度的隨機字串
        # 相比 random 模組，這能有效降低在批次處理時的碰撞機率
        short_id = ''.join(secrets.choice(alphabet) for _ in range(id_length))
        
        # 4. 建構新名稱與安全性檢查
        new_name = f"{prefix}_{short_id}{file_path.suffix}"
        new_file_path = file_path.with_name(new_name)

        # 防止邏輯偏差：如果極低機率發生重名衝突，則跳過或報錯
        if new_file_path.exists():
            print(f"[Warning] 衝突偵測：{new_name} 已存在，跳過檔案 {file_path.name}")
            continue

        try:
            # 5. 執行原子性重命名 (Atomic Rename)
            file_path.rename(new_file_path)
            print(f"[Success] {file_path.name} -> {new_name}")
        except PermissionError:
            print(f"[Error] 權限不足：無法修改 {file_path.name}，請檢查檔案是否被佔用。")
        except Exception as e:
            print(f"[Error] 處理 {file_path.name} 時發生未知錯誤: {e}")

if __name__ == "__main__":
    batch_rename_with_short_uuid()