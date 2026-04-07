import os
import pathlib
import secrets
import string
import re
from typing import List

def batch_rename_with_short_uuid(target_dir: str = ".", id_length: int = 7):
    """
    執行 PNG 檔案重命名任務。
    功能：
    1. 跳過已符合 {4位}_{7位隨機碼}.png 格式的檔案。
    2. 攔截檔名不足 4 位的檔案並回報。
    3. 執行原子性重新命名。
    """
    base_path = pathlib.Path(target_dir).resolve()
    
    # 1. 獲取所有 PNG 檔案並轉為 List (避免迭代時檔案系統變動影響結果)
    png_files: List[pathlib.Path] = [f for f in base_path.glob("*.png") if f.is_file()]
    
    if not png_files:
        print("[Log] 此目錄下無 PNG 檔案，任務終止。")
        return

    # 2. 定義正規表示式模式 (等冪性檢查關鍵)
    # ^.{4} : 開頭任意 4 個字元
    # _     : 底線
    # [a-zA-Z0-9]{id_length} : 精確匹配 7 位隨機字元
    # \.png$ : 結尾
    pattern_str = rf"^.{{4}}_[a-zA-Z0-9]{{{id_length}}}\.png$"
    processed_pattern = re.compile(pattern_str)

    alphabet = string.ascii_letters + string.digits
    short_file_list = [] # 異常清單：長度不足 4 字元
    success_count = 0    # 計數器：成功更名數量
    skip_count = 0       # 計數器：已格式化而跳過數量

    print(f"[Log] 開始處理，目標目錄：{base_path}")

    for file_path in png_files:
        current_name = file_path.name
        original_stem = file_path.stem  # 不含副檔名的檔名

        # 3. 邏輯閘門 A：檢查長度是否足以提取前綴
        if len(original_stem) < 4:
            short_file_list.append(current_name)
            continue

        # 4. 邏輯閘門 B：等冪性檢查 (已格式化則跳過)
        if processed_pattern.match(current_name):
            skip_count += 1
            continue

        # 5. 執行更名邏輯
        prefix = original_stem[:4]
        short_id = ''.join(secrets.choice(alphabet) for _ in range(id_length))
        new_name = f"{prefix}_{short_id}{file_path.suffix}"
        new_file_path = file_path.with_name(new_name)

        # 安全檢查：避免覆蓋現有檔案
        if new_file_path.exists():
            print(f"[Warning] 衝突：{new_name} 已存在，跳過檔案 {current_name}")
            continue

        try:
            file_path.rename(new_file_path)
            print(f"[Success] {current_name} -> {new_name}")
            success_count += 1
        except Exception as e:
            print(f"[Error] 重新命名 {current_name} 失敗: {e}")

    # 6. 最終執行回報
    print("\n" + "="*50)
    print("【執行任務完成回報】")
    print(f"總偵測檔案數：{len(png_files)}")
    print(f"成功更新數量：{success_count}")
    print(f"已符合格式跳過：{skip_count}")
    
    if short_file_list:
        print(f"\n[異常警告] 偵測到 {len(short_file_list)} 個檔名不足 4 字元的檔案：")
        for name in short_file_list:
            print(f"  - {name}")
        print("請手動核對上述檔案。")
    
    print("="*50)

if __name__ == "__main__":
    batch_rename_with_short_uuid()