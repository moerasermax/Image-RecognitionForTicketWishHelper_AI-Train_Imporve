import os
import shutil
import random
import pathlib
from typing import Dict, List, Set

def execute_constrained_sample_copy(
    source_rel_path: str = "../../raw_data/Tixcraft", 
    limit: int = 30
):
    """
    從來源目錄抽取 30 個檔案。
    約束 1：第 6-9 位字元 (Index 5:9) 在結果集中必須唯一。
    命名：test_{前4位}.png (自動處理命名碰撞)。
    """
    # 1. 路徑初始化 (使用 pathlib 封裝，類比 C# 的 Path.Combine 避免斜線偏差)
    src_dir = pathlib.Path(source_rel_path).resolve()
    dest_dir = pathlib.Path(".").resolve()

    if not src_dir.exists():
        print(f"[Fatal] 來源路徑不存在：{src_dir}")
        return

    # 2. 檔案預選與分組 (Grouping)
    # 建立字典以「第 6-9 位」為 Key，確保該區段的絕對唯一性
    # 類比 C# LINQ: source.Where(f => f.Length >= 9).GroupBy(f => f.Substring(5, 4))
    unique_key_map: Dict[str, List[pathlib.Path]] = {}
    
    for file in src_dir.glob("*"):
        # 僅處理檔案且檔名長度需滿足 6-9 位 (Index 5-8 共 4 碼，故長度需至少 9)
        if file.is_file() and len(file.stem) >= 9:
            # 取得第 6 到第 9 位字元 (Python 索引 5:9)
            unique_key = file.stem[5:9]
            
            if unique_key not in unique_key_map:
                unique_key_map[unique_key] = []
            unique_key_map[unique_key].append(file)

    # 3. 執行隨機抽樣 (Random Sampling)
    unique_keys = list(unique_key_map.keys())
    if len(unique_keys) < limit:
        print(f"[Warning] 唯一樣本不足 (僅 {len(unique_keys)} 組)，將改為全數提取。")
        limit = len(unique_keys)

    # 隨機選取 30 個不重複的 key
    selected_keys = random.sample(unique_keys, limit)
    
    # 從每個選中的 key 中隨機挑選一個實體檔案
    candidates = [random.choice(unique_key_map[k]) for k in selected_keys]

    print(f"[Status] 已選定 {len(candidates)} 個符合 6-9 位唯一約束的檔案。")

    # 4. 複製與重命名邏輯 (含命名衝突防禦)
    used_dest_names: Set[str] = set()

    for file_path in candidates:
        # 取得原始前四位
        prefix = file_path.stem[:4]
        base_new_name = f"test_{prefix}"
        ext = ".png" # 強制副檔名
        
        # 處理「前四位」相同導致的命名衝突 (Naming Collision)
        # 邏輯：若 test_AAAA.png 已存在，則嘗試 test_AAAA_1.png
        final_new_name = f"{base_new_name}{ext}"
        counter = 1
        while final_new_name in used_dest_names or (dest_dir / final_new_name).exists():
            final_new_name = f"{base_new_name}_{counter}{ext}"
            counter += 1
        
        used_dest_names.add(final_new_name)
        target_path = dest_dir / final_new_name

        try:
            # 執行複製 (類比 C# File.Copy)
            # 使用 shutil.copy2 確保 Metadata (如修改時間) 被保留
            shutil.copy2(file_path, target_path)
            print(f"[Success] 原檔 {file_path.name} (Key:{file_path.stem[5:9]}) -> {final_new_name}")
        except Exception as e:
            print(f"[Error] 無法處理檔案 {file_path.name}: {e}")

if __name__ == "__main__":
    execute_constrained_sample_copy()