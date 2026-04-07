import os
import time
import uuid
from pathlib import Path
from playwright.sync_api import sync_playwright

# =================================================================
# ⚙️ 配置區 (Configuration)
# =================================================================
# 目標練習網址 (包含座位、價格、顏色參數)
TARGET_URL = "https://ticket-training.onrender.com/checking?seat=特3區&price=7800&color=%23dc64a1"
# 儲存資料夾 (Raw Data)
SAVE_DIR = Path("raw_captcha")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 抓取設定
TOTAL_IMAGES = 1000   # 建議第一波抓 1000-2000 張
DELAY_BETWEEN = 1.0   # 每次抓取間隔 (秒)，保護對方伺服器
HEADLESS_MODE = True  # 是否隱藏瀏覽器視窗 (True 為隱藏)

# =================================================================
# 🛠️ 執行邏輯 (Execution Logic)
# =================================================================

def run_scraper():
    """
    使用 Playwright 自動化瀏覽器模擬真實渲染並截圖驗證碼
    """
    with sync_playwright() as p:
        # 1. 啟動瀏覽器 (Chromium)
        print(f"🌐 正在啟動瀏覽器...")
        browser = p.chromium.launch(headless=HEADLESS_MODE)
        context = browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        print(f"🚀 開始採集任務... 目標數量: {TOTAL_IMAGES}")
        
        success_count = 0
        while success_count < TOTAL_IMAGES:
            try:
                # A. 導航至頁面並等待網路閒置
                page.goto(TARGET_URL, wait_until="networkidle")
                
                # B. 等待驗證碼圖片標籤出現 (對應你提供的 HTML 結構)
                # <img id="captcha-image" src="..." data-answer="xxxx">
                img_selector = "img#captcha-image"
                page.wait_for_selector(img_selector, timeout=5000)
                
                # C. 取得圖片元素與其對應的答案 (data-answer)
                captcha_element = page.query_selector(img_selector)
                answer = captcha_element.get_attribute("data-answer")
                
                if answer:
                    answer = answer.lower().strip()
                    # 生成唯一檔名：標籤_隨機ID.png
                    unique_id = uuid.uuid4().hex[:6]
                    file_path = SAVE_DIR / f"{answer}_{unique_id}.png"
                    
                    # D. 截圖該元素 (這比下載 URL 更能保留真實的渲染特徵，如抗鋸齒)
                    captcha_element.screenshot(path=str(file_path))
                    
                    success_count += 1
                    # 強制刷新終端機輸出，讓你即時看到進度
                    print(f"✅ [{success_count}/{TOTAL_IMAGES}] 已儲存: {file_path.name}", flush=True)
                else:
                    print("⚠️ 找不到 data-answer 屬性，重試中...")

                # 禮貌延遲，避免被網站封鎖
                time.sleep(DELAY_BETWEEN)

            except Exception as e:
                print(f"❌ 抓取過程發生異常: {e}")
                time.sleep(2) # 發生錯誤時停頓一下再繼續

        # 任務結束，關閉瀏覽器
        browser.close()
        print(f"\n🏁 採集完成！所有真實樣本已存放在: {SAVE_DIR.absolute()}")

if __name__ == "__main__":
    # 在執行前，請確保已安裝: pip install playwright && playwright install chromium
    run_scraper()