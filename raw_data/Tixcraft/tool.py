import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pathlib import Path

# 自動偵測腳本所在路徑，或是手動指定
# 使用 r'path' 原始字串格式避開轉義字元問題
CURRENT_DIR = Path(__file__).parent.absolute() 

class Labeler:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR 標註專家 - 快速命名工具")
        
        # 取得資料夾內所有圖片
        self.file_list = [f for f in os.listdir(CURRENT_DIR) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                         and f.lower() != 'tool.py']
        
        if not self.file_list:
            messagebox.showinfo("提示", "此資料夾內沒有找到圖片檔！")
            root.destroy()
            return

        self.index = 0

        # UI 佈局
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=20)

        self.info_label = tk.Label(root, text="", font=("Microsoft JhengHei", 10))
        self.info_label.pack()

        # 輸入框（大字體方便視覺確認）
        self.entry = tk.Entry(root, font=("Consolas", 28), justify='center')
        self.entry.pack(pady=20, padx=50)
        self.entry.bind("<Return>", self.save_and_next)
        self.entry.focus_set()

        self.load_image()

    def load_image(self):
        if self.index < len(self.file_list):
            filename = self.file_list[self.index]
            self.img_path = CURRENT_DIR / filename
            
            # 開啟並調整顯示尺寸 (維持比例)
            img = Image.open(self.img_path)
            # 放大顯示以便看清細節，適合標註
            display_size = (300, 100) 
            img = img.resize(display_size, Image.Resampling.LANCZOS)
            
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img)
            self.info_label.config(text=f"進度: {self.index + 1}/{len(self.file_list)} | 原始檔名: {filename}")
            self.entry.delete(0, tk.END)
        else:
            messagebox.showinfo("完成", "恭喜！所有圖片標註完畢。")
            self.root.destroy()

    def save_and_next(self, event):
        raw_text = self.entry.get().strip()
        if not raw_text:
            return # 防止空標註
            
        ext = Path(self.file_list[self.index]).suffix
        new_filename = f"{raw_text}{ext}"
        new_path = CURRENT_DIR / new_filename
        
        # 衝突處理：若檔名已存在則加後綴
        counter = 1
        final_path = new_path
        while final_path.exists():
            final_path = CURRENT_DIR / f"{raw_text}_{counter}{ext}"
            counter += 1
            
        try:
            # 關閉目前圖片的指標，否則 Windows 無法重命名
            self.img_label.config(image='')
            os.rename(self.img_path, final_path)
            self.index += 1
            self.load_image()
        except Exception as e:
            messagebox.showerror("錯誤", f"無法重命名: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    # 視窗置中
    root.eval('tk::PlaceWindow . center')
    app = Labeler(root)
    root.mainloop()