import torch
import torch.onnx
from Step03_model_architecture import TixcraftCRNN

# =================================================================
# ⚙️ 配置區
# =================================================================
MODEL_PATH = "tixcraft_ocr_v1.pth"
ONNX_EXPORT_PATH = "tixcraft_ocr.onnx"
NUM_CLASS = 27  # 26字母 + 1 blank

def export():
    # 1. 重新載入模型大腦
    model = TixcraftCRNN(NUM_CLASS)
    
    # 2. 讀取你剛訓練好的權重
    print(f"📂 正在讀取權重: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # 3. 準備一個「假數據 (Dummy Input)」
    # 這是為了讓 PyTorch 跑一遍流程，記錄下所有的數學算式
    # 格式跟訓練時一樣: [Batch, Channel, Height, Width]
    dummy_input = torch.randn(1, 3, 60, 160)

    # 4. 開始導出
    print(f"🚀 正在轉換為 ONNX 格式...")
    torch.onnx.export(
        model,               # 你的大腦
        dummy_input,         # 告訴 ONNX 輸入的形狀
        ONNX_EXPORT_PATH,    # 輸出的檔名
        export_params=True,  # 包含訓練好的權重
        opset_version=12,    # ONNX 版本，建議 12 以上
        do_constant_folding=True, # 優化開關：把常數運算預先算好
        input_names=['input'],    # 給 C# 呼叫用的輸入端名稱
        output_names=['output'],  # 給 C# 呼叫用的輸出端名稱
        dynamic_axes={            # 允許 Batch Size 動力變化
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ 導出成功！檔案位於: {ONNX_EXPORT_PATH}")
    print(f"💡 之後在 C# 中，你只需要這個 .onnx 檔案，連 Python 都不用裝了！")

if __name__ == "__main__":
    export()