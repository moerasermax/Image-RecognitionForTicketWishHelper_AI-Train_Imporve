import torch
import torch.nn as nn

# =================================================================
# 🧬 輔助元件：雙向 LSTM 層 (這是 Sub-Service)
# =================================================================
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        # batch_first=True 讓輸入格式為 [Batch, Seq, Feature]
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        # 1. 進入 LSTM 運算
        recurrent, _ = self.rnn(x)
        
        # 2. 取得維度並重新排布記憶體 (解決你剛才遇到的 RuntimeError)
        b, t, h = recurrent.size()
        t_rec = recurrent.reshape(b * t, h) # 這裡用 reshape 最安全
        
        # 3. 通過線性層進行維度映射
        output = self.embedding(t_rec)
        
        # 4. 還原成序列格式 [Batch, TimeSteps, nOut]
        output = output.reshape(b, t, -1)
        return output

# =================================================================
# 🧠 核心大腦：CRNN 模型 (這是 Main Controller)
# =================================================================
class TixcraftCRNN(nn.Module):
    def __init__(self, num_class):
        super(TixcraftCRNN, self).__init__()
        
        # CNN: 負責「看」圖片特徵
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, kernel_size=(3, 2), stride=1, padding=0),
            nn.ReLU(True)
        )
        
        # RNN: 負責「理解」字元順序
        # 這裡會依序通過兩個 BidirectionalLSTM 元件
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_class)
        )

    def forward(self, x):
        # 1. 圖像特徵提取 -> 輸出 [Batch, 512, 1, 39]
        conv = self.cnn(x)
        
        # 2. 維度轉換 (這是最關鍵的一步！)
        b, c, h, w = conv.size()
        assert h == 1, f"高度必須為 1, 當前為 {h}"
        
        # [Batch, 512, 1, 39] -> [Batch, 512, 39]
        conv = conv.squeeze(2)
        # [Batch, 512, 39] -> [Batch, 39, 512] (這才是 RNN 想要的格式)
        conv = conv.permute(0, 2, 1)
        
        # 3. 進入 RNN 邏輯層 -> 輸出 [Batch, 39, 27]
        output = self.rnn(conv)
        
        # 4. 最後套用 LogSoftmax，讓輸出的數值變成機率分布 (CTC Loss 要求)
        return output.log_softmax(2)

# =================================================================
# 🧪 測試驗證
# =================================================================
if __name__ == "__main__":
    model = TixcraftCRNN(num_class=27)
    fake_input = torch.randn(1, 3, 60, 160)
    output = model(fake_input)
    
    print(f"✅ 模型初始化與前向傳播成功！")
    print(f"📊 最終輸出形狀: {output.shape} (符合期待的 [1, 39, 27])")