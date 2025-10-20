# 檔名: configs/base_config.py

import torch
import os

class BaseConfig:
    def __init__(self):
        # ⭐ 視頻採樣配置 (FPS-based)
        self.target_fps = 12  # 每秒12幀
        self.min_video_duration = 2  # 最短處理2秒
        # ⭐⭐⭐ 關鍵改動 1：大幅縮短最大處理時長 ⭐⭐⭐
        self.max_video_duration = 60  # 從 15 秒進一步降至 10 秒以確保安全
        self.max_frames = self.target_fps * self.max_video_duration  # 現在是 12 * 10 = 120 幀上限
        
        # 圖像配置
        # ⭐⭐⭐ 關鍵改動 2：降低解析度 ⭐⭐⭐
        self.target_height = 360  # 原 540
        self.target_width = 640   # 原 960
        
        # 智能裁切配置
        self.content_retain = 0.92  # 保留92%中心內容
        self.aspect_ratio_tolerance = 0.08  # 8%誤差內直接縮放
        
        # 目錄配置
        self.feature_dir = './features'
        self.model_dir = './models'
        self.data_dir = './features'
        self.video_dir='./test-videos'
        
        # 特徵提取配置
        self.backbone_name = 'efficientnet_v2_s'
        self.feature_dim = 1344
        
        # 模型配置
        self.lstm_hidden_dim = 256
        self.num_heads = 8
        self.num_layers = 2
        self.dropout = 0.1
        
        # 訓練配置
        # ⭐⭐⭐ 關鍵改動 3：設定「有效批次大小」 ⭐⭐⭐
        self.batch_size = 8  # 這是我們期望通過梯度累積模擬的「有效批次大小」
        self.learning_rate = 0.001
        self.num_epochs = 20
        self.use_amp = True
        
        # 設備配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ⭐ 增量學習配置（在全量訓練中不啟用）
        self.use_incremental = False
        self.use_kd = False
        self.lambda_kd = 2.0
        self.kd_temperature = 2.5
        self.replay_ratio = 0.6
        self.head_warmup_epochs = 5
        self.freeze_backbone = True
        
        # 創建目錄
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"⚙️  配置加載:")
        print(f"   分辨率: {self.target_width}x{self.target_height}")
        print(f"   採樣: {self.target_fps} FPS")
        print(f"   處理範圍: {self.min_video_duration}-{self.max_video_duration}秒")
        print(f"   最大幀數: {self.max_frames} 幀")