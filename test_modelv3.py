import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
from configs.base_config import BaseConfig

# ⭐ 尝试导入 Decord (快速)
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
    print("✅ Decord 可用 - 将使用快速模式")
except ImportError:
    DECORD_AVAILABLE = False
    print("⚠️  Decord 未安装 - 使用 OpenCV (较慢)")
    print("💡 安装 Decord 可提速 3-5倍: pip install decord")

# ⭐ PyTorch Canny 边缘检测 (GPU加速)
class PytorchCanny(nn.Module):
    """PyTorch 实现的 Canny 边缘检测 (GPU 加速)"""
    def __init__(self, low_threshold=50, high_threshold=150):
        super(PytorchCanny, self).__init__()
        self.low = low_threshold
        self.high = high_threshold

        # 高斯核 (5x5, sigma=1)
        gaussian = torch.tensor([
            [2, 4, 5, 4, 2],
            [4, 9, 12, 9, 4],
            [5, 12, 15, 12, 5],
            [4, 9, 12, 9, 4],
            [2, 4, 5, 4, 2]
        ], dtype=torch.float32).view(1, 1, 5, 5) / 159.0
        self.register_buffer('gaussian', gaussian)

        # Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        x = x.float() / 255.0
        smoothed = F.conv2d(x, self.gaussian, padding=2)
        gx = F.conv2d(smoothed, self.sobel_x, padding=1)
        gy = F.conv2d(smoothed, self.sobel_y, padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        ang = torch.atan2(gy, gx)
        
        mag_sup = self._non_max_suppression(mag, ang)
        
        high = (mag_sup > self.high / 255.0).float()
        low = ((mag_sup >= self.low / 255.0) & (mag_sup < self.high / 255.0)).float()
        
        edges = high.clone()
        kernel = torch.ones(1, 1, 3, 3, device=x.device)
        for _ in range(2):
            neighbor_count = F.conv2d(edges, kernel, padding=1)
            connected = (neighbor_count > 0).float() * low
            edges = edges + connected
            low = low * (1 - connected)
        
        edges = torch.clamp(edges, 0, 1)
        edges = (edges * 255).byte()
        mag = mag * 255
        ang = torch.rad2deg((ang + torch.pi) % (2 * torch.pi)) % 180
        
        return edges, mag, ang

    def _non_max_suppression(self, mag, ang):
        b, _, h, w = mag.shape
        mag_sup = torch.zeros_like(mag)
        ang_deg = torch.rad2deg((ang + torch.pi) % (2 * torch.pi))
        angle_quant = torch.round(ang_deg / 45) % 8 * 45
        
        mag_sup[:, :, 0, :] = 0
        mag_sup[:, :, -1, :] = 0
        mag_sup[:, :, :, 0] = 0
        mag_sup[:, :, :, -1] = 0
        
        mask_h = (angle_quant % 180 < 22.5) | (angle_quant % 180 > 157.5)
        n_right = torch.roll(mag, shifts=-1, dims=3)
        n_left = torch.roll(mag, shifts=1, dims=3)
        is_max_h = (mag > n_right) & (mag > n_left)
        mag_sup[mask_h & is_max_h] = mag[mask_h & is_max_h]
        
        mask_v = (angle_quant % 180 >= 67.5) & (angle_quant % 180 < 112.5)
        n_down = torch.roll(mag, shifts=-1, dims=2)
        n_up = torch.roll(mag, shifts=1, dims=2)
        is_max_v = (mag > n_down) & (mag > n_up)
        mag_sup[mask_v & is_max_v] = mag[mask_v & is_max_v]
        
        mask_d1 = (angle_quant % 180 >= 22.5) & (angle_quant % 180 < 67.5)
        n_diag1 = torch.roll(torch.roll(mag, shifts=-1, dims=2), shifts=-1, dims=3)
        n_diag2 = torch.roll(torch.roll(mag, shifts=1, dims=2), shifts=1, dims=3)
        is_max_d1 = (mag > n_diag1) & (mag > n_diag2)
        mag_sup[mask_d1 & is_max_d1] = mag[mask_d1 & is_max_d1]
        
        mask_d2 = (angle_quant % 180 >= 112.5) & (angle_quant % 180 < 157.5)
        n_diag3 = torch.roll(torch.roll(mag, shifts=-1, dims=2), shifts=1, dims=3)
        n_diag4 = torch.roll(torch.roll(mag, shifts=1, dims=2), shifts=-1, dims=3)
        is_max_d2 = (mag > n_diag3) & (mag > n_diag4)
        mag_sup[mask_d2 & is_max_d2] = mag[mask_d2 & is_max_d2]
        
        return mag_sup

# ⭐ TCN + BiLSTM 模块 (支持边缘特征 1344维)
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, dropout=0.3, lstm_hidden_dim=256, edge_dim=64):
        super(TemporalConvNet, self).__init__()
        self.edge_dim = edge_dim
        self.core_dim = input_dim - edge_dim
        
        self.layers = nn.ModuleList()
        dilations = [1, 2, 4, 8]
        
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else dilations[-1]
            layer = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, 
                         dilation=dilation, padding=dilation),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)
        
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        x_tcn = x.transpose(1, 2)
        
        for layer in self.layers:
            residual = x_tcn
            x_tcn = layer(x_tcn)
            if x_tcn.shape == residual.shape:
                x_tcn = x_tcn + residual
        
        x_lstm = x_tcn.transpose(1, 2)
        lstm_out, _ = self.bilstm(x_lstm)
        global_feature = lstm_out.mean(dim=1)
        output = self.classifier(global_feature)
        
        return output, global_feature

# ⭐ 特征比对模块 (支持边缘特征 1344维)
class FeatureMatchingModule(nn.Module):
    def __init__(self, feature_dim=1344, num_classes=10, edge_dim=64):
        super(FeatureMatchingModule, self).__init__()
        self.feature_dim = feature_dim
        self.edge_dim = edge_dim
        self.core_dim = feature_dim - edge_dim
        self.num_classes = num_classes
        
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.class_prototypes)
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        self.core_projection = nn.Sequential(
            nn.Linear(self.core_dim, self.core_dim),
            nn.ReLU(),
            nn.Linear(self.core_dim, self.core_dim)
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(self.edge_dim, self.edge_dim),
            nn.ReLU(),
            nn.Linear(self.edge_dim, self.edge_dim)
        )
    
    def forward(self, features):
        batch_size, seq_len, feature_dim = features.shape
        video_feature = features.mean(dim=1)
        
        core_features = video_feature[:, :self.core_dim]
        edge_features = video_feature[:, self.core_dim:]
        
        projected_core = self.core_projection(core_features)
        projected_edge = self.edge_projection(edge_features)
        projected_feature = torch.cat([projected_core, projected_edge], dim=1)
        
        projected_feature = F.normalize(projected_feature, p=2, dim=1)
        prototypes = F.normalize(self.class_prototypes, p=2, dim=1)
        
        similarity = torch.mm(projected_feature, prototypes.t())
        similarity = similarity / self.temperature
        
        return similarity, projected_feature

# ⭐ 混合模型 (支持边缘特征 1344维)
class HybridFeatureTCNModel(nn.Module):
    def __init__(self, config, num_classes, edge_dim=64):
        super(HybridFeatureTCNModel, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.edge_dim = edge_dim
        self.core_dim = 1280
        
        self.tcn = TemporalConvNet(
            input_dim=self.core_dim + self.edge_dim,
            output_dim=num_classes,
            num_layers=3,
            dropout=0.3,
            lstm_hidden_dim=config.lstm_hidden_dim,
            edge_dim=self.edge_dim
        )
        
        self.feature_matching = FeatureMatchingModule(
            feature_dim=self.core_dim + self.edge_dim,
            num_classes=num_classes,
            edge_dim=self.edge_dim
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor([0.6, 0.4]))
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim * 2 + (self.core_dim + self.edge_dim), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, return_features=False):
        tcn_logits, tcn_features = self.tcn(x)
        matching_logits, matching_features = self.feature_matching(x)
        
        weights = F.softmax(self.fusion_weight, dim=0)
        weighted_logits = weights[0] * tcn_logits + weights[1] * matching_logits
        
        combined_features = torch.cat([tcn_features, matching_features], dim=1)
        fusion_logits = self.fusion_classifier(combined_features)
        
        final_logits = 0.5 * weighted_logits + 0.5 * fusion_logits
        
        if return_features:
            return final_logits, {
                'tcn_logits': tcn_logits,
                'matching_logits': matching_logits,
                'tcn_features': tcn_features,
                'matching_features': matching_features,
                'fusion_weights': weights
            }
        
        return final_logits

# ⭐ 旧版单一模型 (向后兼容, 不支持边缘)
class FeatureTCNModel(nn.Module):
    def __init__(self, config, num_classes):
        super(FeatureTCNModel, self).__init__()
        self.config = config
        self.tcn = TemporalConvNet(
            input_dim=1280,
            output_dim=num_classes,
            num_layers=3,
            dropout=0.3,
            lstm_hidden_dim=config.lstm_hidden_dim,
            edge_dim=0  # 无边缘
        )
    
    def forward(self, x):
        output, _ = self.tcn(x)
        return output

class AnimatorPredictor:
    def __init__(self, model_path='./models/best_model.pth'):
        """初始化预测器"""
        self.config = BaseConfig()
        self.device = self.config.device
        self.use_decord = DECORD_AVAILABLE
        
        # 载入模型
        print("🔥 载入训练好的模型...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.num_classes = checkpoint['num_classes']
        self.label_to_idx = checkpoint['label_mapping']['label_to_idx']
        
        idx_to_label_raw = checkpoint['label_mapping']['idx_to_label']
        self.idx_to_label = {int(k): v for k, v in idx_to_label_raw.items()}
        
        # ⭐ 检测模型类型 (是否支持边缘特征)
        model_type = checkpoint.get('model_type', 'single')
        edge_dim = checkpoint.get('edge_dim', 0)
        
        if model_type == 'hybrid_edge':
            print(f"📌 检测到混合模型 (边缘版本, {edge_dim}维边缘特征)")
            self.model = HybridFeatureTCNModel(self.config, self.num_classes, edge_dim=edge_dim)
            self.is_hybrid = True
            self.use_edge_features = True
            self.edge_dim = edge_dim
        elif model_type == 'hybrid':
            print("📌 检测到混合模型 (无边缘版本)")
            self.model = HybridFeatureTCNModel(self.config, self.num_classes, edge_dim=0)
            self.is_hybrid = True
            self.use_edge_features = False
            self.edge_dim = 0
        else:
            print("📌 检测到单一模型 (旧版)")
            self.model = FeatureTCNModel(self.config, self.num_classes)
            self.is_hybrid = False
            self.use_edge_features = False
            self.edge_dim = 0
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型载入完成! 可识别 {self.num_classes} 个原画师")
        print(f"📋 原画师列表: {list(self.idx_to_label.values())}")
        print(f"⚡ 特征维度: {1280 + self.edge_dim} (EfficientNet {1280} + 边缘 {self.edge_dim})")
        
        # 设置特征提取器
        self.setup_feature_extractor()
    
    def setup_feature_extractor(self):
        """设置 EfficientNet 特征提取器"""
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        
        print("🔥 载入 EfficientNetV2-S 特征提取器...")
        efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.feature_extractor = efficientnet.features
        self.feature_extractor.eval()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # ⭐ 初始化 Canny (如果需要边缘特征)
        if self.use_edge_features:
            self.canny_model = PytorchCanny(low_threshold=50, high_threshold=150).to(self.device)
            print("✅ 边缘检测器 (PyTorch GPU Canny) 已初始化")
        
        target_size = (self.config.target_height, self.config.target_width)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self._smart_resize(img, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _smart_resize(self, image, target_size, content_retain=0.92):
        """智能调整大小"""
        target_h, target_w = target_size
        orig_w, orig_h = image.size
        
        target_ratio = target_w / target_h
        orig_ratio = orig_w / orig_h
        
        tolerance = getattr(self.config, 'aspect_ratio_tolerance', 0.08)
        if abs(target_ratio - orig_ratio) / orig_ratio < tolerance:
            return image.resize((target_w, target_h), Image.LANCZOS)
        
        if orig_ratio > target_ratio:
            scale = target_h / orig_h
            new_w = int(orig_w * scale)
            new_h = target_h
            resized = image.resize((new_w, new_h), Image.LANCZOS)
            
            crop_w = int(new_w * content_retain)
            crop_w = min(crop_w, target_w)
            
            left = (new_w - crop_w) // 2
            cropped = resized.crop((left, 0, left + crop_w, new_h))
            
            if crop_w != target_w:
                return cropped.resize((target_w, target_h), Image.LANCZOS)
            return cropped
        else:
            scale = target_w / orig_w
            new_w = target_w
            new_h = int(orig_h * scale)
            resized = image.resize((new_w, new_h), Image.LANCZOS)
            
            crop_h = int(new_h * content_retain)
            crop_h = min(crop_h, target_h)
            
            top = (new_h - crop_h) // 2
            cropped = resized.crop((0, top, new_w, top + crop_h))
            
            if crop_h != target_h:
                return cropped.resize((target_w, target_h), Image.LANCZOS)
            return cropped
    
    def _extract_edge_features(self, frame_rgb):
        """⭐ 提取边缘特征 (64维) - 与训练时一致"""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).byte().to(self.device)
        edges_tensor, mag_tensor, ang_tensor = self.canny_model(gray_tensor)
        edges = edges_tensor.squeeze().cpu().numpy()
        mag = mag_tensor.squeeze().cpu().numpy()
        ang_deg = ang_tensor.squeeze().cpu().numpy()
        
        edge_density = np.sum(edges > 0) / (h * w)
        features = [edge_density]
        
        hist, _ = np.histogram(ang_deg[edges > 0], bins=8, range=(0, 180))
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-6)
        features.extend(hist.tolist())
        
        edge_mags = mag[edges > 0]
        if len(edge_mags) > 0:
            features.extend([
                np.mean(edge_mags),
                np.std(edge_mags),
                np.max(edge_mags),
                np.min(edge_mags)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
        total_area = sum(cv2.contourArea(c) for c in contours) if contours else 0
        avg_area = total_area / max(num_contours, 1)
        fill_ratio = total_area / (h * w + 1e-6)
        features.extend([num_contours / 100.0, total_perimeter / (h * w), avg_area / (h * w), fill_ratio])
        
        edge_feat = np.array(features, dtype=np.float32)
        if len(edge_feat) < 64:
            edge_feat = np.pad(edge_feat, (0, 64 - len(edge_feat)))
        else:
            edge_feat = edge_feat[:64]
        
        return edge_feat
    
    def _calculate_sample_indices(self, total_frames, fps):
        """计算采样帧索引(动态长度,与训练时一致)"""
        if fps <= 0:
            fps = 24
        
        video_duration = total_frames / fps
        target_fps = getattr(self.config, 'target_fps', 12)
        min_duration = getattr(self.config, 'min_video_duration', 2)
        max_duration = getattr(self.config, 'max_video_duration', 15)
        
        process_duration = max(min_duration, min(video_duration, max_duration))
        target_frames = int(process_duration * target_fps)
        
        frame_interval = fps / target_fps
        sample_indices = []
        current_pos = 0.0
        
        while len(sample_indices) < target_frames and int(current_pos) < total_frames:
            sample_indices.append(int(current_pos))
            current_pos += frame_interval
        
        if not sample_indices:
            sample_indices = [0]
        
        return sample_indices, len(sample_indices), process_duration
    
    def extract_video_features_decord(self, video_path):
        """⭐ 使用 Decord 提取特征 (1344维: 1280 + 64)"""
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            
            if total_frames == 0 or video_fps <= 0:
                print(f"❌ 影片无效: {os.path.basename(video_path)}")
                return None
            
            duration = total_frames / video_fps
            sample_indices, actual_frames, process_duration = self._calculate_sample_indices(total_frames, video_fps)
            target_fps = getattr(self.config, 'target_fps', 12)
            
            print(f"  📹 {os.path.basename(video_path)}")
            print(f"     长度: {duration:.1f}秒 | 采样: {actual_frames} 帧 ({process_duration:.1f}秒 @ {target_fps}FPS)")
            
            # 批量读取帧
            frames_batch = vr.get_batch(sample_indices).asnumpy()
            
            frames_rgb = []
            for frame_bgr in frames_batch:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
            
            # ⭐ 批量提取 EfficientNet 特征
            frames_pil = [Image.fromarray(frame) for frame in frames_rgb]
            frame_tensors = [self.transform(pil) for pil in frames_pil]
            
            efficientnet_features = []
            batch_size = 16
            with torch.no_grad():
                for i in range(0, len(frame_tensors), batch_size):
                    batch_frames = frame_tensors[i:i+batch_size]
                    batch_tensor = torch.stack(batch_frames).to(self.device)
                    
                    batch_features = self.feature_extractor(batch_tensor)
                    batch_features = self.global_pool(batch_features)
                    batch_features = batch_features.flatten(1).cpu().numpy()
                    
                    efficientnet_features.extend(batch_features)
            
            efficientnet_array = np.array(efficientnet_features)  # [N, 1280]
            
            # ⭐ 如果需要边缘特征,逐帧提取
            if self.use_edge_features:
                edge_features = []
                for frame_rgb in frames_rgb:
                    edge_feat = self._extract_edge_features(frame_rgb)
                    edge_features.append(edge_feat)
                edge_array = np.array(edge_features)  # [N, 64]
                
                # 拼接特征 [N, 1344]
                combined_features = np.concatenate([efficientnet_array, edge_array], axis=1)
                print(f"     ✅ 提取完成: {len(combined_features)} 个特征向量 (1344维)")
            else:
                combined_features = efficientnet_array  # [N, 1280]
                print(f"     ✅ 提取完成: {len(combined_features)} 个特征向量 (1280维)")
            
            return combined_features
            
        except Exception as e:
            print(f"❌ Decord 提取失败: {e}")
            print(f"   尝试使用 OpenCV 备用方法...")
            return self.extract_video_features_opencv(video_path)
    
    def extract_video_features_opencv(self, video_path):
        """⭐ 使用 OpenCV 提取特征 (1344维: 1280 + 64)"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 无法打开影片: {os.path.basename(video_path)}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                print(f"❌ 影片帧数为0: {os.path.basename(video_path)}")
                cap.release()
                return None
            
            if video_fps <= 0:
                video_fps = 24
            
            duration = total_frames / video_fps
            sample_indices, actual_frames, process_duration = self._calculate_sample_indices(total_frames, video_fps)
            target_fps = getattr(self.config, 'target_fps', 12)
            
            print(f"  📹 {os.path.basename(video_path)}")
            print(f"     长度: {duration:.1f}秒 | 采样: {actual_frames} 帧 ({process_duration:.1f}秒 @ {target_fps}FPS)")
            
            # 逐帧提取
            combined_features = []
            
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total_frames-1))
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tensor = self.transform(frame_pil)
                    
                    # ⭐ 提取 EfficientNet 特征 [1280]
                    with torch.no_grad():
                        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
                        feat = self.feature_extractor(frame_tensor)
                        feat = self.global_pool(feat)
                        feat = feat.flatten(1).cpu().numpy()
                    
                    # ⭐ 如果需要边缘特征,提取并拼接
                    if self.use_edge_features:
                        edge_feat = self._extract_edge_features(frame_rgb)
                        combined_feat = np.concatenate([feat.flatten(), edge_feat])  # [1344]
                    else:
                        combined_feat = feat.flatten()  # [1280]
                    
                    combined_features.append(combined_feat)
                else:
                    if combined_features:
                        combined_features.append(combined_features[-1])
                    else:
                        feature_dim = 1280 + (self.edge_dim if self.use_edge_features else 0)
                        combined_features.append(np.zeros(feature_dim))
            
            cap.release()
            
            if not combined_features:
                print(f"❌ 未能提取任何帧")
                return None
            
            features_array = np.array(combined_features)
            feature_dim = 1280 + (self.edge_dim if self.use_edge_features else 0)
            print(f"     ✅ 提取完成: {len(features_array)} 个特征向量 ({feature_dim}维)")
            
            return features_array
            
        except Exception as e:
            print(f"❌ OpenCV 提取失败: {e}")
            return None
    
    def extract_video_features(self, video_path):
        """提取影片特征(自动选择最快方法)"""
        if self.use_decord:
            return self.extract_video_features_decord(video_path)
        else:
            return self.extract_video_features_opencv(video_path)
    
    def predict(self, video_path):
        """预测单一影片"""
        print("\n" + "="*60)
        print("🎯 开始原画师识别")
        print(f"📹 影片: {os.path.basename(video_path)}")
        print("="*60)
        
        # 提取特征
        features = self.extract_video_features(video_path)
        if features is None:
            return None
        
        # ⭐ 转换为张量 [1, seq_len, feature_dim] (1280 或 1344)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # ⭐ 预测 (支持混合模型)
        print("🤖 模型推理中...")
        with torch.no_grad():
            if self.is_hybrid:
                outputs, branch_outputs = self.model(features_tensor, return_features=True)
                
                # 各分支的机率
                tcn_probs = F.softmax(branch_outputs['tcn_logits'], dim=1)[0]
                matching_probs = F.softmax(branch_outputs['matching_logits'], dim=1)[0]
                fusion_weights = branch_outputs['fusion_weights']
            else:
                outputs = self.model(features_tensor)
            
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # 获取预测结果
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_idx = predicted_idx.item()
        predicted_animator = self.idx_to_label[predicted_idx]
        
        # 所有机率
        results = {animator: probabilities[idx].item() for idx, animator in self.idx_to_label.items()}
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # ⭐ 显示结果
        print("\n" + "="*60)
        print("📊 识别结果")
        print("="*60)
        print(f"\n🎨 预测原画师: {predicted_animator}")
        print(f"🎯 置信度: {confidence.item()*100:.2f}%")
        
        # ⭐ 混合模型详细分析
        if self.is_hybrid:
            print("\n🔍 混合模型分析:")
            print("-" * 60)
            
            # 各分支的预测
            tcn_conf, tcn_pred_idx = torch.max(tcn_probs, 0)
            matching_conf, matching_pred_idx = torch.max(matching_probs, 0)
            
            tcn_pred_animator = self.idx_to_label[tcn_pred_idx.item()]
            matching_pred_animator = self.idx_to_label[matching_pred_idx.item()]
            
            print(f"  TCN+BiLSTM 分支:")
            print(f"    预测: {tcn_pred_animator} (置信度: {tcn_conf.item()*100:.2f}%)")
            print(f"  特征比对分支:")
            print(f"    预测: {matching_pred_animator} (置信度: {matching_conf.item()*100:.2f}%)")
            print(f"  融合权重:")
            print(f"    TCN: {fusion_weights[0].item()*100:.1f}%, 特征比对: {fusion_weights[1].item()*100:.1f}%")
            
            if self.use_edge_features:
                print(f"  边缘特征: ✅ 已整合 ({self.edge_dim}维)")
        
        print("\n📈 所有原画师的机率分布:")
        print("-" * 60)
        for animator, prob in sorted_results:
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            marker = "👉" if animator == predicted_animator else "   "
            print(f"{marker} {animator:15s} | {bar} | {prob*100:6.2f}%")
        
        # ⭐ 混合模型各分支的TOP-3
        if self.is_hybrid:
            print("\n🔎 各分支TOP-3对比:")
            print("-" * 60)
            
            tcn_results = {animator: tcn_probs[idx].item() for idx, animator in self.idx_to_label.items()}
            matching_results = {animator: matching_probs[idx].item() for idx, animator in self.idx_to_label.items()}
            
            tcn_top3 = sorted(tcn_results.items(), key=lambda x: x[1], reverse=True)[:3]
            matching_top3 = sorted(matching_results.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print("  TCN+BiLSTM:")
            for i, (animator, prob) in enumerate(tcn_top3, 1):
                print(f"    {i}. {animator}: {prob*100:.2f}%")
            
            print("\n  特征比对:")
            for i, (animator, prob) in enumerate(matching_top3, 1):
                print(f"    {i}. {animator}: {prob*100:.2f}%")
        
        print("="*60 + "\n")
        
        result = {
            'video_path': video_path,
            'predicted_animator': predicted_animator,
            'confidence': confidence.item(),
            'probabilities': results,
            'sorted_results': sorted_results,
            'feature_dim': features.shape[1],
            'use_edge_features': self.use_edge_features
        }
        
        # ⭐ 添加混合模型分析
        if self.is_hybrid:
            result['hybrid_analysis'] = {
                'tcn_prediction': tcn_pred_animator,
                'tcn_confidence': tcn_conf.item(),
                'matching_prediction': matching_pred_animator,
                'matching_confidence': matching_conf.item(),
                'fusion_weights': {
                    'tcn': fusion_weights[0].item(),
                    'matching': fusion_weights[1].item()
                }
            }
        
        return result

def main():
    """主测试函数"""
    video_paths = [
        './无标题视频——使用Clipchamp制作 (1).mp4',
        './yutaka_nakamura_69068.mp4',
        './yoshimichi_kameda_215914_000.mp4',
        './龜.mp4',
    ]
    
    # 初始化预测器
    predictor = AnimatorPredictor('./models/best_model.pth')
    
    # 批量预测
    all_results = []
    for video_path in video_paths:
        if os.path.exists(video_path):
            result = predictor.predict(video_path)
            if result:
                all_results.append(result)
        else:
            print(f"⚠️  文件不存在: {video_path}")
    
    # 汇总统计
    if all_results:
        print("\n" + "="*60)
        print("📊 批量测试汇总")
        print("="*60)
        print(f"测试影片数: {len(all_results)}")
        avg_confidence = sum(r['confidence'] for r in all_results) / len(all_results)
        print(f"平均置信度: {avg_confidence*100:.2f}%")
        
        # ⭐ 显示特征维度信息
        if all_results[0].get('use_edge_features'):
            print(f"特征维度: {all_results[0]['feature_dim']} (含 {predictor.edge_dim} 维边缘特征)")
        else:
            print(f"特征维度: {all_results[0]['feature_dim']} (无边缘特征)")
        
        if predictor.is_hybrid and 'hybrid_analysis' in all_results[0]:
            print("\n🔍 混合模型整体分析:")
            tcn_correct = sum(1 for r in all_results if r['hybrid_analysis']['tcn_prediction'] == r['predicted_animator'])
            matching_correct = sum(1 for r in all_results if r['hybrid_analysis']['matching_prediction'] == r['predicted_animator'])
            print(f"  TCN分支与最终结果一致: {tcn_correct}/{len(all_results)}")
            print(f"  特征比对与最终结果一致: {matching_correct}/{len(all_results)}")
        
        print("="*60)

if __name__ == '__main__':
    main()