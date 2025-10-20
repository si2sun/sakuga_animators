import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F  # ⭐ 新增：F for conv2d 等

# ⭐ 嘗試導入 Decord (快速)
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
    print("✅ Decord 可用 - 將使用快速模式")
except ImportError:
    DECORD_AVAILABLE = False
    print("⚠️  Decord 未安裝 - 使用 OpenCV (較慢)")
    print("💡 安裝 Decord 可提速 3-5倍: pip install decord")

def detect_aspect_ratio(video_path):
    """檢測影片比例（獨立函數，避免循環導入）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "unknown"
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if width == 0 or height == 0:
        return "unknown"
    
    aspect_ratio = width / height
    
    if abs(aspect_ratio - 16/9) < 0.1:
        return "16:9"
    elif abs(aspect_ratio - 4/3) < 0.1:
        return "4:3"
    elif abs(aspect_ratio - 1) < 0.1:
        return "1:1"
    elif abs(aspect_ratio - 936/480) < 0.1:
        return "936:480"
    elif abs(aspect_ratio - 1136/480) < 0.1:
        return "1136:480"
    else:
        return "other"

class PytorchCanny(nn.Module):
    """
    PyTorch 實現的 Canny 邊緣檢測 (GPU 加速)
    返回: edges (二值邊緣圖), mag (梯度幅度), ang (梯度角度, 度數 0-180)
    """
    def __init__(self, low_threshold=50, high_threshold=150):
        super(PytorchCanny, self).__init__()
        self.low = low_threshold
        self.high = high_threshold

        # 高斯核 (5x5, sigma=1)
        k = 5
        gaussian = torch.tensor([
            [2, 4, 5, 4, 2],
            [4, 9, 12, 9, 4],
            [5, 12, 15, 12, 5],
            [4, 9, 12, 9, 4],
            [2, 4, 5, 4, 2]
        ], dtype=torch.float32).view(1, 1, k, k) / 159.0
        self.register_buffer('gaussian', gaussian)

        # Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):  # x: [B, 1, H, W] uint8 (0-255)
        x = x.float() / 255.0  # 正規化

        # 高斯平滑
        smoothed = F.conv2d(x, self.gaussian, padding=2)

        # 梯度
        gx = F.conv2d(smoothed, self.sobel_x, padding=1)
        gy = F.conv2d(smoothed, self.sobel_y, padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        ang = torch.atan2(gy, gx)  # 弧度 (-pi ~ pi)

        # 非最大抑制
        mag_sup = self._non_max_suppression(mag, ang)

        # 雙閾值遲滯
        high = (mag_sup > self.high / 255.0).float()
        low = ((mag_sup >= self.low / 255.0) & (mag_sup < self.high / 255.0)).float()

        # 遲滯：弱邊連通強邊 (用 3x3 鄰域檢查，迭代 2 次)
        edges = high.clone()
        kernel = torch.ones(1, 1, 3, 3, device=x.device)
        for _ in range(2):
            neighbor_count = F.conv2d(edges, kernel, padding=1)
            connected = (neighbor_count > 0).float() * low
            edges = edges + connected
            low = low * (1 - connected)
        
        edges = torch.clamp(edges, 0, 1)  # 確保 [0,1]
        edges = (edges * 255).byte()  # [B, 1, H, W] 0/255
        mag = mag * 255  # 轉回 0-255 範圍
        ang = torch.rad2deg((ang + torch.pi) % (2 * torch.pi)) % 180  # 弧度 → 度數 0-180°

        return edges, mag, ang

    def _non_max_suppression(self, mag, ang):
        # 向量化實現 (避免慢迴圈)
        b, _, h, w = mag.shape
        mag_sup = torch.zeros_like(mag)

        # 角度量化到 0/45/90/135°
        ang_deg = torch.rad2deg((ang + torch.pi) % (2 * torch.pi))  # 0-360
        angle_quant = torch.round(ang_deg / 45) % 8 * 45  # 0/45/.../315

        # 邊界處理 (簡單設 0)
        mag_sup[:, :, 0, :] = 0
        mag_sup[:, :, -1, :] = 0
        mag_sup[:, :, :, 0] = 0
        mag_sup[:, :, :, -1] = 0

        # 水平 (0°/180°)
        mask_h = (angle_quant % 180 < 22.5) | (angle_quant % 180 > 157.5)
        n_right = torch.roll(mag, shifts=-1, dims=3)
        n_left = torch.roll(mag, shifts=1, dims=3)
        is_max_h = (mag > n_right) & (mag > n_left)
        mag_sup[mask_h & is_max_h] = mag[mask_h & is_max_h]

        # 垂直 (90°)
        mask_v = (angle_quant % 180 >= 67.5) & (angle_quant % 180 < 112.5)
        n_down = torch.roll(mag, shifts=-1, dims=2)
        n_up = torch.roll(mag, shifts=1, dims=2)
        is_max_v = (mag > n_down) & (mag > n_up)
        mag_sup[mask_v & is_max_v] = mag[mask_v & is_max_v]

        # 45°
        mask_d1 = (angle_quant % 180 >= 22.5) & (angle_quant % 180 < 67.5)
        n_diag1 = torch.roll(torch.roll(mag, shifts=-1, dims=2), shifts=-1, dims=3)
        n_diag2 = torch.roll(torch.roll(mag, shifts=1, dims=2), shifts=1, dims=3)
        is_max_d1 = (mag > n_diag1) & (mag > n_up)
        mag_sup[mask_d1 & is_max_d1] = mag[mask_d1 & is_max_d1]

        # 135°
        mask_d2 = (angle_quant % 180 >= 112.5) & (angle_quant % 180 < 157.5)
        n_diag3 = torch.roll(torch.roll(mag, shifts=-1, dims=2), shifts=1, dims=3)
        n_diag4 = torch.roll(torch.roll(mag, shifts=1, dims=2), shifts=-1, dims=3)
        is_max_d2 = (mag > n_diag3) & (mag > n_diag4)
        mag_sup[mask_d2 & is_max_d2] = mag[mask_d2 & is_max_d2]

        return mag_sup

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.use_decord = DECORD_AVAILABLE
        self.setup_feature_extractor()
    
    def setup_feature_extractor(self):
        """設置特徵提取器"""
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        
        print("🔥 載入 EfficientNetV2-S 進行特徵提取...")
        efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.feature_extractor = efficientnet.features
        self.feature_extractor.eval()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 凍結權重
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_extractor = self.feature_extractor.to(self.config.device)
        
        # ⭐ 初始化 PyTorch Canny (GPU 加速)
        self.canny_model = PytorchCanny(low_threshold=50, high_threshold=150).to(self.config.device)
        
        # 使用智能調整大小的變換
        target_size = (self.config.target_height, self.config.target_width)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self._smart_resize(img, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # ⭐ 預熱模型 (加速首次運行)
        if self.config.device.type == 'cuda':
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.config.target_height, self.config.target_width).to(self.config.device)
                self.feature_extractor(dummy_input)
                self.canny_model(torch.zeros(1, 1, self.config.target_height, self.config.target_width, dtype=torch.uint8).to(self.config.device))
            torch.cuda.empty_cache()
    
    def _smart_resize(self, image, target_size):
        """智能調整大小：優先縮放，必要時智能裁切（保留92%中心內容）"""
        target_h, target_w = target_size
        orig_w, orig_h = image.size
        
        target_ratio = target_w / target_h
        orig_ratio = orig_w / orig_h
        
        # 獲取配置參數
        content_retain = getattr(self.config, 'content_retain', 0.92)
        tolerance = getattr(self.config, 'aspect_ratio_tolerance', 0.08)
        
        # 如果比例接近（誤差<8%），直接縮放（無裁切）
        if abs(target_ratio - orig_ratio) / orig_ratio < tolerance:
            return image.resize((target_w, target_h), Image.LANCZOS)
        
        # 比例差異大時，使用智能裁切
        if orig_ratio > target_ratio:
            # 原圖更寬，需要裁切左右
            scale = target_h / orig_h
            new_w = int(orig_w * scale)
            new_h = target_h
            resized = image.resize((new_w, new_h), Image.LANCZOS)
            
            # 計算保留區域（中心92%）
            crop_w = int(new_w * content_retain)
            crop_w = min(crop_w, target_w)
            
            # 居中裁切
            left = (new_w - crop_w) // 2
            cropped = resized.crop((left, 0, left + crop_w, new_h))
            
            # 如果裁切後仍不是目標尺寸，再縮放
            if crop_w != target_w:
                return cropped.resize((target_w, target_h), Image.LANCZOS)
            return cropped
        else:
            # 原圖更高，需要裁切上下
            scale = target_w / orig_w
            new_w = target_w
            new_h = int(orig_h * scale)
            resized = image.resize((new_w, new_h), Image.LANCZOS)
            
            # 計算保留區域（中心92%）
            crop_h = int(new_h * content_retain)
            crop_h = min(crop_h, target_h)
            
            # 居中裁切
            top = (new_h - crop_h) // 2
            cropped = resized.crop((0, top, new_w, top + crop_h))
            
            # 如果裁切後仍不是目標尺寸，再縮放
            if crop_h != target_h:
                return cropped.resize((target_w, target_h), Image.LANCZOS)
            return cropped
    
    def _extract_edge_features(self, frame_rgb):
        """
        ⭐ 新增：提取邊緣特徵（64維），使用 PyTorch Canny (GPU 加速)
        
        特徵組成:
        - 邊緣密度 (1維)
        - 邊緣方向直方圖 (8維, 0-180°)
        - 邊緣強度統計 (mean, std, max, min, 4維)
        - 輪廓統計 (數量, 總長度, 平均面積, 填充率, 4維)
        - 填充到64維
        
        Args:
            frame_rgb: (H, W, 3) numpy array
            
        Returns:
            edge_feat: (64,) numpy array
        """
        # 轉灰度 (CPU, 快)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # PyTorch Canny + Sobel (GPU)
        gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).byte().to(self.config.device)  # [1,1,H,W]
        edges_tensor, mag_tensor, ang_tensor = self.canny_model(gray_tensor)
        edges = edges_tensor.squeeze().cpu().numpy()  # [H, W] 0/255
        mag = mag_tensor.squeeze().cpu().numpy()  # [H, W] 0-255
        ang_deg = ang_tensor.squeeze().cpu().numpy()  # [H, W] 0-180°
        
        # 1. 邊緣密度
        edge_density = np.sum(edges > 0) / (h * w)
        features = [edge_density]
        
        # 2. 邊緣方向直方圖 (使用 PyTorch ang 在邊緣處)
        hist, _ = np.histogram(ang_deg[edges > 0], bins=8, range=(0, 180))
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-6)  # 歸一化
        features.extend(hist.tolist())
        
        # 3. 邊緣強度統計 (mag 在邊緣處)
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
        
        # 4. 輪廓統計 (仍用 OpenCV, CPU)
        contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
        total_area = sum(cv2.contourArea(c) for c in contours) if contours else 0
        avg_area = total_area / max(num_contours, 1)
        fill_ratio = total_area / (h * w + 1e-6)
        features.extend([num_contours / 100.0, total_perimeter / (h * w), avg_area / (h * w), fill_ratio])
        
        # 填充到64維
        edge_feat = np.array(features, dtype=np.float32)
        if len(edge_feat) < 64:
            edge_feat = np.pad(edge_feat, (0, 64 - len(edge_feat)))
        else:
            edge_feat = edge_feat[:64]
        
        return edge_feat
    
    def _detect_aspect_ratio(self, video_path):
        """檢測影片比例（使用獨立函數）"""
        return detect_aspect_ratio(video_path)
    
    def _calculate_sample_indices(self, total_frames, fps):
        """
        ⭐ 計算採樣幀索引（動態長度）
        
        Returns:
            sample_indices: 採樣幀的索引列表
            actual_frames: 實際採樣幀數
            process_duration: 實際處理時長
        """
        if fps <= 0:
            fps = 24  # 默認24fps
        
        # 計算視頻實際時長
        video_duration = total_frames / fps
        
        # 獲取配置參數
        target_fps = getattr(self.config, 'target_fps', 12)
        min_duration = getattr(self.config, 'min_video_duration', 2)
        max_duration = getattr(self.config, 'max_video_duration', 15)
        
        # 限制處理範圍
        process_duration = max(min_duration, min(video_duration, max_duration))
        
        # ⭐ 動態計算目標幀數（而非固定值）
        target_frames = int(process_duration * target_fps)
        
        # 計算幀間隔
        frame_interval = fps / target_fps
        
        # 生成採樣索引
        sample_indices = []
        current_pos = 0.0
        
        while len(sample_indices) < target_frames and int(current_pos) < total_frames:
            sample_indices.append(int(current_pos))
            current_pos += frame_interval
        
        # 確保至少有一幀
        if not sample_indices:
            sample_indices = [0]
        
        return sample_indices, len(sample_indices), process_duration
    
    def _extract_edge_features_batch(self, frames_rgb_list):
        """
        ⭐ 批量提取边缘特征 (GPU加速) - 不改变数值
        
        Args:
            frames_rgb_list: List of (H, W, 3) numpy arrays
            
        Returns:
            edge_features: (N, 64) numpy array
        """
        batch_size = 32  # 增大batch
        all_edge_features = []
        
        for i in range(0, len(frames_rgb_list), batch_size):
            batch_frames = frames_rgb_list[i:i+batch_size]
            
            # 1. 批量转灰度 (CPU, 快速) - 預先轉uint8避免後續轉型
            gray_batch = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.uint8) for frame in batch_frames]  # 提前astype
            h, w = gray_batch[0].shape
            
            # 2. 批量转GPU tensor [B, 1, H, W] - 直接從uint8
            gray_tensors = torch.from_numpy(np.stack(gray_batch)).unsqueeze(1).byte().to(self.config.device)  # stack一次，少轉換
            
            # 3. 批量Canny检测 (GPU)
            edges_batch, mag_batch, ang_batch = self.canny_model(gray_tensors)
            
            # 4. 批量提取特征 - 預先轉回CPU/numpy，批量處理統計
            edges_np = edges_batch.squeeze(1).cpu().numpy().astype(np.uint8)  # [B, H, W] uint8，一次轉
            mag_np = mag_batch.squeeze(1).cpu().numpy()  # [B, H, W] float
            ang_deg_np = ang_batch.squeeze(1).cpu().numpy()  # [B, H, W] float
            
            batch_features = []
            for j in range(len(batch_frames)):
                edges = edges_np[j]
                mag = mag_np[j]
                ang_deg = ang_deg_np[j]
                
                # 邊緣密度 & 方向直方圖 - 向量化（numpy快）
                edge_mask = edges > 0
                edge_density = np.sum(edge_mask) / (h * w)
                features = [edge_density]
                
                if np.sum(edge_mask) > 0:
                    hist, _ = np.histogram(ang_deg[edge_mask], bins=8, range=(0, 180))
                    hist = hist.astype(np.float32) / (np.sum(hist) + 1e-6)
                else:
                    hist = np.zeros(8, dtype=np.float32)
                features.extend(hist.tolist())
                
                # 強度統計 - 向量化
                edge_mags = mag[edge_mask]
                if len(edge_mags) > 0:
                    features.extend([
                        np.mean(edge_mags), np.std(edge_mags),
                        np.max(edge_mags), np.min(edge_mags)
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
                
                # 輪廓統計 - 優化：單幀cv2，預計算總和避免重複sum
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                num_contours = len(contours)
                if num_contours > 0:
                    perimeters = [cv2.arcLength(c, True) for c in contours]  # 預存list
                    areas = [cv2.contourArea(c) for c in contours]
                    total_perimeter = sum(perimeters)
                    total_area = sum(areas)
                    avg_area = total_area / num_contours  # 直接除，避免max(1)
                else:
                    total_perimeter = total_area = avg_area = 0.0
                fill_ratio = total_area / (h * w + 1e-6)
                features.extend([num_contours / 100.0, total_perimeter / (h * w), avg_area / (h * w), fill_ratio])
                
                # 填充到64維
                edge_feat = np.array(features, dtype=np.float32)
                if len(edge_feat) < 64:
                    edge_feat = np.pad(edge_feat, (0, 64 - len(edge_feat)))
                else:
                    edge_feat = edge_feat[:64]
                
                batch_features.append(edge_feat)
            
            all_edge_features.extend(batch_features)
        
        return np.array(all_edge_features)


    def extract_video_features_decord(self, video_path):
        """
        ⭐ 优化版 Decord 提取 (批量边缘特征)
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            fps = 24
            
            aspect_ratio = self._detect_aspect_ratio(video_path)
            duration = total_frames / fps
            
            print(f"  📹 {os.path.basename(video_path)}")
            print(f"     比例: {aspect_ratio} | FPS: {fps} | 长度: {duration:.1f}秒")
            
            sample_indices, actual_frames, process_duration = self._calculate_sample_indices(total_frames, fps)
            target_fps = getattr(self.config, 'target_fps', 12)
            print(f"     采样: {actual_frames} 帧 ({process_duration:.1f}秒 @ {target_fps}FPS)")
            
            # 读取帧
            frames_rgb = []
            for idx in sample_indices:
                if idx < total_frames:
                    frame = vr[idx].asnumpy()
                    if len(frame.shape) == 3:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    frames_rgb.append(frame_rgb)
                else:
                    if frames_rgb:
                        frames_rgb.append(frames_rgb[-1])
                    else:
                        h, w = self.config.target_height, self.config.target_width
                        frames_rgb.append(np.zeros((h, w, 3), dtype=np.uint8))
            
            # ⭐ 批量提取 EfficientNet 特征
            frames_pil = [Image.fromarray(frame) for frame in frames_rgb]
            frame_tensors = [self.transform(pil) for pil in frames_pil]
            
            efficientnet_features = []
            batch_size = 32  # 增大batch
            
            with torch.no_grad():
                for i in range(0, len(frame_tensors), batch_size):
                    batch_frames = frame_tensors[i:i+batch_size]
                    batch_tensor = torch.stack(batch_frames).to(self.config.device)
                    
                    batch_features = self.feature_extractor(batch_tensor)
                    batch_features = self.global_pool(batch_features)
                    batch_features = batch_features.flatten(1).cpu().numpy()
                    
                    efficientnet_features.extend(batch_features)
            
            efficientnet_array = np.array(efficientnet_features)
            
            # ⭐ 批量提取边缘特征 (优化重点)
            edge_array = self._extract_edge_features_batch(frames_rgb)
            
            # 拼接特征
            combined_features = np.concatenate([efficientnet_array, edge_array], axis=1)
            
            print(f"     ✅ 提取完成: {len(combined_features)} 个特征向量 (1344维)")
            
            # ⭐ 釋放GPU內存
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return combined_features
            
        except Exception as e:
            print(f"❌ Decord 提取失败: {e}")
            print(f"   尝试使用 OpenCV 备用方法...")
            return self.extract_video_features_opencv(video_path)


    def extract_video_features_opencv(self, video_path):
        """
        ⭐ 优化版 OpenCV 提取 (先读所有帧, 再批量处理)
        """
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
            
            aspect_ratio = self._detect_aspect_ratio(video_path)
            duration = total_frames / video_fps
            
            print(f"  📹 {os.path.basename(video_path)}")
            print(f"     比例: {aspect_ratio} | FPS: {video_fps:.1f} | 长度: {duration:.1f}秒")
            
            sample_indices, actual_frames, process_duration = self._calculate_sample_indices(total_frames, video_fps)
            target_fps = getattr(self.config, 'target_fps', 12)
            print(f"     采样: {actual_frames} 帧 ({process_duration:.1f}秒 @ {target_fps}FPS)")
            
            # ⭐ 先读取所有帧 (避免重复seek)
            frames_rgb = []
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total_frames-1))
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_rgb.append(frame_rgb)
                else:
                    if frames_rgb:
                        frames_rgb.append(frames_rgb[-1])
                    else:
                        h, w = self.config.target_height, self.config.target_width
                        frames_rgb.append(np.zeros((h, w, 3), dtype=np.uint8))
            
            cap.release()
            
            if not frames_rgb:
                print(f"❌ 未能提取任何帧")
                return None
            
            # ⭐ 批量提取 EfficientNet 特征
            frames_pil = [Image.fromarray(frame) for frame in frames_rgb]
            frame_tensors = [self.transform(pil) for pil in frames_pil]
            
            efficientnet_features = []
            batch_size = 32  # 增大batch
            
            with torch.no_grad():
                for i in range(0, len(frame_tensors), batch_size):
                    batch_frames = frame_tensors[i:i+batch_size]
                    batch_tensor = torch.stack(batch_frames).to(self.config.device)
                    
                    batch_features = self.feature_extractor(batch_tensor)
                    batch_features = self.global_pool(batch_features)
                    batch_features = batch_features.flatten(1).cpu().numpy()
                    
                    efficientnet_features.extend(batch_features)
            
            efficientnet_array = np.array(efficientnet_features)
            
            # ⭐ 批量提取边缘特征
            edge_array = self._extract_edge_features_batch(frames_rgb)
            
            # 拼接特征
            combined_features = np.concatenate([efficientnet_array, edge_array], axis=1)
            
            print(f"     ✅ 提取完成: {len(combined_features)} 个特征向量 (1344维)")
            
            # ⭐ 釋放GPU內存
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return combined_features
            
        except Exception as e:
            print(f"❌ OpenCV 提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_video_features(self, video_path):
        """
        ⭐ 提取影片特徵（自動選擇最快方法，輸出1344維）
        """
        if self.use_decord:
            return self.extract_video_features_decord(video_path)
        else:
            return self.extract_video_features_opencv(video_path)
    
    def extract_batch_features(self, video_samples):
        """批量提取特徵"""
        print("🚀 開始批量特徵提取...")
        print(f"📐 目標尺寸: {self.config.target_height}x{self.config.target_width}")
        
        target_fps = getattr(self.config, 'target_fps', 12)
        min_duration = getattr(self.config, 'min_video_duration', 2)
        max_duration = getattr(self.config, 'max_video_duration', 15)
        content_retain = getattr(self.config, 'content_retain', 0.92)
        
        print(f"🎬 採樣率: {target_fps} FPS")
        print(f"⏱️  處理範圍: {min_duration}-{max_duration}秒")
        print(f"✂️  處理策略: 比例接近直接縮放，比例差異大時智能裁切（保留{content_retain*100:.0f}%中心畫面）")
        print(f"📁 目標特徵目錄: {self.config.feature_dir}")
        print(f"⚡ 使用方法: {'Decord (快速)' if self.use_decord else 'OpenCV (較慢)'}")
        print(f"⭐ 輸出維度: 1344 (EfficientNet 1280 + 邊緣 64)")
        print(f"🚀 邊緣檢測: PyTorch GPU Canny (加速)")
        
        new_feature_samples = []
        success_count = 0
        
        for video_path, animator in tqdm(video_samples, desc="提取特徵"):
            # 生成特徵文件名
            video_name = os.path.basename(video_path)
            feature_file = f"{os.path.splitext(video_name)[0]}.npy"
            feature_path = os.path.join(self.config.feature_dir, feature_file)
            
            # 如果特徵已存在，跳過
            if os.path.exists(feature_path):
                new_feature_samples.append((feature_path, animator))
                success_count += 1
                continue
            
            # 提取新特徵
            features = self.extract_video_features(video_path)
            if features is not None:
                np.save(feature_path, features)
                new_feature_samples.append((feature_path, animator))
                success_count += 1
        
        print(f"✅ 特徵提取完成: {success_count}/{len(video_samples)} 成功")
        return new_feature_samples