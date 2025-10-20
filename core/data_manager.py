import os
import json
import numpy as np
from collections import defaultdict
import cv2
import re

def detect_aspect_ratio(video_path):
    """檢測影片比例(獨立函數)"""
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

def extract_base_video_name(filename):
    """
    提取影片的基礎名稱,去除片段編號
    例如:
    - yutaka_nakamura_123_001.mp4 -> yutaka_nakamura_123
    - video_part1.mp4 -> video
    - animation_01.mp4 -> animation
    """
    # 移除副檔名
    name = os.path.splitext(filename)[0]
    
    # 嘗試匹配常見的片段模式
    patterns = [
        r'(.+)_\d{3}$',           # name_001, name_002
        r'(.+)_part\d+$',         # name_part1, name_part2
        r'(.+)_segment\d+$',      # name_segment1
        r'(.+)-\d+$',             # name-1, name-2
        r'(.+)_\d+$',             # name_1, name_2
    ]
    
    for pattern in patterns:
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # 如果沒有匹配到,返回原名稱
    return name

# ### 新增：檢查影片有效性（長度 2-15s）
def is_valid_video(video_path, min_duration=2, max_duration=15):
    """檢查影片是否有效（長度在 min-max 之間）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps <= 0 or frames <= 0:
        return False
    
    duration = frames / fps
    return min_duration <= duration <= max_duration

class DataManager:
    def __init__(self, config):
        self.config = config
        self.metadata_file = os.path.join(config.feature_dir, 'metadata.json')
        self.load_metadata()
        # ### 新增：初始化後自動檢查/更新（可選，設 force=False 避免每次跑）
        self.update_metadata_valid_only(force=False)  # force=True 強制重掃
    
    def load_metadata(self):
        """載入元數據"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'animators': {},
                'video_count': 0,
                'feature_count': 0,
                'last_updated': None,
                'video_groups': {},  # 新增: 記錄影片分組信息
                'fixed_split': {}  # 新增: 固定的訓練/測試分割
            }
        
        # 標籤映射
        self.label_to_idx = {}
        self.idx_to_label = {}
        self._update_label_mapping()
    
    # ### 新增：修正 metadata，只保留有效影片
    def update_metadata_valid_only(self, force=False):
        """更新 metadata，只保留有效影片（長度 2-15s + 存在特徵）"""
        if not force and self.metadata.get('last_valid_update'):  # 已更新過，跳過
            print("📊 metadata 已為有效影片更新過，跳過")
            return
        
        print("🔄 更新 metadata: 只保留有效影片...")
        min_duration = getattr(self.config, 'min_video_duration', 2)
        max_duration = getattr(self.config, 'max_video_duration', 15)
        
        updated_animators = {}
        total_valid_videos = 0
        removed_count = 0
        
        for animator, info in self.metadata['animators'].items():
            valid_video_paths = []
            valid_feature_files = []
            valid_aspect_ratios = {}
            valid_video_groups = {}
            
            for i, (video_path, feature_file) in enumerate(zip(info['video_paths'], info['feature_files'])):
                video_name = os.path.basename(video_path)
                
                # 檢查 1: 影片有效（長度）
                if not is_valid_video(video_path, min_duration, max_duration):
                    removed_count += 1
                    continue
                
                # 檢查 2: 特徵檔案存在
                feature_path = os.path.join(self.config.feature_dir, feature_file)
                if not os.path.exists(feature_path):
                    removed_count += 1
                    continue
                
                # 保留
                valid_video_paths.append(video_path)
                valid_feature_files.append(feature_file)
                
                # 重新算 aspect（如果舊的 unknown，重算）
                aspect = info['aspect_ratios'].get(video_name, "unknown")
                if aspect == "unknown":
                    aspect = detect_aspect_ratio(video_path)
                valid_aspect_ratios[video_name] = aspect
                
                # 保留 group
                base_name = extract_base_video_name(video_name)
                valid_video_groups[video_name] = base_name
            
            # 更新 info
            updated_info = {
                'video_count': len(valid_video_paths),
                'video_paths': valid_video_paths,
                'feature_files': valid_feature_files,
                'aspect_ratios': valid_aspect_ratios,
                'video_groups': valid_video_groups,
                'added_date': info.get('added_date')  # 保留舊資訊
            }
            updated_animators[animator] = updated_info
            total_valid_videos += len(valid_video_paths)
        
        # 更新總計
        self.metadata['animators'] = updated_animators
        self.metadata['video_count'] = total_valid_videos
        self.metadata['feature_count'] = total_valid_videos  # 假設每個有效都有 .npy
        self.metadata['last_valid_update'] = str(np.datetime64('now'))  # 標記已更新
        
        # 重新更新 label_mapping（萬一有空 animator）
        self._update_label_mapping()
        
        self.save_metadata()
        
        print(f"✅ 更新完成: 保留 {total_valid_videos} 個有效影片，移除 {removed_count} 個無效")
        print(f"📊 新 video_count: {total_valid_videos} (與特徵檔案一致)")
    
    def _update_label_mapping(self):
        """從元數據更新標籤映射"""
        all_animators = sorted(self.metadata['animators'].keys())
        self.label_to_idx = {animator: idx for idx, animator in enumerate(all_animators)}
        self.idx_to_label = {idx: animator for animator, idx in self.label_to_idx.items()}
    
    def save_metadata(self):
        """保存元數據"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def scan_data_directory(self, data_root):
        """掃描數據目錄,發現新的原畫師和影片"""
        print("🔍 掃描數據目錄...")
        
        existing_animators = set(self.metadata['animators'].keys())
        new_animators = set()
        all_samples = []
        
        for root, dirs, files in os.walk(data_root):
            animator = os.path.basename(root)
            video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]
            
            if animator and video_files:
                if animator not in existing_animators:
                    new_animators.add(animator)
                    print(f"🎯 發現新原畫師: {animator}")
                
                # 收集所有樣本
                for file in video_files:
                    video_path = os.path.join(root, file)
                    all_samples.append((video_path, animator))
        
        print(f"📊 發現 {len(new_animators)} 個新原畫師, {len(all_samples)} 個影片")
        return list(new_animators), all_samples
    
    def register_animator(self, animator, video_paths):
        """註冊原畫師和影片"""
        if animator not in self.metadata['animators']:
            self.metadata['animators'][animator] = {
                'video_count': 0,
                'video_paths': [],
                'feature_files': [],
                'aspect_ratios': {},
                'video_groups': {},  # 新增: 記錄每個影片屬於哪個組
                'added_date': None
            }
        
        # ### 修改：註冊前過濾有效影片
        valid_paths = [p for p in video_paths if is_valid_video(p)]  # 只註冊有效
        print(f"註冊 {animator}: {len(valid_paths)}/{len(video_paths)} 有效影片")
        
        for video_path in valid_paths:
            if video_path not in self.metadata['animators'][animator]['video_paths']:
                video_name = os.path.basename(video_path)
                feature_file = f"{os.path.splitext(video_name)[0]}.npy"
                
                # 提取基礎影片名稱
                base_name = extract_base_video_name(video_name)
                
                # ⭐ 直接使用獨立函數檢測比例,不載入模型
                aspect_ratio = detect_aspect_ratio(video_path)
                
                self.metadata['animators'][animator]['video_paths'].append(video_path)
                self.metadata['animators'][animator]['feature_files'].append(feature_file)
                self.metadata['animators'][animator]['aspect_ratios'][video_name] = aspect_ratio
                self.metadata['animators'][animator]['video_groups'][video_name] = base_name
                self.metadata['animators'][animator]['video_count'] += 1
                self.metadata['video_count'] += 1
        
        self.metadata['last_updated'] = str(np.datetime64('now'))
        self._update_label_mapping()
        self.save_metadata()
    
    def get_all_feature_samples(self):
        """獲取所有特徵樣本"""
        feature_samples = []
        for animator, info in self.metadata['animators'].items():
            for feature_file in info['feature_files']:
                feature_path = os.path.join(self.config.feature_dir, feature_file)
                if os.path.exists(feature_path):
                    feature_samples.append((feature_path, animator))
        
        self.metadata['feature_count'] = len(feature_samples)
        self.save_metadata()
        
        return feature_samples
    def get_fixed_split_video_samples(self):
        """
        根據固定分割獲取訓練/測試的 **影片樣本**
        
        Returns:
            (train_samples, test_samples): 兩個列表,每個元素為 (video_path, animator)
        """
        if not self.metadata.get('fixed_split'):
            raise ValueError("❌ 尚未創建固定分割,請先運行 create_fixed_train_test_split()")

        fixed_split = self.metadata['fixed_split']
        train_groups_dict = fixed_split['train_groups']
        test_groups_dict = fixed_split['test_groups']
        
        train_samples = []
        test_samples = []

        for animator, info in self.metadata['animators'].items():
            video_groups = info.get('video_groups', {})
            
            # 獲取該動畫師的訓練/測試組
            train_groups = set(train_groups_dict.get(animator, []))
            test_groups = set(test_groups_dict.get(animator, []))
            
            for video_path in info['video_paths']:
                video_name = os.path.basename(video_path)
                group_id = video_groups.get(video_name)
                
                # 組合 animator + group_id 作為全局唯一組標識
                global_group_id = f"{animator}_{group_id}"

                if global_group_id in train_groups:
                    train_samples.append((video_path, animator))
                elif global_group_id in test_groups:
                    test_samples.append((video_path, animator))
                # 注意：這裡我們忽略了未被分組的新影片，因為全量訓練時應該先分割好所有數據
        
        return train_samples, test_samples
    def get_feature_samples_with_groups(self):
        """
        獲取所有特徵樣本,並附帶分組信息
        返回: [(feature_path, animator, group_id), ...]
        """
        feature_samples = []
        for animator, info in self.metadata['animators'].items():
            video_groups = info.get('video_groups', {})
            
            for i, feature_file in enumerate(info['feature_files']):
                feature_path = os.path.join(self.config.feature_dir, feature_file)
                if os.path.exists(feature_path):
                    # 從特徵文件名反推原始影片名
                    video_name = feature_file.replace('.npy', '')
                    
                    # 找到對應的影片名(帶副檔名)
                    matching_video = None
                    for video_path in info['video_paths']:
                        if os.path.splitext(os.path.basename(video_path))[0] == video_name:
                            matching_video = os.path.basename(video_path)
                            break
                    
                    # 獲取分組ID
                    if matching_video and matching_video in video_groups:
                        group_id = video_groups[matching_video]
                    else:
                        # 如果沒有分組信息,使用特徵文件名作為唯一ID
                        group_id = video_name
                    
                    # 組合 animator + group_id 作為全局唯一組標識
                    global_group_id = f"{animator}_{group_id}"
                    
                    feature_samples.append((feature_path, animator, global_group_id))
        
        return feature_samples
    
    def get_num_classes(self):
        """獲取當前類別數量"""
        return len(self.label_to_idx)
    
    def get_aspect_ratio_stats(self):
        """獲取比例統計"""
        aspect_ratio_counts = {}
        for animator, info in self.metadata['animators'].items():
            for video_name, ratio in info['aspect_ratios'].items():
                if ratio not in aspect_ratio_counts:
                    aspect_ratio_counts[ratio] = 0
                aspect_ratio_counts[ratio] += 1
        return aspect_ratio_counts
    
    def create_fixed_train_test_split(self, train_ratio=0.8, force_resplit=False):
        """
        創建固定的訓練/測試集分割
        
        Args:
            train_ratio: 訓練集比例
            force_resplit: 是否強制重新分割(警告:會改變已有的分割)
        """
        # 檢查是否已有固定分割
        if self.metadata.get('fixed_split') and not force_resplit:
            print("✅ 使用已存在的固定訓練/測試分割")
            return
        
        if force_resplit:
            print("⚠️  警告: 強制重新分割會改變原有的訓練/測試集!")
        
        print("🔒 創建固定的訓練/測試集分割...")
        
        # 獲取帶分組信息的樣本
        all_samples_with_groups = self.get_feature_samples_with_groups()
        
        if not all_samples_with_groups:
            print("❌ 沒有特徵樣本,無法創建分割")
            return
        
        # 按原畫師和影片組分組
        animator_video_groups = {}
        for feature_path, animator, group_id in all_samples_with_groups:
            if animator not in animator_video_groups:
                animator_video_groups[animator] = {}
            
            if group_id not in animator_video_groups[animator]:
                animator_video_groups[animator][group_id] = []
            
            animator_video_groups[animator][group_id].append(feature_path)
        
        # 創建固定分割
        fixed_split = {
            'train_groups': {},  # {animator: [group_ids]}
            'test_groups': {},   # {animator: [group_ids]}
            'created_at': str(np.datetime64('now')),
            'train_ratio': train_ratio
        }
        
        print("\n📊 分割結果:")
        for animator, video_groups in animator_video_groups.items():
            group_ids = list(video_groups.keys())
            
            # 使用固定的隨機種子確保可重現
            rng = np.random.RandomState(42)
            rng.shuffle(group_ids)
            
            # 按組分割
            num_groups = len(group_ids)
            train_group_count = max(1, int(num_groups * train_ratio))
            
            train_groups = group_ids[:train_group_count]
            test_groups = group_ids[train_group_count:]
            
            fixed_split['train_groups'][animator] = train_groups
            fixed_split['test_groups'][animator] = test_groups
            
            # 統計片段數
            train_count = sum(len(video_groups[g]) for g in train_groups)
            test_count = sum(len(video_groups[g]) for g in test_groups)
            
            print(f"  {animator}:")
            print(f"    影片組: {num_groups} 個 (訓練 {len(train_groups)}, 測試 {len(test_groups)})")
            print(f"    片段數: 訓練 {train_count}, 測試 {test_count}")
        
        # 保存固定分割
        self.metadata['fixed_split'] = fixed_split
        self.save_metadata()
        
        print("\n🔒 固定分割已保存到 metadata.json")
    
    def get_fixed_split_samples(self):
        """
        根據固定分割獲取訓練/測試樣本
        
        Returns:
            (train_samples, test_samples): 兩個列表,每個元素為 (feature_path, animator)
        """
        if not self.metadata.get('fixed_split'):
            raise ValueError("❌ 尚未創建固定分割,請先運行 create_fixed_train_test_split()")
        
        fixed_split = self.metadata['fixed_split']
        train_groups_dict = fixed_split['train_groups']
        test_groups_dict = fixed_split['test_groups']
        train_ratio = fixed_split.get('train_ratio', 0.8)
        
        # 獲取當前所有樣本
        all_samples_with_groups = self.get_feature_samples_with_groups()
        
        # 按原畫師和組分類
        animator_group_samples = {}
        for feature_path, animator, group_id in all_samples_with_groups:
            if animator not in animator_group_samples:
                animator_group_samples[animator] = {}
            
            if group_id not in animator_group_samples[animator]:
                animator_group_samples[animator][group_id] = []
            
            animator_group_samples[animator][group_id].append((feature_path, animator))
        
        # 根據固定分割收集樣本
        train_samples = []
        test_samples = []
        new_groups_by_animator = {}  # 記錄每個原畫師的新增組
        
        for animator, group_samples in animator_group_samples.items():
            # 獲取該原畫師的固定分割
            train_groups = train_groups_dict.get(animator, [])
            test_groups = test_groups_dict.get(animator, [])
            
            # 收集新增的組
            new_groups = []
            
            for group_id, samples in group_samples.items():
                if group_id in train_groups:
                    train_samples.extend(samples)
                elif group_id in test_groups:
                    test_samples.extend(samples)
                else:
                    # 新增的組
                    new_groups.append((group_id, samples))
            
            if new_groups:
                new_groups_by_animator[animator] = new_groups
        
        # 對新增組進行分割
        if new_groups_by_animator:
            print(f"\n🆕 發現新影片組,按比例 {train_ratio:.0%}/{(1-train_ratio):.0%} 分割:")
            
            for animator, new_groups in new_groups_by_animator.items():
                # 使用固定隨機種子 + 原畫師名稱確保可重現
                seed = hash(animator) % (2**32)
                rng = np.random.RandomState(seed)
                
                # 打亂新組
                rng.shuffle(new_groups)
                
                # 按比例分割
                num_new_groups = len(new_groups)
                train_count = max(1, int(num_new_groups * train_ratio)) if num_new_groups > 1 else 1
                
                new_train_groups = new_groups[:train_count]
                new_test_groups = new_groups[train_count:]
                
                # 更新固定分割
                if animator not in fixed_split['train_groups']:
                    fixed_split['train_groups'][animator] = []
                if animator not in fixed_split['test_groups']:
                    fixed_split['test_groups'][animator] = []
                
                # 添加到訓練集
                train_sample_count = 0
                for group_id, samples in new_train_groups:
                    train_samples.extend(samples)
                    fixed_split['train_groups'][animator].append(group_id)
                    train_sample_count += len(samples)
                
                # 添加到測試集
                test_sample_count = 0
                for group_id, samples in new_test_groups:
                    test_samples.extend(samples)
                    fixed_split['test_groups'][animator].append(group_id)
                    test_sample_count += len(samples)
                
                print(f"  {animator}:")
                print(f"    新增影片組: {num_new_groups} 個 (訓練 {len(new_train_groups)}, 測試 {len(new_test_groups)})")
                print(f"    新增片段數: 訓練 {train_sample_count}, 測試 {test_sample_count}")
            
            # 保存更新後的分割
            self.save_metadata()
            print("\n🔒 新數據的分割已保存到 metadata.json")
        
        return train_samples, test_samples