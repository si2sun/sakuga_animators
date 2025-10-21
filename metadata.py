import json
import os
import cv2
import numpy as np
from collections import Counter, defaultdict
import shutil  # 新增：用於刪除檔案
import re  # 新增：用於 extract_base_video_name

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

def detect_aspect_ratio(video_path):
    """檢測影片比例"""
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

def scan_and_add_missing_videos(video_dir, feature_dir, metadata_path, min_duration=2, max_duration=60):
    """
    掃描 test-videos 資料夾，找遺漏的視頻（.npy 存在但 metadata 沒記錄），補回 metadata.json
    
    Args:
        video_dir: test-videos 根目錄 (e.g., 'C:\\Users\\litsu\\Desktop\\sakuga\\test-videos')
        feature_dir: 特徵目錄 (e.g., './features')
        metadata_path: metadata.json 路徑
        min_duration, max_duration: 長度過濾
    
    Returns:
        bool: 是否成功補加
    """
    if not os.path.exists(metadata_path):
        print(f"❌ metadata.json 不存在: {metadata_path}")
        return False
    
    # 載入現有 metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 現有 feature_files 集合（用於比對遺漏）
    existing_features = set()
    for animator, info in metadata.get('animators', {}).items():
        existing_features.update(info.get('feature_files', []))
    
    print(f"📊 現有 metadata 記錄 .npy: {len(existing_features)} 個")
    
    # 掃描 video_dir，按資料夾分類
    animator_videos = defaultdict(list)  # {animator: [video_paths]}
    for root, dirs, files in os.walk(video_dir):
        animator = os.path.basename(root)
        if not animator:  # 根目錄跳過
            continue
        video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for video_file in video_files:
            video_path = os.path.join(root, video_file)
            animator_videos[animator].append(video_path)
    
    print(f"🔍 掃描發現 {sum(len(paths) for paths in animator_videos.values())} 個視頻 (分類: {list(animator_videos.keys())})")
    
    # 比對並補加遺漏
    added_count = 0
    valid_feature_names = set(existing_features)  # 最終有效名
    aspect_stats = Counter()
    
    for animator, video_paths in animator_videos.items():
        info = metadata.get('animators', {}).get(animator)
        if info is None:
            info = {
                'video_count': 0,
                'video_paths': [],
                'feature_files': [],
                'aspect_ratios': {},
                'video_groups': {},
                'added_date': str(np.datetime64('now'))
            }

        new_video_paths = []
        new_feature_files = []
        new_aspect_ratios = {}
        new_video_groups = {}
        current_feature_files = set(info.get('feature_files', []))
        
        new_video_paths = []
        new_feature_files = []
        new_aspect_ratios = {}
        new_video_groups = {}
        
        for video_path in video_paths:
            video_name = os.path.basename(video_path)
            feature_file = os.path.splitext(video_name)[0] + '.npy'
            feature_path = os.path.join(feature_dir, feature_file)
            
            # 檢查: .npy 存在 且 不在現有 metadata
            if os.path.exists(feature_path) and feature_file not in existing_features:
                # 檢查長度有效
                if is_valid_video(video_path, min_duration, max_duration):
                    # 補加
                    new_video_paths.append(video_path)
                    new_feature_files.append(feature_file)
                    valid_feature_names.add(feature_file)
                    
                    # 算 aspect
                    aspect = detect_aspect_ratio(video_path)
                    new_aspect_ratios[video_name] = aspect
                    aspect_stats[aspect] += 1
                    
                    # 算 group
                    base_name = extract_base_video_name(video_name)
                    new_video_groups[video_name] = base_name
                    
                    added_count += 1
                    print(f"✅ 補加遺漏: {feature_file} (animator: {animator}, 長度 OK)")
                else:
                    print(f"⚠️ 遺漏但長度無效: {video_name} (animator: {animator})")
            else:
                if not os.path.exists(feature_path):
                    print(f"⚠️ .npy 不存在: {feature_file} (animator: {animator})")
                # 已在 metadata，跳過
        
        # 合併到現有
        if new_video_paths:
            if animator not in metadata['animators']:
                print(f"🆕 新動畫師: {animator} (新增 {len(new_video_paths)} 個影片)")
                metadata['animators'][animator] = info
            info['video_paths'].extend(new_video_paths)
            info['feature_files'].extend(new_feature_files)
            info['aspect_ratios'].update(new_aspect_ratios)
            info['video_groups'].update(new_video_groups)
            info['video_count'] += len(new_video_paths)
            print(f"  {animator}: 補加 {len(new_video_paths)} 個新條目")
    
    # 更新總計
    metadata['video_count'] = sum(info['video_count'] for info in metadata['animators'].values())
    metadata['feature_count'] = len(valid_feature_names)
    metadata['last_valid_update'] = str(np.datetime64('now'))
    
    # 保存
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 掃描補加完成: 新增 {added_count} 個遺漏條目")
    print(f"📊 新 video_count: {metadata['video_count']}")
    print(f"📊 比例統計 (新增):")
    for ratio, count in aspect_stats.items():
        print(f"  {ratio}: {count} 個影片")
    
    return True

def fix_metadata(metadata_path, feature_dir, min_duration=2, max_duration=60, clean_extra_npy=True, skip_duration_check=False, restore_from_backup=False):
    """
    修正 metadata.json，只保留有效影片（長度 2-15s + 存在 .npy）
    
    Args:
        metadata_path: metadata.json 路徑
        feature_dir: 特徵目錄
        min_duration: 最小長度
        max_duration: 最大長度
        clean_extra_npy: 是否刪除多餘 .npy（預設 True）
        skip_duration_check: 是否跳過長度檢查（預設 False；設 True 保留短視頻）
        restore_from_backup: 是否從 backup 恢復多餘 .npy（預設 False）
    
    Returns:
        bool: 是否成功修正
    """
    if not os.path.exists(metadata_path):
        print(f"❌ metadata.json 不存在: {metadata_path}")
        return False
    
    # 載入 metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"📊 原始 video_count: {metadata.get('video_count', 0)}")
    if skip_duration_check:
        print("⚠️  跳過長度檢查（保留所有短視頻）")
    
    updated_animators = {}
    total_valid_videos = 0
    removed_count = 0
    aspect_stats = Counter()
    
    # 收集所有有效 feature_files（用於比對多餘 .npy）
    valid_feature_names = set()
    
    for animator, info in metadata.get('animators', {}).items():
        valid_video_paths = []
        valid_feature_files = []
        valid_aspect_ratios = {}
        valid_video_groups = {}
        
        # 假設 video_paths 和 feature_files 對應
        video_paths = info.get('video_paths', [])
        feature_files = info.get('feature_files', [])
        
        for i, (video_path, feature_file) in enumerate(zip(video_paths, feature_files)):
            video_name = os.path.basename(video_path)
            
            # 檢查 1: 影片有效（長度） - 可跳過
            if not skip_duration_check:
                if not is_valid_video(video_path, min_duration, max_duration):
                    removed_count += 1
                    print(f"跳過無效長度: {video_name} (animator: {animator})")
                    continue
            
            # 檢查 2: 特徵檔案存在
            feature_path = os.path.join(feature_dir, feature_file)
            if not os.path.exists(feature_path):
                removed_count += 1
                print(f"跳過無 .npy: {feature_file} (animator: {animator})")
                continue
            
            # 保留
            valid_video_paths.append(video_path)
            valid_feature_files.append(feature_file)
            valid_feature_names.add(feature_file)  # 收集用於比對
            
            # 重新算 aspect（如果舊的 unknown，重算）
            old_aspect = info.get('aspect_ratios', {}).get(video_name, "unknown")
            if old_aspect == "unknown":
                aspect = detect_aspect_ratio(video_path)
                print(f"重算 aspect: {video_name} -> {aspect}")
            else:
                aspect = old_aspect
            valid_aspect_ratios[video_name] = aspect
            aspect_stats[aspect] += 1
            
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
    metadata['animators'] = updated_animators
    metadata['video_count'] = total_valid_videos
    metadata['feature_count'] = total_valid_videos  # 假設每個有效都有 .npy
    metadata['last_valid_update'] = str(np.datetime64('now'))  # 標記已更新
    
    # 保存
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 修正完成: 保留 {total_valid_videos} 個有效影片，移除 {removed_count} 個無效")
    print(f"📊 新 video_count: {total_valid_videos}")
    print(f"📊 比例統計:")
    for ratio, count in aspect_stats.items():
        print(f"  {ratio}: {count} 個影片")
    
    # 檢查實際 .npy 數
    all_npy_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    actual_npy_count = len(all_npy_files)
    print(f"📊 實際 .npy 檔案: {actual_npy_count}")
    
    # ### 處理多餘 .npy
    extra_npy_count = actual_npy_count - total_valid_videos
    if extra_npy_count > 0:
        print(f"⚠️  發現 {extra_npy_count} 個多餘 .npy（metadata 沒記錄的）")
        
        extra_npy_files = [f for f in all_npy_files if f not in valid_feature_names]
        print(f"多餘檔案示例: {extra_npy_files[:5]}...")  # 只印前5
        
        if clean_extra_npy:
            backup_dir = os.path.join(feature_dir, 'backup_extra_npy')
            os.makedirs(backup_dir, exist_ok=True)
            
            deleted_count = 0
            for extra_file in extra_npy_files:
                src = os.path.join(feature_dir, extra_file)
                dst = os.path.join(backup_dir, extra_file)
                shutil.move(src, dst)  # 移到 backup，避免誤刪
                deleted_count += 1
                if deleted_count <= 10:  # 限印，避免刷屏
                    print(f"移到備份: {extra_file}")
            
            print(f"✅ 已移 {deleted_count} 個多餘 .npy 到 {backup_dir}")
            print(f"📊 目錄現在 .npy: {total_valid_videos}")
        else:
            print("💡 設 clean_extra_npy=True 來自動清理（移到備份）")
    else:
        print("✅ .npy 數與 metadata 一致，無多餘")
    
    # ### 新增：從 backup 恢復（如果設 True）
    if restore_from_backup:
        backup_dir = os.path.join(feature_dir, 'backup_extra_npy')
        if os.path.exists(backup_dir):
            restored_count = 0
            for file in os.listdir(backup_dir):
                if file.endswith('.npy'):
                    src = os.path.join(backup_dir, file)
                    dst = os.path.join(feature_dir, file)
                    shutil.move(src, dst)
                    restored_count += 1
                    print(f"恢復從備份: {file}")
            print(f"✅ 已恢復 {restored_count} 個 .npy 從 backup")
        else:
            print("ℹ️  無 backup 目錄，跳過恢復")
    
    return True


# 使用範例
if __name__ == '__main__':
    metadata_path = './features/metadata.json'  # 修改為你的路徑
    feature_dir = './features' # 你的 config.feature_dir
    video_dir = r'C:\Users\litsu\Desktop\sakuga\test-videos'  # 新增：視頻根目錄
    
    # 先掃描補加遺漏
    scan_and_add_missing_videos(video_dir, feature_dir, metadata_path, min_duration=1)  # 降低 min=1，保留短片
    
    # 再修正過濾無效（跳過長度檢查，保留補加的）
    fix_metadata(metadata_path, feature_dir, min_duration=2, skip_duration_check=True, clean_extra_npy=True, restore_from_backup=False)