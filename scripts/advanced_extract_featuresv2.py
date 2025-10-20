import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs.base_config import BaseConfig
from core.data_manager import DataManager
from core.advanced_feature_extractorv2 import FeatureExtractor

def main():
    """特徵提取主流程"""
    print("🎬 開始特徵提取流程...")
    print("🔥 載入 EfficientNetV2-S 進行特徵提取...")
    # 配置
    config = BaseConfig()
    data_manager = DataManager(config)
    
    # 掃描數據
    data_root = r"C:\Users\litsu\Desktop\sakuga\test-videos"  # 修改為你的路徑
    new_animators, all_samples = data_manager.scan_data_directory(data_root)
    
    if not all_samples:
        print("❌ 沒有找到影片文件")
        return
    
    # 提取特徵
    feature_extractor = FeatureExtractor(config)
    
    feature_samples = feature_extractor.extract_batch_features(all_samples)
    
    # 註冊所有原畫師和影片
    animator_videos = {}
    for video_path, animator in all_samples:
        if animator not in animator_videos:
            animator_videos[animator] = []
        animator_videos[animator].append(video_path)
    
    for animator, videos in animator_videos.items():
        data_manager.register_animator(animator, videos)
    
    # 顯示比例統計
    print(f"\n📊 比例統計:")
    aspect_ratio_stats = data_manager.get_aspect_ratio_stats()
    for ratio, count in aspect_ratio_stats.items():
        print(f"  {ratio}: {count} 個影片")
    
    print(f"\n✅ 特徵提取完成!")
    print(f"📊 總計: {len(data_manager.metadata['animators'])} 個原畫師")
    print(f"📊 總計: {data_manager.metadata['video_count']} 個影片")
    print(f"📊 總計: {data_manager.metadata['feature_count']} 個特徵文件")

if __name__ == '__main__':
    main()