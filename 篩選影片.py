import os
import cv2
import numpy as np
from multiprocessing import Pool
from argparse import ArgumentParser
from tqdm import tqdm
import shutil
from pathlib import Path

def filter_single_video(args):
    """篩選單片：Optical Flow + 差異 (加慢動作區分)"""
    video_path = args
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 50:  # 太短直接過
        cap.release()
        return video_path
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    sample_interval = max(1, total_frames // 20)  # 樣本 20 幀，加速
    
    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        return None
    gray_prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    flow_mags = []
    diffs = []
    frame_idx = 0
    while ret:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret or frame_idx % sample_interval != 0:
            continue
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_mags.append(np.mean(mag))
        
        # 畫面差異
        diffs.append(np.mean(np.abs(gray_curr - gray_prev)))
        
        gray_prev = gray_curr
    
    cap.release()
    
    if not flow_mags:
        return None
    
    avg_flow = np.mean(flow_mags)
    flow_std = np.std(flow_mags)  # 新增：flow 變異 (低=慢動作)
    avg_diff = np.mean(diffs)
    
    # 速度類型判斷
    if avg_flow < 0.4 and flow_std < 0.5:
        speed_type = "慢動作"
    elif avg_flow > 0.75 or flow_std > 1:
        speed_type = "快動作"
    else:
        speed_type = "中速動作"
    
    # 靜態過濾 (原邏輯)
    if avg_flow < 0.7 and avg_diff < 30:
        print(f"❌ 靜態跳過: {os.path.basename(video_path)} (flow: {avg_flow:.3f}, std: {flow_std:.3f}, diff: {avg_diff:.1f}, type: {speed_type})")
        return None
    if avg_flow < 0.5 and avg_diff < 40:
        print(f"❌ 靜態跳過: {os.path.basename(video_path)} (flow: {avg_flow:.3f}, std: {flow_std:.3f}, diff: {avg_diff:.1f}, type: {speed_type})")
        return None
    
    # 新增：慢動作過濾 (可調或註解掉保留)
    if speed_type == "慢動作":
        print(f"❌ 慢動作跳過: {os.path.basename(video_path)} (flow: {avg_flow:.3f}, std: {flow_std:.3f}, diff: {avg_diff:.1f}, type: {speed_type})")
        return None  # 改成 return video_path 來保留
    
    elif avg_flow < 0.1:
        print(f"❌ 靜態跳過: {os.path.basename(video_path)} (flow: {avg_flow:.3f}, std: {flow_std:.3f}, diff: {avg_diff:.1f}, type: {speed_type})")
        return None
    elif avg_diff < 10:
        print(f"❌ 靜態跳過: {os.path.basename(video_path)} (flow: {avg_flow:.3f}, std: {flow_std:.3f}, diff: {avg_diff:.1f}, type: {speed_type})")
        return None
    else:
        print(f"✅ {speed_type} 通過: {os.path.basename(video_path)} (flow: {avg_flow:.3f}, std: {flow_std:.3f}, diff: {avg_diff:.1f})")
        return f"{video_path}:{speed_type}"  # 存檔時加類型標籤

def move_videos(good_videos, target_base_dir, sakuga_artist):
    """
    移動通過的影片到目標目錄。
    
    Args:
        good_videos: 通過的影片列表 (格式: "完整路徑:速度類型")
        target_base_dir: 目標基礎目錄，如 "C:/Users/litsu/Desktop/sakuga/test-videos"
        sakuga_artist: 藝術家名稱，如 "松本憲生"
    """
    target_dir = Path(target_base_dir) / sakuga_artist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    skipped_count = 0
    
    for line_num, result in enumerate(good_videos, 1):
        # 解析路徑和類型 (假設格式: "完整路徑:速度類型")
        # 更安全的分割：使用 rsplit 從右邊分割一次，避免路徑中多個 :
        if ':' in result:
            parts = result.rsplit(':', 1)  # 從右邊分割一次
            video_path_str = parts[0].strip()  # 路徑部分
            speed_type = parts[1].strip() if len(parts) > 1 else ''
        else:
            video_path_str = result.strip()
            speed_type = ''
        
        # 轉為 Path 物件，處理可能的路徑問題 (如 Windows / vs \)
        # 替換可能的錯誤反斜杠為正斜杠，或直接用 Path 處理
        video_path_str = video_path_str.replace('\\', '/').replace('\\\\', '/')
        video_path = Path(video_path_str)
        
        # 檢查檔案是否存在
        if not video_path.exists():
            print(f"❌ 第 {line_num} 行路徑不存在，跳過: {video_path_str} (類型: {speed_type})")
            skipped_count += 1
            continue
        
        filename = video_path.name
        target_path = target_dir / filename
        
        # 如果目標已存在，跳過或覆蓋 (這裡跳過)
        if target_path.exists():
            print(f"⚠️  第 {line_num} 行目標已存在，跳過: {filename} (類型: {speed_type})")
            skipped_count += 1
            continue
        
        try:
            # 移動檔案 (保留元數據)
            shutil.move(str(video_path), str(target_path))
            print(f"✅ 移動: {filename} -> {target_dir} (類型: {speed_type})")
            moved_count += 1
        except Exception as e:
            print(f"❌ 移動失敗 (第 {line_num} 行): {filename} - 錯誤: {e} (類型: {speed_type})")
            skipped_count += 1
    
    print(f"\n📊 移動完成: 總 {moved_count + skipped_count} 行，成功 {moved_count} 個，跳過 {skipped_count} 個")
    print(f"📁 目標目錄: {target_dir}")

def main():
    sakuga_artist = "吉原達矢"
    folder_path = f"C:/Users/litsu/Desktop/sakuga/videos/{sakuga_artist}"
    target_base_dir = "C:/Users/litsu/Desktop/sakuga/test-videos"
    
    parser = ArgumentParser(description="篩選 sakuga 影片")
    parser.add_argument('--dir', default=folder_path, help='影片資料夾路徑 (預設: {})'.format(folder_path))
    parser.add_argument('--workers', type=int, default=4, help='並行 worker 數')
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"❌ 資料夾不存在: {args.dir}")
        return
    
    video_paths = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv','.webm'))]  # 加 mkv 支持
    
    if not video_paths:
        print(f"❌ 資料夾 {args.dir} 無影片檔案")
        return
    
    print(f"🚀 開始篩選 {len(video_paths)} 個影片 (資料夾: {args.dir})")
    
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(filter_single_video, video_paths), 
                            total=len(video_paths), desc="篩選影片"))
    
    good_videos = [r for r in results if r is not None]
    skipped = len(video_paths) - len(good_videos)
    print(f"\n📊 總 {len(video_paths)} 片，跳過 {skipped} 片，通過 {len(good_videos)} 片 (剩餘率: {len(good_videos)/len(video_paths)*100:.1f}%)")
    
    # 移動通過的影片
    move_videos(good_videos, target_base_dir, sakuga_artist)
    
    # 存清單 (含速度類型)
    output_file = os.path.join(args.dir, 'good_videos.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(good_videos))
    print(f"✅ 通過清單存至 {output_file} (格式: 路徑:速度類型)")

if __name__ == '__main__':
    main()