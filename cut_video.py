import os
import subprocess
import json
from pathlib import Path

def get_video_duration(video_path):
    """使用 ffprobe 獲取影片時長（秒）"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', '-show_streams', video_path
        ]
        # ⭐ 修復：指定 UTF-8 編碼，避免 Windows cp949 解碼錯誤
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.stdout is None or result.stdout.strip() == '':
            raise ValueError("ffprobe 返回空輸出")
        data = json.loads(result.stdout)
        
        # 獲取時長（字串格式，可能是浮點數）
        duration_str = data['format']['duration']
        duration = float(duration_str)
        return duration
    except json.JSONDecodeError as e:
        print(f"JSON 解析失敗 {video_path}: {e}")
        print(f"ffprobe 輸出: {result.stdout[:200] if 'result' in locals() else '無'}")
        return None
    except Exception as e:
        print(f"無法獲取影片時長 {video_path}: {e}")
        return None

def split_video(input_path, output_pattern, segment_duration=10):
    """使用 ffmpeg 切割影片"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c', 'copy',  # 直接複製流，不重新編碼
            '-f', 'segment',
            '-segment_time', str(segment_duration),
            '-reset_timestamps', '1',
            output_pattern
        ]
        
        # ⭐ 修復：指定 UTF-8 編碼，避免 Windows cp949 解碼錯誤
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print(f"成功切割: {input_path}")
            return True
        else:
            print(f"切割失敗 {input_path}: {result.stderr}")
            return False
    except Exception as e:
        print(f"切割過程出錯 {input_path}: {e}")
        return False

def process_video_folder(folder_path, max_duration=15, segment_duration=10, min_segment_duration=5):
    """處理資料夾中的所有影片檔案"""
    video_extensions = {'.mp4', '.webm', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v','.webm'}
    
    # 確保 ffmpeg 和 ffprobe 可用
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, encoding='utf-8')
        subprocess.run(['ffprobe', '-version'], capture_output=True, text=True, encoding='utf-8')
    except FileNotFoundError:
        print("錯誤: 請先安裝 ffmpeg 並確保在 PATH 中")
        return
    
    processed_count = 0
    skipped_count = 0
    
    for file_path in Path(folder_path).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            print(f"\n處理檔案: {file_path.name}")
            
            # 獲取影片時長
            duration = get_video_duration(str(file_path))
            if duration is None:
                print(f"跳過 {file_path.name} (無法讀取時長)")
                skipped_count += 1
                continue
            
            print(f"影片時長: {duration:.2f} 秒")
            
            # 檢查是否需要切割
            if duration <= max_duration:
                print(f"跳過 {file_path.name} (時長未超過 {max_duration} 秒)")
                skipped_count += 1
                continue
            
            # 計算切割段數
            total_segments = int(duration // segment_duration)
            remaining_seconds = duration % segment_duration
            
            print(f"將切割為 {total_segments} 段 (每段 {segment_duration} 秒)")
            if remaining_seconds < min_segment_duration:
                print(f"最後 {remaining_seconds:.1f} 秒不足 {min_segment_duration} 秒，將被捨棄")
            else:
                total_segments += 1
                print(f"最後一段: {remaining_seconds:.1f} 秒")
            
            # 準備輸出模式
            output_pattern = str(file_path.with_suffix('')) + '_%03d' + file_path.suffix
            
            # 切割影片
            if split_video(str(file_path), output_pattern, segment_duration):
                # 刪除原檔案
                try:
                    file_path.unlink()
                    print(f"已刪除原檔案: {file_path.name}")
                except Exception as e:
                    print(f"刪除原檔案失敗: {e}")
                
                processed_count += 1
            else:
                skipped_count += 1
    
    print(f"\n處理完成!")
    print(f"成功切割: {processed_count} 個影片")
    print(f"跳過處理: {skipped_count} 個影片")

def main():
    """主函數"""
    # 設定參數 
    sakuga_artist = "吉成曜"
    folder_path = f"C:/Users/litsu/Desktop/sakuga/videos/{sakuga_artist}"
    max_duration = 15      # 超過此時長才切割 (秒)
    segment_duration = 10  # 每段時長 (秒)
    min_segment_duration = 5  # 最小段時長，少於此值則捨棄 (秒)
    
    print("影片自動切割工具")
    print(f"處理資料夾: {folder_path}")
    print(f"切割條件: 時長 > {max_duration}秒 的影片將被切割為 {segment_duration}秒 的片段")
    print(f"捨棄條件: 最後片段 < {min_segment_duration}秒 將被捨棄")
    print("-" * 50)
    
    if not os.path.exists(folder_path):
        print(f"錯誤: 資料夾不存在 - {folder_path}")
        return
    
    process_video_folder(folder_path, max_duration, segment_duration, min_segment_duration)

if __name__ == "__main__":
    main()