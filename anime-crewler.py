import time
import requests
import os
import threading
from queue import Queue
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse

# 全局變量
download_queue = Queue()
downloading_files = set()
download_complete = []
download_lock = threading.Lock()

def download_worker():
    """下載工作線程"""
    while True:
        item = download_queue.get()
        if item is None:  # 終止信號
            break
            
        video_url, download_folder, filename = item
        download_video(video_url, download_folder, filename)
        download_queue.task_done()

def download_video(video_url, download_folder, filename):
    """下載影片文件"""
    try:
        filepath = os.path.join(download_folder, filename)
        
        # 檢查文件是否已存在或正在下載
        with download_lock:
            if filename in downloading_files or os.path.exists(filepath):
                print(f"文件已存在或正在下載，跳過: {filename}")
                return True
            downloading_files.add(filename)
        
        print(f"[下載開始] {filename}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.sakugabooru.com/'
        }
        
        # 先發送HEAD請求獲取文件資訊
        try:
            head_response = requests.head(video_url, headers=headers, timeout=10)
            content_type = head_response.headers.get('content-type', '')
            content_length = head_response.headers.get('content-length', 0)
            print(f"[下載資訊] {filename} - 類型: {content_type}, 大小: {int(content_length)/1024/1024:.1f}MB")
        except:
            content_type = ""
            content_length = 0
            print(f"[下載資訊] {filename} - 無法獲取文件資訊，繼續下載...")
        
        response = requests.get(video_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(filepath)
        
        with download_lock:
            downloading_files.remove(filename)
            download_complete.append(filename)
        
        print(f"[下載完成] {filename} ({file_size/1024/1024:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"[下載失敗] {filename}: {e}")
        # 刪除可能損壞的文件
        if os.path.exists(filepath):
            os.remove(filepath)
        
        with download_lock:
            if filename in downloading_files:
                downloading_files.remove(filename)
        
        return False

def get_video_format(video_url):
    """從影片URL確定文件格式"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.sakugabooru.com/'
        }
        
        # 發送HEAD請求獲取Content-Type
        head_response = requests.head(video_url, headers=headers, timeout=10)
        content_type = head_response.headers.get('content-type', '').lower()
        
        # 根據Content-Type確定副檔名
        if 'mp4' in content_type or 'video/mp4' in content_type:
            return '.mp4'
        elif 'webm' in content_type or 'video/webm' in content_type:
            return '.webm'
        elif 'x-matroska' in content_type or 'video/x-matroska' in content_type:
            return '.mkv'
        elif 'quicktime' in content_type or 'video/quicktime' in content_type:
            return '.mov'
        elif 'avi' in content_type or 'video/avi' in content_type:
            return '.avi'
        else:
            # 從URL路徑提取副檔名
            parsed_url = urlparse(video_url)
            path = parsed_url.path.lower()
            
            # 檢查常見影片副檔名
            video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v', '.mka', '.3gp', '.ogv']
            for ext in video_extensions:
                if ext in path:
                    return ext
            
            # 預設使用mp4
            return '.mp4'
            
    except Exception as e:
        print(f"無法確定影片格式: {e}")
        # 預設使用mp4
        return '.mp4'

def print_download_status():
    """定期打印下載狀態"""
    while True:
        time.sleep(5)
        with download_lock:
            current_downloading = list(downloading_files)
            completed_count = len(download_complete)
        
        if current_downloading or completed_count > 0:
            print(f"\n[下載狀態] 進行中: {len(current_downloading)}, 已完成: {completed_count}")
            if current_downloading:
                print(f"正在下載: {', '.join(current_downloading)}")

def get_all_page_urls(driver, base_url):
    """獲取所有分頁的URL"""
    try:
        print("尋找分頁導航...")
        
        # 等待分頁元素載入
        wait = WebDriverWait(driver, 10)
        pagination = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.pagination"))
        )
        
        # 找到所有頁面連結
        page_links = pagination.find_elements(By.CSS_SELECTOR, "a, em.current")
        
        page_urls = []
        
        for link in page_links:
            # 如果是當前頁面（em標籤）
            if link.tag_name == 'em' and 'current' in link.get_attribute('class'):
                page_num = link.text.strip()
                page_urls.append((int(page_num), base_url))
                print(f"找到當前頁面: {page_num}")
            # 如果是頁面連結（a標籤）
            elif link.tag_name == 'a':
                href = link.get_attribute('href')
                page_text = link.text.strip()
                if page_text.isdigit():
                    page_num = int(page_text)
                    page_urls.append((page_num, href))
                    print(f"找到頁面: {page_num}")
        
        # 按頁碼排序
        page_urls.sort(key=lambda x: x[0])
        
        # 找出最大頁數
        if page_urls:
            max_page = max([p[0] for p in page_urls])
        else:
            max_page = 1
        
        print(f"從元素中找到 {len(page_urls)} 個頁面，最大頁數: {max_page}")
        
        # 生成所有頁面的URL，從1到max_page
        full_page_urls = []
        # 第1頁使用base_url
        full_page_urls.append((1, base_url))
        print(f"生成頁面1: {base_url}")
        
        # 從2到max_page生成URL
        for i in range(2, max_page + 1):
            # 構造URL: base_url + f"&page={i}"
            page_url = base_url + f"&page={i}"
            full_page_urls.append((i, page_url))
            print(f"自動生成頁面 {i}: {page_url}")
        
        # 按頁碼排序
        full_page_urls.sort(key=lambda x: x[0])
        
        print(f"總共生成 {len(full_page_urls)} 個頁面URL")
        return full_page_urls
        
    except Exception as e:
        print(f"獲取分頁資訊失敗: {e}")
        # 如果找不到分頁，返回第一頁
        return [(1, base_url)]

def get_post_urls_from_page(driver, page_url):
    """從單一頁面獲取所有帖子的URL"""
    try:
        print(f"訪問頁面獲取帖子URL: {page_url}")
        driver.get(page_url)
        
        # 等待帖子列表載入
        wait = WebDriverWait(driver, 15)
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.content ul li"))
        )
        
        # 獲取所有帖子元素
        posts = driver.find_elements(By.CSS_SELECTOR, "div.content ul li")
        print(f"本頁找到 {len(posts)} 個帖子")
        
        # 收集帖子URL
        post_urls = []
        for post in posts:
            try:
                post_id = post.get_attribute('id').replace('p', '')
                detail_link = post.find_element(By.CSS_SELECTOR, "a.thumb")
                detail_url = detail_link.get_attribute('href')
                post_urls.append((post_id, detail_url))
            except Exception as e:
                print(f"收集帖子URL時出錯: {e}")
                continue
        
        return post_urls
        
    except Exception as e:
        print(f"從頁面獲取帖子URL失敗: {e}")
        return []

def process_single_post(driver, post_id, post_url, sakuga_artist, download_folder):
    """處理單一帖子"""
    try:
        print(f"處理帖子 {post_id}: {post_url}")
        
        # 進入詳情頁
        driver.get(post_url)
        
        # 等待影片元素載入
        wait = WebDriverWait(driver, 10)
        video_element = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "video.vjs-tech"))
        )
        
        # 獲取影片URL
        video_src = video_element.get_attribute('src')
        if not video_src:
            # 如果沒有直接src，嘗試從source標籤獲取
            try:
                source_element = driver.find_element(By.CSS_SELECTOR, "video source")
                video_src = source_element.get_attribute('src')
            except:
                print("無法從source標籤獲取影片URL")
        
        if video_src:
            print(f"找到影片: {video_src}")
            
            # 確定影片格式
            file_extension = get_video_format(video_src)
            print(f"檢測到影片格式: {file_extension}")
            
            # 創建文件名：藝術家_帖子ID.副檔名
            filename = f"{sakuga_artist}_{post_id}{file_extension}"
            
            # 將下載任務加入隊列（非阻塞）
            download_queue.put((video_src, download_folder, filename))
            print(f"[任務已加入隊列] {filename}")
            return True
        else:
            print("未找到影片URL")
            return False
            
    except Exception as e:
        print(f"處理帖子 {post_id} 時出錯: {e}")
        return False

# 設定下載目錄
sakuga_artist = "tatsuya_yoshihara"
# kazuto_nakazawa
# ryo_araki
# masahiro_sato
# yoh_yoshinari
# hiroyuki_imaishi
sakuga_artist_ch=""
download_folder = fr"C:\Users\litsu\Desktop\sakuga\videos\{sakuga_artist_ch}"
os.makedirs(download_folder, exist_ok=True)
print(f"下載目錄: {download_folder}")

# 啟動下載線程
num_download_workers = 3  # 同時下載的線程數
download_threads = []
for i in range(num_download_workers):
    t = threading.Thread(target=download_worker)
    t.daemon = True
    t.start()
    download_threads.append(t)

# 啟動狀態打印線程
status_thread = threading.Thread(target=print_download_status)
status_thread.daemon = True
status_thread.start()

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--log-level=3') # 減少控制台不必要的日誌
options.add_argument('--blink-settings=imagesEnabled=false')

try:
    driver = webdriver.Chrome(options=options)
    
    # 訪問搜索頁面
    base_url = f"https://www.sakugabooru.com/post?tags={sakuga_artist}+"
    print(f"訪問搜索頁面: {base_url}")
    driver.get(base_url)
    
    # 獲取所有頁面URL
    page_urls = get_all_page_urls(driver, base_url)
    
    total_success = 0
    total_fail = 0
    
    # 限制處理的頁面數（測試用）
    # max_pages_to_process = 1  # 只處理第一頁
    max_pages_to_process = len(page_urls)  # 處理所有頁面
    
    # 處理每個頁面
    for page_num, page_url in page_urls[:max_pages_to_process]:
        print(f"\n{'#'*60}")
        print(f"開始處理第 {page_num} 頁")
        print(f"{'#'*60}")
        
        # 從當前頁面獲取所有帖子URL
        post_urls = get_post_urls_from_page(driver, page_url)
        
        # 限制每頁處理的帖子數量（測試用）
        # max_posts_per_page = 5  # 只處理前5個帖子
        max_posts_per_page = len(post_urls)  # 處理所有帖子
        
        page_success = 0
        page_fail = 0
        
        # 處理當前頁面的每個帖子
        for i, (post_id, post_url) in enumerate(post_urls[:max_posts_per_page]):
            print(f"\n--- 處理第 {i+1}/{len(post_urls[:max_posts_per_page])} 個帖子 ---")
            
            if process_single_post(driver, post_id, post_url, sakuga_artist, download_folder):
                page_success += 1
            else:
                page_fail += 1
            
            # 帖子間暫停
            time.sleep(2)
        
        total_success += page_success
        total_fail += page_fail
        
        print(f"\n第 {page_num} 頁處理完成: 成功 {page_success} 個, 失敗 {page_fail} 個")
        
        # 頁面間暫停
        if page_num < len(page_urls[:max_pages_to_process]):
            print(f"\n準備進入下一頁...")
            time.sleep(3)
    
    print(f"\n{'='*60}")
    print("所有頁面處理完成，等待下載完成...")
    
    # 等待所有下載任務完成
    download_queue.join()
    
    # 發送終止信號給下載線程
    for _ in range(num_download_workers):
        download_queue.put(None)
    
    for t in download_threads:
        t.join()
    
    print(f"\n{'='*60}")
    print("下載總結:")
    print(f"總共處理頁面: {len(page_urls[:max_pages_to_process])} 頁")
    print(f"成功加入隊列: {total_success} 個")
    print(f"處理失敗: {total_fail} 個")
    print(f"實際完成下載: {len(download_complete)} 個")
    print(f"下載目錄: {download_folder}")
    print(f"{'='*60}")
    
    # 顯示下載的文件列表
    print("\n下載的文件:")
    for file in os.listdir(download_folder):
        filepath = os.path.join(download_folder, file)
        if os.path.isfile(filepath):
            file_size = os.path.getsize(filepath) / 1024 / 1024
            print(f"  - {file} ({file_size:.1f} MB)")

except Exception as e:
    print(f"程式執行錯誤: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 確保所有下載線程終止
    for _ in range(num_download_workers):
        download_queue.put(None)
    
    input("按 Enter 結束...")
    driver.quit()
# import time
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# options = Options()
# options.add_argument('--disable-gpu')

# try:
#     driver = webdriver.Chrome(options=options)
#     url = "https://www.sakugabooru.com/post?tags=arifumi_imai+"
#     driver.get(url)
    
#     wait = WebDriverWait(driver, 15)
    
#     # 使用成功的選擇器
#     selector = "div#post-list li"
#     li_elements = wait.until(
#         EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
#     )
    
#     print(f"找到 {len(li_elements)} 個帖子")
    
#     # 詳細檢查前3個帖子的結構
#     for i, li in enumerate(li_elements[:3]):
#         print(f"\n=== 詳細檢查帖子 {i+1} ===")
        
#         # 獲取整個 li 的 HTML 結構
#         li_html = li.get_attribute('outerHTML')
#         print("li 的 HTML 結構:")
#         print(li_html[:500] + "..." if len(li_html) > 500 else li_html)  # 只顯示前500字元
        
#         print("\n內部元素分析:")
        
#         # 檢查所有子元素
#         all_children = li.find_elements(By.XPATH, ".//*")
#         print(f"總共有 {len(all_children)} 個子元素")
        
#         for child_index, child in enumerate(all_children[:10]):  # 只檢查前10個子元素
#             tag_name = child.tag_name
#             class_name = child.get_attribute('class') or ""
#             text = child.text.strip()
            
#             print(f"  元素 {child_index+1}: <{tag_name}> class='{class_name}'")
#             if text:
#                 print(f"      文字: {text[:100]}{'...' if len(text) > 100 else ''}")
            
#             # 如果是圖片，顯示更多資訊
#             if tag_name == 'img':
#                 src = child.get_attribute('src')
#                 data_src = child.get_attribute('data-src')
#                 print(f"      src: {src}")
#                 print(f"      data-src: {data_src}")
        
#         print("=" * 50)

# except Exception as e:
#     print(f"錯誤: {e}")
#     import traceback
#     traceback.print_exc()

# finally:
#     input("按 Enter 結束...")
#     driver.quit()

