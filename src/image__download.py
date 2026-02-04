import pandas as pd
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- variable ---
CSV_FILE = "youtube_shorts_data_CLEAN.csv"
IMAGE_DIR = "shorts_thumbnails"

MAX_WORKERS = 10 
# ---Download---

def download_image(video_id: str, url: str, target_dir: str):
    """
    Thumbnail image download and save.
    """
    filepath = os.path.join(target_dir, f"{video_id}.jpg")
    
    # if file exist -> pass .
    if os.path.exists(filepath):
        # print(f"[INFO] Already Exist: {video_id}")
        return video_id, True, "Skipped (Exists)"

    try:
        # User-Agent 
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        # HTTP status check 
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return video_id, True, "Success"
        else:
            return video_id, False, f"HTTP Error {response.status_code}"

    except requests.exceptions.RequestException as e:
        return video_id, False, f"Network Error: {e}"
    except Exception as e:
        return video_id, False, f"Unknown Error: {e}"

def main_downloader():
    """
    Read URL  then download image in CSV File.
    """
    print(f"--- Thumbnail download start: CSV File Path = {CSV_FILE} ---")
    
    # 1. CSV 파일 존재 확인
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] CSV file '{CSV_FILE}' Not Found. First Collect please.")
        return

    # 2. image save directory generate
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"[INFO] Image Save directory '{IMAGE_DIR}' complete to generate.")

    # 3. Dataload
    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
        # drop duplicate ID,  correct URL Filtering .
        df.drop_duplicates(subset=['videoId'], inplace=True)
        df.dropna(subset=['thumbnail_url'], inplace=True)
        
        tasks = [(row['videoId'], row['thumbnail_url'], IMAGE_DIR) 
                 for index, row in df.iterrows()]
        
        total_tasks = len(tasks)
        print(f"[INFO] {total_tasks} Thumbnail Download begin.")
    
    except Exception as e:
        print(f"[ERROR] CSV File read Error: {e}")
        return

    # 4. Download execute
    start_time = time.time()
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(lambda p: download_image(*p), tasks))

    # 5. Total
    for video_id, status, message in results:
        if status:
            success_count += 1
        # Fail message print.
        elif not status:
            print(f"[FAIL] ID: {video_id} | URL Download Fail: {message}")

    end_time = time.time()
    
    print("\n----------------------------------------------------")
    print(f"*** Download Complete ***")
    print(f"Total download attempt: {total_tasks}개")
    print(f"Success / Fail: {success_count}개 / {total_tasks - success_count}개")
    print(f"Total Time: {end_time - start_time:.2f}초")
    print(f"Save Path: ./{IMAGE_DIR}")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main_downloader()
