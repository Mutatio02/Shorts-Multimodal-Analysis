import os
import time
import csv
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

API_KEY = "" # Your API Key
YT = build("youtube", "v3", developerKey=API_KEY)

VIEW_BINS = [
    (0, 1_000),
    (1_000, 10_000),
    (10_000, 100_000),
    (100_000, 500_000),
    (500_000, 1_000_000),
    (1_000_000, 10_000_000)
]

MAX_RESULTS = 50
SAVE_FILE = "" # File Name
collected_ids = set()

def yt_request_with_backoff(func, **kwargs):
    backoff = 1
    while True:
        try:
            return func(**kwargs).execute()
        except HttpError as e:
            print(f"[WARN] API Error: {e}")
            if e.resp.status in [403, 500, 503]:
                print(f"[INFO] {backoff}sec wait then retry")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                raise

# Text Filter
def filter_title(title: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z가-힣\s]", "", title)  # Englis or Korean, 
    cleaned = re.sub(r"\s+", " ", cleaned).strip()   
    return cleaned

def fetch_video_details(video_ids):
    details = []
    for i in range(0, len(video_ids), 50):
        response = yt_request_with_backoff(
            YT.videos().list,
            part="snippet,statistics",
            id=",".join(video_ids[i:i+50])
        )
        for item in response.get("items", []):
            vid = item["id"]
            raw_title = item["snippet"]["title"]
            title = filter_title(raw_title)  
            if not title:  # title empty?
                continue
            thumbnail = item["snippet"]["thumbnails"]["high"]["url"]
            views = int(item["statistics"].get("viewCount", 0))
            details.append((vid, title, thumbnail, views))
    return details

def search_videos(query, max_results=50):
    try:
        response = yt_request_with_backoff(
            YT.search().list,
            part="snippet",        
            q=query,# if don't want -> erase
            type="video",
            maxResults=max_results,
            order="viewCount" # date and viewCount Control
        )
        video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
        return video_ids
    except Exception as e:
        print(f"[ERROR] Search Fail ({query}): {e}")
        return []


def save_to_csv(data):
    write_header = not os.path.exists(SAVE_FILE)
    with open(SAVE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["videoId", "title", "thumbnail", "viewCount", "bin"])
        writer.writerows(data)

def collect_youtube_data(queries, target_per_bin=50):
    bin_counts = {b: 0 for b in VIEW_BINS}

    for query in queries:
        print(f"[INFO] Keword '{query}' Collect Begin")
        video_ids = search_videos(query)
        details = fetch_video_details(video_ids)

        data_to_save = []
        for vid, title, thumb, views in details:
            for b in VIEW_BINS:
                if b[0] <= views < b[1] and bin_counts[b] < target_per_bin:
                    data_to_save.append((vid, title, thumb, views, f"{b[0]}~{b[1]}"))
                    bin_counts[b] += 1
                    break

        save_to_csv(data_to_save)
        print(f"[INFO] Current Collect: {bin_counts}")

        if all(c >= target_per_bin for c in bin_counts.values()):
            break

if __name__ == "__main__":
    queries = ["IT","IPhone","Galaxy","war","Trump","China","USA"] # Example
    collect_youtube_data(queries, target_per_bin=100)
