import requests

def get_video_title(video_id: str) -> str:
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("title", "Unknown Title")
    except Exception:
        return "Unknown Title"
