import requests

def get_video_metadata(video_id: str) -> dict:
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            "title": data.get("title", "Unknown Title"),
            "channel_name": data.get("author_name", "Unknown Channel"),
            "channel_url": data.get("author_url", "")
        }
    except Exception:
        return {"title": "Unknown Title", "channel_name": "Unknown Channel", "channel_url": ""}
