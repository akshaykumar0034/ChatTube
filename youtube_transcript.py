from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=["en", "hi"])

        # Flatten it to plain text
        transcript = " ".join(chunk.text for chunk in transcript_list)
        #print(transcript)
        return transcript

    except TranscriptsDisabled:
        print("No captions available for this video.")
        return None

