from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# def get_transcript(video_id):
#     try:
#         api = YouTubeTranscriptApi()
#         transcript_list = api.fetch(video_id, languages=["en", "hi"])

#         # Flatten it to plain text
#         transcript = " ".join(chunk.text for chunk in transcript_list)
#         return transcript

#     except TranscriptsDisabled:
#         return "No captions available for this video."
#     except Exception as e:
#         return f"An error occurred: {str(e)}"


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def get_transcript(video_id: str):
    """
    Fetches the transcript for a given YouTube video ID.
    Returns a dictionary with 'success' and either 'transcript' or 'error'.
    """
    try:
        # Create an instance of the class first.
        api = YouTubeTranscriptApi()
        
        # Then, use the fetch() method on the instance.
        # This returns an iterable object of transcript snippets.
        fetched_transcript = api.fetch(video_id, languages=["en", "hi"])
        
        # Check if the fetched transcript is empty
        if not fetched_transcript:
            return {"success": False, "error": "Transcript is empty."}

        # Iterate over the fetched transcript snippets and join their text.
        transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
        
        return {"success": True, "transcript": transcript_text}
        
    except TranscriptsDisabled:
        return {"success": False, "error": "No captions available for this video. Captions may be disabled by the video owner."}
    except NoTranscriptFound:
        return {"success": False, "error": "No transcript found for this video. This may happen if the video is not in the specified languages."}
    except Exception as e:
        return {"success": False, "error": str(e)}