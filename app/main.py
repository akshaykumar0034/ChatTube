#main.py
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from youtube_url import extract_video_id
from youtube_transcript import get_transcript
from chatbot import ChatbotManager
from config import get_google_api_key
from youtube_video_metadata import get_video_metadata
import threading, time
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="ChatTube API")

# Allow CORS for communication with the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot_manager = ChatbotManager()
active_sessions = {} 
def cleanup_task():
    while True:
        chatbot_manager.cleanup_old_indexes()
        time.sleep(3600)  # run every hour

threading.Thread(target=cleanup_task, daemon=True).start()
class VideoPayload(BaseModel):
    video_url: str

class ChatPayload(BaseModel):
    session_id: str
    question: str

@app.get("/")
def read_root():
    return {"message": "ChatTube API is running!"}

@app.post("/api/load_video")
async def load_video(request: Request):
    try:
        get_google_api_key()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    body = await request.json()
    video_url = body.get("video_url")
    video_id = extract_video_id(video_url)

    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL provided.")

    # ✅ get transcript
    result = get_transcript(video_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    transcript = result["transcript"]

    # ✅ get video metadata (title, channel name, channel url)
    metadata = get_video_metadata(video_id)

    # ✅ session handling
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = chatbot_manager.build_chatbot_chain(transcript, video_id)

    return {
        "message": "Video loaded successfully.",
        "session_id": session_id,
        "title": metadata["title"],
        "channel_name": metadata["channel_name"],
        "channel_url": metadata["channel_url"],
    }



@app.post("/api/chat")
async def chat_with_video(payload: ChatPayload):
    if payload.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please load a video first.")
    
    chatbot = active_sessions[payload.session_id]
    
    try:
        response = chatbot.invoke(
            {"question": payload.question},
            config={"configurable": {"session_id": payload.session_id}}
        )

        # ✅ Log history after each chat
        history = chatbot_manager.get_history(payload.session_id).messages
        print(f"\n--- Chat History for session {payload.session_id} ---")
        for msg in history:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            print(f"{role}: {msg.content}")
        print("------------------------------------------------\n")

        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during chat: {str(e)}")
