#Chatbot.py
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, get_google_api_key
from questionAnswerPrompt import qa_prompt

import os, time, shutil

CACHE_TTL = 60 * 60 * 24  # 24 hours in seconds

class ChatbotManager:
    def __init__(self):
        self.chat_histories = {}
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash-latest",
            temperature=0.2,
            google_api_key=get_google_api_key()
        )
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=get_google_api_key()
        )
        self.vectorstore_dir = "vectorstores"
        self.last_access = {}  # video_id -> last used timestamp

        if not os.path.exists(self.vectorstore_dir):
            os.makedirs(self.vectorstore_dir)

    def get_history(self, session_id: str):
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = InMemoryChatMessageHistory()
        return self.chat_histories[session_id]

    def build_chatbot_chain(self, transcript_text: str, video_id: str):
        """Builds or loads a chatbot chain for a given transcript/video."""
        vectorstore_path = os.path.join(self.vectorstore_dir, video_id)

        if os.path.exists(vectorstore_path):
            vector_db = FAISS.load_local(
                vectorstore_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            transcript_chunks = splitter.create_documents([transcript_text])

            vector_db = FAISS.from_documents(transcript_chunks, self.embedding_model)
            vector_db.save_local(vectorstore_path)

        # ✅ update last access time
        self.last_access[video_id] = time.time()

        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )

        base_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

        return RunnableWithMessageHistory(
            base_chain,
            self.get_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    def cleanup_old_indexes(self):
        """Deletes FAISS indexes older than TTL."""
        now = time.time()
        for video_id, last_used in list(self.last_access.items()):
            if now - last_used > CACHE_TTL:
                vectorstore_path = os.path.join(self.vectorstore_dir, video_id)
                if os.path.exists(vectorstore_path):
                    shutil.rmtree(vectorstore_path)
                del self.last_access[video_id]
