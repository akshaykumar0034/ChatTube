#Chatbot.py
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, get_google_api_key
from questionAnswerPrompt import qa_prompt

class ChatbotManager:
    """Manages chatbot instances for different sessions."""
    
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

    def get_history(self, session_id: str):
        """Retrieves or creates an in-memory chat history for a session."""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = InMemoryChatMessageHistory()
        return self.chat_histories[session_id]

    def build_chatbot_chain(self, transcript_text: str):
        """Builds a new chatbot chain for a given transcript."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        transcript_chunks = splitter.create_documents([transcript_text])

        vector_db = FAISS.from_documents(transcript_chunks, self.embedding_model)
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