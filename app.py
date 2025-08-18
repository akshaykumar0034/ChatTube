# ------------------- Imports ------------------- #
from dotenv import load_dotenv
from youtube_url import extract_video_id
from youtube_transcript import transcript

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# ------------------- 1. Setup ------------------- #
load_dotenv()

# Input YouTube video URL
youtube_url = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"

# Extract video ID
video_id = extract_video_id(youtube_url)


# ------------------- 2. Transcript ------------------- #
# Get raw transcript text for video
video_transcript = transcript(video_id)

# Split transcript into chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
transcript_chunks = text_splitter.create_documents([video_transcript])


# ------------------- 3. Embeddings & Vector DB ------------------- #
# Create embeddings using Google Generative AI
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store transcript chunks in FAISS vector database
vector_database = FAISS.from_documents(transcript_chunks, embedding_model)

# Create retriever to fetch top relevant chunks
transcript_retriever = vector_database.as_retriever(search_type="similarity", search_kwargs={"k": 8})


# ------------------- 4. LLM ------------------- #
# Initialize Google Gemini model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)


# ------------------- 5. Prompt Template ------------------- #
qa_prompt = PromptTemplate(
    template="""  
        You are a highly accurate, knowledgeable, and helpful assistant.
        Your goal is to provide consistent, detailed, and reliable answers
        based on the given video transcript.

        **Instructions:**
        1. Always combine information from ALL provided context snippets.
        2. If the same entity is mentioned in multiple contexts, merge them into
           one unified and consistent answer.
        3. Use ONLY the transcript context first.  
        4. If the context does not contain sufficient information, respond with:  
           **"Information is not provided in the video, but based on my knowledge, [give best answer]."**
        5. Answers must remain consistent for the same question, regardless of small variations in wording.
        6. Be clear and structured:
            - Use **numbered points** for multiple facts.  
            - Use a **single bullet point** if it’s one fact.  

        Context:
        {context}

        Question:
        {question}

        Answer:
    """,
    input_variables=['context', 'question']
)


# ------------------- 6. Memory ------------------- #
conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# ------------------- 7. Conversational QA Chain ------------------- #
chatbot = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=transcript_retriever,
    memory=conversation_memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)


# ------------------- 8. Run Chatbot ------------------- #
if __name__ == "__main__":
    while True:
        user_question = input("\nUser: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        result = chatbot.invoke({"question": user_question})
        print("\nAI:", result["answer"])
