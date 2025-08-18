# ------------------- Imports ------------------- #
from dotenv import load_dotenv
from youtube_url import extract_video_id
from youtube_transcript import transcript

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# ------------------- Setup ------------------- #
# Load environment variables (API keys, etc.)
load_dotenv()

# Input YouTube video URL
url = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"

# Extract video ID from URL
video_id = extract_video_id(url)


# ------------------- Step 1: Transcript ------------------- #
# Fetch transcript for the given YouTube video
raw_transcript = transcript(video_id)

# Split transcript into smaller chunks for better embedding & retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([raw_transcript])


# ------------------- Step 2: Embeddings & Vector Store ------------------- #
# Create embeddings using Google Generative AI
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store transcript chunks into FAISS vector store
vector_store = FAISS.from_documents(chunks, embeddings)

# Create retriever (fetch top 4 most relevant chunks per query)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})


# ------------------- Step 3: LLM ------------------- #
# Initialize Google Gemini model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)


# ------------------- Step 4: Prompt Template ------------------- #
# Custom prompt with clear instructions for responses
prompt = PromptTemplate(
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

# ------------------- Step 5: Helper Function ------------------- #
# Format retrieved documents into plain text for the LLM
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ------------------- Step 6: QA Chain ------------------- #
# Build the RAG pipeline
qa_chain_with_memory = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

if __name__ == "__main__":
    while True:
        user_question = input("\nUser: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        result = qa_chain_with_memory.invoke({"question": user_question})
        print("\nAI:", result["answer"])