import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

docs = PyPDFLoader("data.pdf").load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100).split_documents(docs)
print(f"Total chunks: {len(chunks)}")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory="db", embedding_function=embeddings)

BATCH = 90
for i in range(0, len(chunks), BATCH):
    db.add_documents(chunks[i:i + BATCH])
    print(f"Stored {min(i + BATCH, len(chunks))}/{len(chunks)}")
    if i + BATCH < len(chunks):
        time.sleep(61)

print("Done!")