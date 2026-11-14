from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory="db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0,
                             max_retries=1, timeout=30)

app = FastAPI()


def to_text(content):
    """Gemini can return a plain string or a list of parts; keep only the text."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


@app.get("/ask")
def ask(q: str):
    docs = retriever.invoke(q)
    ctx = "\n\n".join(d.page_content for d in docs)
    prompt = f"""Answer only using the context below.
If the answer is not there, say "I don't know".

Context:
{ctx}

Question: {q}"""
    answer = to_text(llm.invoke(prompt).content)
    pages = [d.metadata.get("page") for d in docs]
    return {"answer": answer, "source_pages": pages}
