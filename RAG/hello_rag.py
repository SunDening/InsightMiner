# 1. Import dependencies (LangChain 1.x paths)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# 2. Load the document
loader = TextLoader("my_knowledge.txt", encoding="utf-8")
documents = loader.load()

# 3. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64
)
splits = text_splitter.split_documents(documents)

# 4. Build vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 原始语料块和嵌入以键值对形式存储
vector_db = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 5. Local LLM via Ollama
llm = OllamaLLM(model="gemma4:e4b")

# 6. RAG pipeline
def rag_ask(question):
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Based on the following information, answer the question:\n\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt)

# Test
print(rag_ask("吕不韦帮谁成为了国君"))

print("="*60)

print(rag_ask("秦非子能建国的原因是什么？"))