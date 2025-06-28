from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

#
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Set embedding model
embedding_model = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

# Load ChromaDB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Custom retriever 
class PatchedRetriever(BaseRetriever):
    def __init__(self, base_retriever):
        super().__init__()
        object.__setattr__(self, "base_retriever", base_retriever)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        for doc in docs:
            if "source" not in doc.metadata:
                doc.metadata["source"] = doc.metadata.get("filename", "unknown")
        return docs

# Turn the vectorstore into a retriever with top-k = 5 
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever = PatchedRetriever(base_retriever)

# response generation
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus"
)

# Q&A pipeline with sources
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

# terminal interface to ask and answer questions
print("\n📘 Multilingual Q&A System (Cohere + Sources) — type 'exit' to quit")
while True:
    query = input("\nAsk your question: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain.invoke({"question": query})
    print("\nAnswer:\n", result['answer'])
    print("\nSources:", result['sources'])
