import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import csv

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

# Retriever and LLM
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever = PatchedRetriever(base_retriever)
llm = ChatCohere(cohere_api_key=COHERE_API_KEY, model="command-r-plus")
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("📘 Multilingual Q&A System with Feedback")
query = st.text_input("Enter your question (English or Arabic):")

if query:
    result = qa_chain.invoke({"question": query})
    answer = result['answer']
    sources = result['sources']

    # Basic Trust Score = based on source count + answer length
    trust_score = min(100, len(answer) + 10 * len(sources.split(", ")))

    st.markdown("### ✅ Answer")
    st.write(answer)
    st.markdown("**📂 Sources:**")
    st.write(sources)
    st.markdown(f"**🔒 Trust Score:** {trust_score}/100")

    st.markdown("---")
    st.markdown("### 📝 Feedback")
    feedback_text = st.text_area("Your comments")
    rating = st.slider("Rate this answer", 1, 5, 3)

    if st.button("Submit Feedback"):
        with open("feedback_log.csv", "a", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([query, answer, sources, trust_score, rating, feedback_text])
        st.success("Thank you! Your feedback has been recorded.")
