# 🌐 Multi-Lingual RAG System

This project implements a **Multilingual Retrieval-Augmented Generation (RAG) System** using English and Arabic business reports. It enables users to ask questions in Arabic or English and receive answers grounded in real documents, with cited sources and a trust score.

---

## 📂 Dataset Structure
The project includes realistic, synthetic reports in both Arabic and English:
```
.
├── Arabic_txt_dataset.zip         # 15 Arabic TXT reports
├── Arabic_pdf_dataset.zip         # 10 Arabic PDF reports
├── English_txt_dataset.zip        # 15 English TXT reports
└── English_pdf_dataset.zip        # 10 English PDF reports
```

---

## 📝 Key Components

### 1. `preprocess_datat_embed.py`
- Extracts and cleans text (PDF + TXT)
- Supports OCR for scanned PDFs
- Performs Arabic normalization
- Infers topic from filenames
- Chunks text into 500-word segments
- Stores embeddings in ChromaDB

### 2. `QA_pipeline.py`
- Loads ChromaDB and Cohere embeddings
- Sets up a retriever with fallback metadata
- Answers questions with citations
- CLI interface for real-time testing

### 3. `QA_streamlit_app.py`
- Provides a Streamlit interface
- Accepts Arabic, English, mix questions
- Displays answer, source, and a trust score
- Collects user feedback with rating and comment
- Saves logs to `feedback_log.csv`

### 4. `requirements.txt`
- Lists all dependencies to set up the environment

---

## ⚖️ Trust Score
A basic score (out of 100) is computed using:
- Number of retrieved source files
- Length of the generated answer

This provides a rough confidence estimate for users.

---

## 🔎 Example Q&A
**Q:** What are the packaging recommendations?

**A:**
1. Strengthen packaging procedures to preserve freshness.  
2. Regularly audit the temperature in storage units.  
3. Enhance weekend shift training for QC staff.

**Sources:** `report_04_packaging.txt`, `report_08_packaging.pdf`  
**Trust Score:** 100/100

---

## 🌟 How to Run
1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Cohere API key in a `.env` file:
   ```
   COHERE_API_KEY=your_api_key_here
   ```
4. Run the preprocessor:
   ```bash
   python preprocess_datat_embed.py
   ```
5. Launch the app:
   ```bash
   streamlit run QA_streamlit_app.py
   ```
---

## 🚀 Improvements & Next Steps
- Use advanced topic classification from content.
- Support PDF/CSV uploads via Streamlit for dynamic use.
- Add multilingual summarization or translation features.
