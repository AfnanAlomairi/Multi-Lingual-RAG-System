import os
import pytesseract
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.docstore.document import Document
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import re
from langdetect import detect

#  path to the Tesseract
pytesseract.pytesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# API KEY 
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
embed_model = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

# Files path
ARABIC_PATHS = {
    "pdf": "./dataset/Arabic/pdf",
    "txt": "./dataset/Arabic/txt"
}
ENGLISH_PATHS = {
    "pdf": "./dataset/English/pdf",
    "txt": "./dataset/English/txt"
}

# Extract text from PDFs + OCR
def extract_text_from_pdf(filepath):
    try:
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        if text.strip():
            return text
    except:
        pass

    print(f"[OCR] Scanned PDF: {filepath}")
    images = convert_from_path(filepath, dpi=300)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='ara+eng')
    return text


def extract_text_from_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Normalizing and removing diacritics
def normalize_arabic(text):
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    return text

# remove links, image markers, and emojis
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text

# Infer general topic category from filename keywords
def infer_topic(filename):
    fname = filename.lower()
    topics = {
        "packaging": ["تغليف", "packaging"],
        "training": ["تدريب", "training"],
        "quality": ["جودة", "quality", "audit"],
        "energy": ["طاقة", "energy", "consumption"],
        "customer": ["عملاء", "رضا", "customer", "satisfaction"],
        "production": ["انتاج", "production"],
        "marketing": ["تسويق", "marketing", "campaign"],
        "storage": ["تخزين", "storage", "warehouse"],
        "transport": ["نقل", "transport", "logistics"],
        "safety": ["سلامة", "safety", "hygiene"]
    }
    for topic, keywords in topics.items():
        for kw in keywords:
            if kw in fname:
                return topic
    return "general"

# Split long text into chunks (for embeddings)
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Preprocess, clean, and chunk the documents, and attach metadata
def process_language(lang_label, paths, normalize=False):
    docs = []
    for file_type, folder in paths.items():
        for file in os.listdir(folder):
            if not file.endswith(('.pdf', '.txt')):
                continue
            filepath = os.path.join(folder, file)
            text = extract_text_from_pdf(filepath) if file.endswith(".pdf") else extract_text_from_txt(filepath)
            if normalize:
                text = normalize_arabic(text)
            text = clean_text(text)
            chunks = chunk_text(text)
            topic = infer_topic(file)
            for i, chunk in enumerate(chunks):
                metadata = {
                    "filename": file,
                    "lang": lang_label,
                    "chunk_id": i,
                    "source_type": file_type,
                    "topic": topic
                }
                docs.append(Document(page_content=chunk, metadata=metadata))
    return docs

# Embed all documents into ChromaDB
def store_in_chroma(all_documents, persist_dir="./chroma_db"):
    db = Chroma.from_documents(all_documents, embed_model, persist_directory=persist_dir)
    db.persist()
    print(f"[INFO] Stored {len(all_documents)} chunks in ChromaDB.")

# Entry point
if __name__ == "__main__":
    print("[INFO] Processing Arabic dataset...")
    arabic_docs = process_language("ar", ARABIC_PATHS, normalize=True)

    print("[INFO] Processing English dataset...")
    english_docs = process_language("en", ENGLISH_PATHS, normalize=False)

    all_docs = arabic_docs + english_docs
    print(f"[INFO] Total chunks: {len(all_docs)}")

    store_in_chroma(all_docs)
