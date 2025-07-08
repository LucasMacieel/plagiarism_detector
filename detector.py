import cv2
import faiss
import fitz
import numpy as np
import pytesseract
import spacy
from PIL import Image
from sentence_transformers import SentenceTransformer
# --- New import for keyword extraction ---
from sklearn.feature_extraction.text import TfidfVectorizer


# --- Pre-load spaCy models efficiently ---
def load_spacy_models():
    """Loads English and Portuguese spaCy models into a dictionary."""
    print("--- Pre-loading spaCy models for sentence segmentation ---")
    models = {}
    try:
        print("Loading spaCy model: en_core_web_sm...")
        models['en'] = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
        print("'en_core_web_sm' loaded successfully.")
        print("Loading spaCy model: pt_core_news_sm...")
        models['pt'] = spacy.load("pt_core_news_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
        print("'pt_core_news_sm' loaded successfully.")
    except OSError as e:
        print(f"Error loading a spaCy model: {e}")
        print("Please run 'python -m spacy download en_core_web_sm' and 'python -m spacy download pt_core_news_sm'")
    print("--- spaCy models loaded. ---")
    return models

PRELOADED_SPACY_MODELS = load_spacy_models()

# --- Document Processing and OCR (Unchanged) ---
def preprocess_image(pil_img):
    try:
        img = np.array(pil_img); img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(img)
    except Exception as e:
        print(f"Image preprocessing failed: {e}"); return pil_img

def ocr_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num); pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text += pytesseract.image_to_string(preprocess_image(img), lang='por+eng') + "\n"
    except Exception as e:
        print(f"OCR failed for {pdf_path}. Error: {e}")
    return text

# --- Text Chunking (Now using preloaded models) ---
def chunk_text_by_sentence(text, lang_code, target_words_per_chunk=75, max_sentences_per_chunk=5):
    sentence_segmentation_model = PRELOADED_SPACY_MODELS.get(lang_code, PRELOADED_SPACY_MODELS.get('en'))
    if not sentence_segmentation_model:
        print("CRITICAL: No spaCy models available for sentence chunking.")
        return []
    if not text or not text.strip(): return []
    text = text.replace('\n', ' ').strip()
    doc = sentence_segmentation_model(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences: return []
    chunks, current_chunk = [], []
    for sentence in sentences:
        current_chunk.append(sentence)
        word_count = sum(len(s.split()) for s in current_chunk)
        if word_count >= target_words_per_chunk or len(current_chunk) >= max_sentences_per_chunk:
            chunks.append(" ".join(current_chunk)); current_chunk = []
    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks

# --- NEW: Keyword Extraction and Similarity ---
class KeywordExtractor:
    def __init__(self, corpus):
        # Note: Removed 'stop_words' to be language-agnostic.
        self.vectorizer = TfidfVectorizer(max_features=1000, lowercase=True)
        self.vectorizer.fit(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def get_top_keywords(self, text, n=10):
        tfidf_vector = self.vectorizer.transform([text])
        sorted_indices = np.argsort(tfidf_vector.toarray().flatten())[::-1]
        return {self.feature_names[i] for i in sorted_indices[:n] if tfidf_vector[0, i] > 0}

def calculate_jaccard_similarity(set1, set2):
    if not set1 and not set2: return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

# --- Embeddings and Faiss (Unchanged) ---
class SentenceEmbedder:
    def __init__(self, preloaded_model):
        self.model = preloaded_model

    def get_embeddings(self, text_chunks):
        if not text_chunks:
            return np.array([])
        embeddings = self.model.encode(text_chunks, show_progress_bar=False)
        return np.array(embeddings).astype('float32')

class SimilaritySearch:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension); self.chunk_map = []
    def build_index(self, embeddings, chunks, doc_name):
        if embeddings.ndim == 1: embeddings = np.expand_dims(embeddings, axis=0)
        if embeddings.shape[0] == 0: return
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        for chunk in chunks: self.chunk_map.append({'doc': doc_name, 'chunk': chunk})
    def search(self, query_embeddings, k=1):
        if query_embeddings.ndim == 1: query_embeddings = np.expand_dims(query_embeddings, axis=0)
        if query_embeddings.shape[0] == 0: return np.array([]), np.array([])
        faiss.normalize_L2(query_embeddings)
        return self.index.search(query_embeddings, k)
