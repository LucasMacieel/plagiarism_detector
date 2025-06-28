import fitz
import pytesseract
from PIL import Image
import numpy as np
import faiss
import cv2
import spacy

# --- Global Variables & Model Loading ---
def load_spacy_models():
    """Loads English and Portuguese spaCy models into a dictionary."""
    print("--- Pre-loading spaCy models for sentence segmentation ---")
    models = {}
    try:
        # Load English model
        print("Loading spaCy model: en_core_web_sm...")
        models['en'] = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
        print("'en_core_web_sm' loaded successfully.")

        # Load Portuguese model
        print("Loading spaCy model: pt_core_news_sm...")
        models['pt'] = spacy.load("pt_core_news_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
        print("'pt_core_news_sm' loaded successfully.")

    except OSError as e:
        print(f"Error loading a spaCy model: {e}")
        print("Please run 'python -m spacy download en_core_web_sm' and 'python -m spacy download pt_core_news_sm'")

    print("--- spaCy models loaded. ---")
    return models

PRELOADED_SPACY_MODELS = load_spacy_models()

# --- 1. Document Processing and OCR ---
def preprocess_image(pil_img):
    """Converts a PIL image to OpenCV format and applies preprocessing for better OCR."""
    try:
        # Convert PIL image to OpenCV format (grayscale)
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply adaptive thresholding to create a binary image
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        return Image.fromarray(img)
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return pil_img  # Return original image if preprocessing fails


def ocr_pdf(pdf_path):
    """Performs OCR on each page of a PDF and returns the extracted text."""
    text = ""
    print(f"Starting OCR for {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)  # Render page at higher DPI for better quality
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            preprocessed_img = preprocess_image(img)
            # Use Tesseract to extract text
            text += pytesseract.image_to_string(preprocessed_img, lang='por') + "\n"
        doc.close()
        print(f"OCR completed for {pdf_path}.")
    except Exception as e:
        print(f"OCR failed for {pdf_path}. Error: {e}")
    return text


# --- 2. Text Preprocessing (Sentence-Aware Chunking) ---
def chunk_text_by_sentence(text, target_words_per_chunk=75, max_sentences_per_chunk=5, lang_code='en'):
    """
    Splits text into sentences and then groups them into chunks for embedding.
    """
    sentence_segmentation_model = PRELOADED_SPACY_MODELS.get(lang_code)

    if not sentence_segmentation_model:
        print("spaCy model is not available. Cannot chunk text.")
        return []
    if not text or not text.strip():
        return []

    # Replace newlines with spaces to help sentence boundary detection.
    text = text.replace('\n', ' ').strip()
    doc = sentence_segmentation_model(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return []

    # Group sentences into chunks
    chunks = []
    current_chunk = []
    for sentence in sentences:
        current_chunk.append(sentence)
        word_count = sum(len(s.split()) for s in current_chunk)

        # Create a new chunk if the current one is large enough or has enough sentences
        if word_count >= target_words_per_chunk or len(current_chunk) >= max_sentences_per_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# --- 3. Sentence-Transformers Embeddings ---
class SentenceEmbedder:
    """A wrapper class for the SentenceTransformer model."""

    def __init__(self, preloaded_model):
        """
        Initializes the embedder with a pre-loaded model object.
        """
        self.model = preloaded_model

    def get_embeddings(self, text_chunks):
        """Generates embeddings for a list of text chunks."""
        if not text_chunks:
            return np.array([])
        embeddings = self.model.encode(text_chunks, show_progress_bar=False)
        return np.array(embeddings).astype('float32')


# --- 4. Faiss Similarity Search ---
class SimilaritySearch:
    """A wrapper for Faiss index and search functionality."""

    def __init__(self, dimension):
        """Initializes the Faiss index."""
        self.index = faiss.IndexFlatL2(dimension)
        self.chunk_map = []  # To map index results back to text

    def build_index(self, embeddings, chunks, doc_name):
        """Adds embeddings to the Faiss index."""
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        if embeddings.shape[0] == 0:
            return

        # Faiss requires L2 normalized vectors for this calculation to be equivalent to cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Store the original text chunk and its document name
        for chunk in chunks:
            self.chunk_map.append({'doc': doc_name, 'chunk': chunk})

    def search(self, query_embeddings, k=1):
        """Searches the index for the most similar vectors."""
        if query_embeddings.ndim == 1:
            query_embeddings = np.expand_dims(query_embeddings, axis=0)

        if query_embeddings.shape[0] == 0:
            return np.array([]), np.array([])

        # Normalize the query vectors as well
        faiss.normalize_L2(query_embeddings)
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices
