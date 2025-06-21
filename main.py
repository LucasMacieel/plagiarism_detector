import fitz
import pytesseract
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import nltk
import cv2


# --- Installation ---
# 1. Tesseract-OCR engine: https://github.com/tesseract-ocr/tesseract
# 2. NLTK's sentence tokenizer data (run this once in a Python interpreter):
#    import nltk
#    nltk.download('punkt')
#    nltk.download('punkt_tab')
# 3. Python libraries (see requirements.txt section below).

# --- 1. Document Processing and OCR ---
def ocr_pdf(pdf_path):
    """Performs OCR on each page of a PDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            preprocessed_img = preprocess_image(img)
            text += pytesseract.image_to_string(preprocessed_img)
        doc.close()
    except Exception as e:
        print(f"OCR failed for {pdf_path}. Error: {e}")
    return text


# --- 2. Text Preprocessing (Sentence-Aware) ---

def chunk_text_by_sentence(text, sentences_per_chunk=1):
    """
    Splits text into sentences and then groups them into chunks.
    """
    # First, split the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Clean up sentences
    sentences = [s.strip().replace("\n", " ") for s in sentences if s.strip()]

    # Group sentences into chunks
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks


# --- 3. Sentence-Transformers Embeddings ---

class SentenceEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)
        print(f"SentenceTransformer model '{model_name}' loaded.")

    def get_embeddings(self, text_chunks):
        """
        Generates embeddings for a list of text chunks.
        """
        embeddings = self.model.encode(text_chunks, show_progress_bar=True)
        return np.array(embeddings)


# --- 4. Faiss Similarity Search ---

class SimilaritySearch:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunk_map = []

    def build_index(self, embeddings, chunks, doc_name):
        """
        Adds embeddings to the Faiss index and maps them to their original text and document.
        """
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        faiss.normalize_L2(embeddings)  # Normalize vectors for cosine similarity
        self.index.add(embeddings)
        for chunk in chunks:
            self.chunk_map.append({'doc': doc_name, 'chunk': chunk})

    def search(self, query_embeddings, k=1):
        """
        Searches the index for the most similar vectors.
        """
        if query_embeddings.ndim == 1:
            query_embeddings = np.expand_dims(query_embeddings, axis=0)

        faiss.normalize_L2(query_embeddings)  # Normalize query vector
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices


def preprocess_image(pil_img):
    """Convert PIL image to OpenCV format and apply preprocessing."""
    # Convert PIL image to OpenCV format
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize to improve DPI (simulate 300+ dpi if needed)
    scale_percent = 150  # e.g., 150% zoom
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    # Apply adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 15, 10)

    # Optional: apply denoising
    img = cv2.fastNlMeansDenoising(img, h=30)

    return Image.fromarray(img)


# --- Main Prototype ---

def main():
    # --- Setup ---
    nltk.data.find('tokenizers/punkt')

    print("Initializing Sentence Embedder...")
    embedder = SentenceEmbedder(model_name='all-MiniLM-L6-v2')
    similarity_searcher = SimilaritySearch(384)

    # --- Indexing Source Documents ---
    print("\n--- Indexing Source Documents ---")
    source_doc_path = "source_docs/source_document_PTBR.pdf"
    print(f"Processing {source_doc_path}...")
    text = ocr_pdf(source_doc_path) if source_doc_path.endswith(".pdf") else open(source_doc_path).read()
    chunks = chunk_text_by_sentence(text)
    if chunks:
        embeddings = embedder.get_embeddings(chunks)
        similarity_searcher.build_index(embeddings, chunks, os.path.basename(source_doc_path))

    print(f"\nIndexed {similarity_searcher.index.ntotal} chunks from {source_doc_path}.")

    # --- Checking a Suspect Document ---
    print("\n--- Checking Suspect Document for Plagiarism ---")
    suspect_path = "suspect_docs/plagiarized_document_PTBR.pdf"

    suspect_text = ocr_pdf(suspect_path)
    suspect_chunks = chunk_text_by_sentence(suspect_text)

    if not suspect_chunks:
        print("No text could be extracted from the suspect document.")
        return

    for suspect_chunk in suspect_chunks:
        if len(suspect_chunk.split()) < 3:
            suspect_chunks.remove(suspect_chunk)

    suspect_embeddings = embedder.get_embeddings(suspect_chunks)

    # --- Perform Similarity Search ---
    print("\nPerforming similarity search...")
    distances, indices = similarity_searcher.search(suspect_embeddings, k=1)

    # --- Report Findings ---
    print("\n--- Plagiarism Detection Report ---")
    print("=" * 50)
    # L2 distance threshold for normalized vectors. A smaller value means higher similarity.
    threshold = 0.6
    similar_chunks = []
    for i, chunk in enumerate(suspect_chunks):
        dist = distances[i][0]

        if dist < threshold:
            idx = indices[i][0]
            similar_chunk_info = similarity_searcher.chunk_map[idx]
            similar_chunks.append(similar_chunk_info)
            # For normalized vectors, cosine similarity = 1 - (L2_distance^2 / 2)
            cosine_similarity = 1 - (dist ** 2 / 2)

            print(f"Potential Plagiarism Found!\n")
            print(f"  Suspect Chunk:      '{chunk}'")
            print(f"  Source Document:    '{similar_chunk_info['doc']}'")
            print(f"  Similar Source Chunk: '{similar_chunk_info['chunk']}'")
            print(f"  Cosine Similarity:  {cosine_similarity:.4f} (L2 Distance: {dist:.4f})\n")
    if len(similar_chunks) == 0:
        print("No Plagiarism Found!\n")


if __name__ == '__main__':
    main()