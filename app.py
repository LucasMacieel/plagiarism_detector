import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer

# Import our custom classes
from detector import (
    SentenceEmbedder,
    SimilaritySearch,
    ocr_pdf,
    chunk_text_by_sentence,
)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# --- Define the specific models to be pre-loaded ---
# 768 dimensions
ENGLISH_MODEL_NAME = 'all-mpnet-base-v2'
# 384 dimensions
MULTILINGUAL_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


# --- Pre-load the English and Multilingual models into memory ---
def load_models():
    """Loads the English and Multilingual models into a dictionary."""
    print("--- Pre-loading SBERT models ---")
    preloaded_models = {}
    try:
        # Load the English model
        print(f"Loading English model: {ENGLISH_MODEL_NAME}...")
        preloaded_models['en'] = SentenceTransformer(ENGLISH_MODEL_NAME)
        print(f"'{ENGLISH_MODEL_NAME}' loaded successfully.")

        # Load the multilingual model
        print(f"Loading Multilingual model: {MULTILINGUAL_MODEL_NAME}...")
        preloaded_models['multi'] = SentenceTransformer(MULTILINGUAL_MODEL_NAME)
        print(f"'{MULTILINGUAL_MODEL_NAME}' loaded successfully.")

    except Exception as e:
        print(f"Fatal error loading models: {e}")

    print("--- All models loaded. ---")
    return preloaded_models


# --- Global variables ---
PRELOADED_MODELS = load_models()

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'


# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_check():
    """Handles file uploads and displays plagiarism results."""
    if len(PRELOADED_MODELS) < 2:  # Check if both models loaded
        flash("One or more models could not be loaded. The service is unavailable.", "error")
        return render_template('index.html', results=None)

    if request.method == 'POST':
        # File handling logic (unchanged)
        if 'source_file' not in request.files or 'suspect_file' not in request.files:
            flash('Both source and suspect files are required!', 'error')
            return redirect(request.url)

        source_file = request.files['source_file']
        suspect_file = request.files['suspect_file']

        if source_file.filename == '' or suspect_file.filename == '':
            flash('Please select both files.', 'error')
            return redirect(request.url)

        if not (allowed_file(source_file.filename) and allowed_file(suspect_file.filename)):
            flash('Invalid file type. Please upload .txt or .pdf files.', 'error')
            return redirect(request.url)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(source_file.filename))
        suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(suspect_file.filename))
        source_file.save(source_path)
        suspect_file.save(suspect_path)

        try:
            source_text = ocr_pdf(source_path) if source_path.endswith(".pdf") else open(source_path, encoding='utf-8',
                                                                                         errors='ignore').read()
            if not source_text.strip():
                flash('Could not extract text from the source document.', 'warning')
                return redirect(request.url)

            # --- DYNAMIC MODEL SELECTION LOGIC (UPDATED) ---
            try:
                lang_code = detect(source_text[:1000])
            except LangDetectException:
                lang_code = 'en'  # Default to English if detection is uncertain

            print(f"Detected language: '{lang_code}'.")

            # Choose the model based on the detected language
            if lang_code == 'en':
                model = PRELOADED_MODELS['en']
            else:
                model = PRELOADED_MODELS['multi']

            # --- Initialize classes with the selected pre-loaded model ---
            # THIS IS THE FIX: Changed 'model_name=model' to 'preloaded_model=model'
            embedder = SentenceEmbedder(preloaded_model=model)
            embedding_dim = model.get_sentence_embedding_dimension()
            similarity_searcher = SimilaritySearch(dimension=embedding_dim)

            # --- Continue with the plagiarism check process (unchanged) ---
            source_chunks = chunk_text_by_sentence(source_text)
            if source_chunks:
                source_embeddings = embedder.get_embeddings(source_chunks)
                similarity_searcher.build_index(source_embeddings, source_chunks, source_file.filename)

            suspect_text = ocr_pdf(suspect_path) if suspect_path.endswith(".pdf") else open(suspect_path,
                                                                                            encoding='utf-8',
                                                                                            errors='ignore').read()
            suspect_chunks = chunk_text_by_sentence(suspect_text)

            if not suspect_chunks:
                flash('Could not extract text from the suspect document.', 'warning')
                return render_template('index.html', results=[])

            suspect_embeddings = embedder.get_embeddings(suspect_chunks)
            distances, indices = similarity_searcher.search(suspect_embeddings, k=1)

            report_results = []
            threshold = 0.6
            for i, chunk in enumerate(suspect_chunks):
                dist = distances[i][0]
                if dist < threshold:
                    cosine_similarity = 1 - (dist ** 2 / 2)
                    if cosine_similarity > 0.85:
                        idx = indices[i][0]
                        similar_chunk_info = similarity_searcher.chunk_map[idx]
                        report_results.append({
                            'suspect_chunk': chunk,
                            'source_doc': similar_chunk_info['doc'],
                            'source_chunk': similar_chunk_info['chunk'],
                            'similarity': f"{cosine_similarity:.2%}"
                        })

            return render_template('index.html', results=report_results)

        except Exception as e:
            flash(f"An error occurred during processing: {e}", "error")
            print(f"Error: {e}")
            return redirect(request.url)
        finally:
            if os.path.exists(source_path): os.remove(source_path)
            if os.path.exists(suspect_path): os.remove(suspect_path)

    return render_template('index.html', results=None)


if __name__ == '__main__':
    app.run(debug=True)
