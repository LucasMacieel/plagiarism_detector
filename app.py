import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from detector import (
    SentenceEmbedder,
    SimilaritySearch,
    ocr_pdf,
    chunk_text_by_sentence,
)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'  # Change this for production


# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Pre-load Models ---
# This is done once when the app starts to avoid reloading on each request.
print("Initializing Sentence Embedder... This may take a moment.")
try:
    embedder = SentenceEmbedder(model_name='all-MiniLM-L6-v2')
    print("Sentence Embedder loaded successfully.")
except Exception as e:
    print(f"Error loading Sentence Embedder: {e}")
    embedder = None


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_check():
    """Handles file uploads and displays plagiarism results."""
    if embedder is None:
        flash('The Sentence Embedder model could not be loaded. The application cannot proceed.', 'error')
        return render_template('index.html', results=None)

    if request.method == 'POST':
        # --- File Handling ---
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

        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        source_filename = secure_filename(source_file.filename)
        suspect_filename = secure_filename(suspect_file.filename)

        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
        suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_filename)

        source_file.save(source_path)
        suspect_file.save(suspect_path)

        # --- Plagiarism Detection Logic ---
        try:
            # 1. Initialize Similarity Search
            # The dimension (384) is specific to the 'all-MiniLM-L6-v2' model.
            similarity_searcher = SimilaritySearch(dimension=384)

            # 2. Process and Index Source Document
            print(f"Processing source document: {source_filename}")
            source_text = ocr_pdf(source_path) if source_path.endswith(".pdf") else open(source_path,
                                                                                         encoding='utf-8').read()
            source_chunks = chunk_text_by_sentence(source_text)

            if source_chunks:
                source_embeddings = embedder.get_embeddings(source_chunks)
                similarity_searcher.build_index(source_embeddings, source_chunks, source_filename)
                print(f"Indexed {similarity_searcher.index.ntotal} chunks from {source_filename}.")
            else:
                flash('Could not extract text from the source document.', 'warning')

            # 3. Process Suspect Document
            print(f"Processing suspect document: {suspect_filename}")
            suspect_text = ocr_pdf(suspect_path) if suspect_path.endswith(".pdf") else open(suspect_path,
                                                                                            encoding='utf-8').read()
            suspect_chunks = chunk_text_by_sentence(suspect_text)

            if not suspect_chunks:
                flash('Could not extract text from the suspect document.', 'warning')
                return render_template('index.html', results=[])

            suspect_embeddings = embedder.get_embeddings(suspect_chunks)

            # 4. Perform Search and Report Findings
            print("Performing similarity search...")
            distances, indices = similarity_searcher.search(suspect_embeddings, k=1)

            # L2 distance threshold for normalized vectors. A smaller value means higher similarity.
            threshold = 0.6
            report_results = []

            for i, chunk in enumerate(suspect_chunks):
                dist = distances[i][0]
                if dist < threshold:
                    idx = indices[i][0]
                    # For normalized vectors, cosine similarity = 1 - (L2_distance^2 / 2)
                    cosine_similarity = 1 - (dist ** 2 / 2)

                    if cosine_similarity > 0.85:  # Additional filter for higher confidence
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
            # Clean up uploaded files
            if os.path.exists(source_path):
                os.remove(source_path)
            if os.path.exists(suspect_path):
                os.remove(suspect_path)

    # Handle GET request
    return render_template('index.html', results=None)


if __name__ == '__main__':
    app.run(debug=True)
