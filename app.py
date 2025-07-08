import os

from flask import Flask, request, render_template, redirect, flash
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

# Import all necessary classes and functions from our detector
from detector import (
    SentenceEmbedder,
    SimilaritySearch,
    ocr_pdf,
    chunk_text_by_sentence,
    KeywordExtractor,
    calculate_jaccard_similarity,
)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
ENGLISH_MODEL_NAME = 'all-mpnet-base-v2'
MULTILINGUAL_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


# --- Pre-load SBERT models ---
def load_sbert_models():
    print("--- Pre-loading SBERT models ---")
    models = {}
    try:
        models['en'] = SentenceTransformer(ENGLISH_MODEL_NAME)
        models['multi'] = SentenceTransformer(MULTILINGUAL_MODEL_NAME)
    except Exception as e:
        print(f"Fatal error loading SBERT models: {e}")
    return models


PRELOADED_SBERT_MODELS = load_sbert_models()

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_and_check():
    if len(PRELOADED_SBERT_MODELS) < 2:
        flash("Models could not be loaded. The service is unavailable.", "error")
        return render_template('index.html', results=None)

    if request.method == 'POST':
        if 'source_file' not in request.files or 'suspect_file' not in request.files:
            return redirect(request.url)
        source_file = request.files['source_file']
        suspect_file = request.files['suspect_file']
        if not source_file.filename or not suspect_file.filename:
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
                flash('Could not extract text from source document.', 'warning')
                return redirect(request.url)

            try:
                lang_code = detect(source_text[:1500])
            except LangDetectException:
                lang_code = 'en'

            model = PRELOADED_SBERT_MODELS['en'] if lang_code == 'en' else PRELOADED_SBERT_MODELS['multi']

            embedder = SentenceEmbedder(preloaded_model=model)
            similarity_searcher = SimilaritySearch(dimension=model.get_sentence_embedding_dimension())

            source_chunks = chunk_text_by_sentence(source_text, lang_code=lang_code)
            if not source_chunks:
                flash('Could not process source document into chunks.', 'warning')
                return redirect(request.url)

            keyword_extractor = KeywordExtractor(corpus=source_chunks)
            source_embeddings = embedder.get_embeddings(source_chunks)
            similarity_searcher.build_index(source_embeddings, source_chunks, source_file.filename)

            suspect_text = ocr_pdf(suspect_path) if suspect_path.endswith(".pdf") else open(suspect_path,
                                                                                            encoding='utf-8',
                                                                                            errors='ignore').read()
            suspect_chunks = chunk_text_by_sentence(suspect_text, lang_code=lang_code)
            if not suspect_chunks:
                flash('Could not process suspect document.', 'warning')
                return render_template('index.html', results=[])

            suspect_embeddings = embedder.get_embeddings(suspect_chunks)
            distances, indices = similarity_searcher.search(suspect_embeddings, k=1)

            report_results = []
            hybrid_threshold = 0.75  # Combined score threshold

            for i, suspect_chunk in enumerate(suspect_chunks):
                dist = distances[i][0]
                semantic_score = 1 - (dist ** 2 / 2)

                if semantic_score > 0.65:  # Pre-filter for semantic relevance
                    idx = indices[i][0]
                    source_chunk_info = similarity_searcher.chunk_map[idx]
                    source_chunk = source_chunk_info['chunk']

                    source_keywords = keyword_extractor.get_top_keywords(source_chunk)
                    suspect_keywords = keyword_extractor.get_top_keywords(suspect_chunk)
                    keyword_score = calculate_jaccard_similarity(source_keywords, suspect_keywords)

                    hybrid_score = (0.7 * semantic_score) + (0.3 * keyword_score)

                    if hybrid_score > hybrid_threshold:
                        report_results.append({
                            'suspect_chunk': suspect_chunk,
                            'source_chunk': source_chunk,
                            'source_doc': source_chunk_info['doc'],
                            'semantic_score': f"{semantic_score:.1%}",
                            'keyword_score': f"{keyword_score:.1%}",
                            'hybrid_score': f"{hybrid_score:.1%}"
                        })

            report_results.sort(key=lambda x: float(x['hybrid_score'].strip('%')), reverse=True)
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
