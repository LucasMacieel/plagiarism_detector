# Hybrid Plagiarism Detection System

A sophisticated, multilingual plagiarism detection tool built with Python and Flask. This application goes beyond simple text matching by implementing a hybrid similarity analysis, combining cutting-edge semantic search with traditional keyword analysis to provide a more accurate and nuanced plagiarism score.

The system is designed to handle PDF and TXT documents in both English and Portuguese, dynamically selecting the optimal language models for each task.

## Features
Hybrid Scoring System: Combines two methods for superior accuracy.

Semantic Similarity: Uses SBERT models (all-mpnet-base-v2 for English, paraphrase-multilingual-MiniLM-L12-v2 for other languages) to understand the contextual meaning of sentences, catching even sophisticated paraphrasing.

Keyword Similarity: Employs TF-IDF and Jaccard Similarity to analyze the overlap of important keywords, effectively detecting direct copy-pasting.

Multilingual Support: Automatically detects the language of the source document (English or Portuguese) and dynamically loads the appropriate, pre-trained language models for both sentence segmentation (spaCy) and embedding generation (SBERT).

PDF & TXT File Support: Processes text directly from .txt files or performs OCR on .pdf documents using Tesseract to extract text for analysis.

Efficient Backend: All language models (SBERT and spaCy) are pre-loaded into memory at startup to ensure fast and responsive processing for every user request.

High-Performance Indexing: Utilizes Facebook AI's Faiss library for incredibly fast similarity searches, even with large source documents.

Modern Web Interface: A clean, intuitive, and responsive dark-mode web UI built with Flask and Tailwind CSS for uploading documents and viewing detailed analysis reports.

## How It Works
The detection process follows a sophisticated pipeline:

File Upload: The user uploads a source document and a suspect document via the web interface.

OCR & Text Extraction: If a file is a PDF, Tesseract OCR is used to extract the raw text. For TXT files, the text is read directly.

Language Detection: The language of the source text is detected using langdetect.

Model Selection: Based on the detected language, the system selects the appropriate pre-loaded spaCy model for sentence splitting and the corresponding SBERT model for embedding.

Sentence Chunking: The text from both documents is intelligently split into overlapping chunks of sentences using the selected spaCy model. This provides context for the semantic analysis.

Keyword Analysis: A TfidfVectorizer is fitted on the source document's chunks to learn important vocabulary.

Semantic Indexing: The source document's chunks are converted into high-dimensional vectors (embeddings) using the selected SBERT model. These vectors are then indexed into a Faiss data structure for rapid searching.

Hybrid Comparison: For each chunk in the suspect document:
a. An embedding is generated.
b. Faiss finds the most semantically similar chunk from the source document.
c. A Semantic Score (Cosine Similarity) is calculated between the two chunks.
d. The top keywords are extracted from both chunks, and a Keyword Score (Jaccard Similarity) is calculated.
e. The two scores are combined into a final Hybrid Score (weighted 70% semantic, 30% keyword).

Report Generation: Matches with a Hybrid Score above a set threshold are presented to the user in a detailed, side-by-side report.

## Tech Stack
### Backend & Machine Learning

Framework: Flask

Semantic Embeddings: sentence-transformers (SBERT)

English Model: all-mpnet-base-v2

Multilingual Model: paraphrase-multilingual-MiniLM-L12-v2

Sentence Segmentation: spaCy

English Model: en_core_web_sm

Portuguese Model: pt_core_news_sm

Similarity Search: faiss-cpu (Facebook AI Similarity Search)

Keyword Extraction: scikit-learn (TfidfVectorizer)

OCR: pytesseract & PyMuPDF (fitz)

Language Detection: langdetect

Core Libraries: numpy, torch

### Frontend

Templating: Jinja2 (via Flask)

Styling: Tailwind CSS

## Setup and Installation
Follow these steps to get the project running on your local machine.

1. Prerequisites
You must have Google's Tesseract OCR Engine installed on your system.

Windows: Download and run the installer from the Tesseract at UB Mannheim page. Important: Make sure to add the Tesseract installation directory to your system's PATH environment variable.

macOS (Homebrew): brew install tesseract tesseract-lang

Linux (Debian/Ubuntu): sudo apt-get install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng

2. Clone the Repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/LucasMacieel/plagiarism_detector.git)
cd plagiarism_detector

3. Set Up a Virtual Environment
It is highly recommended to use a virtual environment.

python -m venv venv

On Windows: venv\Scripts\activate
On macOS/Linux: source venv/bin/activate

4. Install Python Dependencies
Install all required libraries from the requirements.txt file.

pip install -r requirements.txt

5. Download spaCy Models
Download the small language models for English and Portuguese that are used for sentence segmentation.

python -m spacy download en_core_web_sm
python -m spacy download pt_core_news_sm

6. Run the Application
Once all dependencies are installed, start the Flask server.

python app.py

The application will start, and the SBERT and spaCy models will be pre-loaded into memory (this may take a minute on the first run). Once you see the "All models loaded" message, open your web browser and navigate to:

https://www.google.com/search?q=http://127.0.0.1:5000

You can now upload your documents and start detecting plagiarism!

## License
This project is licensed under the MIT License. See the LICENSE file for details.
