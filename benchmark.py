import os
from sklearn.metrics import precision_score, recall_score, f1_score

from main import chunk_text_by_sentence, SentenceEmbedder, SimilaritySearch, ocr_pdf

benchmark_cases = [
    {
        "source": "source_docs/source_document.pdf",
        "suspect": "suspect_docs/direct-plagiarism.pdf",
        "is_plagiarized": 1
    },
    {
        "source": "source_docs/source_document.pdf",
        "suspect": "suspect_docs/mosaic-plagiarism.pdf",
        "is_plagiarized": 1
    },
]

def evaluate_benchmark(cases, model_name='all-MiniLM-L6-v2', threshold=0.6):
    embedder = SentenceEmbedder(model_name=model_name)
    y_true, y_pred = [], []

    for case in cases:
        print(f"\n[TEST CASE] {case['suspect']} against {case['source']}")

        source_text = ocr_pdf(case["source"]) if case["source"].endswith(".pdf") else open(case["source"]).read()
        suspect_text = ocr_pdf(case["suspect"]) if case["suspect"].endswith(".pdf") else open(case["suspect"]).read()

        source_chunks = chunk_text_by_sentence(source_text)
        suspect_chunks = chunk_text_by_sentence(suspect_text)

        if not source_chunks or not suspect_chunks:
            print("Empty document. Skipping.")
            continue

        source_embeddings = embedder.get_embeddings(source_chunks)
        suspect_embeddings = embedder.get_embeddings(suspect_chunks)

        similarity_search = SimilaritySearch(dimension=source_embeddings.shape[1])
        similarity_search.build_index(source_embeddings, source_chunks, os.path.basename(case['source']))

        distances, indices = similarity_search.search(suspect_embeddings, k=1)

        found_plagiarism = False
        for i, dist in enumerate(distances):
            if dist[0] < threshold:
                found_plagiarism = True
                break

        y_true.append(case['is_plagiarized'])
        y_pred.append(1 if found_plagiarism else 0)

        result_str = "✅ Detected" if found_plagiarism else "❌ Not Detected"
        print(f"  Ground Truth: {case['is_plagiarized']} | Prediction: {result_str}")

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == '__main__':
    evaluate_benchmark(benchmark_cases)
