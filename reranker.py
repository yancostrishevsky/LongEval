import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import json
import os

RESULTS_PATH = 'dense_results_top50.csv'
QUERIES_PATH = 'data/queries.txt'
QRELS_PATH = 'data/qrels.txt'
DOCUMENTS_DIR = 'data/documents/'

reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

results_df = pd.read_csv(RESULTS_PATH)
queries = pd.read_csv(QUERIES_PATH, sep='\t', names=['qid', 'query'])
qrels = pd.read_csv(QRELS_PATH, sep=' ', names=['qid', 'iter', 'docid', 'relevance'])

query_map = dict(zip(queries['qid'], queries['query']))
document_texts = {}

print("Loading document texts...")
all_jsonl_files = [os.path.join(DOCUMENTS_DIR, f) for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.jsonl')]
for file in tqdm(all_jsonl_files):
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            docid = str(doc['id'])
            title = doc.get('title') or ''
            abstract = doc.get('abstract') or ''
            text = (title + " " + abstract).strip()
            if text:
                document_texts[docid] = text

print(f"Loaded {len(document_texts)} documents.")
print("Loading cross-encoder model...")
model = CrossEncoder(reranker_model_name, device=device)

reranked_results = []

print("Reranking with cross-encoder...")
for qid in tqdm(results_df['qid'].unique()):
    query = query_map.get(qid, None)
    if not query:
        continue

    top_docs = results_df[results_df['qid'] == qid]['docid'].astype(str).tolist()
    pairs = []

    for docid in top_docs:
        text = document_texts.get(docid)
        if text:
            pairs.append([query, text])
        else:
            print(f"Warning: Missing text for docid {docid}")

    if not pairs:
        print(f"Skipping qid {qid}: No valid document texts found.")
        continue

    scores = model.predict(pairs, batch_size=32)

    for rank, (docid, score) in enumerate(sorted(zip(top_docs, scores), key=lambda x: x[1], reverse=True)):
        reranked_results.append({
            'qid': qid,
            'docid': docid,
            'rank': rank + 1,
            'score': float(score)
        })

if not reranked_results:
    raise RuntimeError("No reranked results were produced. Check document IDs and query mappings.")

reranked_df = pd.DataFrame(reranked_results)
reranked_df.to_csv('reranked_results.csv', index=False)
print("Reranked results saved.")

qrels['docid'] = qrels['docid'].astype(str)
qrels_grouped = qrels.groupby('qid').apply(lambda g: dict(zip(g['docid'], g['relevance']))).to_dict()

ndcg_scores = []
for qid in reranked_df['qid'].unique():
    retrieved_docs = reranked_df[reranked_df['qid'] == qid]['docid'].tolist()
    true_relevances = []
    scores = []

    if qid in qrels_grouped:
        qrel_docs = qrels_grouped[qid]
        for doc in retrieved_docs:
            rel = qrel_docs.get(doc, 0)
            true_relevances.append(rel)
            scores.append(1)

        if sum(true_relevances) > 0:
            ndcg = ndcg_score([true_relevances], [scores])
            ndcg_scores.append(ndcg)

if ndcg_scores:
    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
else:
    mean_ndcg = 0.0

print(f"Mean nDCG@10 (Cross-Encoder Reranking): {mean_ndcg:.4f}")
