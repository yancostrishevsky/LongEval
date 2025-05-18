import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import time

QUERIES_PATH = 'data/queries.txt'
QRELS_PATH = 'data/qrels.txt'
SAVE_EMBEDDINGS_PATH = 'document_embeddings_finetuned_qrels.npy'
SAVE_DOCIDS_PATH = 'document_ids_finetuned.npy'

MODEL_NAME = 'e5base-finetuned-model'

K = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print('Loading precomputed document embeddings...')
document_embeddings = torch.tensor(np.load(SAVE_EMBEDDINGS_PATH)).to(device)
doc_ids = np.load(SAVE_DOCIDS_PATH).astype(str)

model = SentenceTransformer(MODEL_NAME, device=device)

queries = pd.read_csv(QUERIES_PATH, sep='\t', names=['qid', 'query'])
qrels = pd.read_csv(QRELS_PATH, sep=' ', names=['qid', 'iter', 'docid', 'relevance'])
qrels['docid'] = qrels['docid'].astype(str)

available_docs_set = set(doc_ids)
qrels_filtered = qrels[qrels['docid'].isin(available_docs_set)]
print(f"Filtered qrels: {len(qrels_filtered)} entries remain.")

qrels_grouped = qrels_filtered.groupby('qid', group_keys=False).apply(lambda g: dict(zip(g['docid'], g['relevance']))).to_dict()

results = []
print('Encoding queries and searching...')
for _, row in tqdm(queries.iterrows(), total=len(queries)):
    qid = row['qid']
    query = row['query']
    formatted_query = f"query: {query.strip().lower()}"
    query_embedding = model.encode(formatted_query, convert_to_tensor=True, device=device)

    cos_scores = util.cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cos_scores, k=K)

    for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
        results.append({
            'qid': qid,
            'docid': doc_ids[idx],
            'rank': rank + 1,
            'score': score.item()
        })

results_df = pd.DataFrame(results)
results_df.to_csv('dense_results_top50.csv', index=False)
print("Dense retrieval completed.")


import pandas as pd
import numpy as np
import json
import os
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from sklearn.metrics import ndcg_score
import time

QUERIES_PATH = 'data/queries.txt'
QRELS_PATH = 'data/qrels.txt'
DOCUMENTS_DIR = 'data/documents/'
RETRIEVAL_RESULTS_PATH = 'dense_results_top50.csv'

queries = pd.read_csv(QUERIES_PATH, sep='\t', names=['qid', 'query'])
qrels = pd.read_csv(QRELS_PATH, sep=' ', names=['qid', 'iter', 'docid', 'relevance'])
qrels['docid'] = qrels['docid'].astype(str)

qrels_grouped = qrels.groupby('qid', group_keys=False).apply(lambda g: dict(zip(g['docid'], g['relevance']))).to_dict()
retrieval_df = pd.read_csv(RETRIEVAL_RESULTS_PATH)

print("Loading document texts...")
document_texts = {}
for fname in os.listdir(DOCUMENTS_DIR):
    if fname.endswith(".jsonl"):
        with open(os.path.join(DOCUMENTS_DIR, fname), 'r') as f:
            for line in f:
                doc = json.loads(line)
                doc_id = str(doc['id'])
                title = doc.get('title') or ''
                abstract = doc.get('abstract') or ''
                document_texts[doc_id] = (title + ' ' + abstract).strip()

print(f"Loaded {len(document_texts)} document texts")

print("Loading cross-encoder model...")
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

print("Reranking with cross-encoder...")
pairs = []
pairs_metadata = []

missing_docs = 0
for _, row in retrieval_df.iterrows():
    qid, docid = row['qid'], str(row['docid'])
    query_row = queries[queries['qid'] == qid]
    if not query_row.empty and docid in document_texts:
        query_text = query_row.iloc[0]['query']
        pairs.append((query_text, document_texts[docid]))
        pairs_metadata.append((qid, docid))
    else:
        missing_docs += 1

print(f"Prepared {len(pairs)} query-document pairs (missing: {missing_docs})")

if not pairs:
    raise ValueError("No query-document pairs to rerank. Check document matching.")

start = time.time()
scores = model.predict(pairs, batch_size=32)
end = time.time()
print(f"Reranking completed in {end - start:.2f} seconds")

reranked = pd.DataFrame(pairs_metadata, columns=['qid', 'docid'])
reranked['score'] = scores
reranked = reranked.sort_values(by=['qid', 'score'], ascending=[True, False])
reranked['rank'] = reranked.groupby('qid').cumcount() + 1
reranked.to_csv('reranked_results.csv', index=False)

ndcg_scores = []
for qid in reranked['qid'].unique():
    docs = reranked[reranked['qid'] == qid]['docid'].tolist()
    true_relevances = []
    pred_scores = reranked[reranked['qid'] == qid]['score'].tolist()

    qrel_docs = qrels_grouped.get(qid, {})
    for doc in docs:
        true_relevances.append(qrel_docs.get(doc, 0))

    if sum(true_relevances) > 0:
        ndcg = ndcg_score([true_relevances], [pred_scores])
        ndcg_scores.append(ndcg)

mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
print(f"Mean nDCG@10 (Cross-Encoder Reranking): {mean_ndcg:.4f}")
