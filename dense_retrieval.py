import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import ndcg_score
from tqdm import tqdm

QUERIES_PATH = 'data/queries.txt'
QRELS_PATH = 'data/qrels.txt'
SAVE_EMBEDDINGS_PATH = 'document_embeddings.npy'
SAVE_DOCIDS_PATH = 'document_ids.npy'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print('Loading precomputed document embeddings...')
document_embeddings = torch.tensor(np.load(SAVE_EMBEDDINGS_PATH)).to(device)
doc_ids = np.load(SAVE_DOCIDS_PATH)
doc_ids = doc_ids.astype(str)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

queries = pd.read_csv(QUERIES_PATH, sep='\t', names=['qid', 'query'])
qrels = pd.read_csv(QRELS_PATH, sep=' ', names=['qid', 'iter', 'docid', 'relevance'])
qrels['docid'] = qrels['docid'].astype(str)

available_docs_set = set(doc_ids)

print('Filtering qrels to available documents...')
qrels_filtered = qrels[qrels['docid'].isin(available_docs_set)]
print(f"Filtered qrels: {len(qrels_filtered)} entries remain.")

qrels_grouped = qrels_filtered.groupby('qid').apply(lambda g: dict(zip(g['docid'], g['relevance']))).to_dict()

results = []

print('Encoding queries and searching...')
for _, row in tqdm(queries.iterrows(), total=len(queries)):
    qid = row['qid']
    query = row['query']
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)

    cos_scores = util.cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cos_scores, k=100)

    for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
        results.append({
            'qid': qid,
            'docid': doc_ids[idx],
            'rank': rank + 1,
            'score': score.item()
        })

results_df = pd.DataFrame(results)
print(results_df.head())

ndcg_scores = []

for qid in results_df['qid'].unique():
    retrieved_docs = results_df[results_df['qid'] == qid]['docid'].tolist()
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

print(f"Mean nDCG@10 (Dense Retrieval, filtered): {mean_ndcg:.4f}")

results_df.to_csv('dense_retrieval_top100.csv', index=False)


print('Dense retrieval finished.')
