# reranker.py

import os
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import torch

def run_cross_encoder_reranker(retrieval_path, documents_dir, queries_path, model_name):
    print("Loading queries and retrieval results...")
    queries = pd.read_csv(queries_path, sep='\t', names=['qid', 'query'])
    query_map = dict(zip(queries['qid'], queries['query']))

    retrieval = pd.read_csv(retrieval_path)

    print("Loading document texts...")
    doc_texts = {}
    for fname in os.listdir(documents_dir):
        if fname.endswith(".jsonl"):
            with open(os.path.join(documents_dir, fname), 'r') as f:
                for line in f:
                    doc = json.loads(line)
                    doc_id = str(doc['id'])
                    title = doc.get('title') or ''
                    abstract = doc.get('abstract') or ''
                    if not isinstance(title, str):
                        title = ''
                    if not isinstance(abstract, str):
                        abstract = ''
                    text = (title + ' ' + abstract).strip()
                    if text:
                        doc_texts[doc_id] = text

    model = CrossEncoder(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

    print("Preparing pairs...")
    rerank_data = []
    for _, row in tqdm(retrieval.iterrows(), total=len(retrieval)):
        qid, docid = row['qid'], str(row['docid'])
        if qid in query_map and docid in doc_texts:
            rerank_data.append((qid, docid, query_map[qid], doc_texts[docid]))

    print(f"Scoring {len(rerank_data)} query-document pairs...")
    scores = model.predict([(q, d) for _, _, q, d in rerank_data], batch_size=32)

    output = pd.DataFrame(
        [(qid, docid, score) for (qid, docid, _, _), score in zip(rerank_data, scores)],
        columns=['qid', 'docid', 'score']
    )
    output = output.sort_values(by=['qid', 'score'], ascending=[True, False])
    output['rank'] = output.groupby('qid').cumcount() + 1
    return output
