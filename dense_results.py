import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

MODEL_E5 = 'intfloat/e5-large-v2'
MODEL_SPECTER = 'sentence-transformers/allenai-specter'
QUERIES_PATH = 'data/queries.txt'
EMB_E5_PATH = 'document_embeddings_e5_qrels.npy'
DOCIDS_E5_PATH = 'document_ids_e5_qrels.npy'
EMB_SPECTER_PATH = 'document_embeddings_specter_qrels.npy'
DOCIDS_SPECTER_PATH = 'document_ids_specter_qrels.npy'
OUTPUT_E5 = 'dense_results_e5.csv'
OUTPUT_SPECTER = 'dense_results_specter.csv'
TOP_K = 200

queries = pd.read_csv(QUERIES_PATH, sep='\t', names=['qid', 'query'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================
# Retrieval Function
# =============================
def run_dense_retrieval(model_name, emb_path, docid_path, output_path, format_query_fn):
    print(f"\nEncoding queries with {model_name}...")
    model = SentenceTransformer(model_name, device=device)

    doc_embeddings = torch.tensor(np.load(emb_path)).to(device)
    doc_ids = np.load(docid_path).astype(str)

    results = []
    for _, row in tqdm(queries.iterrows(), total=len(queries)):
        qid, query = row['qid'], row['query']
        formatted = format_query_fn(query)
        q_emb = model.encode(formatted, convert_to_tensor=True, device=device)
        scores = util.cos_sim(q_emb, doc_embeddings)[0]
        top_k = torch.topk(scores, k=TOP_K)

        for rank, (score, idx) in enumerate(zip(top_k.values, top_k.indices)):
            results.append({
                'qid': qid,
                'docid': doc_ids[idx],
                'score': score.item(),
                'rank': rank + 1
            })

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

run_dense_retrieval(
    model_name=MODEL_E5,
    emb_path=EMB_E5_PATH,
    docid_path=DOCIDS_E5_PATH,
    output_path=OUTPUT_E5,
    format_query_fn=lambda q: f"query: {q.strip().lower()}"
)

run_dense_retrieval(
    model_name=MODEL_SPECTER,
    emb_path=EMB_SPECTER_PATH,
    docid_path=DOCIDS_SPECTER_PATH,
    output_path=OUTPUT_SPECTER,
    format_query_fn=lambda q: q.strip()
)
