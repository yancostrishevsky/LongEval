# encode_documents_qrels.py

import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = 'data/documents/'
QRELS_PATH = 'data/qrels.txt'
SAVE_EMBEDDINGS_PATH = 'document_embeddings_qrels.npy'
SAVE_DOCIDS_PATH = 'document_ids_qrels.npy'

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 512

device = 'cpu'
model = SentenceTransformer(MODEL_NAME, device=device)

print('Loading relevant document IDs from qrels...')
qrels = pd.read_csv(QRELS_PATH, sep=' ', names=['qid', 'iter', 'docid', 'relevance'])
required_doc_ids = set(qrels['docid'].astype(str).tolist())
print(f"Need to find {len(required_doc_ids)} documents.")

documents = []
doc_ids = []

all_jsonl_files = [os.path.join(DOCUMENTS_DIR, f) for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.jsonl')]

print('Scanning documents...')
found_count = 0
for doc_file in tqdm(all_jsonl_files):
    with open(doc_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc['id'])
            if doc_id in required_doc_ids:
                title = doc.get('title') or ''
                abstract = doc.get('abstract') or ''
                text = (title + ' ' + abstract).strip()
                if text:
                    documents.append(text.lower())
                    doc_ids.append(doc_id)
                    found_count += 1
                    if found_count % 5000 == 0:
                        print(f"Found {found_count} relevant documents so far...")

print(f"Total found documents: {len(documents)} out of {len(required_doc_ids)} needed.")

print('Encoding relevant documents...')
all_embeddings = []

for i in tqdm(range(0, len(documents), BATCH_SIZE)):
    batch_texts = documents[i:i+BATCH_SIZE]
    embeddings = model.encode(batch_texts, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings)

np.save(SAVE_EMBEDDINGS_PATH, all_embeddings)
np.save(SAVE_DOCIDS_PATH, np.array(doc_ids))

print('Document embeddings (only qrels relevant) saved successfully.')
