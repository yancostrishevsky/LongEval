# encode_documents.py

import os
import json
import random
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DOCUMENTS_DIR = 'data/documents/'
SAVE_EMBEDDINGS_PATH = 'document_embeddings.npy'
SAVE_DOCIDS_PATH = 'document_ids.npy'

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MAX_DOCUMENTS = 100000
BATCH_SIZE = 1024

model = SentenceTransformer(MODEL_NAME, device='cpu')

print('Loading documents (limited subset)...')
documents = []
doc_ids = []

all_jsonl_files = [os.path.join(DOCUMENTS_DIR, f) for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.jsonl')]

for doc_file in tqdm(all_jsonl_files):
    with open(doc_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            title = doc.get('title') or ''
            abstract = doc.get('abstract') or ''
            text = (title + ' ' + abstract).strip()
            if text:
                documents.append(text.lower())
                doc_ids.append(doc['id'])

print(f"Total loaded documents before sampling: {len(documents)}")

if len(documents) > MAX_DOCUMENTS:
    sampled_indices = random.sample(range(len(documents)), MAX_DOCUMENTS)
    documents = [documents[i] for i in sampled_indices]
    doc_ids = [doc_ids[i] for i in sampled_indices]
    print(f"Sampled {MAX_DOCUMENTS} documents.")

print(f"Final number of documents to encode: {len(documents)}")

print('Encoding documents...')
all_embeddings = []

for i in tqdm(range(0, len(documents), BATCH_SIZE)):
    batch_texts = documents[i:i+BATCH_SIZE]
    embeddings = model.encode(batch_texts, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings)

np.save(SAVE_EMBEDDINGS_PATH, all_embeddings)
np.save(SAVE_DOCIDS_PATH, np.array(doc_ids))

print('Document embeddings saved successfully.')
