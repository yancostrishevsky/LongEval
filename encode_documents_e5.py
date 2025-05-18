import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
DOCUMENTS_DIR = 'data/documents/'
QRELS_PATH = 'data/qrels.txt'
SAVE_EMBEDDINGS_PATH = 'document_embeddings_finetuned_qrels.npy'
SAVE_DOCIDS_PATH = 'document_ids_finetuned.npy'

# Model Specter
MODEL_NAME = 'e5base-finetuned-model'
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SentenceTransformer(MODEL_NAME, device=device)

print("Loading qrels...")
qrels = pd.read_csv(QRELS_PATH, sep=' ', names=['qid', 'iter', 'docid', 'relevance'])
qrel_docids = set(qrels['docid'].astype(str))
print(f"Unique docids in qrels: {len(qrel_docids)}")

documents = []
doc_ids = []

print("Loading matching documents from disk...")
all_jsonl_files = [os.path.join(DOCUMENTS_DIR, f) for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.jsonl')]

for doc_file in tqdm(all_jsonl_files):
    with open(doc_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc.get('id'))
            if doc_id in qrel_docids:
                title = doc.get('title') or ''
                abstract = doc.get('abstract') or ''
                text = (title + ' ' + abstract).strip()
                if text:
                    formatted = f"passage: {text.lower()}"
                    documents.append(formatted)
                    doc_ids.append(doc_id)

print(f"Filtered documents to encode: {len(documents)}")

print("Encoding documents...")
all_embeddings = []

for i in tqdm(range(0, len(documents), BATCH_SIZE)):
    batch = documents[i:i + BATCH_SIZE]
    embeddings = model.encode(batch, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings)

np.save(SAVE_EMBEDDINGS_PATH, all_embeddings)
np.save(SAVE_DOCIDS_PATH, np.array(doc_ids))

print("Document embeddings saved successfully.")
