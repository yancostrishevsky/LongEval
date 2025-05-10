import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd

# === Ścieżki ===
DOCUMENTS_DIR = "data/documents/"
QRELS_PATH = "data/qrels.txt"
SAVE_EMBEDDINGS_PATH = "document_embeddings_filtered.npy"
SAVE_DOCIDS_PATH = "document_ids_filtered.npy"

# === Ustawienia ===
MODEL_NAME = "intfloat/e5-base-v2"
BATCH_SIZE = 128
NOISE_DOCS = 100000  # liczba losowych dokumentów do dodania

# === Wczytaj model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer(MODEL_NAME, device=device)

# === Wczytaj dokumenty z qrels
print("Loading qrels...")
qrels = pd.read_csv(QRELS_PATH, sep=" ", names=["qid", "iter", "docid", "relevance"])
qrel_doc_ids = set(qrels["docid"].astype(str))

# === Wczytaj dokumenty z JSONL
print("Loading documents from JSONL files...")
all_docs = {}
all_jsonl_files = [os.path.join(DOCUMENTS_DIR, f) for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".jsonl")]

for doc_file in tqdm(all_jsonl_files):
    with open(doc_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc.get("id"))
            if not doc_id or doc_id in all_docs:
                continue
            title = doc.get("title") or ""
            abstract = doc.get("abstract") or ""
            text = (title + " " + abstract).strip()
            if text:
                all_docs[doc_id] = text.lower()

print(f"Total unique documents loaded: {len(all_docs)}")

# === Wybierz dokumenty z qrels
filtered_docs = {doc_id: all_docs[doc_id] for doc_id in qrel_doc_ids if doc_id in all_docs}
print(f"Documents from qrels found: {len(filtered_docs)}")

# === Dodaj losowe dokumenty jako "noise"
remaining_ids = list(set(all_docs.keys()) - set(filtered_docs.keys()))
noise_ids = random.sample(remaining_ids, min(NOISE_DOCS, len(remaining_ids)))

for doc_id in noise_ids:
    filtered_docs[doc_id] = all_docs[doc_id]

print(f"Final document count (qrels + noise): {len(filtered_docs)}")

# === Kodowanie
texts = list(filtered_docs.values())
doc_ids = list(filtered_docs.keys())
embeddings = []

print("Encoding documents...")
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_embeds = model.encode(batch_texts, batch_size=BATCH_SIZE, convert_to_numpy=True)
    embeddings.append(batch_embeds)

embeddings = np.vstack(embeddings)

# === Zapisz
np.save(SAVE_EMBEDDINGS_PATH, embeddings)
np.save(SAVE_DOCIDS_PATH, np.array(doc_ids))

print(f"Embeddings saved. Shape: {embeddings.shape}")
