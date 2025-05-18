# This script assumes the following:
# - Your documents are in 'data/documents/*.jsonl'
# - Your queries are in 'data/queries.txt' (tab-separated)
# - Your qrels are in 'data/qrels.txt' (space-separated)
# - You want to train intfloat/e5-large-v2 using MultipleNegativesRankingLoss

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os
import json
from tqdm import tqdm

DOCUMENTS_DIR = 'data/documents'
QUERIES_PATH = 'data/queries.txt'
QRELS_PATH = 'data/qrels.txt'
OUTPUT_PATH = 'e5base-finetuned-model'
BATCH_SIZE = 8
EPOCHS = 100

print("Loading documents...")
doc_texts = {}
for file in os.listdir(DOCUMENTS_DIR):
    if file.endswith(".jsonl"):
        with open(os.path.join(DOCUMENTS_DIR, file), 'r') as f:
            for line in f:
                doc = json.loads(line)
                doc_id = str(doc['id'])
                title = doc.get('title') or ''
                abstract = doc.get('abstract') or ''
                fulltext = (title + ' ' + abstract).strip()
                if fulltext:
                    doc_texts[doc_id] = fulltext

print("Loading queries...")
queries = pd.read_csv(QUERIES_PATH, sep='\t', names=['qid', 'query'])
query_map = dict(zip(queries["qid"].astype(str), queries["query"]))


print("Building training pairs from qrels...")
examples = []
qrels = pd.read_csv(QRELS_PATH, sep=' ', names=['qid', 'snapshot', 'docid', 'rel'])
qrels = qrels[qrels.rel > 0]  # only positives
for _, row in tqdm(qrels.iterrows(), total=len(qrels)):
    qid = str(row.qid)
    docid = str(row.docid)
    if qid in query_map and docid in doc_texts:
        query = query_map[qid].strip()
        document = doc_texts[docid].strip()
        examples.append(InputExample(texts=[f"query: {query}", f"passage: {document}"]))

print(f"Loaded {len(examples)} positive pairs. Starting training...")
model = SentenceTransformer('intfloat/e5-base-v2')
train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=1000,
    show_progress_bar=True,
    output_path=OUTPUT_PATH
)

print(f"Model saved to {OUTPUT_PATH}")
