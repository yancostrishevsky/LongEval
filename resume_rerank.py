#!/usr/bin/env python3
import gzip
import os
import json
import heapq
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm

import click

def get_topk_similar_docs(query_vec, doc_embeddings, doc_ids, topk=100, batch_size=1000, device='cpu'):
    top_candidates = []

    for i in range(0, len(doc_embeddings), batch_size):
        batch_emb = torch.tensor(doc_embeddings[i:i+batch_size]).to(device)
        with torch.no_grad():
            scores = util.cos_sim(query_vec, batch_emb)[0].cpu().numpy()
        for j, score in enumerate(scores):
            heapq.heappush(top_candidates, (score, i + j))
            if len(top_candidates) > topk:
                heapq.heappop(top_candidates)

    top_candidates.sort(reverse=True)
    return [(score, doc_ids[idx]) for score, idx in top_candidates]

@click.command()
@click.option('--queries', type=click.Path(exists=True), required=True, help='Plik z zapytaniami TSV')
@click.option('--doc-embeddings', type=click.Path(exists=True), required=True, help='Plik .npy z embeddingami dokumentów')
@click.option('--doc-ids', type=click.Path(exists=True), required=True, help='Plik .npy z ID dokumentów')
@click.option('--documents-dir', type=click.Path(exists=True), required=True, help='Folder z dokumentami JSONL')
@click.option('--output', type=click.Path(), required=True, help='Folder wyjściowy z plikiem run.txt.gz')
@click.option('--topk', default=100, help='Liczba dokumentów do rerankingu na zapytanie')
@click.option('--batch-size', default=16, help='Batch size do rerankingu')
@click.option('--dense-batch', default=1000, help='Batch size do dense retrieval (na GPU)')
def run_rerank(queries, doc_embeddings, doc_ids, documents_dir, output, topk, batch_size, dense_batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, 'run.txt.gz')

    print("Loading document embeddings...")
    doc_emb = np.load(doc_embeddings)
    doc_ids_arr = np.load(doc_ids).astype(str)

    print("Loading queries...")
    queries_df = pd.read_csv(queries, sep='\t', names=['qid', 'query'])
    query_encoder = SentenceTransformer('intfloat/e5-base-v2', device=device)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)

    print("Preparing document texts...")
    doc_texts = {}
    for fname in os.listdir(documents_dir):
        if fname.endswith('.jsonl'):
            with open(os.path.join(documents_dir, fname)) as f:
                for line in f:
                    d = json.loads(line)
                    text = (d.get('title') or '') + ' ' + (d.get('abstract') or '')
                    if text.strip():
                        doc_texts[str(d['id'])] = text.strip()

    print("Running dense + reranking...")
    with gzip.open(output_path, 'wt') as outf:
        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
            qid, qtext = row['qid'], row['query']
            qvec = query_encoder.encode(f"query: {qtext.strip().lower()}", convert_to_tensor=True, device=device)

            top_docs = get_topk_similar_docs(qvec, doc_emb, doc_ids_arr, topk=topk, batch_size=dense_batch, device=device)
            pairs = [[qtext, doc_texts.get(docid, '')] for _, docid in top_docs]
            rerank_scores = reranker.predict(pairs, batch_size=batch_size)
            ranked = sorted(zip([docid for _, docid in top_docs], rerank_scores), key=lambda x: x[1], reverse=True)

            for rank, (docid, score) in enumerate(ranked):
                outf.write(f"{qid} Q0 {docid} {rank+1} {score} e5-reranked\n")

    print(f"✅ Saved TREC run file to: {output_path}")

if __name__ == '__main__':
    run_rerank()
