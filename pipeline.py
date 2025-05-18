# run_pipeline.py

import os
import pandas as pd
from fusion_utils import run_fusion_of_experts, run_rrf
from reranker import run_cross_encoder_reranker
from evaluate import evaluate


RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

FILES = {
    'e5': 'dense_results_e5.csv',
    'specter': 'dense_results_specter.csv',
    'bm25': 'bm25_results.csv'
}


print("\n Running Fusion of Experts...")
fused_df = run_fusion_of_experts(
    e5_path=FILES['e5'],
    specter_path=FILES['specter'],
    bm25_path=FILES['bm25'],
    weights=(0.8, 0.1, 0.1)
)
fused_df.to_csv(os.path.join(RESULTS_DIR, 'fused_experts.csv'), index=False)


print("\n Running Reciprocal Rank Fusion (RRF)...")
rrf_df = run_rrf([
    FILES['e5'], FILES['specter'], FILES['bm25']
])
rrf_df.to_csv(os.path.join(RESULTS_DIR, 'rrf_fused.csv'), index=False)


print("\n Running Cross-Encoder Reranker...")
reranked_df = run_cross_encoder_reranker(
    retrieval_path=os.path.join(RESULTS_DIR, 'rrf_fused.csv'),
    documents_dir='data/documents/',
    queries_path='data/queries.txt',
    model_name='cross-encoder/ms-marco-MiniLM-L-12-v2'
)
reranked_df.to_csv(os.path.join(RESULTS_DIR, 'reranked_final.csv'), index=False)


print("\n Evaluating final results...")
score = evaluate(reranked_df, 'data/qrels.txt')
print(f"\n Final Mean nDCG@10: {score:.4f}")
