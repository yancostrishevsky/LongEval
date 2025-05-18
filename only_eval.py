# eval_only.py

from evaluate import evaluate
import pandas as pd

results_df = pd.read_csv('results/reranked_final.csv')
score = evaluate(results_df, 'data/qrels.txt')

print(f"Final Mean nDCG@10: {score:.4f}")
