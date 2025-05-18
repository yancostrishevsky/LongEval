import pandas as pd
from evaluate import evaluate

# Wczytaj run
run_df = pd.read_csv("output/2024-11/run.txt.gz", sep=' ', names=['qid', 'Q0', 'docid', 'rank', 'score', 'tag'])

# Ścieżka do odpowiadającego qrels
qrels_path = "data/qrels.txt"

# Licz nDCG
score = evaluate(run_df, qrels_path)
print(f"nDCG@10: {score:.4f}")
