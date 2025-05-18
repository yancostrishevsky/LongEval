import pandas as pd
from sklearn.metrics import ndcg_score

def evaluate(results_df, qrels_path):
    qrels = pd.read_csv(qrels_path, sep=' ', names=['qid', 'iter', 'docid', 'relevance'])

    qrels['docid'] = qrels['docid'].astype(str)
    results_df['docid'] = results_df['docid'].astype(str)

    qrels_grouped = qrels.groupby('qid').apply(lambda g: dict(zip(g['docid'], g['relevance']))).to_dict()

    ndcg_scores = []
    for qid in results_df['qid'].unique():
        retrieved = results_df[results_df['qid'] == qid]
        retrieved_docs = retrieved['docid'].tolist()
        pred_scores = retrieved['score'].tolist()

        true_relevances = [qrels_grouped.get(qid, {}).get(docid, 0) for docid in retrieved_docs]

        if sum(true_relevances) > 0:
            ndcg = ndcg_score([true_relevances], [pred_scores])
            ndcg_scores.append(ndcg)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0