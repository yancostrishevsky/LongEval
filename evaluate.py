import pandas as pd
from sklearn.metrics import ndcg_score

def evaluate(results_df, qrels_path):
    qrels = pd.read_csv(qrels_path, sep=' ', names=['qid', 'iter', 'docid', 'relevance'])
    qrels_grouped = qrels.groupby('qid').apply(lambda g: dict(zip(g['docid'], g['relevance']))).to_dict()

    ndcg_scores = []
    for qid in results_df['qid'].unique():
        retrieved_docs = results_df[results_df['qid'] == qid]['docid'].tolist()
        true_relevances = []
        scores = []
        if qid in qrels_grouped:
            qrel_docs = qrels_grouped[qid]
            for doc in retrieved_docs:
                rel = qrel_docs.get(doc, 0)
                true_relevances.append(rel)
                scores.append(1)
            ndcg = ndcg_score([true_relevances], [scores])
            ndcg_scores.append(ndcg)

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    return mean_ndcg
