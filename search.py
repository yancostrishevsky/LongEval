import pandas as pd
from bm25 import bm25_score

def search(queries_df, documents, inverted_index, doc_lengths, avg_doc_len):
    results = []
    for _, row in queries_df.iterrows():
        qid = row['qid']
        query_terms = row['query'].lower().split()
        candidate_docs = set()

        for term in query_terms:
            if term in inverted_index:
                candidate_docs.update([doc_id for doc_id, _ in inverted_index[term]])

        doc_scores = [(doc_id, bm25_score(query_terms, doc_id, inverted_index, doc_lengths, avg_doc_len, documents)) for doc_id in candidate_docs]
        ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:10]

        for rank, (docid, score) in enumerate(ranked_docs):
            results.append({'qid': qid, 'docid': docid, 'rank': rank + 1, 'score': score})
    return pd.DataFrame(results)
