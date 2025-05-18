import os
import json
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import math

QUERIES_PATH = 'data/queries.txt'
QRELS_PATH = 'data/qrels.txt'
DOCS_DIR = 'data/documents/'
OUTPUT_PATH = 'bm25_results.csv'
TOP_K = 200

print("Loading qrels docids...")
qrels = pd.read_csv(QRELS_PATH, sep=' ', names=['qid', 'iter', 'docid', 'relevance'])
qrel_docids = set(qrels['docid'].astype(str))

doc_texts = {}
term_doc_freq = defaultdict(list)
doc_lengths = {}
total_terms = 0

print("Loading documents...")
for fname in tqdm(os.listdir(DOCS_DIR)):
    if not fname.endswith('.jsonl'):
        continue
    with open(os.path.join(DOCS_DIR, fname), 'r') as f:
        for line in f:
            doc = json.loads(line)
            docid = str(doc['id'])
            if docid not in qrel_docids:
                continue
            title = doc.get('title') or ''
            abstract = doc.get('abstract') or ''
            if not isinstance(title, str):
                title = ''
            if not isinstance(abstract, str):
                abstract = ''
            text = (title + ' ' + abstract).lower()
            terms = text.split()
            doc_texts[docid] = text
            doc_lengths[docid] = len(terms)
            term_freq = Counter(terms)
            for term, freq in term_freq.items():
                term_doc_freq[term].append((docid, freq))
            total_terms += len(terms)

avg_doc_len = total_terms / len(doc_texts)

def bm25_score(query_terms, docid, k1=1.5, b=0.75):
    score = 0
    doc_len = doc_lengths.get(docid, 0)
    N = len(doc_lengths)
    for term in query_terms:
        docs_with_term = term_doc_freq.get(term, [])
        df = len(docs_with_term)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        f = dict(docs_with_term).get(docid, 0)
        denom = f + k1 * (1 - b + b * doc_len / avg_doc_len)
        score += idf * f * (k1 + 1) / denom
    return score

print("Scoring with BM25...")
queries = pd.read_csv(QUERIES_PATH, sep='\t', names=['qid', 'query'])
results = []

for _, row in tqdm(queries.iterrows(), total=len(queries)):
    qid = row['qid']
    terms = row['query'].lower().split()
    candidate_docs = set()
    for t in terms:
        candidate_docs.update([d for d, _ in term_doc_freq.get(t, [])])

    scored = [(docid, bm25_score(terms, docid)) for docid in candidate_docs]
    top_docs = sorted(scored, key=lambda x: x[1], reverse=True)[:TOP_K]
    for rank, (docid, score) in enumerate(top_docs):
        results.append({
            'qid': qid,
            'docid': docid,
            'score': score,
            'rank': rank + 1
        })

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")
