import os
import json
from collections import defaultdict, Counter

def load_documents(documents_dir):
    documents = {}
    for doc_file in os.listdir(documents_dir):
        if doc_file.endswith('.jsonl'):
            with open(os.path.join(documents_dir, doc_file), 'r') as f:
                for line in f:
                    doc = json.loads(line)
                    title = doc.get('title') or ''
                    abstract = doc.get('abstract') or ''
                    documents[doc['id']] = (title + ' ' + abstract).lower()
    return documents

def build_inverted_index(documents):
    inverted_index = defaultdict(list)
    doc_lengths = {}

    for doc_id, text in documents.items():
        terms = text.split()
        doc_lengths[doc_id] = len(terms)
        term_freqs = Counter(terms)
        for term, freq in term_freqs.items():
            inverted_index[term].append((doc_id, freq))

    avg_doc_len = sum(doc_lengths.values()) / len(doc_lengths)

    return inverted_index, doc_lengths, avg_doc_len

