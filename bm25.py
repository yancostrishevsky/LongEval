import math

def bm25_score(query_terms, doc_id, inverted_index, doc_lengths, avg_doc_len, documents, k1=1.5, b=0.75):
    score = 0.0
    for term in query_terms:
        if term in inverted_index:
            doc_freq = len(inverted_index[term])
            idf = math.log((len(documents) - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            term_freq = dict(inverted_index[term]).get(doc_id, 0)
            denom = term_freq + k1 * (1 - b + b * doc_lengths[doc_id] / avg_doc_len)
            score += idf * (term_freq * (k1 + 1)) / denom
    return score
