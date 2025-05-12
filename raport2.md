# LongEval 2025 Dense Retrieval and Re-ranking Experiments

## 1. Introduction

This report summarizes a sequence of information retrieval experiments conducted for the LongEval 2025 shared task. The focus was on exploring the performance of dense retrievers and cross-encoder re-rankers compared to traditional BM25-based retrieval. Our goal was to incrementally improve retrieval quality on the LongEval scientific document collection.

## 2. Baseline Comparison: BM25 vs Dense Retriever (MiniLM)

Initially, we compared the official BM25 baseline provided by the LongEval organizers with a dense retrieval model using `sentence-transformers/all-MiniLM-L6-v2`. Both models were evaluated on the full training collection of over **2 million documents**.

* **BM25 Baseline (official)**: nDCG\@10 ≈ **0.45**
* **Dense Retriever (MiniLM-L6-v2)**: nDCG\@10 ≈ **0.52**

This result encouraged us to explore further improvements using stronger dense encoders.

## 3. Scaling Up: Switching to E5-Large-v2

We upgraded to a larger model: `intfloat/e5-large-v2`, known for strong retrieval performance. However, generating embeddings for all 2M documents with this model was computationally expensive (estimated time: **\~70 hours** on RTX 2070 SUPER), which was infeasible.

### 3.1 Document Set Reduction

To mitigate this, we applied a targeted filtering strategy:

* Included **all documents mentioned in QRELs** (\~4,000 unique docs).
* Added **50,000–200,000 randomly sampled noisy documents** to simulate realistic retrieval scenarios.

Despite this compromise, the initial dense retrieval performance **decreased to ≈ 0.42 nDCG\@10**, likely due to reduced document diversity.

## 4. Introducing Cross-Encoder Re-Ranking

To enhance retrieval further, we implemented **cross-encoder re-ranking** of top-K candidates from dense retrieval.

Our retrieval pipeline consists of two stages: a dense retriever followed by a cross-encoder re-ranker. Each of these stages serves a distinct purpose and leverages a specialized pretrained model.
Dense Retriever (intfloat/e5-large-v2)

    Model: `intfloat/e5-large-v2`

    Function: This model encodes queries and documents into fixed-size dense vectors (embeddings).

    Query Format: Input queries are prepended with "query: " as per model documentation.

    Document Format: Documents are prepended with "passage: " and consist of the concatenated title and abstract.

    Retrieval Mechanism: Uses cosine similarity between the query embedding and each document embedding to rank documents.


Cross-Encoder Re-Ranker (cross-encoder/ms-marco-MiniLM-L-12-v2)

    Model: cross-encoder/ms-marco-MiniLM-L-12-v2 (~66M parameters)

    Function: This model scores the relevance of a specific (query, document) pair jointly.

    Mechanism: Unlike the dense retriever, which encodes inputs independently, the cross-encoder processes the query and document together in a single transformer pass, enabling it to model fine-grained interactions.

    Usage: We use it to rerank the top-K (e.g., 200) documents retrieved by the dense retriever.




### 4.1 Models Tested

We tested several cross-encoder models using the `sentence-transformers` CrossEncoder API:

| Cross-Encoder Model                                 | Top-K | Mean nDCG\@10 |
| --------------------------------------------------- | ----- | ------------- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2`              | 200    | 0.42          |
| `cross-encoder/ms-marco-Electra-base`               | 200    | 0.70          |
| `cross-encoder/ms-marco-TinyBERT-L-6-v2`            | 200    | 0.63          |
| **`cross-encoder/ms-marco-MiniLM-L-12-v2`** | 200   | **0.7448**    |

> Note: K refers to the number of top documents returned by the dense retriever before re-ranking.

### 4.2 Final Setup (Best Performance)

* **Dense Retriever**: `intfloat/e5-large-v2` (filtered + sampled docs)
* **Cross-Encoder Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
* **Top-K Candidates**: 200
* **Final nDCG\@10**: **0.7448**
* **Inference Time**: \~5 minutes for 78,600 pairs on RTX 2070 SUPER

## 5. Summary and Future Work

Our experiments show that combining powerful dense encoders (E5 family) with cross-encoder re-ranking yields substantial gains over the BM25 baseline. Future improvements may include:

* Fine-tuning E5 or CrossEncoders on LongEval-specific data.
* Using smarter document sampling strategies.
* Applying hybrid methods (BM25 + Dense fusion).

These steps are likely to yield further gains in nDCG and real-world retrieval quality.
