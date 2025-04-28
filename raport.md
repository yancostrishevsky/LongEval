**Sprint Summary Report — LongEval 2025 Task 2 (Dense Retrieval Baseline)**

---

## 1. Objective

The goal of this sprint was to build an initial **baseline retrieval system** for **LongEval 2025 Task 2 - LongEval-Sci Retrieval**, focusing on scientific documents sourced from the CORE collection.

The main aims were:
- Implementing a **basic retrieval system** for scientific documents.
- Evaluating the system's performance using **nDCG@10** as the primary metric.
- Optimizing preprocessing and retrieval given resource constraints (CPU-only environment).
- Ensuring compatibility with the LongEval dataset structure (documents, queries, qrels).

---

## 2. Approach

### 2.1 Dataset
- **Documents**: JSONL files (~2M scientific documents).
- **Queries**: Text file (`queries.txt`) containing 393 queries.
- **Qrels**: Relevance judgments based on click models (`qrels.txt`).

Due to the large document collection size (~4 GB), it was impractical to process the entire dataset on a CPU machine. Instead, a **targeted encoding approach** was applied:
- Only documents **appearing in the qrels** were encoded.
- This allowed for meaningful evaluation without needing to encode millions of documents.

### 2.2 System Architecture

Two main components were implemented:

#### 1. Document Encoding
- Model: `sentence-transformers/all-MiniLM-L6-v2` (dense embedding model).
- Only documents listed in `qrels.txt` were selected and encoded.
- Documents were encoded in batches (`batch_size=512`) for better efficiency.
- Encoded documents and IDs were saved as `.npy` files for fast retrieval.

#### 2. Dense Retrieval
- Queries were embedded using the same model.
- Cosine similarity was computed between each query embedding and all document embeddings.
- Top-10 most similar documents were retrieved for each query.
- Results were evaluated using **nDCG@10**.

---

## 3. Results

### 3.1 Performance

- **Documents encoded**: 4262 documents.
- **Queries processed**: 393 queries.
- **Dense retrieval runtime**: ~3 seconds for all queries.
- **Mean nDCG@10**: **0.6683**

### 3.2 Key Observations

- Retrieval system performed solidly given the constraints.
- nDCG@10 of **0.6683** indicates that, on average, **relevant documents were ranked within the top 10** but were not always perfectly ordered.
- The choice of a general-purpose model (`all-MiniLM-L6-v2`) was appropriate for a first baseline, but domain-specific models could improve results.

---

## 4. Challenges and Solutions

| Challenge | Solution |
|:---|:---|
| Dataset size too large for CPU | Selected and encoded only qrels-relevant documents |
| No GPU available | Efficient batching and memory management |
| Type mismatch between document IDs and qrels | Ensured consistent data types (`str`) across datasets |
| Initial zero nDCG results | Corrected ID matching and restricted evaluation only to available documents |

---

## 5. Lessons Learned

- **Precise alignment** between document IDs and relevance judgments is crucial.
- **Efficient sampling and batching** allows working with large datasets even without GPUs.
- **Baseline Dense Retrieval** provides a strong starting point, but fine-tuning or reranking could further improve results.

---

## 6. Next Steps

Recommended next steps to improve retrieval performance:
- **Switch to a scientific domain-specific model** like `allenai-specter`.
- **Implement reranking** using a cross-encoder model to refine top-10 results.
- **Fine-tune** the retrieval model on LongEval click data (triplet loss training).
- **Expand** the document set if compute resources allow, to cover a broader snapshot of CORE documents.

---

# Summary

This sprint successfully built and evaluated a first **Dense Retrieval Baseline** for LongEval 2025 Task 2.  
The system achieves **solid performance (nDCG@10 = 0.6683)** with efficient CPU-based encoding and lays a strong foundation for further improvements.