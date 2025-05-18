# fusion_utils.py

import pandas as pd

def run_fusion_of_experts(e5_path, specter_path, bm25_path, weights=(0.4, 0.4, 0.2)):
    e5 = pd.read_csv(e5_path)
    specter = pd.read_csv(specter_path)
    bm25 = pd.read_csv(bm25_path)

    merged = e5.merge(specter, on=['qid', 'docid'], suffixes=('_e5', '_specter'), how='outer')
    merged = merged.merge(bm25, on=['qid', 'docid'], how='outer')
    merged = merged.rename(columns={'score': 'score_bm25'})

    for col in ['score_e5', 'score_specter', 'score_bm25']:
        merged[col] = merged[col].fillna(0.0)

    w_e5, w_specter, w_bm25 = weights
    merged['fused_score'] = (
        merged['score_e5'] * w_e5 +
        merged['score_specter'] * w_specter +
        merged['score_bm25'] * w_bm25
    )

    merged = merged.sort_values(by=['qid', 'fused_score'], ascending=[True, False])
    merged['rank'] = merged.groupby('qid').cumcount() + 1

    return merged[['qid', 'docid', 'fused_score', 'rank']]

def run_rrf(paths, k=60):
    all_dfs = []
    for path in paths:
        df = pd.read_csv(path)
        if 'rank' not in df.columns:
            df = df.sort_values(by=['qid', 'score'], ascending=[True, False])
            df['rank'] = df.groupby('qid').cumcount() + 1
        df['rrf_score'] = 1 / (df['rank'] + k)
        all_dfs.append(df[['qid', 'docid', 'rrf_score']])

    merged = pd.concat(all_dfs)
    fused = merged.groupby(['qid', 'docid'])['rrf_score'].sum().reset_index()
    fused = fused.sort_values(by=['qid', 'rrf_score'], ascending=[True, False])
    fused['rank'] = fused.groupby('qid').cumcount() + 1
    return fused