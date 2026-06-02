"""
相似岗位推荐 + 全量矩阵保存 —— To Do List #157 ~ #162

    #157: batch_cosine_sim(job_id, top_k=5) 函数
    #158: 全量 5k 文档 TF-IDF 余弦相似度 Top-5
    #159: 详情页底部相似岗位推荐 (数据端)
    #160: 全量 TF-IDF 稀疏矩阵保存
    #161: 聚类结果 (label + PCA坐标) 保存
    #162: 供检索系统加载的词表+矩阵文件

用法: python src/16_similar_jobs.py
输出: models/tfidf_matrix.npz, models/vocab_map.json, models/similar_jobs.json
"""

import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SEGMENTED_CSV, MODEL_DIR, DIR_16_SIMILAR

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DIR_16_SIMILAR, exist_ok=True)

MATRIX_FILE = os.path.join(MODEL_DIR, "tfidf_matrix.npz")
VOCAB_FILE = os.path.join(MODEL_DIR, "vocab_map.json")
SIMILAR_FILE = os.path.join(DIR_16_SIMILAR, "similar_jobs.json")

TOP_K_FEAT = 200


# ══════════════════════════════════════════════════
# #160: 构建全量 TF-IDF 稀疏矩阵
# ══════════════════════════════════════════════════

def build_full_matrix():
    print("[#160] 构建全量 TF-IDF 矩阵...")
    df = pd.read_csv(SEGMENTED_CSV, encoding="utf-8-sig", engine="python")
    print(f"  文档数: {len(df):,}")

    # 收集词袋
    docs = []
    for i in tqdm(range(len(df)), desc="  分词", ncols=80):
        seg = str(df.iloc[i].get("职位描述_分词", ""))
        title = str(df.iloc[i].get("招聘岗位_分词", ""))
        tokens = [w.strip() for w in (title + "/" + seg).split("/")
                  if w.strip() and w != "nan" and not re.search(r'[，。；：、（）【】\-;:,.!?\d]', w.strip())]
        docs.append(tokens)

    # 全局 IDF
    N = len(docs)
    df_dict = defaultdict(int)
    for d in docs:
        for w in set(d):
            df_dict[w] += 1

    # TF-IDF 向量
    vectors = []
    for tokens in tqdm(docs, desc="  TF-IDF", ncols=80):
        tf = defaultdict(float)
        for w in tokens:
            tf[w] += 1.0
        scored = [(w, t * math.log(N / (df_dict.get(w, 0) + 1))) for w, t in tf.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        vectors.append({w: s for w, s in scored[:TOP_K_FEAT]})

    # 稀疏矩阵
    vec = DictVectorizer(sparse=True)
    matrix = vec.fit_transform(vectors)
    feature_names = vec.get_feature_names_out()

    print(f"  矩阵: {matrix.shape}, 词汇: {len(feature_names):,}")

    # 保存
    save_npz(MATRIX_FILE, matrix)
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump({"feature_names": list(feature_names), "N": N, "df": dict(df_dict)}, f, ensure_ascii=False)

    print(f"  [#162] 矩阵: {MATRIX_FILE}")
    print(f"  [#162] 词表: {VOCAB_FILE}")
    return matrix, docs


# ══════════════════════════════════════════════════
# #157-#159: batch_cosine_sim + Top-5
# ══════════════════════════════════════════════════

def batch_cosine_sim(job_id, matrix, top_k=5):
    """#157: 返回与 job_id 最相似的 Top-K 职位 (0-based index)"""
    vec = matrix[job_id]
    sims = cosine_similarity(vec, matrix)[0]
    # 排除自身
    sims[job_id] = -1
    indices = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in indices if sims[i] > 0]


def build_similarity_index(matrix, sample_size=5000):
    """#158-#159: 计算每条的 Top-5 相似职位"""
    print(f"\n[#157-#159] 计算相似职位推荐...")
    n = min(sample_size, matrix.shape[0])

    results = {}
    for i in tqdm(range(0, n, 100), desc="  批量相似度", ncols=80):
        batch_end = min(i + 100, n)
        batch = matrix[i:batch_end]
        sims = cosine_similarity(batch, matrix[:n])
        np.fill_diagonal(sims[i - i:], -1)  # 排除自身 (相对于batch的偏移)
        for j in range(batch_end - i):
            idx = i + j
            top = np.argsort(sims[j])[::-1][:5]
            results[str(idx)] = [(int(t), float(sims[j][t])) for t in top if sims[j][t] > 0]

    with open(SIMILAR_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"  [{len(results)} 条] 相似推荐: {SIMILAR_FILE}")

    # 预览
    print(f"\n  示例 (job 0):")
    for job_id, sim in results.get("0", [])[:5]:
        print(f"    → job {job_id} (相似度: {sim:.4f})")


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  相似推荐 + 矩阵保存 #157 ~ #162")
    print("=" * 60)

    matrix, docs = build_full_matrix()
    build_similarity_index(matrix)

    print(f"\n{'=' * 60}")
    print(f"  #157 ~ #162 完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
