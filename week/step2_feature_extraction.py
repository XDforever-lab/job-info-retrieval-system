import pandas as pd
import math
import json
import os
import networkx as nx
from collections import defaultdict

# ================= 配置区 =================
# 输入的分词文件列表 (排列组合的第一维)
SEG_FILES = {
    "baseline": "baseline_seg_result.csv",
    "crf": "crf_seg_result.csv",
    "ngram": "ngram_seg_result.csv"
}

OUTPUT_DIR = "vectors/"
TOP_K_KEYWORDS = 200   # 调大到 200，增强聚类时的特征重叠度
TEXTRANK_WINDOW = 5    
# ==========================================

def calculate_global_df(docs):
    df_dict = defaultdict(int)
    for doc_words in docs:
        for word in set(doc_words):
            df_dict[word] += 1
    return df_dict

def extract_tfidf_vector(doc_words, df_dict, N, top_k):
    word_tf = defaultdict(float)
    for word in doc_words: word_tf[word] += 1.0
    scored = []
    for word, tf in word_tf.items():
        df = df_dict.get(word, 0)
        idf = math.log(N / (df + 1))
        scored.append((word, tf * idf))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {word: score for word, score in scored[:top_k]}

def extract_textrank_vector(doc_words, df_dict, N, top_k, window_size):
    if not doc_words: return {}
    # 先用TF-IDF初筛，提高TextRank构建图的效率
    candidates = set(extract_tfidf_vector(doc_words, df_dict, N, top_k * 2).keys())
    graph = nx.Graph()
    for word in candidates: graph.add_node(word)
    
    seq_len = len(doc_words)
    for i in range(seq_len):
        wA = doc_words[i]
        if wA not in candidates: continue
        for j in range(i + 1, min(i + window_size, seq_len)):
            wB = doc_words[j]
            if wB in candidates and wA != wB:
                if graph.has_edge(wA, wB): graph[wA][wB]['weight'] += 1.0
                else: graph.add_edge(wA, wB, weight=1.0)
    try:
        pr = nx.pagerank(graph, alpha=0.85, weight='weight')
    except: return {}
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return {word: score for word, score in sorted_pr[:top_k]}

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for seg_name, file_path in SEG_FILES.items():
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过 {seg_name}: 找不到文件 {file_path}")
            continue
            
        print(f"\n🚀 正在处理分词模式: {seg_name.upper()}")
        df = pd.read_csv(file_path)
        df['words'] = df['分词后内容'].apply(lambda x: str(x).split('/') if pd.notnull(x) else [])
        docs = df['words'].tolist()
        ids = df['序号'].tolist()
        N = len(docs)
        df_dict = calculate_global_df(docs)

        # 组合 1: TF-IDF
        print(f"  -> 抽取 TF-IDF (Top {TOP_K_KEYWORDS})...")
        tfidf_vecs = [{"序号": i, "向量": extract_tfidf_vector(d, df_dict, N, TOP_K_KEYWORDS)} for i, d in zip(ids, docs)]
        with open(f"{OUTPUT_DIR}{seg_name}_tfidf.json", 'w', encoding='utf-8') as f:
            json.dump(tfidf_vecs, f, ensure_ascii=False, indent=4)

        # 组合 2: TextRank
        print(f"  -> 抽取 TextRank (Top {TOP_K_KEYWORDS})...")
        tr_vecs = [{"序号": i, "向量": extract_textrank_vector(d, df_dict, N, TOP_K_KEYWORDS, TEXTRANK_WINDOW)} for i, d in zip(ids, docs)]
        with open(f"{OUTPUT_DIR}{seg_name}_textrank.json", 'w', encoding='utf-8') as f:
            json.dump(tr_vecs, f, ensure_ascii=False, indent=4)

    print(f"\n✨ 所有排列组合处理完成！请在 {OUTPUT_DIR} 目录查看生成的 6 个向量文件。")

if __name__ == "__main__":
    run_pipeline()