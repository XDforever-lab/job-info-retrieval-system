"""
关键词抽取全流程 —— 覆盖 To Do List #97 ~ #108

    #97:  加载分词数据，构建文档-语词矩阵
    #98:  计算全局 IDF
    #99:  逐文档 TF-IDF 得分
    #100: 抽取 TF-IDF Top-10 关键词
    #101: 保存 TF-IDF 结果
    #102: TextRank 算法实现 (滑动窗口=5, 阻尼=0.85)
    #103: 构建词共现图
    #104: PageRank 迭代至收敛
    #105: 抽取 TextRank Top-10 关键词
    #106: 保存 TextRank 结果
    #107: Pooling 融合 (TF-IDF + TextRank 并集)
    #108: 选取 100 条生成人工标注候选文件

用法:
    conda activate my_project
    python src/10_keyword_extraction.py

输出:
    output/10_keywords/tfidf_keywords.json        TF-IDF 关键词 (全量)
    output/10_keywords/textrank_keywords.json     TextRank 关键词 (抽样)
    output/10_keywords/pooled_keywords.json       Pooling 融合结果
    output/10_keywords/gold_100_for_annotation.txt  人工标注候选文件
"""

import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEGMENTED_CSV, DIR_10_KW, STOPWORDS_FILE

# ── 输出路径 ──
TFIDF_OUT = os.path.join(DIR_10_KW, "tfidf_keywords.json")
TEXTRANK_OUT = os.path.join(DIR_10_KW, "textrank_keywords.json")
POOLED_OUT = os.path.join(DIR_10_KW, "pooled_keywords.json")
GOLD_100_FILE = os.path.join(DIR_10_KW, "gold_100_for_annotation.txt")
PROCESSED_JSON = os.path.join(DIR_10_KW, "processed_corpus.json")

TITLE_WEIGHT = 2.0           # 标题字段权重加成
TOP_K = 10                   # 关键词抽取数
CANDIDATE_K = 30             # TextRank 候选词池大小
WINDOW_SIZE = 5              # TextRank 共现窗口
DAMPING = 0.85               # PageRank 阻尼系数


# ══════════════════════════════════════════════════
# #97: 加载数据 + 构建文档-语词矩阵
# ══════════════════════════════════════════════════

def load_stopwords():
    path = STOPWORDS_FILE
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(w.strip() for w in f if w.strip() and not w.startswith("#"))


def is_valid_token(w):
    """过滤 CRF 分词碎片：含标点、纯数字、单字、明显切碎片段"""
    if not w or len(w) == 0:
        return False
    # ① 含标点符号的碎片
    if re.search(r'[，。；：、（）【】"".！？…—\-;:,.!?()（）\[\]{}《》#@&*+/=]', w):
        return False
    # ② 纯数字 / 小数 / 百分比
    if re.match(r'^\d+(\.\d+)?%?$', w):
        return False
    # ③ 纯英文单字母 / 纯英文短串
    if re.match(r'^[a-zA-Z]{1,2}$', w):
        return False
    # ④ 单字词 — 关键词无意义
    if len(w) <= 1:
        return False
    # ⑤ CRF 碎片库 (2字词含常见虚词/介词/量词/方位字)
    FRAGMENT_CHARS = {'的', '了', '是', '在', '和', '与', '或', '及', '将', '相',
                      '而', '所', '对', '能', '被', '把', '从', '以', '可', '会',
                      '要', '就', '也', '都', '很', '到', '出', '过', '着', '上',
                      '下', '中', '里', '外', '前', '后', '时', '为', '向', '于',
                      '不', '没', '有', '来', '去', '这', '那', '个', '各', '每',
                      '司', '应', '作', '用', '如', '因', '其', '已', '由', '但',
                      '只', '还', '又', '之', '此', '者', '内', '等', '第', '头'}
    if len(w) == 2 and (w[0] in FRAGMENT_CHARS or w[1] in FRAGMENT_CHARS):
        # 白名单：真正的2字专业词
        WHITELIST = {'目的', '出口', '进口', '上市', '下水', '中风', '前后', '上下',
                     '中外', '以上', '以下', '以前', '以后', '将来', '过去', '出来',
                     '进去', '上来', '下来', '出去', '上来', '回来', '过来', '起来'}
        if w not in WHITELIST:
            return False
    return True


def load_docs():
    """#97: 加载分词CSV，转为 week5 格式的文档列表"""
    print("[#97] 加载分词数据并构建文档-语词矩阵...")
    stopwords = load_stopwords()
    print(f"  停用词: {len(stopwords)} 个")

    df = pd.read_csv(SEGMENTED_CSV, encoding="utf-8-sig", engine="python")
    print(f"  文档总数: {len(df):,}")

    docs = []
    skipped = 0
    for i in range(len(df)):
        title_raw = str(df.iloc[i].get("招聘岗位_分词", ""))
        body_raw = str(df.iloc[i].get("职位描述_分词", ""))

        def tokenize(text):
            if not text or text == "nan":
                return []
            tokens = []
            for w in str(text).split("/"):
                w = w.strip()
                if not w:
                    continue
                if w in stopwords:
                    continue
                if not is_valid_token(w):
                    continue
                tokens.append(w)
            return tokens

        title_tokens = tokenize(title_raw)
        body_tokens = tokenize(body_raw)

        if not body_tokens:
            skipped += 1
            continue

        docs.append({
            "id": i,
            "title": title_tokens,
            "body": body_tokens,
        })

    print(f"  有效文档: {len(docs):,} (跳过 {skipped} 条无内容)")
    return docs


# ══════════════════════════════════════════════════
# #98-#101: TF-IDF 关键词抽取 (复用 week5/step2_tfidf.py)
# ══════════════════════════════════════════════════

def calc_df(documents):
    """#98: 计算全局 DF"""
    print("[#98] 计算全局 IDF...")
    df_dict = defaultdict(int)
    for doc in tqdm(documents, desc="  统计DF", ncols=80):
        unique = set(doc["title"]) | set(doc["body"])
        for w in unique:
            df_dict[w] += 1
    print(f"  词汇总数: {len(df_dict):,}")
    return df_dict


def extract_tfidf(doc, df_dict, N, top_k=TOP_K):
    """#99-#100: 单文档 TF-IDF Top-K 抽取"""
    tf = defaultdict(float)
    for w in doc["body"]:
        tf[w] += 1.0
    for w in doc["title"]:
        tf[w] += TITLE_WEIGHT

    scored = []
    for w, t in tf.items():
        df = df_dict.get(w, 0)
        idf = math.log(N / (df + 1))
        scored.append((w, t * idf))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:top_k]]


def run_tfidf(docs):
    """#101: 全量 TF-IDF 抽取 + 保存"""
    print("\n[#98-#101] TF-IDF 关键词抽取...")
    N = len(docs)
    df_dict = calc_df(docs)

    results = []
    for doc in tqdm(docs, desc="  TF-IDF", ncols=80):
        kws = extract_tfidf(doc, df_dict, N, TOP_K)
        results.append({"id": doc["id"], "keywords": kws})

    with open(TFIDF_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  已保存 {len(results):,} 条 → {TFIDF_OUT}")
    return results, df_dict


# ══════════════════════════════════════════════════
# #102-#106: TextRank 关键词抽取 (复用 week5/step3_textrank.py)
# ══════════════════════════════════════════════════

def get_candidates_from_tfidf(doc, df_dict, N, candidate_k=CANDIDATE_K):
    """从 TF-IDF 结果中获取 TextRank 候选词池"""
    tf = defaultdict(float)
    for w in doc["body"]:
        tf[w] += 1.0
    for w in doc["title"]:
        tf[w] += TITLE_WEIGHT

    scored = []
    for w, t in tf.items():
        df = df_dict.get(w, 0)
        idf = math.log(N / (df + 1))
        scored.append((w, t * idf))
    scored.sort(key=lambda x: x[1], reverse=True)
    return set(w for w, _ in scored[:candidate_k])


def extract_textrank(doc, candidates, top_k=TOP_K):
    """#102-#105: 单文档 TextRank"""
    full_seq = doc["title"] + doc["body"]
    if not full_seq:
        return []

    graph = nx.Graph()
    for w in candidates:
        graph.add_node(w)

    n = len(full_seq)
    for i in range(n):
        wa = full_seq[i]
        if wa not in candidates:
            continue
        for j in range(i + 1, min(i + WINDOW_SIZE, n)):
            wb = full_seq[j]
            if wb in candidates and wa != wb:
                if graph.has_edge(wa, wb):
                    graph[wa][wb]["weight"] += 1.0
                else:
                    graph.add_edge(wa, wb, weight=1.0)

    try:
        pr = nx.pagerank(graph, alpha=DAMPING, weight="weight")
    except nx.PowerIterationFailedConvergence:
        return []

    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]


def run_textrank(docs, df_dict):
    """#106: 抽样 TextRank 抽取 + 保存"""
    print(f"\n[#102-#106] TextRank 关键词抽取 (全量 {len(docs)} 条)...")
    N = len(docs)
    sample = docs  # 5000条全量跑

    results = []
    for i, doc in enumerate(tqdm(sample, desc="  TextRank", ncols=80)):
        candidates = get_candidates_from_tfidf(doc, df_dict, N)
        kws = extract_textrank(doc, candidates)
        results.append({"id": doc["id"], "keywords": kws})

    with open(TEXTRANK_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  已保存 {len(results):,} 条 → {TEXTRANK_OUT}")
    return results


# ══════════════════════════════════════════════════
# #107: Pooling 融合 (TF-IDF + TextRank)
# ══════════════════════════════════════════════════

def run_pooling(docs, tfidf_results, textrank_results):
    """#107: TF-IDF + TextRank 并集 Pooling，生成人工标注候选池"""
    print(f"\n[#107] Pooling 融合...")

    # 建索引
    tfidf_map = {r["id"]: set(r["keywords"]) for r in tfidf_results}
    tr_map = {r["id"]: set(r["keywords"]) for r in textrank_results}

    pooled = []
    common_ids = set(tfidf_map.keys()) & set(tr_map.keys())
    for did in sorted(common_ids):
        union = sorted(tfidf_map[did] | tr_map[did])
        pooled.append({"id": did, "keywords": union})

    with open(POOLED_OUT, "w", encoding="utf-8") as f:
        json.dump(pooled, f, ensure_ascii=False, indent=2)
    print(f"  已保存 {len(pooled):,} 条 → {POOLED_OUT}")
    print(f"  每条候选池平均大小: {np.mean([len(r['keywords']) for r in pooled]):.1f}")
    return pooled


# ══════════════════════════════════════════════════
# #108: 生成人工标注候选文件
# ══════════════════════════════════════════════════

def generate_gold_standard_file(docs, pooled_results, sample_size=100):
    """#108: 从 Pooling 结果中选取 100 条，生成供人工筛选金标准的文本文件"""
    print(f"\n[#108] 生成人工标注候选文件 (选取 {sample_size} 条)...")

    # 均匀抽样
    indices = np.linspace(0, len(pooled_results) - 1, sample_size, dtype=int)
    sampled = [pooled_results[i] for i in indices]

    # 构建 doc 索引
    doc_map = {d["id"]: d for d in docs}

    lines = []
    lines.append("=" * 60)
    lines.append("  上市公司招聘数据 — 关键词人工标注金标准")
    lines.append("  说明: 阅读职位描述原文，从候选词池中选出最能概括")
    lines.append("        该岗位核心内容的词（宁缺毋滥），用逗号分隔。")
    lines.append("=" * 60)
    lines.append("")

    for idx, item in enumerate(sampled):
        did = item["id"]
        doc = doc_map.get(did)
        if not doc:
            continue

        # 拼接原文（用分词还原形式，可读性较差但保留信息）
        body_text = " ".join(doc["body"])
        if len(body_text) > 500:
            body_text = body_text[:500] + "..."

        lines.append(f"### 文档 {idx + 1} (原始ID: {did})")
        lines.append(f"职位描述: {body_text}")
        lines.append(f"候选词池: {', '.join(item['keywords'])}")
        lines.append(f"金标准关键词: ")
        lines.append("")
        lines.append("-" * 50)
        lines.append("")

    with open(GOLD_100_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  已保存 → {GOLD_100_FILE}")
    print(f"  请在文件中填写每条的 '金标准关键词:' 行")


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════

def main():
    os.makedirs(DIR_10_KW, exist_ok=True)
    t0 = time.time()
    print("=" * 60)
    print("  关键词抽取 #97 ~ #108")
    print("=" * 60)

    # #97
    docs = load_docs()

    # #98-#101
    tfidf_results, df_dict = run_tfidf(docs)

    # #102-#106
    textrank_results = run_textrank(docs, df_dict)

    # #107
    pooled_results = run_pooling(docs, tfidf_results, textrank_results)

    # #108
    generate_gold_standard_file(docs, pooled_results, sample_size=100)

    # 预览
    print(f"\n{'=' * 60}")
    print("  TF-IDF 示例 (前3条):")
    for r in tfidf_results[:3]:
        print(f"    ID={r['id']}: {', '.join(r['keywords'][:8])}")

    print(f"\n  TextRank 示例 (前3条):")
    for r in textrank_results[:3]:
        print(f"    ID={r['id']}: {', '.join(r['keywords'][:8])}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  #97 ~ #108 完成! 耗时 {elapsed / 60:.1f} 分钟")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
