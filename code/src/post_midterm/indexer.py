"""
倒排索引构建 —— To Do List 第八阶段 #171 ~ #189

对应理论课: 第3章 顺排文档与倒排文档检索 + 第6章 TF-IDF 标引加权

功能:
  #171-#173: 顺排文档检索（基线对比）
  #174-#181: 倒排索引构建（Lexicon + Posting List）
  #182-#186: TF-IDF 加权标引（ltc 公式 + 萨顿加权方案对比）
  #187-#189: 词-文档矩阵验证

分词格式: 词/词/词 (CRF 分词，/ 分隔)
数据量: 5,000 条

用法:
    conda activate my_project
    python src/post_midterm/indexer.py

输出:
    models/inverted_index.pkl    倒排索引
    models/weighted_index.pkl    TF-IDF 加权索引
"""

import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SEGMENTED_CSV, STOPWORDS_FILE, MODEL_DIR, OUTPUT_DIR

INDEX_FILE = os.path.join(MODEL_DIR, 'inverted_index.pkl')
WEIGHTED_INDEX_FILE = os.path.join(MODEL_DIR, 'weighted_index.pkl')
SEQ_BENCHMARK_FILE = os.path.join(OUTPUT_DIR, 'seq_vs_inverted_benchmark.json')


# ═══════════════════════════════════════════════════════════════
# 0. 数据加载
# ═══════════════════════════════════════════════════════════════

def load_segmented_data():
    df = pd.read_csv(SEGMENTED_CSV, encoding='utf-8-sig')
    # 手动获取列名，兼容 BOM
    cols = list(df.columns)
    desc_col = cols[-2]   # 职位描述_分词
    title_col = cols[-1]  # 招聘岗位_分词
    return df, desc_col, title_col


def load_stopwords():
    stops = set()
    with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if w:
                stops.add(w)
    return stops


def parse_segmented(seg_text):
    """将'词1/词2/词3'格式的分词结果解析为词列表"""
    if not isinstance(seg_text, str):
        return []
    return [w.strip() for w in seg_text.split('/') if w.strip()]


# ═══════════════════════════════════════════════════════════════
# 1. 顺排文档检索 (#171-#173)
# ═══════════════════════════════════════════════════════════════

def sequential_search(query_terms, doc_words_list):
    """
    顺排文档检索: 遍历所有文档，检查是否包含所有查询词 (AND 逻辑)

    对应第3章 顺排文档检索：
    - 读入一篇文档 → 判断是否匹配 → 读入下一篇
    - 时间复杂度: O(N × L)  N=文档数 L=平均文档长度

    Args:
        query_terms: 查询词列表
        doc_words_list: [(doc_id, [word_list]), ...]

    Returns:
        [doc_id, ...] 按文档顺序排列的匹配文档ID列表
    """
    results = []
    for doc_id, words in doc_words_list:
        word_set = set(words)
        if all(term in word_set for term in query_terms):
            results.append(doc_id)
    return results


def benchmark_sequential(query_terms, doc_words_list, n_runs=3):
    """测试顺排文档检索耗时"""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sequential_search(query_terms, doc_words_list)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


# ═══════════════════════════════════════════════════════════════
# 2. 倒排索引构建 (#174-#181)
# ═══════════════════════════════════════════════════════════════

def build_inverted_index(df, desc_col, title_col, stopwords, verbose=True):
    """
    构建倒排索引 —— 对应第3章 倒排文档检索

    数据结构:
      Lexicon (词典):  {词条 → (文档频率DF, 倒排列表起始位置)}
      Posting List:    [(doc_id, 词频TF, [position1, position2, ...])]

    构建流程 (对应教材 8.1.1 节):
      原始文档集 → CRF 分词 + 去停用词 → 词条序列
        → 按 (词条, docID, TF, 位置) 组织 → 倒排索引

    Args:
        df: 包含分词字段的 DataFrame
        desc_col: 职位描述分词列名
        title_col: 招聘岗位分词列名
        stopwords: 停用词集合
        verbose: 是否打印进度

    Returns:
        lexicon:  {词条 → {'df': int, 'postings': [(doc_id, tf, [positions])]}}
        stats: 索引统计信息
    """
    lexicon = defaultdict(lambda: {'df': 0, 'postings': []})
    total_tokens = 0
    total_valid = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        doc_id = idx  # 用行号作为内部 doc_id

        # 解析分词后的职位描述
        desc_tokens = parse_segmented(str(row[desc_col]))
        title_tokens = parse_segmented(str(row[title_col]))

        # 合并去重计数 (描述+标题, 分别记录位置以区分字段来源)
        all_tokens = desc_tokens + title_tokens
        total_tokens += len(all_tokens)

        # 统计各词在本文档中的词频和位置
        token_positions = defaultdict(list)
        for pos, token in enumerate(all_tokens):
            if token in stopwords or len(token) < 2:
                continue
            token_positions[token].append(pos)

        for token, positions in token_positions.items():
            lexicon[token]['df'] += 1
            lexicon[token]['postings'].append((doc_id, len(positions), positions))
            total_valid += 1

        if verbose and (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1}/{len(df)} 篇文档, 词典规模 {len(lexicon)} 词条")

    # 转回普通 dict
    lexicon = dict(lexicon)

    stats = {
        'total_docs': len(df),
        'total_tokens_raw': total_tokens,
        'total_tokens_valid': total_valid,
        'lexicon_size': len(lexicon),
        'avg_df': sum(v['df'] for v in lexicon.values()) / max(len(lexicon), 1),
        'avg_posting_len': sum(len(v['postings']) for v in lexicon.values()) / max(len(lexicon), 1),
    }
    return lexicon, stats


def benchmark_inverted(query_terms, lexicon):
    """
    倒排索引检索: 利用词典+倒排列表快速定位候选文档 (AND 逻辑)

    时间复杂度: O(|q| × log|L|) |q|=查询词数 L=词典大小
    对比顺排检索 O(N×L) 有显著优势

    Returns:
        [doc_id, ...] 包含所有查询词的文档ID列表
    """
    if not query_terms:
        return []

    # 取第一个词的倒排列表
    first_term = query_terms[0]
    if first_term not in lexicon:
        return []

    doc_sets = [set(doc_id for doc_id, _, _ in lexicon[first_term]['postings'])]

    # 与其他词的倒排列表取交集 (AND 逻辑)
    for term in query_terms[1:]:
        if term not in lexicon:
            return []
        doc_sets.append(set(doc_id for doc_id, _, _ in lexicon[term]['postings']))

    return list(set.intersection(*doc_sets))


def verify_index(lexicon, doc_words_list, n_samples=100):
    """验证倒排索引完整性: 随机抽查100个词条，检查倒排列表正确性"""
    import random
    random.seed(42)

    terms = list(lexicon.keys())
    sample_terms = random.sample(terms, min(n_samples, len(terms)))

    errors = 0
    for term in sample_terms:
        idx_docs = set(doc_id for doc_id, _, _ in lexicon[term]['postings'])

        # 用顺排方式验证
        seq_docs = set()
        for doc_id, words in doc_words_list:
            if term in words:
                seq_docs.add(doc_id)

        if idx_docs != seq_docs:
            errors += 1
            if errors <= 3:
                print(f"  [不一致] 词条'{term}': 索引={sorted(idx_docs)[:5]}... 顺排={sorted(seq_docs)[:5]}...")

    accuracy = (n_samples - errors) / n_samples * 100
    print(f"  抽查 {n_samples} 词条, {n_samples - errors}/{n_samples} 一致, 准确率 {accuracy:.1f}%")
    return accuracy


# ═══════════════════════════════════════════════════════════════
# 3. TF-IDF 加权标引 (#182-#186)
# ═══════════════════════════════════════════════════════════════

def compute_idf(lexicon, N):
    """计算每个词条的 IDF 值: IDF = log(N/DF)"""
    idf = {}
    for term, info in lexicon.items():
        df = info['df']
        idf[term] = math.log(N / df) if df > 0 else 0.0
    return idf


def build_weighted_index(lexicon, idf, N):
    """
    构建 TF-IDF 加权倒排索引 (#183-#185)

    加权公式 (ltc):
      - 词频因子: 1 + log(tf)   (对数归一化)
      - 逆文档频率: log(N/df)
      - 文档向量: L2 归一化

    Args:
        lexicon: 倒排索引
        idf: {词条 → IDF值}
        N: 文档总数

    Returns:
        weighted_postings: {doc_id → {词条 → tfidf_weight}}
        doc_norms: {doc_id → L2范数}
    """
    print("  计算 TF-IDF 权重 (ltc 公式)...")

    # 每文档的 TF-IDF 向量 (稀疏表示)
    doc_vectors = defaultdict(dict)

    for term, info in lexicon.items():
        term_idf = idf[term]
        for doc_id, tf, positions in info['postings']:
            # ltc: (1 + log(tf)) × log(N/df)
            weight = (1 + math.log(tf)) * term_idf if tf > 0 else 0
            doc_vectors[doc_id][term] = weight

    # L2 归一化
    doc_norms = {}
    for doc_id, vec in doc_vectors.items():
        norm = math.sqrt(sum(w * w for w in vec.values()))
        doc_norms[doc_id] = norm
        if norm > 0:
            for term in vec:
                vec[term] /= norm

    doc_vectors = dict(doc_vectors)

    # 构建倒排结构的加权索引: {词条 → [(doc_id, tfidf_weight)]}
    weighted_index = defaultdict(list)
    for term, info in lexicon.items():
        for doc_id, tf, positions in info['postings']:
            if doc_id in doc_vectors and term in doc_vectors[doc_id]:
                weighted_index[term].append((doc_id, doc_vectors[doc_id][term]))

    weighted_index = dict(weighted_index)

    stats = {
        'total_docs': N,
        'weighted_terms': len(weighted_index),
        'avg_norm': sum(doc_norms.values()) / max(len(doc_norms), 1),
    }
    return weighted_index, doc_vectors, doc_norms, stats


def sutton_weight_comparison(lexicon, N):
    """
    萨顿加权公式对比实验 (#186)

    对比几种加权方案:
      - ltc: (1+log(tf)) × log(N/df), L2归一化  (教材第6章推荐)
      - nfc: tf × log(N/df), L2归一化
      - bfx: 0/1 二元权重 × log(N/df)
      - tfx: tf × 1.0 (不加IDF), L2归一化
    """
    print("  萨顿加权公式对比...")

    def compute_scheme(name, tf_func, use_idf, use_l2):
        """按指定方案计算 doc_vectors"""
        doc_vectors = defaultdict(dict)
        idf = {term: math.log(N / info['df']) for term, info in lexicon.items()}

        for term, info in lexicon.items():
            term_idf = idf[term] if use_idf else 1.0
            for doc_id, tf, _ in info['postings']:
                w = tf_func(tf) * term_idf
                doc_vectors[doc_id][term] = w

        if use_l2:
            for doc_id, vec in doc_vectors.items():
                norm = math.sqrt(sum(w * w for w in vec.values()))
                if norm > 0:
                    for term in vec:
                        vec[term] /= norm

        return dict(doc_vectors)

    schemes = {
        'ltc (1+log(tf))*IDF, L2': (lambda tf: 1 + math.log(tf) if tf > 0 else 0, True, True),
        'nfc (tf)*IDF, L2':        (lambda tf: tf, True, True),
        'bfx (0/1)*IDF':            (lambda tf: 1.0, True, False),
        'tfx (tf)*1.0, L2':         (lambda tf: tf, False, True),
    }

    results = {}
    for name, (tf_func, use_idf, use_l2) in schemes.items():
        vecs = compute_scheme(name, tf_func, use_idf, use_l2)
        # 统计指标: 平均非零权重数 / 权重稀疏度
        n_nonzero = sum(len(v) for v in vecs.values())
        avg_nonzero = n_nonzero / max(len(vecs), 1)
        results[name] = {
            'avg_nonzero_per_doc': round(avg_nonzero, 2),
            'total_nonzero': n_nonzero,
        }
        print(f"    {name}: 平均每文档 {avg_nonzero:.1f} 个非零权重")

    return results


# ═══════════════════════════════════════════════════════════════
# 4. 序列化保存
# ═══════════════════════════════════════════════════════════════

def save_index(lexicon, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(lexicon, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("倒排索引与顺排文档检索 —— 第八阶段")
    print("=" * 60)

    # ── 0. 加载数据 ──
    print("\n[0] 加载数据...")
    df, desc_col, title_col = load_segmented_data()
    stopwords = load_stopwords()
    N = len(df)
    print(f"  文档数: {N}")
    print(f"  停用词数: {len(stopwords)}")

    # 构建词列表 (供顺排检索和验证用)
    doc_words_list = []
    for idx, (_, row) in enumerate(df.iterrows()):
        desc_tokens = parse_segmented(str(row[desc_col]))
        title_tokens = parse_segmented(str(row[title_col]))
        all_tokens = desc_tokens + title_tokens
        valid_tokens = [t for t in all_tokens if t not in stopwords and len(t) >= 2]
        doc_words_list.append((idx, valid_tokens))

    # ── 1. 顺排文档检索基线 ──
    print("\n" + "=" * 60)
    print("[1] 顺排文档检索 (#171-#173)")
    print("=" * 60)

    test_queries = [
        ['Python', '开发'],
        ['财务', '审计'],
        ['数据分析'],
        ['电气', '工程'],
        ['机器学习'],
    ]

    for query in test_queries:
        n_runs = 3
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            results = sequential_search(query, doc_words_list)
            times.append(time.perf_counter() - t0)
        avg_time = sum(times) / len(times) * 1000  # ms
        print(f"  查询 '{' '.join(query)}': 命中 {len(results)} 篇, "
              f"平均耗时 {avg_time:.2f}ms (顺排, {n_runs}次)")

    # ── 2. 倒排索引构建 ──
    print("\n" + "=" * 60)
    print("[2] 倒排索引构建 (#174-#181)")
    print("=" * 60)

    t0 = time.perf_counter()
    lexicon, idx_stats = build_inverted_index(df, desc_col, title_col, stopwords)
    build_time = time.perf_counter() - t0

    print(f"\n  构建完成, 耗时 {build_time:.1f}s")
    print(f"  词典大小: {idx_stats['lexicon_size']:,} 词条")
    print(f"  原始 token 数: {idx_stats['total_tokens_raw']:,}")
    print(f"  有效 token 数 (去停用词+短词): {idx_stats['total_tokens_valid']:,}")
    print(f"  平均 DF: {idx_stats['avg_df']:.1f}")
    print(f"  平均倒排列表长度: {idx_stats['avg_posting_len']:.1f}")

    # 倒排检索 vs 顺排检索对比
    print("\n  倒排 vs 顺排 检索对比:")
    for query in test_queries:
        # 倒排检索
        t0 = time.perf_counter()
        inv_results = benchmark_inverted(query, lexicon)
        inv_time = (time.perf_counter() - t0) * 1000

        # 顺排检索
        seq_time = 0
        for _ in range(3):
            t0 = time.perf_counter()
            seq_results = sequential_search(query, doc_words_list)
            seq_time += (time.perf_counter() - t0)
        seq_time = seq_time / 3 * 1000

        assert set(inv_results) == set(seq_results), \
            f"结果不一致! 倒排={len(inv_results)} 顺排={len(seq_results)}"
        speedup = seq_time / inv_time if inv_time > 0 else float('inf')
        print(f"    查询 '{' '.join(query)}': "
              f"顺排 {seq_time:.2f}ms | 倒排 {inv_time:.3f}ms | 加速比 {speedup:.0f}× | 命中 {len(inv_results)}")

    # 索引完整性验证
    print("\n  索引完整性验证 (#180):")
    verify_index(lexicon, doc_words_list)

    # ── 3. 保存倒排索引 ──
    print(f"\n[3] 保存倒排索引 → {INDEX_FILE}")
    save_index(lexicon, INDEX_FILE)
    fsize = os.path.getsize(INDEX_FILE) / 1024 / 1024
    print(f"  文件大小: {fsize:.1f} MB")

    # ── 4. TF-IDF 加权 ──
    print("\n" + "=" * 60)
    print("[4] TF-IDF 加权标引 (#182-#186)")
    print("=" * 60)

    idf = compute_idf(lexicon, N)
    weighted_index, doc_vectors, doc_norms, w_stats = build_weighted_index(lexicon, idf, N)
    print(f"  加权词条数: {w_stats['weighted_terms']:,}")
    print(f"  平均 L2 范数: {w_stats['avg_norm']:.4f}")

    # 萨顿加权公式对比
    print()
    sutton_results = sutton_weight_comparison(lexicon, N)

    # ── 5. 保存加权索引 ──
    print(f"\n[5] 保存加权倒排索引 → {WEIGHTED_INDEX_FILE}")
    save_index(weighted_index, WEIGHTED_INDEX_FILE)
    fsize = os.path.getsize(WEIGHTED_INDEX_FILE) / 1024 / 1024
    print(f"  文件大小: {fsize:.1f} MB")

    # ── 6. 词-文档矩阵验证 (#187-#189) ──
    print("\n" + "=" * 60)
    print("[6] 词-文档矩阵验证 (#187-#189)")
    print("=" * 60)
    # 读取已有 TF-IDF 矩阵验证一致性
    tfidf_matrix_path = os.path.join(MODEL_DIR, 'tfidf_matrix.npz')
    vocab_map_path = os.path.join(MODEL_DIR, 'vocab_map.json')

    if os.path.exists(tfidf_matrix_path) and os.path.exists(vocab_map_path):
        from scipy.sparse import load_npz
        matrix = load_npz(tfidf_matrix_path)
        with open(vocab_map_path, 'r', encoding='utf-8') as f:
            vm = json.load(f)
        vocab_terms = set(vm.get('feature_names', []))
        print(f"  TF-IDF 矩阵: {matrix.shape[0]} 文档 × {matrix.shape[1]} 词")
        print(f"  词汇映射表: {len(vocab_terms)} 词")

        # 验证: vocab_map 中的词是否在倒排索引中出现
        index_terms = set(lexicon.keys())
        common = vocab_terms & index_terms
        only_vocab = vocab_terms - index_terms
        only_index = index_terms - vocab_terms
        print(f"  交集: {len(common)} 词")
        print(f"  仅在vocab_map中: {len(only_vocab)} 词")
        print(f"  仅在倒排索引中: {len(only_index)} 词")
        print(f"  一致性: {len(common)/len(vocab_terms)*100:.1f}%")
    else:
        print("  (TF-IDF 矩阵或 vocab_map 不存在，跳过验证)")

    # ── 7. 汇总报告 ──
    print("\n" + "=" * 60)
    print("第八阶段完成 —— 汇总")
    print("=" * 60)
    print(f"  顺排文档检索: 已实现 (基线对比)")
    print(f"  倒排索引: {idx_stats['lexicon_size']:,} 词条, 构建耗时 {build_time:.1f}s")
    print(f"  TF-IDF 加权: ltc 公式, L2 归一化")
    print(f"  萨顿加权对比: {len(sutton_results)} 种方案")
    print(f"\n  输出文件:")
    print(f"    {INDEX_FILE}      倒排索引 ({os.path.getsize(INDEX_FILE)/1024/1024:.1f}MB)")
    print(f"    {WEIGHTED_INDEX_FILE}  TF-IDF 加权索引 ({os.path.getsize(WEIGHTED_INDEX_FILE)/1024/1024:.1f}MB)")


if __name__ == '__main__':
    main()
