"""
词汇聚类（分布聚类/概念聚类）—— To Do List #141 ~ #156

核心思路: 不聚文档（短文本区分度太低），改聚词汇。
         哪些技能词天然一起出现？形成"技能族群"。
         例如 {Python, 机器学习, 算法} 和 {财务, 报表, 审计} 各自成簇。

    #141-#143: 构建词-文档共现矩阵 + 词向量
    #144-#146: K-Means 对词向量聚类, 肘部+轮廓确定最佳 K
    #147-#149: 等级聚类 + 树状图
    #150-#152: 评估簇内紧密度 / 簇间分离度
    #153-#156: 词云矩阵图 + 分析报告

用法: python src/15_clustering.py
"""

import json, math, os, re, sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SEGMENTED_CSV, STOPWORDS_FILE, DIR_15_CLUSTER

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = DIR_15_CLUSTER
SVD_DIM = 80           # 词向量维度
WORD_MIN_DF = 10       # 词至少出现在10篇文档中
WORD_TOP_N = 800       # 取最有区分度的前800个词
K_RANGE = list(range(6, 13))

os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════
# 构建词向量（词在文档维度上的 TF-IDF 向量）
# ══════════════════════════════════════════════════

def build_word_vectors():
    """#141-#143: 构建词-文档矩阵, 每行=词, 每列=文档, 值=TF-IDF"""
    print("[#141-#143] 构建词-文档矩阵...")
    # 加载停用词
    sw = set()
    if os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, encoding="utf-8") as f:
            sw = {l.strip() for l in f if l.strip() and not l.startswith("#")}

    # 加载分词
    df = pd.read_csv(SEGMENTED_CSV, encoding="utf-8-sig", engine="python")
    print(f"  加载 {len(df):,} 条文档")

    # 逐文档统计词频
    doc_word_counts = []
    all_words = Counter()
    for i in tqdm(range(len(df)), desc="  统计词频", ncols=80):
        seg = str(df.iloc[i].get("职位描述_分词", ""))
        title = str(df.iloc[i].get("招聘岗位_分词", ""))
        tokens = [w.strip() for w in (title + "/" + seg).split("/")
                  if w.strip() and w != "nan" and w not in sw
                  and not re.search(r'[，。；：、（）【】\-;:,.!?\d]', w.strip())
                  and len(w.strip()) >= 2]
        doc_word_counts.append(Counter(tokens))
        all_words.update(doc_word_counts[-1].keys())

    # 选高频高分词: min_df + top_n by TF-IDF variance
    N = len(doc_word_counts)
    words_above_thresh = {w for w, c in all_words.items() if c >= WORD_MIN_DF}
    print(f"  DF>={WORD_MIN_DF} 的词: {len(words_above_thresh):,}")

    # 为每个候选词构建文档TF向量 → 算 IDF → 取top
    word_tfidf_var = {}
    for w in tqdm(words_above_thresh, desc="  算区分度", ncols=80):
        df_w = sum(1 for dc in doc_word_counts if w in dc)
        idf = math.log(N / (df_w + 1))
        scores = [dc.get(w, 0) * idf for dc in doc_word_counts]
        word_tfidf_var[w] = np.var(scores)  # 方差越大=区分度越高

    top_words = sorted(word_tfidf_var, key=word_tfidf_var.get, reverse=True)[:WORD_TOP_N]
    print(f"  选取区分度最高的 {len(top_words)} 个词")

    # 构建词-文档矩阵 (word × doc), 值=TF-IDF
    word_vectors = {}
    for w in top_words:
        df_w = sum(1 for dc in doc_word_counts if w in dc)
        idf = math.log(N / (df_w + 1))
        vec = np.zeros(N)
        for i, dc in enumerate(doc_word_counts):
            if w in dc:
                vec[i] = dc[w] * idf
        word_vectors[w] = vec

    word_list = list(word_vectors.keys())
    X = np.array([word_vectors[w] for w in word_list])

    # 不降维，直接用原始词-文档向量 + L2归一化（余弦距离=欧氏距离）
    from sklearn.preprocessing import normalize
    X_norm = normalize(X, norm='l2')
    print(f"  词向量: {X.shape}, L2归一化后直接聚类")

    return X_norm, word_list, N


# ══════════════════════════════════════════════════
# K-Means 词聚类
# ══════════════════════════════════════════════════

def run_kmeans_words(X):
    print("\n[#144-#146] K-Means 对词聚类...")
    sse_vals, sil_vals = [], []
    all_labels = {}

    for k in tqdm(K_RANGE, desc="  K-Means", ncols=80):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sse_vals.append(km.inertia_)
        sil = silhouette_score(X, labels)
        sil_vals.append(sil)
        all_labels[k] = labels

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(K_RANGE, sse_vals, "bo-")
    ax1.set_xlabel("K"); ax1.set_ylabel("SSE")
    ax1.set_title("时部法则"); ax1.grid(alpha=0.3)
    ax2.plot(K_RANGE, sil_vals, "go-")
    ax2.set_xlabel("K"); ax2.set_ylabel("轮廓系数")
    ax2.set_title("轮廓系数 vs K"); ax2.grid(alpha=0.3)
    best_sil_k = K_RANGE[np.argmax(sil_vals)]
    ax2.axvline(x=best_sil_k, color="red", linestyle="--", label=f"最佳 K={best_sil_k}")
    ax2.legend()
    fig.suptitle("词聚类 — K-Means 最佳 K 值", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "word_kmeans_elbow.png"), dpi=150)
    plt.close()

    best_idx = np.argmax(sil_vals)
    best_k = K_RANGE[best_idx]
    print(f"  最佳 K={best_k}, 轮廓系数={sil_vals[best_idx]:.4f}")
    return best_k, all_labels[best_k], sil_vals[best_idx]


# ══════════════════════════════════════════════════
# 等级聚类
# ══════════════════════════════════════════════════

def run_hierarchical_words(X, words, best_k_kmeans):
    print(f"\n[#147-#149] 等级聚类 (K={best_k_kmeans})...")
    methods = [
        ("ward", "euclidean", "Ward"),
        ("single", "cosine", "Single"),
        ("complete", "cosine", "Complete"),
    ]

    results = {}
    for link, metric_name, display in methods:
        agg = AgglomerativeClustering(n_clusters=best_k_kmeans, metric=metric_name, linkage=link)
        labels = agg.fit_predict(X)
        sil = silhouette_score(X, labels)
        results[display] = {"labels": labels, "silhouette": round(sil, 4)}
        print(f"  {display:<10s}: sil={sil:.4f}")

    # 树状图 (ward, 取80个词展示)
    sample_n = min(80, len(words))
    Z = linkage(X[:sample_n], method="ward")
    fig, ax = plt.subplots(figsize=(16, 5))
    dendrogram(Z, ax=ax, labels=[words[i][:6] for i in range(sample_n)],
              leaf_rotation=90, leaf_font_size=8)
    ax.set_title("词聚类树状图 (Ward, 前80词)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "word_dendrogram.png"), dpi=150)
    plt.close()

    best_name = max(results, key=lambda n: results[n]["silhouette"])
    print(f"  最佳: {best_name} (sil={results[best_name]['silhouette']:.4f})")
    return results, best_name


# ══════════════════════════════════════════════════
# #150-#152: 簇质量评估
# ══════════════════════════════════════════════════

def evaluate_clusters(X, labels, words, best_k):
    """评估簇内紧密度、簇间分离度，输出每簇关键词"""
    print(f"\n[#150-#152] 簇质量评估 (K={best_k})...")

    # 簇内紧密度 = 平均余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    intra_sims = []
    for k in range(best_k):
        mask = labels == k
        if mask.sum() < 2:
            continue
        sims = cosine_similarity(X[mask])
        upper = np.triu_indices_from(sims, k=1)
        intra_sims.append(np.mean(sims[upper]))
    intra = np.mean(intra_sims) if intra_sims else 0

    # 簇间分离度
    centroids = np.array([X[labels == k].mean(axis=0) for k in range(best_k)])
    inter_sims = cosine_similarity(centroids)
    upper = np.triu_indices_from(inter_sims, k=1)
    inter = 1 - np.mean(inter_sims[upper]) if len(upper[0]) > 0 else 0

    # 每簇大小与关键词
    print(f"\n  {'簇':<6} {'大小':>5} {'簇内紧密度':>10} {'关键词'}")
    print(f"  {'-'*60}")
    cluster_info = {}
    for k in range(best_k):
        mask = labels == k
        size = mask.sum()
        cluster_words = [words[i] for i in np.where(mask)[0]]
        cluster_info[str(k)] = {"size": int(size), "words": cluster_words}
        # 簇内平均相似度
        if size >= 2:
            sim = cosine_similarity(X[mask])
            up = np.triu_indices_from(sim, k=1)
            ci = np.mean(sim[up])
        else:
            ci = 0
        print(f"  K{k+1:<5} {size:>5} {ci:>10.4f}   {', '.join(cluster_words[:8])}")

    print(f"\n  平均簇内紧密度: {intra:.4f}")
    print(f"  平均簇间分离度: {inter:.4f}")

    with open(os.path.join(OUT_DIR, "word_cluster_info.json"), "w", encoding="utf-8") as f:
        json.dump({"intra_sim": round(float(intra), 4), "inter_dist": round(float(inter), 4),
                   "clusters": cluster_info}, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════
# #153-#156: 词云 + 散点图 + 报告
# ══════════════════════════════════════════════════

def visualize_word_clusters(X, labels, words, best_k):
    print("\n[#153-#156] 可视化...")

    # PCA 2D 散点图
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, best_k))
    for k in range(best_k):
        mask = labels == k
        ax.scatter(coords[mask, 0], coords[mask, 1], s=40, alpha=0.8,
                  color=colors[k], label=f"簇 {k+1} ({mask.sum()}词)")
        # 标注词（仅标注高分词）
        cluster_coords = coords[mask]
        cluster_words = [words[i] for i in np.where(mask)[0]]
        for j in range(len(cluster_words)):
            if j < 5:  # 每簇标注前5个
                ax.annotate(cluster_words[j], (cluster_coords[j, 0], cluster_coords[j, 1]),
                           fontsize=7, alpha=0.8)
    ax.set_title(f"词汇聚类结果 — {WORD_TOP_N} 个技能词 → {best_k} 个技能族群", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "word_cluster_pca.png"), dpi=150)
    plt.close()

    # 词云矩阵
    fig, axes = plt.subplots((best_k + 2) // 3, 3, figsize=(15, 3 * ((best_k + 2) // 3)))
    axes = axes.flatten() if best_k > 1 else [axes]
    from wordcloud import WordCloud
    try:
        font_path = 'C:/Windows/Fonts/simhei.ttf'
    except:
        font_path = None

    for k in range(best_k):
        cluster_words = [words[i] for i in np.where(labels == k)[0]]
        freq = {w: (len(cluster_words) - i) for i, w in enumerate(cluster_words)}
        try:
            wc = WordCloud(font_path=font_path, width=300, height=200,
                          background_color='white', max_words=30,
                          colormap='tab20')
            wc.generate_from_frequencies(freq)
            axes[k].imshow(wc)
            axes[k].set_title(f"族群 {k+1} ({len(cluster_words)}词)", fontsize=11)
            axes[k].axis('off')
        except:
            axes[k].text(0.5, 0.5, f"族群 {k+1}\n" + "\n".join(cluster_words[:5]),
                        ha='center', va='center', fontsize=8)
            axes[k].axis('off')
    for k in range(best_k, len(axes)):
        fig.delaxes(axes[k])
    fig.suptitle("招聘领域技能词族群 (Word Clusters)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "word_cloud_matrix.png"), dpi=150)
    plt.close()

    # 报告
    lines = [
        "招聘数据词汇聚类分析报告",
        "=" * 50,
        f"候选词: {WORD_TOP_N} 个 (DF>={WORD_MIN_DF}, 取 TF-IDF 方差最高)",
        f"最佳簇数: {best_k}",
        "",
        "各技能族群:",
    ]
    for k in range(best_k):
        cluster_words = [words[i] for i in np.where(labels == k)[0]]
        lines.append(f"  族群 {k+1}: {', '.join(cluster_words[:12])}")
    lines.extend([
        "",
        "实验意义:",
        "  不同于文档聚类（短文本失效），词汇聚类基于词-文档共现分布,",
        "  发现招聘数据中天然存在的'技能族群'——即经常同一岗位要求中",
        "  一起出现的技能词组合。这为求职者提供了'如果掌握了X，",
        "  还应该关注Y'的跨技能参考价值。",
        "",
        "与实验五的对比:",
        "  分类(有监督) = 给岗位贴行业标签",
        "  词汇聚类(无监督) = 发现技能之间的天然共生关系",
        "  两者互补而非重复。",
    ])
    with open(os.path.join(OUT_DIR, "word_cluster_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  词汇聚类实验 #141 ~ #156")
    print("=" * 60)

    X, words, N_docs = build_word_vectors()

    best_k, kmeans_labels, kmeans_sil = run_kmeans_words(X)
    hier_results, hier_best = run_hierarchical_words(X, words, best_k)

    # ── 保存所有聚类方法的标签 ──
    all_labels = {"K-Means": kmeans_labels.tolist()}
    for name in ["Ward", "Single", "Complete"]:
        all_labels[name] = hier_results[name]["labels"].tolist()
    all_labels["best_k"] = best_k
    all_labels["words"] = words
    with open(os.path.join(OUT_DIR, "cluster_labels.json"), "w", encoding="utf-8") as f:
        json.dump(all_labels, f, ensure_ascii=False)

    # 用轮廓系数更高的
    hier_sil = hier_results[hier_best]["silhouette"]
    if hier_sil > kmeans_sil:
        final_labels = hier_results[hier_best]["labels"]
        method = f"等级聚类({hier_best})"
    else:
        final_labels = kmeans_labels
        method = "K-Means"

    evaluate_clusters(X, final_labels, words, best_k)
    visualize_word_clusters(X, final_labels, words, best_k)

    print(f"\n{'=' * 60}")
    print(f"  #141 ~ #156 完成 ({method}, K={best_k})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
