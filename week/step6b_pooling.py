"""
Step 6b — Pooling 池化 + 人工标注导出
运行前请先执行 step6a 确定 K 值，然后修改下面的 BEST_K

运行：python step6b_pooling.py
输出：pooling_for_annotation.csv
"""
import os, json, warnings
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# ==================== 配置（运行前修改 BEST_K）====================
BEST_K        = 5                  # ← 根据 step6a 的结果修改
DATA_SOURCE   = "crf_textrank"     # 最优数据源
VECTORS_DIR   = "vectors/"
PLOTS_DIR     = "plots/"
N_CHAMPIONS   = 3                  # 冠军模型数（奇数，方便投票）
# =================================================================

os.makedirs(PLOTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
#  第1步：加载向量 + 选取冠军模型
# ═══════════════════════════════════════════════════

def load_vectors():
    path = os.path.join(VECTORS_DIR, f"{DATA_SOURCE}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vectors = [item["向量"] for item in data]
    ids     = [item["序号"] for item in data]
    dv = DictVectorizer(sparse=True)
    matrix = dv.fit_transform(vectors)
    return vectors, matrix, ids


def define_champions(k):
    """
    定义 3 个冠军模型（不同路径，相同K值）
    模型A: 等级聚类 Ward
    模型B: K-Means k-means++ 余弦
    模型C: K-Means Buckshot 余弦
    """
    champions = [
        {
            "名称": "等级聚类 Ward",
            "算法": "hierarchical",
            "参数": f"linkage=ward, K={k}",
        },
        {
            "名称": "K-Means k-means++",
            "算法": "kmeans_pp",
            "参数": f"init=k-means++, distance=cosine, K={k}",
        },
        {
            "名称": "K-Means Buckshot",
            "算法": "buckshot",
            "参数": f"init=buckshot, distance=cosine, K={k}",
        },
    ]
    return champions


# ═══════════════════════════════════════════════════
#  第2步：运行各模型
# ═══════════════════════════════════════════════════

def run_model(champion, matrix):
    """根据冠军配置运行聚类，返回 labels"""
    k = BEST_K
    algo = champion["算法"]

    if algo == "hierarchical":
        dense = matrix.toarray()
        model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        labels = model.fit_predict(dense)

    elif algo == "buckshot":
        from step5_kmeans_turbo import get_buckshot_centers
        mat = normalize(matrix, norm="l2")
        init = get_buckshot_centers(mat, k)
        model = KMeans(n_clusters=k, init=init, n_init=1, random_state=42)
        labels = model.fit_predict(mat)

    else:  # kmeans_pp
        mat = normalize(matrix, norm="l2")
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = model.fit_predict(mat)

    return labels


# ═══════════════════════════════════════════════════
#  第3步：标签对齐（匈牙利算法）
# ═══════════════════════════════════════════════════

def align_all_labels(all_labels):
    """
    以模型A为锚点，将B、C的标签映射到A的编号体系
    思路：对 B→A 和 C→A 分别运行匈牙利算法，最大化匹配文档数
    """
    n_docs, n_models = all_labels.shape
    anchor = all_labels[:, 0].copy()
    mappings = [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}]

    for mi in range(1, n_models):
        tgt = all_labels[:, mi]
        nc = BEST_K
        cost = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                cost[i, j] = -np.sum((anchor == i) & (tgt == j))
        row, col = linear_sum_assignment(cost)
        mapping = {old: new for old, new in zip(col, row)}
        mappings.append(mapping)
        all_labels[:, mi] = np.array([mapping.get(l, l) for l in tgt])

    return all_labels, mappings


# ═══════════════════════════════════════════════════
#  第4步：共识计算
# ═══════════════════════════════════════════════════

def compute_consensus(all_labels, champions, ids, vectors):
    """逐文档计算模型共识度"""
    n_docs = len(ids)
    records = []

    for i in range(n_docs):
        votes = all_labels[i, :]
        top_label, top_count = Counter(votes).most_common(1)[0]
        ratio = top_count / len(champions)

        records.append({
            "序号": ids[i],
            "共识簇": int(top_label),
            "共识度": ratio,
            "完全一致": len(set(votes)) == 1,
            "各模型标签": str(list(votes)),
        })

    df = pd.DataFrame(records)
    df = df.sort_values(["完全一致", "共识簇", "共识度"],
                         ascending=[False, True, False]).reset_index(drop=True)

    # 统计
    n_full = df["完全一致"].sum()
    print(f"\n  共识统计：")
    print(f"    三个模型完全一致: {n_full} 篇 ({n_full/n_docs*100:.1f}%)")
    print(f"    两个模型一致:     {(df['共识度'] > 0.5) & (~df['完全一致'])} 篇"
          f"({((df['共识度']>0.5)&(~df['完全一致'])).sum()/n_docs*100:.1f}%)")
    print(f"    三个模型各执一词: {(df['共识度'] < 0.6).sum()} 篇 "
          f"({(df['共识度']<0.6).sum()/n_docs*100:.1f}%)")

    # 提取各共识簇的关键词
    cluster_kw = {}
    for label in sorted(df["共识簇"].unique()):
        indices = df[df["共识簇"] == label].index
        all_words = Counter()
        for idx in indices:
            if idx < len(vectors):
                all_words.update(vectors[idx])
        cluster_kw[label] = ", ".join([w for w, _ in all_words.most_common(15)])

    return df, cluster_kw


# ═══════════════════════════════════════════════════
#  第5步：导出 + 绘图
# ═══════════════════════════════════════════════════

def export_and_plot(df, cluster_kw, champions):
    """导出 CSV 并绘制共识度分布图"""

    # --- 加载文档预览 ---
    doc_previews = {}
    seg_file = "crf_seg_result.csv"
    if os.path.exists(seg_file):
        seg = pd.read_csv(seg_file)
        for _, row in seg.iterrows():
            words = str(row["分词后内容"]).split("/")[:80]
            doc_previews[row["序号"]] = "/".join(words)

    # --- 导出 CSV ---
    rows = []
    for _, r in df.iterrows():
        label = int(r["共识簇"])
        rows.append({
            "序号": r["序号"],
            "文章内容预览(前80词)": doc_previews.get(r["序号"], ""),
            "共识簇": label,
            "簇核心关键词": cluster_kw.get(label, ""),
            "共识度": f"{r['共识度']:.2f}",
            "完全一致": "是" if r["完全一致"] else "否",
            "各模型标签": r["各模型标签"],
            "需要人工审核": "" if r["完全一致"] else "★",
        })

    export_df = pd.DataFrame(rows)
    export_df.to_csv("pooling_for_annotation.csv", index=False, encoding="utf-8-sig")
    print(f"\n  [保存] pooling_for_annotation.csv ({len(export_df)} 行)")

    # --- 打印各簇特征（供命名参考）---
    print(f"\n  各共识簇特征词（供命名参考）：")
    print(f"  {'簇':<6} {'文档数':<8} Top 15 关键词")
    print(f"  {'-'*55}")
    for label in sorted(df["共识簇"].unique()):
        cnt = (df["共识簇"] == label).sum()
        print(f"  {label:<6} {cnt:<8} {cluster_kw[label][:80]}")

    # --- 共识度分布图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(df["共识度"], bins=15, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax1.axvline(x=1.0, color="red", ls="--", label=f"完全一致 (共识度=1.0)")
    ax1.set_xlabel("共识度", fontsize=11)
    ax1.set_ylabel("文档数", fontsize=11)
    ax1.set_title("模型共识度分布", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)

    sizes = df.groupby("共识簇").size()
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    ax2.bar(range(len(sizes)), sizes.values, color=colors, edgecolor="white")
    ax2.set_xlabel("共识簇编号", fontsize=11)
    ax2.set_ylabel("文档数", fontsize=11)
    ax2.set_title("各共识簇文档数", fontsize=13, fontweight="bold")
    for i, v in enumerate(sizes.values):
        ax2.text(i, v + 10, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "step6b_consensus.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [图] step6b_consensus.png")


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"  Step 6b — Pooling 池化  (K = {BEST_K})")
    print("=" * 60)

    # 1. 加载
    print("\n[1/5] 加载向量数据 ...")
    vectors, matrix, ids = load_vectors()
    print(f"  文档数: {len(ids)}, 特征维数: {matrix.shape[1]}")

    # 2. 定义冠军模型
    print("\n[2/5] 定义冠军模型 ...")
    champions = define_champions(BEST_K)
    for i, c in enumerate(champions):
        print(f"  模型{i+1}: {c['名称']} ({c['参数']})")

    # 3. 运行 + 对齐
    print("\n[3/5] 运行模型并标签对齐 ...")
    all_labels = np.zeros((len(ids), len(champions)), dtype=int)
    for mi, champ in enumerate(champions):
        labels = run_model(champ, matrix)
        all_labels[:, mi] = labels
        dist = Counter(labels)
        print(f"  {champ['名称']}: {dict(sorted(dist.items()))}")

    print("\n  标签对齐（锚点: 模型1）...")
    all_labels, mappings = align_all_labels(all_labels)
    for mi in range(1, len(champions)):
        print(f"  模型{mi+1} → 模型1 映射: {mappings[mi]}")

    # 4. 共识计算
    print("\n[4/5] 计算共识度 ...")
    df_consensus, cluster_kw = compute_consensus(
        all_labels, champions, ids, vectors
    )

    # 5. 导出
    print("\n[5/5] 导出结果 ...")
    export_and_plot(df_consensus, cluster_kw, champions)

    print(f"\n{'='*60}")
    print("  Step 6b 完成！")
    print(f"  下一步：打开 pooling_for_annotation.csv，")
    print(f"  1) 查看各簇关键词，为每个共识簇命名（2-6字）")
    print(f"  2) 对标记 ★ 的文章进行人工裁决")
    print(f"  3) 全部确定后，即为标准答案（Ground Truth）")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
