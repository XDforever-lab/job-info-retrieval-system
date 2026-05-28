"""
Step 7 — 全组合终极评测：用标准答案检验所有路径
对 6 种数据源 × 多种聚类配置 逐一评测，找到真正的最优路径
"""
import os, json, warnings, sys
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
GROUND_TRUTH_FILE = "ground_truth_final.csv"
VECTORS_DIR       = "vectors/"
PLOTS_DIR         = "plots/"
K_CLUSTERS        = 5
# =============================================

os.makedirs(PLOTS_DIR, exist_ok=True)

# 6 种数据源
ALL_SOURCES = [
    "baseline_tfidf", "baseline_textrank",
    "crf_tfidf", "crf_textrank",
    "ngram_tfidf", "ngram_textrank",
]


def load_ground_truth():
    """加载标准答案，返回 y_true 数组和文章序号"""
    gt = pd.read_csv(GROUND_TRUTH_FILE)
    return gt["序号"].tolist(), np.array(gt["最终类别编号"].astype(int))


def load_vectors(source):
    """加载某个数据源的向量文件"""
    path = os.path.join(VECTORS_DIR, f"{source}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vec_dict = {item["序号"]: item["向量"] for item in data}
    return vec_dict


def align_to_truth(y_true, y_pred, n_clusters):
    """匈牙利算法：把模型标签映射到标准答案标签"""
    cost = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost[i, j] = -np.sum((y_pred == i) & (y_true == j))
    row, col = linear_sum_assignment(cost)
    mapping = {p: t for p, t in zip(row, col)}
    return np.array([mapping.get(l, l) for l in y_pred]), mapping


def compute_metrics(y_true, y_pred):
    """成对指标 + 聚类分布"""
    pcm = pair_confusion_matrix(y_true, y_pred)
    tn, fp = pcm[0, 0], pcm[0, 1]
    fn, tp = pcm[1, 0], pcm[1, 1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    error     = (fp + fn) / (tp + tn + fp + fn)
    return precision, recall, f1, error


def run_all_evaluations(gt_ids, y_true):
    """
    对所有数据源 × 多种算法进行评测
    算法列表：
      - Hierarchical Ward (euclidean)
      - Hierarchical Complete (cosine)
      - KMeans k-means++ cosine
      - KMeans random cosine
      - KMeans Buckshot cosine
    """
    all_results = []
    all_aligned = {}  # key: "数据源|算法名" -> aligned labels

    for source_name in ALL_SOURCES:
        print(f"\n{'='*60}")
        print(f"  数据源: {source_name}")
        print(f"{'='*60}")

        # 加载向量
        vec_dict = load_vectors(source_name)

        # 对齐：确保向量顺序与 ground truth 一致
        vectors_aligned = []
        y_true_aligned = []
        for i, doc_id in enumerate(gt_ids):
            if doc_id in vec_dict:
                vectors_aligned.append(vec_dict[doc_id])
                y_true_aligned.append(y_true[i])

        if len(vectors_aligned) == 0:
            print(f"  [跳过] 无匹配文档")
            continue

        y_true_sub = np.array(y_true_aligned)
        dv = DictVectorizer(sparse=True)
        matrix = dv.fit_transform(vectors_aligned)
        dense = matrix.toarray()
        matrix_l2 = normalize(matrix, norm="l2")

        # ---- 算法1: Hierarchical Ward ----
        print("  运行: Hierarchical Ward ...")
        hw = AgglomerativeClustering(n_clusters=K_CLUSTERS, metric="euclidean", linkage="ward")
        pred_hw = hw.fit_predict(dense)
        pred_hw_aligned, _ = align_to_truth(y_true_sub, pred_hw, K_CLUSTERS)
        p, r, f1, err = compute_metrics(y_true_sub, pred_hw_aligned)
        all_results.append({
            "数据源": source_name, "算法": "Hierarchical Ward",
            "准确率": round(p, 4), "全面率": round(r, 4),
            "F1得分": round(f1, 4), "错误率": round(err, 4),
        })
        all_aligned[f"{source_name}|Hierarchical Ward"] = pred_hw_aligned
        print(f"    → F1={f1:.4f}  准确率={p:.4f}  全面率={r:.4f}")

        # ---- 算法2: Hierarchical Complete (cosine) ----
        print("  运行: Hierarchical Complete ...")
        from sklearn.metrics import pairwise_distances
        dist_cos = pairwise_distances(matrix, metric="cosine")
        hc = AgglomerativeClustering(n_clusters=K_CLUSTERS, metric="precomputed", linkage="complete")
        pred_hc = hc.fit_predict(dist_cos)
        pred_hc_aligned, _ = align_to_truth(y_true_sub, pred_hc, K_CLUSTERS)
        p, r, f1, err = compute_metrics(y_true_sub, pred_hc_aligned)
        all_results.append({
            "数据源": source_name, "算法": "Hierarchical Complete",
            "准确率": round(p, 4), "全面率": round(r, 4),
            "F1得分": round(f1, 4), "错误率": round(err, 4),
        })
        all_aligned[f"{source_name}|Hierarchical Complete"] = pred_hc_aligned
        print(f"    → F1={f1:.4f}  准确率={p:.4f}  全面率={r:.4f}")

        # ---- 算法3: KMeans k-means++ cosine ----
        print("  运行: KMeans k-means++ cosine ...")
        km = KMeans(n_clusters=K_CLUSTERS, init="k-means++", n_init=10, random_state=42)
        pred_km = km.fit_predict(matrix_l2)
        pred_km_aligned, _ = align_to_truth(y_true_sub, pred_km, K_CLUSTERS)
        p, r, f1, err = compute_metrics(y_true_sub, pred_km_aligned)
        all_results.append({
            "数据源": source_name, "算法": "KMeans k-means++",
            "准确率": round(p, 4), "全面率": round(r, 4),
            "F1得分": round(f1, 4), "错误率": round(err, 4),
        })
        all_aligned[f"{source_name}|KMeans k-means++"] = pred_km_aligned
        print(f"    → F1={f1:.4f}  准确率={p:.4f}  全面率={r:.4f}")

        # ---- 算法4: KMeans Buckshot cosine ----
        print("  运行: KMeans Buckshot cosine ...")
        from step5_kmeans_turbo import get_buckshot_centers
        init_b = get_buckshot_centers(matrix_l2, K_CLUSTERS)
        kmb = KMeans(n_clusters=K_CLUSTERS, init=init_b, n_init=1, random_state=42)
        pred_kmb = kmb.fit_predict(matrix_l2)
        pred_kmb_aligned, _ = align_to_truth(y_true_sub, pred_kmb, K_CLUSTERS)
        p, r, f1, err = compute_metrics(y_true_sub, pred_kmb_aligned)
        all_results.append({
            "数据源": source_name, "算法": "KMeans Buckshot",
            "准确率": round(p, 4), "全面率": round(r, 4),
            "F1得分": round(f1, 4), "错误率": round(err, 4),
        })
        all_aligned[f"{source_name}|KMeans Buckshot"] = pred_kmb_aligned
        print(f"    → F1={f1:.4f}  准确率={p:.4f}  全面率={r:.4f}")

        # ---- 算法5: KMeans random cosine ----
        print("  运行: KMeans random cosine ...")
        kmr = KMeans(n_clusters=K_CLUSTERS, init="random", n_init=10, random_state=42)
        pred_kmr = kmr.fit_predict(matrix_l2)
        pred_kmr_aligned, _ = align_to_truth(y_true_sub, pred_kmr, K_CLUSTERS)
        p, r, f1, err = compute_metrics(y_true_sub, pred_kmr_aligned)
        all_results.append({
            "数据源": source_name, "算法": "KMeans random",
            "准确率": round(p, 4), "全面率": round(r, 4),
            "F1得分": round(f1, 4), "错误率": round(err, 4),
        })
        all_aligned[f"{source_name}|KMeans random"] = pred_kmr_aligned
        print(f"    → F1={f1:.4f}  准确率={p:.4f}  全面率={r:.4f}")

    return all_results, all_aligned


def print_final_ranking(results):
    """打印综合排名"""
    df = pd.DataFrame(results)
    df["排名"] = df["F1得分"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("F1得分", ascending=False).reset_index(drop=True)

    print(f"\n{'='*90}")
    print("  全组合评测最终排名 (按F1降序)")
    print(f"{'='*90}")
    print(df.to_string(index=False))

    # 按数据源聚合
    print(f"\n{'='*90}")
    print("  各数据源平均F1 (跨算法)")
    print(f"{'='*90}")
    agg = df.groupby("数据源")["F1得分"].agg(["mean", "max"]).sort_values("mean", ascending=False)
    print(agg.to_string())

    # 按算法聚合
    print(f"\n{'='*90}")
    print("  各算法平均F1 (跨数据源)")
    print(f"{'='*90}")
    agg2 = df.groupby("算法")["F1得分"].agg(["mean", "max"]).sort_values("mean", ascending=False)
    print(agg2.to_string())

    return df


def plot_ranking_heatmap(df):
    """热力图：行=数据源，列=算法，值=F1"""
    pivot = df.pivot_table(index="数据源", columns="算法", values="F1得分", aggfunc="max")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                linewidths=1, cbar_kws={"label": "F1得分"})
    ax.set_title("全组合评测 F1 得分热力图 (行=数据源, 列=算法)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "step7_full_ranking_heatmap.png"), dpi=150)
    plt.close()
    print("\n  [图] step7_full_ranking_heatmap.png")


def plot_best_confusion(y_true, y_pred, model_name, source_name):
    """最佳模型的混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    names = ["党政人事", "社会民生", "思想文化", "经济科技", "国际外交"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names)
    ax.set_ylabel("标准答案", fontsize=12)
    ax.set_xlabel(f"模型预测 ({source_name} / {model_name})", fontsize=12)
    ax.set_title(f"最优模型混淆矩阵", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "step7_best_confusion.png"), dpi=150)
    plt.close()
    print("  [图] step7_best_confusion.png")


def plot_tsne(vectors_2d, y_true, y_pred, model_name, source_name):
    """t-SNE 可视化"""
    names = ["党政人事", "社会民生", "思想文化", "经济科技", "国际外交"]
    y_true_text = [names[l] for l in y_true]
    y_pred_text = [names[l] for l in y_pred]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1],
                    hue=y_true_text, palette="tab10", ax=ax1, s=20, alpha=0.7)
    ax1.set_title("标准答案 (Ground Truth)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=7, loc="lower right")

    sns.scatterplot(x=vectors_2d[:, 0], y=vectors_2d[:, 1],
                    hue=y_pred_text, palette="tab10", ax=ax2, s=20, alpha=0.7)
    ax2.set_title(f"模型预测 ({source_name} / {model_name})", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "step7_tsne.png"), dpi=200)
    plt.close()
    print("  [图] step7_tsne.png")


def main():
    print("=" * 70)
    print("  Step 7 — 全组合终极评测")
    print(f"  6 数据源 × 5 算法 = 30 种组合")
    print("=" * 70)

    # 1. 加载标准答案
    print("\n[1] 加载标准答案 ...")
    gt_ids, y_true = load_ground_truth()
    print(f"  {len(gt_ids)} 篇文档, 分布: {dict(sorted(Counter(y_true).items()))}")

    # 2. 全组合评测
    print("\n[2] 开始全组合评测 ...")
    all_results, all_aligned = run_all_evaluations(gt_ids, y_true)

    # 3. 排名
    print("\n[3] 生成排名 ...")
    df = print_final_ranking(all_results)
    plot_ranking_heatmap(df)

    # 4. 最佳模型详细分析
    df_sorted = df.sort_values("F1得分", ascending=False)
    best = df_sorted.iloc[0]
    best_key = f"{best['数据源']}|{best['算法']}"
    best_pred = all_aligned[best_key]

    print(f"\n{'='*70}")
    print(f"  冠军: {best['数据源']} + {best['算法']}")
    print(f"  F1={best['F1得分']:.4f}  准确率={best['准确率']:.4f}  全面率={best['全面率']:.4f}")
    print(f"{'='*70}")

    plot_best_confusion(y_true, best_pred, best['算法'], best['数据源'])

    # 5. t-SNE (用最优数据源)
    print("\n[5] t-SNE 降维可视化 (可能需要几十秒) ...")
    vec_dict = load_vectors(best['数据源'])
    vectors_list = [vec_dict[did] for did in gt_ids if did in vec_dict]
    dv = DictVectorizer(sparse=True)
    matrix = dv.fit_transform(vectors_list)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    vectors_2d = tsne.fit_transform(matrix.toarray())
    plot_tsne(vectors_2d, y_true, best_pred, best['算法'], best['数据源'])

    print(f"\n{'='*70}")
    print("  评测完成！输出:")
    print(f"    plots/step7_full_ranking_heatmap.png  — 全组合F1热力图")
    print(f"    plots/step7_best_confusion.png        — 最优模型混淆矩阵")
    print(f"    plots/step7_tsne.png                  — t-SNE空间分布")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
