"""
Step 6a — 确定最优 K 值
使用课本基础方法：肘部法则 + 轮廓系数 + leaderboard 对比

运行：python step6a_determine_k.py
输出：plots/step6a_elbow.png 、plots/step6a_silhouette.png
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
VECTORS_DIR   = "vectors/"
H_LEADERBOARD = "hierarchical_leaderboard.csv"
K_LEADERBOARD = "kmeans_leaderboard.csv"
PLOTS_DIR     = "plots/"
DATA_SOURCE   = "crf_textrank"   # 使用表现最好的数据源
K_RANGE       = range(2, 11)      # K = 2 到 10
# ==============================================

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_best_vectors():
    """加载 crf_textrank 向量"""
    path = os.path.join(VECTORS_DIR, f"{DATA_SOURCE}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vectors = [item["向量"] for item in data]
    dv = DictVectorizer(sparse=True)
    matrix = dv.fit_transform(vectors)
    matrix_norm = normalize(matrix, norm="l2")
    return vectors, matrix, matrix_norm


def method1_elbow(matrix_norm):
    """
    方法一：肘部法则
    对 L2 归一化后的矩阵跑 KMeans (k=2..10)，画 SSE 曲线，找拐点
    """
    ks = list(K_RANGE)
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(matrix_norm)
        inertias.append(km.inertia_)
        print(f"  K={k}, SSE={km.inertia_:.2f}")

    # 找肘部：连接首尾直线，曲线上距离该直线最远的点
    x, y = np.array(ks), np.array(inertias)
    line = np.array([x[-1] - x[0], y[-1] - y[0]])
    line = line / np.linalg.norm(line)
    vec = np.column_stack([x - x[0], y - y[0]])
    proj = np.dot(vec, line)
    proj_pts = np.outer(proj, line) + np.array([x[0], y[0]])
    dists = np.sqrt(np.sum((np.column_stack([x, y]) - proj_pts) ** 2, axis=1))
    elbow_k = ks[np.argmax(dists[1:-1]) + 1]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, inertias, "o-", color="#E91E63", lw=2, ms=10)
    ax.axvline(x=elbow_k, color="gray", ls="--", lw=1.5,
               label=f"肘部位置 K={elbow_k}")
    for k, sse in zip(ks, inertias):
        ax.annotate(f"{sse:.0f}", (k, sse), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center")
    ax.set_xlabel("聚类数 K", fontsize=12)
    ax.set_ylabel("SSE（误差平方和）", fontsize=12)
    ax.set_title(f"肘部法则 — {DATA_SOURCE}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "step6a_elbow.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [图] step6a_elbow.png（肘部建议 K={elbow_k}）")
    return elbow_k, inertias


def method2_silhouette(vectors, matrix_norm):
    """
    方法二：轮廓系数
    对 KMeans (余弦) 和 等级聚类 (Ward) 分别计算各 K 下的轮廓系数
    """
    ks = list(K_RANGE)
    km_sils = []
    hc_sils = []

    print("\n  K    KMeans轮廓    等级聚类轮廓")
    print("  " + "-" * 38)
    for k in ks:
        # KMeans (余弦等效)
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km_labels = km.fit_predict(matrix_norm)
        km_sil = silhouette_score(matrix_norm, km_labels, metric="cosine")
        km_sils.append(km_sil)

        # 等级聚类 (Ward)
        dense = matrix_norm.toarray() if hasattr(matrix_norm, "toarray") else matrix_norm
        hc = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        hc_labels = hc.fit_predict(dense)
        hc_sil = silhouette_score(dense, hc_labels, metric="cosine")
        hc_sils.append(hc_sil)

        print(f"  K={k:<3} {km_sil:>10.4f}     {hc_sil:>10.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, km_sils, "s-", color="#FF9800", lw=2, ms=10,
            label="K-Means（余弦）")
    ax.plot(ks, hc_sils, "o-", color="#2196F3", lw=2, ms=10,
            label="等级聚类（Ward）")
    ax.set_xlabel("聚类数 K", fontsize=12)
    ax.set_ylabel("轮廓系数", fontsize=12)
    ax.set_title(f"轮廓系数 vs K — {DATA_SOURCE}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "step6a_silhouette.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [图] step6a_silhouette.png")
    return km_sils, hc_sils


def method3_leaderboard_summary():
    """
    方法三：直接看 leaderboard 中各K的表现
    分别展示等级聚类 (ward) 和 KMeans (cosine) 在各K下的最佳综合得分
    """
    h_df = pd.read_csv(H_LEADERBOARD)
    k_df = pd.read_csv(K_LEADERBOARD)

    print("\n  从 leaderboard 提取各K最优结果：")
    print(f"\n  {'K':<5} {'等级-综合得分':>14} {'等级-均衡比':>12} | "
          f"{'KMeans-综合得分':>14} {'KMeans-均衡比':>12}")
    print("  " + "-" * 65)

    for k in K_RANGE:
        h_best = h_df[(h_df["数据源"] == DATA_SOURCE) &
                       (h_df["链接方法"] == "ward") &
                       (h_df["聚类数_K"] == k)]
        k_best = k_df[(k_df["数据源"] == DATA_SOURCE) &
                       (k_df["初始化方法"] == "k-means++") &
                       (k_df["等效距离度量"] == "cosine") &
                       (k_df["聚类数_K"] == k)]

        h_score = h_best["综合得分"].max() if len(h_best) > 0 else 0
        h_bal = h_best["均衡比"].max() if len(h_best) > 0 else 0
        k_score = k_best["综合得分"].max() if len(k_best) > 0 else 0
        k_bal = k_best["均衡比"].max() if len(k_best) > 0 else 0

        print(f"  K={k:<3} {h_score:>14.4f} {h_bal:>12.4f} | "
              f"{k_score:>14.4f} {k_bal:>12.4f}")


def main():
    print("=" * 60)
    print("  Step 6a — 确定最优 K 值")
    print("=" * 60)

    # 加载数据
    print(f"\n[加载] 数据源: {DATA_SOURCE}")
    vectors, matrix, matrix_norm = load_best_vectors()

    # 方法一：肘部法则
    print("\n" + "=" * 60)
    print("  方法一：肘部法则")
    print("  原理：K增大时SSE下降幅度会骤减，拐点即为最优K")
    print("=" * 60)
    elbow_k, inertias = method1_elbow(matrix_norm)

    # 方法二：轮廓系数
    print("\n" + "=" * 60)
    print("  方法二：轮廓系数")
    print("  原理：轮廓系数越高，簇内紧密度和簇间分离度的平衡越好")
    print("=" * 60)
    km_sils, hc_sils = method2_silhouette(vectors, matrix_norm)

    # 方法三：leaderboard 汇总
    print("\n" + "=" * 60)
    print("  方法三：Leaderboard 汇总")
    print("  直接查看等级聚类和KMeans在各K下的最优综合得分")
    print("=" * 60)
    method3_leaderboard_summary()

    # 综合推荐
    print("\n" + "=" * 60)
    print("  综合推荐")
    print("=" * 60)
    print(f"  肘部法则建议:       K = {elbow_k}")
    print(f"  KMeans轮廓最高:      K = {K_RANGE[np.argmax(km_sils)]} "
          f"(轮廓={max(km_sils):.4f})")
    print(f"  等级聚类轮廓最高:    K = {K_RANGE[np.argmax(hc_sils)]} "
          f"(轮廓={max(hc_sils):.4f})")
    print(f"\n  → 建议在 K = {max(3, elbow_k-1)} ~ {min(10, elbow_k+1)} 范围内选择")
    print(f"  → 同时参考《人民日报》实际板块数（通常 4~6 个）")
    print(f"\n  确定K值后，修改 step6b_pooling.py 中的 BEST_K 变量并运行 step6b")


if __name__ == "__main__":
    main()
