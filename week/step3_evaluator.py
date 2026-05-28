import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


def dicts_to_matrix(vector_dicts):
    """将字典格式向量转化为 sklearn 稀疏矩阵"""
    vec = DictVectorizer(sparse=True)
    matrix = vec.fit_transform(vector_dicts)
    return matrix


def calculate_intra_cluster_similarity(matrix, labels):
    """
    簇内相似度：各簇内文档对之间的平均余弦相似度，取所有簇的均值
    越高越好：说明聚在一起的文章内容一致
    """
    unique_labels = [l for l in set(labels) if l != -1]
    if len(unique_labels) <= 1:
        return 0.0

    cluster_sims = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) < 2:
            continue
        sub_matrix = matrix[indices]
        sim_matrix = cosine_similarity(sub_matrix)
        upper = np.triu_indices_from(sim_matrix, k=1)
        cluster_sims.append(np.mean(sim_matrix[upper]))

    return np.mean(cluster_sims) if cluster_sims else 0.0


def calculate_inter_cluster_distance(matrix, labels):
    """
    簇间差异度：各簇中心之间的平均余弦距离 (1 - 余弦相似度)
    越高越好：说明不同簇之间界限分明
    """
    unique_labels = [l for l in set(labels) if l != -1]
    if len(unique_labels) < 2:
        return 0.0

    centroids = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        sub_matrix = matrix[indices]
        centroid = np.asarray(sub_matrix.mean(axis=0))
        centroids.append(centroid[0])

    centroids_matrix = np.array(centroids)
    sim_matrix = cosine_similarity(centroids_matrix)
    dist_matrix = 1 - sim_matrix
    upper = np.triu_indices_from(dist_matrix, k=1)
    return np.mean(dist_matrix[upper])


def calculate_davies_bouldin(matrix, labels):
    """
    Davies-Bouldin 指数（余弦距离版本）
    逻辑：对每个簇 i，找到与其最相似的簇 j，计算 (σ_i + σ_j) / d(ci, cj)
    越低越好：说明簇内紧密、簇间分离
    """
    unique_labels = [l for l in set(labels) if l != -1]
    if len(unique_labels) < 2:
        return float('inf')

    dense = matrix.toarray() if hasattr(matrix, 'toarray') else matrix

    centroids = []
    intra_dists = []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_points = dense[indices]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)

        if len(indices) > 1:
            dists = pairwise_distances(cluster_points, [centroid], metric='cosine')
            intra_dists.append(float(np.mean(dists)))
        else:
            intra_dists.append(0.0)

    centroids = np.array(centroids)
    inter_dists = pairwise_distances(centroids, metric='cosine')

    n = len(unique_labels)
    db_sum = 0.0
    for i in range(n):
        max_ratio = 0.0
        for j in range(n):
            if i != j and inter_dists[i, j] > 1e-10:
                ratio = (intra_dists[i] + intra_dists[j]) / inter_dists[i, j]
                max_ratio = max(max_ratio, ratio)
        db_sum += max_ratio

    return db_sum / n


def calculate_composite_score(silhouette, intra_sim, inter_dist, counts_dict):
    """
    综合得分：均衡性主导 + 有效簇比例修正

    设计理念：
    - 均衡性权重 50%：防止巨型簇与 singleton 作弊
    - 轮廓/簇间差异权重受 valid_ratio 修正：< 5 篇的"簇"不参与质量计算
    - 簇内相似度不受 valid_ratio 影响（它反映簇内聚合度，大簇也有效）

    最终得分 0-1，越高越好
    """
    n_clusters = len(counts_dict)
    total = sum(counts_dict.values())
    min_sz = min(counts_dict.values())
    ideal_min = total / n_clusters
    balance_norm = min(1.0, min_sz / ideal_min)

    # 有效簇：>= 5 篇文档才算真正的簇
    valid_clusters = sum(1 for sz in counts_dict.values() if sz >= 5)
    valid_ratio = valid_clusters / n_clusters

    sil_norm = max(0.0, silhouette)
    inter_norm = min(1.0, inter_dist)

    # silhouette 和 inter_dist 乘以 valid_ratio 防 singleton 虚高
    return (sil_norm * 0.15 * valid_ratio
            + intra_sim * 0.15
            + inter_norm * 0.20 * valid_ratio
            + balance_norm * 0.50)


def evaluate_clustering(vector_dicts, labels):
    """
    综合评估函数：返回多维度指标
    """
    unique_labels = [l for l in set(labels) if l != -1]
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return {
            "轮廓系数": -1.0,
            "簇内相似度": 0.0,
            "簇间差异度": 0.0,
            "DB指数": float('inf'),
            "均衡比": 0.0,
            "聚类分布": str(dict(Counter(labels))),
            "综合得分": 0.0,
        }

    matrix = dicts_to_matrix(vector_dicts)

    # 1. 轮廓系数（余弦距离）
    silhouette = silhouette_score(matrix, labels, metric='cosine')

    # 2. 簇内相似度
    intra_sim = calculate_intra_cluster_similarity(matrix, labels)

    # 3. 簇间差异度
    inter_dist = calculate_inter_cluster_distance(matrix, labels)

    # 4. Davies-Bouldin 指数（越低越好）
    db_index = calculate_davies_bouldin(matrix, labels)

    # 5. 聚类分布与均衡性
    counts = Counter(int(l) for l in labels)
    dist_dict = dict(sorted(counts.items()))
    min_sz = min(counts.values())
    max_sz = max(counts.values())
    balance_ratio = min_sz / max_sz  # 0 到 1，1 = 完全均衡

    # 6. 综合得分（传入 counts 用于计算理想均衡比）
    composite = calculate_composite_score(
        silhouette, intra_sim, inter_dist, counts
    )

    return {
        "轮廓系数": round(float(silhouette), 4),
        "簇内相似度": round(float(intra_sim), 4),
        "簇间差异度": round(float(inter_dist), 4),
        "DB指数": round(float(db_index), 4),
        "均衡比": round(float(balance_ratio), 4),
        "聚类分布": str(dist_dict),
        "综合得分": round(float(composite), 4),
    }


# --- 测试代码 ---
if __name__ == "__main__":
    mock_vectors = [
        {"经济": 0.5, "发展": 0.5},
        {"经济": 0.4, "金融": 0.6},
        {"篮球": 0.8, "体育": 0.2},
        {"足球": 0.7, "体育": 0.3},
    ]
    mock_labels = np.array([0, 0, 1, 1])

    results = evaluate_clustering(mock_vectors, mock_labels)
    print("--- 测评模块内部测试 ---")
    for k, v in results.items():
        print(f"  {k}: {v}")
