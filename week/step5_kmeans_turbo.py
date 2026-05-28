import os
import json
import numpy as np
import warnings
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from step3_evaluator import evaluate_clustering, dicts_to_matrix
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ================= 配置区 =================
INPUT_DIR = "代码/vectors/"
OUTPUT_CSV = "kmeans_leaderboard.csv"

K_RANGE = [4, 5, 6, 7, 8]
INIT_METHODS = ['k-means++', 'random', 'buckshot']
N_INIT = 10
DISTANCE_EQUIV = ['euclidean', 'cosine']
# ==========================================


def get_buckshot_centers(sparse_matrix, k, random_seed=42):
    """
    Buckshot 算法：随机抽取 sqrt(k * n) 样本，等级聚类找初始中心点
    """
    n_samples = sparse_matrix.shape[0]
    subset_size = int(np.sqrt(k * n_samples))
    subset_size = max(k, min(subset_size, n_samples))

    np.random.seed(random_seed)
    indices = np.random.choice(n_samples, subset_size, replace=False)
    subset_matrix = sparse_matrix[indices].toarray()

    hier_model = AgglomerativeClustering(
        n_clusters=k, metric='cosine', linkage='average'
    )
    subset_labels = hier_model.fit_predict(subset_matrix)

    centroids = []
    for i in range(k):
        cluster_points = subset_matrix[subset_labels == i]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
        else:
            centroid = subset_matrix[np.random.choice(subset_size)]
        centroids.append(centroid)

    return np.array(centroids)


def run_kmeans_search():
    if not os.path.exists(INPUT_DIR):
        print(f"找不到输入目录 {INPUT_DIR}")
        return

    all_results = []
    vector_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"找到 {len(vector_files)} 个向量数据源，启动 K-Means 网格搜索...\n")

    for filename in vector_files:
        filepath = os.path.join(INPUT_DIR, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vectors = [item['向量'] for item in data]
        sparse_matrix = dicts_to_matrix(vectors)

        print(f"\n处理数据源: {filename} "
              f"(文档:{sparse_matrix.shape[0]} x 特征:{sparse_matrix.shape[1]})")

        valid_combinations = [
            (k, init, dist)
            for k in K_RANGE
            for init in INIT_METHODS
            for dist in DISTANCE_EQUIV
        ]

        for k, init_method, dist_equiv in tqdm(valid_combinations,
                                                desc="  K-Means 迭代中",
                                                unit="组合"):
            try:
                if dist_equiv == 'cosine':
                    matrix_to_use = normalize(sparse_matrix, norm='l2')
                else:
                    matrix_to_use = sparse_matrix

                if init_method == 'buckshot':
                    init_param = get_buckshot_centers(matrix_to_use, k)
                    current_n_init = 1
                else:
                    init_param = init_method
                    current_n_init = N_INIT

                model = KMeans(
                    n_clusters=k,
                    init=init_param,
                    n_init=current_n_init,
                    random_state=42,
                )

                labels = model.fit_predict(matrix_to_use)
                eval_scores = evaluate_clustering(vectors, labels)

                all_results.append({
                    "数据源": filename.replace('.json', ''),
                    "聚类数_K": k,
                    "初始化方法": init_method,
                    "等效距离度量": dist_equiv,
                    "轮廓系数": eval_scores["轮廓系数"],
                    "簇内相似度": eval_scores["簇内相似度"],
                    "簇间差异度": eval_scores["簇间差异度"],
                    "DB指数": eval_scores["DB指数"],
                    "均衡比": eval_scores["均衡比"],
                    "聚类分布": eval_scores["聚类分布"],
                    "综合得分": eval_scores["综合得分"],
                })
            except Exception as e:
                pass

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values(by="综合得分", ascending=False).reset_index(drop=True)
    df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("  K-Means 网格搜索完成")
    print("=" * 60)
    print("\n  Top 10 最佳组合如下 (按综合得分排序)：")
    print(df_results.head(10)[[
        "数据源", "聚类数_K", "初始化方法", "等效距离度量",
        "轮廓系数", "簇内相似度", "均衡比", "综合得分"
    ]].to_string(index=False))
    print(f"\n  (共 {len(df_results)} 条记录，已保存至 {OUTPUT_CSV})")


if __name__ == "__main__":
    run_kmeans_search()
