import os
import json
import warnings
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from step3_evaluator import evaluate_clustering, dicts_to_matrix

warnings.filterwarnings("ignore")

# ================= 配置区 =================
INPUT_DIR = "代码/vectors/"
OUTPUT_CSV = "hierarchical_leaderboard.csv"

K_RANGE = [4, 5, 6, 7, 8]
METRICS = ['cosine', 'euclidean']
LINKAGES = ['average', 'complete', 'single', 'ward']
# ==========================================


def run_turbo_search():
    if not os.path.exists(INPUT_DIR):
        print(f"找不到输入目录 {INPUT_DIR}")
        return

    all_results = []
    vector_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"找到 {len(vector_files)} 个向量数据源，开启 Turbo 搜索...\n")

    for filename in vector_files:
        filepath = os.path.join(INPUT_DIR, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vectors = [item['向量'] for item in data]
        sparse_matrix = dicts_to_matrix(vectors)

        print(f"\n处理数据源: {filename} "
              f"(文档:{sparse_matrix.shape[0]} x 特征:{sparse_matrix.shape[1]})")
        print("  预计算全局距离矩阵...")

        dist_cosine = pairwise_distances(sparse_matrix, metric='cosine')
        dist_euclidean = pairwise_distances(sparse_matrix, metric='euclidean')

        dense_matrix_for_ward = None

        valid_combinations = []
        for k in K_RANGE:
            for metric in METRICS:
                for linkage in LINKAGES:
                    if linkage == 'ward' and metric != 'euclidean':
                        continue
                    valid_combinations.append((k, metric, linkage))

        for k, metric, linkage in tqdm(valid_combinations,
                                        desc=f"  高速演算中",
                                        unit="组合"):
            try:
                if linkage == 'ward':
                    if dense_matrix_for_ward is None:
                        dense_matrix_for_ward = sparse_matrix.toarray()
                    model = AgglomerativeClustering(
                        n_clusters=k, metric='euclidean', linkage='ward'
                    )
                    labels = model.fit_predict(dense_matrix_for_ward)
                else:
                    model = AgglomerativeClustering(
                        n_clusters=k, metric='precomputed', linkage=linkage
                    )
                    if metric == 'cosine':
                        labels = model.fit_predict(dist_cosine)
                    else:
                        labels = model.fit_predict(dist_euclidean)

                eval_scores = evaluate_clustering(vectors, labels)

                all_results.append({
                    "数据源": filename.replace('.json', ''),
                    "聚类数_K": k,
                    "距离度量": metric,
                    "链接方法": linkage,
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
    print("  Turbo 网格搜索完成")
    print("=" * 60)
    print("\n  Top 10 最佳组合如下 (按综合得分排序)：")
    print(df_results.head(10)[[
        "数据源", "聚类数_K", "链接方法", "轮廓系数",
        "簇内相似度", "簇间差异度", "DB指数", "均衡比", "综合得分"
    ]].to_string(index=False))
    print(f"\n  (共 {len(df_results)} 条记录，已保存至 {OUTPUT_CSV})")


if __name__ == "__main__":
    run_turbo_search()
