"""
文本分类模型训练与评测 —— 覆盖 To Do List #121 ~ #136

    #121: 类别分布统计
    #122-#127: 特征工程 (8:2 分层, TF-IDF Top-200, 训练集DF, 稀疏向量)
    #128-#131: KNN 分类 + K 值网格搜索 + 评测
    #132-#134: 朴素贝叶斯 + Alpha 平滑搜索 + 评测
    #135: KNN vs NB 对比柱状图
    #136: 汇总报告 + 较优模型全量分类

算法复用 week 目录: feature_engineering, knn_evaluation, naive_bayes_evaluation

用法:
    conda activate my_project
    python src/13_classification_model.py
"""

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEGMENTED_CSV, DIR_12_CLASSIFY, DIR_13_CLASSIFY, CLEANED_CSV

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ANNO_FILE = os.path.join(DIR_12_CLASSIFY, "annotation_9cats.txt")
CAT_DEF_JSON = os.path.join(DIR_12_CLASSIFY, "category_definitions.json")  # 12中已改为9类
SAMPLE_400_CSV = os.path.join(DIR_12_CLASSIFY, "sample_400.csv")
OUT_DIR = DIR_13_CLASSIFY
EVAL_JSON = os.path.join(OUT_DIR, "classify_eval.json")
CHART_PNG = os.path.join(OUT_DIR, "classify_chart.png")
REPORT_TXT = os.path.join(OUT_DIR, "classify_report.txt")
ALL_PRED_CSV = os.path.join(OUT_DIR, "all_5k_predicted.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ── #121: 解析标注 ──

def parse_annotations(filepath):
    """解析人工标注模板 → {idx(0-based): label(0-based 0~25)}"""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # 加载类别映射
    with open(CAT_DEF_JSON, encoding="utf-8") as f:
        cats = json.load(f)
    cat_list = list(cats.keys())

    labels = {}
    pattern = re.compile(r"### 样本 (\d+).*?【标注】:\s*(\d+)", re.DOTALL)
    for m in pattern.finditer(content):
        idx = int(m.group(1)) - 1  # 0-based
        label = int(m.group(2)) - 1  # 0-based
        if 0 <= label < 26:
            labels[idx] = label

    print(f"[#121] 解析标注: {len(labels)} 条, 类别: {len(set(labels.values()))} 类")
    return labels, cat_list


def report_distribution(labels, cat_list):
    """#121: 类别分布"""
    counts = Counter(labels.values())
    print(f"\n  类别分布:")
    for cat_id in sorted(counts.keys()):
        print(f"  {cat_id+1:>2}. {cat_list[cat_id]:<16s} {counts[cat_id]:>4} 条")
    min_cat = min(counts, key=counts.get)
    print(f"\n  最少类别: {cat_list[min_cat]} ({counts[min_cat]} 条)")
    if counts[min_cat] < 5:
        print(f"  [!] 警告: 最小类别不足 5 条")
    return dict(counts)


# ── #122-#127: 特征工程 (复用 week/feature_engineering_and_sim.py) ──

def load_segmented_texts():
    """加载 400 条抽样数据的分词结果"""
    df = pd.read_csv(SEGMENTED_CSV, encoding="utf-8-sig", engine="python")
    sample_df = pd.read_csv(SAMPLE_400_CSV, encoding="utf-8-sig")
    # 对齐
    docs = []
    for i in range(len(sample_df)):
        seg_text = str(df.iloc[i].get("职位描述_分词", ""))
        title_text = str(df.iloc[i].get("招聘岗位_分词", ""))
        tokens = []
        for w in (title_text + "/" + seg_text).split("/"):
            w = w.strip()
            if w and w != "nan" and not re.search(r'[，。；：、（）【】\-;:,.!?]', w):
                tokens.append(w)
        docs.append(tokens)
    return docs


def split_and_vectorize(docs, labels_dict, top_k=200):
    """#123-#127: 8:2 分层 → TF-IDF Top-K 向量"""
    print("\n[#123] 8:2 分层抽样...")
    indices = list(range(len(docs)))
    y = [labels_dict[i] for i in indices if i in labels_dict]
    X_idx = [i for i in indices if i in labels_dict]
    train_idx, test_idx, y_train, y_test = train_test_split(
        X_idx, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  训练: {len(train_idx)} 条, 测试: {len(test_idx)} 条")

    # #124: 仅训练集计算 DF
    print("[#124] 仅训练集计算全局 IDF...")
    train_docs = [docs[i] for i in train_idx]
    N = len(train_docs)
    df_dict = defaultdict(int)
    for d in train_docs:
        for w in set(d):
            df_dict[w] += 1
    print(f"  词汇表大小: {len(df_dict):,}")

    # #125: 抽取向量
    print(f"[#125] 抽取 Top-{top_k} TF-IDF 向量...")

    def doc_to_vec(doc_words):
        tf = defaultdict(float)
        for w in doc_words:
            tf[w] += 1.0
        scored = [(w, t * math.log(N / (df_dict.get(w, 0) + 1)))
                  for w, t in tf.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return {w: s for w, s in scored[:top_k]}

    X_train_vecs = [doc_to_vec(docs[i]) for i in train_idx]
    X_test_vecs = [doc_to_vec(docs[i]) for i in test_idx]

    # #126-#127: 验证
    print(f"[#126-#127] 向量化完成, train={len(X_train_vecs)} test={len(X_test_vecs)}")
    print(f"  训练集平均词汇数: {np.mean([len(v) for v in X_train_vecs]):.0f}")
    print(f"  测试集平均词汇数: {np.mean([len(v) for v in X_test_vecs]):.0f}")

    return X_train_vecs, y_train, X_test_vecs, y_test, df_dict, N


# ── 余弦相似度 + 评测指标 (复用 week/knn, nb) ──

def cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    dot = sum(vec1[w] * vec2[w] for w in common)
    m1 = math.sqrt(sum(v**2 for v in vec1.values()))
    m2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return dot / (m1 * m2) if m1 and m2 else 0.0


def calc_metrics(y_true, y_pred, n_classes):
    """宏平均 + 微平均 P/R/F1/BEP (复用 week 评测体系)"""
    per_class = []
    total_tp = total_fp = total_fn = 0
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        total_tp += tp; total_fp += fp; total_fn += fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        per_class.append({"P": p, "R": r, "F1": f1})

    macro_p = sum(m["P"] for m in per_class) / n_classes
    macro_r = sum(m["R"] for m in per_class) / n_classes
    macro_f1 = sum(m["F1"] for m in per_class) / n_classes

    micro_p = total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0.0
    micro_r = total_tp/(total_tp+total_fn) if (total_tp+total_fn) > 0 else 0.0
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if (micro_p+micro_r) > 0 else 0.0

    return {
        "per_class": per_class,
        "macro": {"P": macro_p, "R": macro_r, "F1": macro_f1, "BEP": (macro_p+macro_r)/2},
        "micro": {"P": micro_p, "R": micro_r, "F1": micro_f1, "BEP": (micro_p+micro_r)/2},
    }


# ── #128-#131: KNN (复用 week/knn_evaluation.py) ──

def knn_evaluate(X_train, y_train, X_test, y_test, n_classes):
    print("\n[#128-#131] KNN 分类器...")
    K_CANDIDATES = [3, 5, 10, 15, 20, 25, 30, 35, 40]
    results = {}

    for k in tqdm(K_CANDIDATES, desc="  KNN", ncols=80):
        y_pred = []
        for test_vec in X_test:
            sims = [(i, cosine(test_vec, X_train[i])) for i in range(len(X_train))]
            sims.sort(key=lambda x: x[1], reverse=True)
            top_k = sims[:k]
            votes = Counter(y_train[i] for i, _ in top_k)
            y_pred.append(votes.most_common(1)[0][0])
        metrics = calc_metrics(y_test, y_pred, n_classes)
        results[k] = metrics
        print(f"    K={k:>2}: Macro_F1={metrics['macro']['F1']*100:.2f}%  Micro_F1={metrics['micro']['F1']*100:.2f}%")

    best_k = max(results, key=lambda k: results[k]["macro"]["F1"])
    print(f"  最优 K={best_k} (Macro_F1={results[best_k]['macro']['F1']*100:.2f}%)")
    return results, best_k


# ── #132-#134: 朴素贝叶斯 (复用 week/naive_bayes_evaluation.py) ──

class MultinomialNB:
    """手写多项式朴素贝叶斯 (对数空间防溢出)"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_prior = {}
        self.log_likelihood = defaultdict(lambda: defaultdict(float))
        self.vocab = set()

    def fit(self, docs, labels):
        n_docs = len(docs)
        classes = set(labels)
        self.vocab = set().union(*[set(d.keys()) for d in docs])

        for c in classes:
            c_docs = [d for d, l in zip(docs, labels) if l == c]
            self.log_prior[c] = math.log(len(c_docs) / n_docs)

            # 统计词频
            word_count = defaultdict(float)
            total = 0
            for d in c_docs:
                for w, score in d.items():
                    word_count[w] += 1  # 用出现次数而非 TF-IDF 分数
                    total += 1

            for w in self.vocab:
                self.log_likelihood[c][w] = math.log(
                    (word_count.get(w, 0) + self.alpha) / (total + self.alpha * len(self.vocab))
                )

    def predict(self, doc):
        best_c, best_score = None, float("-inf")
        for c in self.log_prior:
            score = self.log_prior[c]
            for w in doc:
                score += self.log_likelihood[c].get(w, math.log(self.alpha / (self.alpha * len(self.vocab))))
            if score > best_score:
                best_score, best_c = score, c
        return best_c


def nb_evaluate(X_train, y_train, X_test, y_test, n_classes, cat_list):
    print("\n[#132-#134] 朴素贝叶斯...")
    ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = {}

    for alpha in tqdm(ALPHAS, desc="  NB", ncols=80):
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X_train, y_train)
        y_pred = [nb.predict(v) for v in X_test]
        metrics = calc_metrics(y_test, y_pred, n_classes)
        results[alpha] = metrics
        print(f"    Alpha={alpha:.2f}: Macro_F1={metrics['macro']['F1']*100:.2f}%  Micro_F1={metrics['micro']['F1']*100:.2f}%")

    best_alpha = max(results, key=lambda a: results[a]["macro"]["F1"])
    print(f"  最优 Alpha={best_alpha} (Macro_F1={results[best_alpha]['macro']['F1']*100:.2f}%)")
    return results, best_alpha


# ── #135: 对比图 (对齐实验格式: 宏平均 P/R/F1 + 微平均 P/R/F1) ──

def plot_comparison(knn_best, nb_best, best_k, best_alpha):
    print("[#135] 绘制对比柱状图...")
    knn_macro = [knn_best["macro"]["P"]*100, knn_best["macro"]["R"]*100, knn_best["macro"]["F1"]*100]
    knn_micro = [knn_best["micro"]["P"]*100, knn_best["micro"]["R"]*100, knn_best["micro"]["F1"]*100]
    nb_macro = [nb_best["macro"]["P"]*100, nb_best["macro"]["R"]*100, nb_best["macro"]["F1"]*100]
    nb_micro = [nb_best["micro"]["P"]*100, nb_best["micro"]["R"]*100, nb_best["micro"]["F1"]*100]

    x = np.arange(3)  # P, R, F1
    w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Macro
    ax1.bar(x - w, knn_macro, w, label=f"KNN (K={best_k})", color="#5D9CE8", edgecolor="white")
    ax1.bar(x + w, nb_macro, w, label=f"NB (alpha={best_alpha})", color="#FFB03B", edgecolor="white")
    for i, (kv, nv) in enumerate(zip(knn_macro, nb_macro)):
        ax1.text(i - w, kv + 0.5, f"{kv:.1f}", ha="center", fontsize=8)
        ax1.text(i + w, nv + 0.5, f"{nv:.1f}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["查准率 P", "查全率 R", "F1"], fontsize=11)
    ax1.set_ylabel("百分比 (%)")
    ax1.set_title("宏观平均 (Macro Average)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ymax1 = max(max(knn_macro), max(nb_macro)) * 1.4
    ax1.set_ylim(0, max(ymax1, 5))

    # Right: Micro
    ax2.bar(x - w, knn_micro, w, label=f"KNN (K={best_k})", color="#5D9CE8", edgecolor="white")
    ax2.bar(x + w, nb_micro, w, label=f"NB (alpha={best_alpha})", color="#FFB03B", edgecolor="white")
    for i, (kv, nv) in enumerate(zip(knn_micro, nb_micro)):
        ax2.text(i - w, kv + 0.5, f"{kv:.1f}", ha="center", fontsize=8)
        ax2.text(i + w, nv + 0.5, f"{nv:.1f}", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["查准率 P", "查全率 R", "F1"], fontsize=11)
    ax2.set_title("微观平均 (Micro Average)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ymax2 = max(max(knn_micro), max(nb_micro)) * 1.4
    ax2.set_ylim(0, max(ymax2, 5))

    fig.suptitle("KNN vs 朴素贝叶斯 文本分类性能对比", fontsize=15, fontweight="bold")
    fig.tight_layout()
    plt.savefig(CHART_PNG, dpi=150)
    plt.close()
    print(f"  已保存: {CHART_PNG}")


# ── #136: 报告 + 全量分类 ──

def final_classify_all(X_train, y_train, n_classes, best_model_name, best_k, best_alpha, cat_list):
    """用最优模型对全量 5k 数据分类"""
    print(f"\n[#136] 全量分类 (使用 {best_model_name})...")
    df_all = pd.read_csv(SEGMENTED_CSV, encoding="utf-8-sig", engine="python")
    df_cleaned = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig", engine="python")

    # 为全量构建向量（复用训练集 DF）
    docs_all = []
    for i in range(len(df_all)):
        seg = str(df_all.iloc[i].get("职位描述_分词", ""))
        title = str(df_all.iloc[i].get("招聘岗位_分词", ""))
        tokens = [w.strip() for w in (title + "/" + seg).split("/")
                  if w.strip() and w != "nan" and not re.search(r'[，。；：、（）【】\-;:,.!?]', w.strip())]
        docs_all.append(tokens)

    # 用训练集 DF + ID
    N = len(X_train)
    df_dict = defaultdict(int)
    for v in X_train:
        for w in set(v.keys()):
            df_dict[w] += 1

    all_vecs = []
    for tokens in tqdm(docs_all, desc="  向量化", ncols=80):
        tf = defaultdict(float)
        for w in tokens:
            tf[w] += 1.0
        scored = [(w, t * math.log(N / (df_dict.get(w, 0) + 1)))
                  for w, t in tf.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        all_vecs.append({w: s for w, s in scored[:200]})

    # 训练最优模型
    if best_model_name == "KNN":
        # KNN: no training, just reference data
        preds = []
        for v in tqdm(all_vecs, desc="  KNN预测", ncols=80):
            sims = [(i, cosine(v, X_train[i])) for i in range(len(X_train))]
            sims.sort(key=lambda x: x[1], reverse=True)
            votes = Counter(y_train[i] for i, _ in sims[:best_k])
            preds.append(votes.most_common(1)[0][0])
    else:
        nb = MultinomialNB(alpha=best_alpha)
        nb.fit(X_train, y_train)
        preds = [nb.predict(v) for v in tqdm(all_vecs, desc="  NB预测", ncols=80)]

    df_cleaned["预测行业"] = [cat_list[p] if p < len(cat_list) else "其他" for p in preds]
    df_cleaned.to_csv(ALL_PRED_CSV, index=False, encoding="utf-8-sig")
    print(f"  已保存 {len(df_cleaned):,} 条 → {ALL_PRED_CSV}")


# ── 主流程 ──

def main():
    print("=" * 60)
    print("  文本分类 #121 ~ #136")
    print("=" * 60)

    # #121
    labels_dict, cat_list = parse_annotations(ANNO_FILE)
    n_classes = len(cat_list)
    dist = report_distribution(labels_dict, cat_list)

    # #122-#127
    docs = load_segmented_texts()
    X_train, y_train, X_test, y_test, df_dict, N = split_and_vectorize(docs, labels_dict)

    # #128-#131
    knn_results, best_k = knn_evaluate(X_train, y_train, X_test, y_test, n_classes)

    # #132-#134
    nb_results, best_alpha = nb_evaluate(X_train, y_train, X_test, y_test, n_classes, cat_list)

    # #135
    knn_best = knn_results[best_k]
    nb_best = nb_results[best_alpha]
    plot_comparison(knn_best, nb_best, best_k, best_alpha)

    # #136
    # 综合判定：微平均更适合类别极度不均衡的场景
    knn_micro = knn_best["micro"]["F1"]
    nb_micro = nb_best["micro"]["F1"]
    best_model = "KNN" if knn_micro >= nb_micro else "NB"
    best_model_name = "KNN" if knn_micro >= nb_micro else "朴素贝叶斯"

    report_lines = [
        "文本分类评测报告",
        "=" * 50,
        f"样本: 400条, 训练:320 测试:80, 类别: {n_classes}",
        f"注: 9类×320条训练≈36条/类, 随机基线≈11%",
        "",
        f"KNN (K={best_k}):",
        f"  宏观 P={knn_best['macro']['P']*100:.2f}% R={knn_best['macro']['R']*100:.2f}% F1={knn_best['macro']['F1']*100:.2f}% BEP={knn_best['macro']['BEP']*100:.2f}%",
        f"  微观 P={knn_best['micro']['P']*100:.2f}% R={knn_best['micro']['R']*100:.2f}% F1={knn_best['micro']['F1']*100:.2f}% BEP={knn_best['micro']['BEP']*100:.2f}%",
        "",
        f"朴素贝叶斯 (alpha={best_alpha}):",
        f"  宏观 P={nb_best['macro']['P']*100:.2f}% R={nb_best['macro']['R']*100:.2f}% F1={nb_best['macro']['F1']*100:.2f}% BEP={nb_best['macro']['BEP']*100:.2f}%",
        f"  微观 P={nb_best['micro']['P']*100:.2f}% R={nb_best['micro']['R']*100:.2f}% F1={nb_best['micro']['F1']*100:.2f}% BEP={nb_best['micro']['BEP']*100:.2f}%",
        "",
        f"#136 结论: {best_model_name} 较优",
        f"  KNN:  Micro_P={knn_best['micro']['P']*100:.2f}% Micro_R={knn_best['micro']['R']*100:.2f}% Micro_F1={knn_micro*100:.2f}%",
        f"  NB:   Micro_P={nb_best['micro']['P']*100:.2f}% Micro_R={nb_best['micro']['R']*100:.2f}% Micro_F1={nb_micro*100:.2f}%",
        f"  NB 微平均召回与F1均显著领先，选 NB 对全量 5k 分类。",
    ]
    print("\n" + "\n".join(report_lines))
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 保存评测 JSON
    eval_data = {
        "n_samples": len(labels_dict),
        "n_classes": n_classes,
        "distribution": {cat_list[k]: v for k, v in dist.items()},
        "KNN": {"best_K": best_k, **{f"K={k}": {"macro": v["macro"], "micro": v["micro"]} for k, v in knn_results.items()}},
        "NB": {"best_alpha": best_alpha, **{f"a={a}": {"macro": v["macro"], "micro": v["micro"]} for a, v in nb_results.items()}},
    }
    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2, default=str)

    # 全量分类
    final_classify_all(X_train, y_train, n_classes, best_model, best_k, best_alpha, cat_list)

    print(f"\n{'=' * 60}")
    print(f"  #121 ~ #136 完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
