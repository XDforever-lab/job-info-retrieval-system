"""
分类模型持久化与预测接口 —— To Do List #139 ~ #140

    #139: 在全量标注数据上训练最优 NB 模型 + 保存 joblib
    #140: 编写 predict_industry(text) 在线预测接口

用法:
    conda activate my_project
    python src/15_save_model_and_predict.py
"""

import json
import joblib
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SEGMENTED_CSV, DIR_12_CLASSIFY, DIR_13_CLASSIFY, MODEL_DIR

ANNO_FILE = os.path.join(DIR_12_CLASSIFY, "annotation_9cats.txt")
CAT_DEF_JSON = os.path.join(DIR_12_CLASSIFY, "category_definitions.json")
MODEL_FILE = os.path.join(MODEL_DIR, "nb_classifier.joblib")
VOCAB_FILE = os.path.join(MODEL_DIR, "classifier_vocab.json")

os.makedirs(MODEL_DIR, exist_ok=True)


class MultinomialNB:
    """手写多项式朴素贝叶斯 (对数空间)"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_prior = {}
        self.log_likelihood = {}  # {class_id: {word: log_prob}}
        self.vocab = set()
        self.classes_ = []
        self.cat_list = []

    def fit(self, docs, labels):
        n_docs = len(docs)
        self.classes_ = sorted(set(labels))
        self.vocab = set().union(*[set(d.keys()) for d in docs])

        for c in self.classes_:
            c_docs = [d for d, l in zip(docs, labels) if l == c]
            self.log_prior[c] = math.log(len(c_docs) / n_docs)
            self.log_likelihood[c] = {}  # 初始化每个类的词概率表
            word_count = defaultdict(float)
            total = 0
            for d in c_docs:
                for w in d:
                    word_count[w] += 1
                    total += 1
            for w in self.vocab:
                self.log_likelihood[c][w] = math.log(
                    (word_count.get(w, 0) + self.alpha) / (total + self.alpha * len(self.vocab))
                )

    def predict(self, doc):
        best_c, best_score = None, float("-inf")
        for c in self.classes_:
            score = self.log_prior[c]
            for w in doc:
                score += self.log_likelihood[c].get(w, math.log(self.alpha / (self.alpha * len(self.vocab))))
            if score > best_score:
                best_score, best_c = score, c
        return best_c

    def predict_proba(self, doc):
        """返回各类别对数概率（用于置信度计算）"""
        scores = {}
        for c in self.classes_:
            score = self.log_prior[c]
            for w in doc:
                score += self.log_likelihood[c].get(w, math.log(self.alpha / (self.alpha * len(self.vocab))))
            scores[c] = score
        return scores


# ── 公共预测接口 (#140) ──

_the_model = None
_the_cats = None
_the_df_dict = None
_the_N = None
_stopwords = None

# 从分词CSV加载停用词
def _load_stopwords():
    global _stopwords
    if _stopwords is None:
        sw_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "output", "stopwords.txt")
        if os.path.exists(sw_file):
            with open(sw_file, encoding="utf-8") as f:
                _stopwords = set(w.strip() for w in f if w.strip() and not w.startswith("#"))
        else:
            _stopwords = set()


def _tokenize(text):
    """对输入文本分词 (使用 CRF 分词结果中的词袋逻辑)"""
    if not text or not isinstance(text, str):
        return []
    _load_stopwords()
    tokens = []
    # 简单 jieba 分词作为 fallback
    try:
        import jieba
        words = list(jieba.cut(text))
    except Exception:
        words = text.split()
    for w in words:
        w = w.strip()
        if not w or w in _stopwords:
            continue
        if re.search(r'[，。；：、（）【】\-;:,.!?\d]', w):
            continue
        tokens.append(w)
    return tokens


def _doc_to_vec(tokens, df_dict, N, top_k=200):
    """将词列表转为 TF-IDF 稀疏向量字典"""
    tf = defaultdict(float)
    for w in tokens:
        tf[w] += 1.0
    scored = [(w, t * math.log(N / (df_dict.get(w, 0) + 1))) for w, t in tf.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {w: s for w, s in scored[:top_k]}


def load_model():
    """加载已保存的分类模型 (系统启动时调用一次)"""
    global _the_model, _the_cats, _the_df_dict, _the_N
    _the_model = joblib.load(MODEL_FILE)
    with open(CAT_DEF_JSON, encoding="utf-8") as f:
        _the_cats = list(json.load(f).keys())
    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab_meta = json.load(f)
        _the_df_dict = defaultdict(int, vocab_meta["df_dict"])
        _the_N = vocab_meta["N"]
    print(f"[分类器] 已加载: {_the_N} 文档, {len(_the_df_dict)} 词汇")


def predict_industry(text):
    """#140: 在线预测接口 — 输入职位描述文本, 返回 (行业名, 置信度)"""
    if _the_model is None:
        raise RuntimeError("模型未加载，请先调用 load_model()")
    tokens = _tokenize(text)
    vec = _doc_to_vec(tokens, _the_df_dict, _the_N)
    label_id = _the_model.predict(vec)
    proba = _the_model.predict_proba(vec)
    # 置信度 = softmax of log probabilities
    log_probs = np.array(list(proba.values()))
    log_probs = log_probs - np.max(log_probs)  # 数值稳定
    probs = np.exp(log_probs)
    probs = probs / probs.sum()
    confidence = float(probs[list(proba.keys()).index(label_id)])
    return _the_cats[label_id], round(confidence, 4)


# ── #139: 训练 + 保存模型 ──

def train_and_save():
    print("=" * 60)
    print("  分类模型持久化 #139 ~ #140")
    print("=" * 60)

    # 解析标注
    with open(ANNO_FILE, encoding="utf-8") as f:
        content = f.read()
    with open(CAT_DEF_JSON, encoding="utf-8") as f:
        cats = json.load(f)
    cat_list = list(cats.keys())

    labels = {}
    for m in re.finditer(r"### 样本 (\d+).*?【标注】:\s*(\d+)", content, re.DOTALL):
        idx = int(m.group(1)) - 1
        label = int(m.group(2)) - 1
        if 0 <= label < 9:
            labels[idx] = label

    # 加载分词数据
    df = pd.read_csv(SEGMENTED_CSV, encoding="utf-8-sig", engine="python")
    sample_df = pd.read_csv(os.path.join(DIR_12_CLASSIFY, "sample_400.csv"), encoding="utf-8-sig")

    docs = []
    for i in range(len(sample_df)):
        seg = str(df.iloc[i].get("职位描述_分词", ""))
        title = str(df.iloc[i].get("招聘岗位_分词", ""))
        tokens = [w.strip() for w in (title + "/" + seg).split("/")
                  if w.strip() and w != "nan" and not re.search(r'[，。；：、（）【】\-;:,.!?\d]', w.strip())]
        docs.append(tokens)

    # 全量 400 条训练 (#139)
    print(f"\n[#139] 用全量 {len(docs)} 条训练最优 NB 模型...")
    N = len(docs)
    df_dict = defaultdict(int)
    for d in docs:
        for w in set(d):
            df_dict[w] += 1

    def to_vec(doc_words):
        tf = defaultdict(float)
        for w in doc_words:
            tf[w] += 1.0
        scored = [(w, t * math.log(N / (df_dict.get(w, 0) + 1))) for w, t in tf.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return {w: s for w, s in scored[:200]}

    X = [to_vec(docs[i]) for i in labels.keys()]
    y = [labels[i] for i in labels.keys()]

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X, y)
    nb.cat_list = cat_list

    # 保存模型
    joblib.dump(nb, MODEL_FILE)
    print(f"  模型已保存: {MODEL_FILE}")

    # 保存词汇信息
    vocab_meta = {
        "df_dict": dict(df_dict),
        "N": N,
        "cat_list": cat_list,
    }
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, ensure_ascii=False)
    print(f"  词汇表已保存: {VOCAB_FILE}")

    # #140: 测试预测接口
    print(f"\n[#140] 测试预测接口...")
    load_model()
    test_texts = [
        "负责嵌入式软件开发和调试，熟悉C语言和RTOS系统",
        "负责银行信贷风险模型的开发与验证",
        "负责新能源汽车电池包的结构设计",
    ]
    for t in test_texts:
        result, conf = predict_industry(t)
        print(f"  '{t[:40]}...' → {result} (置信度: {conf:.2%})")

    print(f"\n{'=' * 60}")
    print(f"  #139 ~ #140 完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_and_save()
