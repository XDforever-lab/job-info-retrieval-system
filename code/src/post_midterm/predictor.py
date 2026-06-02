"""
分类预测接口 —— 第九阶段 #197-#199

封装朴素贝叶斯分类器的加载与在线预测

用法:
    from src.post_midterm.predictor import IndustryPredictor
    predictor = IndustryPredictor()
    predictor.load()
    result = predictor.predict("职位描述文本...")

测试:
    python src/post_midterm/predictor.py
"""

import json
import math
import os
import sys

import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_DIR, SEGMENTED_CSV

CLASSIFIER_FILE = os.path.join(MODEL_DIR, 'nb_classifier.joblib')
VOCAB_FILE = os.path.join(MODEL_DIR, 'classifier_vocab.json')

# 9 大行业标签
INDUSTRY_NAMES = [
    '互联网/软硬件技术',
    '销售/市场/客服',
    '人力/财务/行政',
    '生产制造/汽车',
    '金融',
    '房地产/建筑',
    '物流/贸易/采购',
    '医疗/能源/环保/农业',
    '文化传媒/教育/生活服务',
]


class IndustryPredictor:
    """朴素贝叶斯行业分类器"""

    def __init__(self):
        self.model = None
        self.vocab = None
        self.stopwords = set()

    def load(self):
        """加载分类模型和词汇表"""
        print("[Predictor] 加载分类模型...")
        self.model = joblib.load(CLASSIFIER_FILE)

        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # 加载停用词
        stopwords_file = os.path.join(MODEL_DIR, '..', 'output', 'stopwords.txt')
        stopwords_file = os.path.normpath(stopwords_file)
        if os.path.exists(stopwords_file):
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())

        print(f"[Predictor] 模型加载完成, {len(self.vocab.get('feature_names', []))} 个特征词")
        return self

    def predict(self, text):
        """
        对输入文本进行行业分类预测

        Args:
            text: 职位描述原文 (未分词)

        Returns:
            {
                'label': str,        # 预测行业名
                'label_id': int,     # 行业ID
                'confidence': float, # 置信度 0~1
                'top3': [(行业名, 概率), ...]
            }
        """
        import jieba

        if not text:
            return {'label': '未知', 'label_id': -1, 'confidence': 0, 'top3': []}

        # 1. 分词 + 去停用词
        words = [w.strip() for w in jieba.cut(text)
                 if w.strip() and len(w.strip()) >= 2 and w.strip() not in self.stopwords]

        # 2. 构建特征向量 (词袋)
        feature_names = self.vocab.get('feature_names', [])
        vec = np.zeros(len(feature_names))
        for w in words:
            if w in self.vocab.get('word_to_idx', {}):
                vec[self.vocab['word_to_idx'][w]] += 1

        # 3. 调用模型预测
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(vec.reshape(1, -1))[0]
        else:
            # 手写 NB 模型 (dict 格式)
            probs = self._handwritten_predict(vec, words)

        # 4. Top-3
        top3_idx = np.argsort(-probs)[:3]
        top3 = []
        for idx in top3_idx:
            label_id = int(idx)
            label_name = INDUSTRY_NAMES[label_id] if label_id < len(INDUSTRY_NAMES) else f'类别{label_id}'
            top3.append((label_name, round(float(probs[idx]), 4)))

        best_label = top3[0][0] if top3 else '未知'
        best_conf = top3[0][1] if top3 else 0
        best_id = int(np.argmax(probs))

        return {
            'label': best_label,
            'label_id': best_id,
            'confidence': best_conf,
            'top3': top3,
        }

    def _handwritten_predict(self, vec, words):
        """手写 NB 模型预测 (兼容 joblib 保存格式)"""
        n_classes = 9
        probs = np.ones(n_classes)
        # 如果模型是 defaultdict -> 展开
        class_log_prior = self.model.get('class_log_prior', np.zeros(n_classes))
        feature_log_prob = self.model.get('feature_log_prob', np.zeros((n_classes, len(self.vocab.get('feature_names', [])))))

        for c in range(n_classes):
            score = class_log_prior[c]
            for w in words:
                w_idx = self.vocab.get('word_to_idx', {}).get(w, -1)
                if w_idx >= 0:
                    score += feature_log_prob[c][w_idx]
            probs[c] = math.exp(score)

        # 归一化
        total = probs.sum()
        if total > 0:
            probs /= total
        return probs


# ── 测试 ──────────────────────────────────────────

def main():
    predictor = IndustryPredictor()
    predictor.load()

    test_texts = [
        "负责嵌入式软件开发与调试，熟悉C语言与RTOS系统，参与物联网产品研发",
        "负责公司财务报表编制，税务申报，预算管理和成本核算工作",
        "负责电商平台运营，直播带货，短视频内容策划与私域流量运营",
    ]

    for i, text in enumerate(test_texts):
        result = predictor.predict(text)
        print(f"\n--- 测试 {i+1} ---")
        print(f"原文: {text[:60]}...")
        print(f"预测行业: {result['label']} (置信度: {result['confidence']:.4f})")
        print(f"Top-3: {result['top3']}")


if __name__ == '__main__':
    main()
