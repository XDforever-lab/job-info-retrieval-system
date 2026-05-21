"""项目全局配置"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_CSV = os.path.join(DATA_DIR, '上市公司招聘数据2026.csv')
PROCESSED_CSV = os.path.join(DATA_DIR, '上市公司招聘数据_processed.csv')
SEGMENTED_CSV = os.path.join(DATA_DIR, '上市公司招聘数据_segmented.csv')

# 模型路径
MODEL_DIR = os.path.join(BASE_DIR, 'models')
INDEX_FILE = os.path.join(MODEL_DIR, 'inverted_index.pkl')
TFIDF_MATRIX = os.path.join(MODEL_DIR, 'tfidf_matrix.npz')
VOCAB_MAP = os.path.join(MODEL_DIR, 'vocab_map.json')
CLASSIFIER_MODEL = os.path.join(MODEL_DIR, 'classifier.joblib')
CLUSTER_MODEL = os.path.join(MODEL_DIR, 'cluster.joblib')
KEYWORDS_FILE = os.path.join(MODEL_DIR, 'keywords.json')

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
STOPWORDS_FILE = os.path.join(OUTPUT_DIR, 'stopwords.txt')
NGRAM_REPORT = os.path.join(OUTPUT_DIR, 'ngram_report.json')
STATS_SUMMARY = os.path.join(OUTPUT_DIR, 'stats_summary.json')
CLUSTER_VIS_DATA = os.path.join(OUTPUT_DIR, 'cluster_vis.json')
EVAL_REPORT = os.path.join(OUTPUT_DIR, 'eval_report.json')

# 检索参数
SEARCH_PAGE_SIZE = 20
TITLE_WEIGHT = 2.0       # 标题字段权重加成
CONTENT_WEIGHT = 1.0     # 正文字段权重
COMPANY_BONUS = 0.1      # 企业名匹配加分
CACHE_TTL = 300          # 检索缓存有效期（秒）

# 分类与聚类参数
CLASSIFICATION_K = 30    # KNN 的 K 值
CLUSTER_N_CLUSTERS = 5   # 聚类簇数
LDA_N_TOPICS = 10        # LDA 主题数
TEXTRANK_WINDOW = 5      # TextRank 共现窗口
TEXTRANK_DAMPING = 0.85  # TextRank 阻尼系数

# Jieba 分词配置
JIEBA_DICT = None         # 自定义词典路径（None 则用默认）
